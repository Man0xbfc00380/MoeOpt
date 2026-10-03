#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TaskEmulator: lightweight generator of MoE-style workloads.

Responsibilities:
- Create initial token blocks (per-request tokens) and expert blocks (per-expert micro-slices)
- Generate binding relations mapping expert -> a set of token block names it consumes
- Provide a stable expert-loading order and a windowed loader API

Behavioral notes:
- Random is seeded for determinism across runs
- Binding sizes are skewed by an exponential decay to emulate popularity differences
- A threshold controls which experts participate in the loading order
"""

from hardware_config import HardwareConfig
from data_block import DataBlock, DataBlockType
import random
from typing import Dict, List, Set


class TaskEmulator:
    """Synthetic task/workload generator for the scheduler.

    This class intentionally focuses on data shape and ordering rather than
    algorithmic fidelity to any particular production MoE system. It supplies
    the scheduler with a stream of token and expert data blocks and a
    precomputed mapping of which tokens are relevant to which expert.
    """

    def __init__(self):
        # Fix random seed
        random.seed(2025)

        # Number of requests equals number of chips
        self.request_num = HardwareConfig.column_number_of_chip * \
            HardwareConfig.row_number_of_chip
        # Reserve two buffer slots so chips have headroom for in-flight movements
        self.token_num_per_request = HardwareConfig.token_buffer_size - 2
        # All token blocks keyed by name, e.g., "R1-T7"
        self.token_blocks: Dict[str, DataBlock] = {}

        # Expert space
        self.expert_number: int = 4
        self.micro_slice_num_per_expert: int = 16
        # All expert blocks keyed by name, e.g., "E2-M5"
        self.expert_blocks: Dict[str, DataBlock] = {}

        # Binding relationships for computation:
        # {expert_id: {token_block_name1, token_block_name2}}
        # where expert_id == DataBlock.seq_or_expert_id for expert blocks
        self.min_activated_token = HardwareConfig.column_number_of_chip * \
            HardwareConfig.row_number_of_chip  # Minimum token activation threshold for expert loading
        self.bind_relations: Dict[int, Set[str]] = {}

        # Expert load management
        # Prefer experts with token count strictly greater than this threshold
        # Keep the original misspelled public attribute for backward compatibility.
        self.token_weight_threshold: int = (
            HardwareConfig.column_number_of_chip * HardwareConfig.row_number_of_chip - 1
        )
        self.sorted_expert_blocks: List[DataBlock] = []
        self.current_load_expert: int = 0  # Load pointer into sorted_expert_blocks

        # 是否启用expert load排序的标志
        self.enable_expert_load_sort: bool = True

        # Dynamic token management
        self.current_alive_token: List[DataBlock] = []

    # Backward-compatibility shim for the historical misspelling
    @property
    def token_weight_threshould(self) -> int:  # type: ignore[override]
        """Alias to token_weight_threshold (kept to avoid breaking demos)."""
        return self.token_weight_threshold

    @token_weight_threshould.setter
    def token_weight_threshould(self, value: int) -> None:  # type: ignore[override]
        self.token_weight_threshold = int(value)

    def init_data(self) -> None:
        """Populate tokens, experts, relations and precompute load order."""
        self._init_token_blocks()
        self._init_expert_blocks()
        self._init_bind_relations()
        # 根据标志决定是否进行expert排序
        self.resort_expert_blocks()

    def _init_token_blocks(self) -> None:
        """Initialize all token data blocks across requests.

        Produces blocks named like "R{request}-T{token}" and stores them in
        self.token_blocks, also populating current_alive_token with the full set.
        """
        self.token_blocks = {}
        # Iterate through each request
        for r in range(1, self.request_num + 1):
            # Iterate through tokens in each request
            for t in range(1, self.token_num_per_request + 1):
                # Generate block name in format "R{request_id}-T{token_id}"
                block_name = f"R{r}-T{t}"
                # Create and store new data block
                data_block = DataBlock(
                    name=block_name,
                    block_type=DataBlockType.SEQ
                )
                data_block.seq_or_expert_id = r
                data_block.mico_slice = t
                self.token_blocks[block_name] = data_block

        # All tokens start as alive/available to schedule
        self.current_alive_token = list(self.token_blocks.values())

    def _init_expert_blocks(self) -> None:
        """Initialize all expert data blocks for each expert's micro-slices."""
        self.expert_blocks = {}
        # Iterate through each expert
        for e in range(1, self.expert_number + 1):
            # Iterate through micro slices of each expert
            for s in range(1, self.micro_slice_num_per_expert + 1):
                # Generate block name in format "E{expert_id}-M{slice_id}"
                block_name = f"E{e}-M{s}"
                # Create and store new data block
                data_block = DataBlock(
                    name=block_name,
                    block_type=DataBlockType.EXPERT
                )
                data_block.seq_or_expert_id = e
                data_block.mico_slice = s
                self.expert_blocks[block_name] = data_block

    def _init_bind_relations(self) -> None:
        """Initialize expert->token binding relationships.

        We draw different token-set sizes per expert following a simple
        exponential-decay curve to emulate load skew. Each expert receives a
        set of token block names selected uniformly at random from the pool of
        available tokens, subject to minimum activation.
        """
        self.bind_relations = {}

        chip_num = HardwareConfig.column_number_of_chip * HardwareConfig.row_number_of_chip

        # Calculate token count for each expert
        expert_token_counts = []
        max_tokens = 4 * chip_num
        for e in range(self.expert_number - 1):
            # Calculate token count using exponential decay
            token_count = int(max_tokens * (0.25 ** (e/self.expert_number)))
            token_count = max(token_count, self.min_activated_token)  # Ensure minimum token count
            expert_token_counts.append(token_count)

        # Set last expert to minimum token count
        expert_token_counts.append(self.min_activated_token)

        # Iterate through experts
        for e in range(1, self.expert_number + 1):
            token_count = expert_token_counts[e-1]
            # Randomly select tokens for binding
            available_tokens = [f"R{r}-T{t}"
                                for r in range(1, self.request_num + 1)
                                for t in range(1, self.token_num_per_request + 1)]
            # Guard: ensure we do not request more samples than available
            token_count = min(token_count, len(available_tokens))
            selected_tokens = set(random.sample(available_tokens, token_count))

            # Assign token set to expert
            self.bind_relations[e] = selected_tokens

    def resort_expert_blocks(self) -> None:
        """Resort expert blocks based on loading order.

        Expert load order is derived from token weights and then we produce the
        flattened sequence of expert micro-slice blocks in that order.
        """
        # Get expert loading order
        expert_order = self.get_expert_load_order()

        # Clear current sorted list
        self.sorted_expert_blocks = []
        self.current_load_expert = 0  # Reset load pointer

        # Sort by expert order and micro_slice number
        for expert_id in expert_order:
            # Get all micro_slice blocks for current expert
            expert_slices = [
                block for block in self.expert_blocks.values()
                if block.seq_or_expert_id == expert_id
            ]

            # Sort by micro_slice number
            expert_slices.sort(key=lambda x: x.mico_slice)

            # Add to sorted list
            self.sorted_expert_blocks.extend(expert_slices)

    # Allow external direct setting of some data blocks
    def set_token_block(self, request_id: int, token_num: int) -> None:
        """Set token blocks for a given request ID.

        Args:
            request_id: ID of the request (1-indexed).
            token_num: Number of tokens per request (unused here).
        """
        for t in range(1, self.token_num_per_request + 1):
            block_name = f"R{request_id}-T{t}"
            if block_name not in self.token_blocks:
                data_block = DataBlock(
                        name=block_name,
                        block_type=DataBlockType.SEQ
                    )
                data_block.seq_or_expert_id = request_id
                data_block.mico_slice = t
                self.token_blocks[block_name] = data_block
    
    def set_expert_block(self, expert_id: int, micro_slice_num: int) -> None:
        """Set expert blocks for a given expert ID.

        Args:
            expert_id: ID of the expert (1-indexed).
            micro_slice_num: Number of micro-slices per expert (unused here).
        """
        for s in range(1, self.micro_slice_num_per_expert + 1):
            block_name = f"E{expert_id}-M{s}"
            if block_name not in self.expert_blocks:
                data_block = DataBlock(
                        name=block_name,
                        block_type=DataBlockType.EXPERT
                    )
                data_block.seq_or_expert_id = expert_id
                data_block.mico_slice = s
                self.expert_blocks[block_name] = data_block
    
    def set_bind_relations(self, expert_id: int, token_block_names: Set[str]) -> None:
        """Allow external code to directly set the binding between a given expert and token blocks.

        Args:
            expert_id: Expert ID (1-based).
            token_block_names: Set of token block names to bind.
        """
        # Overwrite the binding relation for this expert
        self.bind_relations[expert_id] = token_block_names

    def set_bind_relations_verse(self, token_id: str, expert_ids: Set[int]) -> None:
        """Allow external code to directly set the binding between a given token and expert blocks.

        This method updates the bind_relations dictionary so that the specified token
        is associated with all given experts. Any previous bindings for this token
        are replaced.

        Args:
            token_id: The token block name (e.g., "R1-T7").
            expert_ids: Set of expert IDs (1-based) to which this token should bind.
        """
        # Ensure the token exists in token_blocks to avoid dangling bindings
        if token_id not in self.token_blocks:
            raise ValueError(f"Token block '{token_id}' does not exist.")

        # Clear existing bindings for this token across all experts
        for expert_id in list(self.bind_relations.keys()):
            self.bind_relations[expert_id].discard(token_id)

        # Add the token to each specified expert's binding set
        for expert_id in expert_ids:
            if expert_id not in self.bind_relations:
                self.bind_relations[expert_id] = set()
            self.bind_relations[expert_id].add(token_id)

    def get_expert_load_order(self) -> List[int]:
        """Compute the expert loading priority order.

        - First, count how many tokens bind to each expert
        - Keep only experts whose token count is strictly greater than the threshold
        - Sort descending by count
        - Interleave most/least to diversify scheduling pressure

        Returns:
            List[int]: Expert IDs by planned loading order.
        """
        # Count tokens bound to each expert
        expert_token_counts = {}
        for expert_id, tokens in self.bind_relations.items():
            expert_token_counts[expert_id] = len(tokens)

        # Filter experts above threshold
        qualified_experts = []
        for expert_id, count in expert_token_counts.items():
            if count > self.token_weight_threshold:
                qualified_experts.append((expert_id, count))

        if self.enable_expert_load_sort:
            # Sort experts by token count descending
            qualified_experts.sort(key=lambda x: x[1], reverse=True)
        else:
            qualified_experts.sort(key=lambda x: x[0])

        # Reorder: most tokens, least tokens, second most, second least...
        reordered_remaining = []
        if self.enable_expert_load_sort:
            while qualified_experts:
                # Add expert with most tokens
                if qualified_experts:
                    reordered_remaining.append(qualified_experts[0][0])
                    qualified_experts.pop(0)

                # Add expert with least tokens
                if qualified_experts:
                    reordered_remaining.append(qualified_experts[-1][0])
                    qualified_experts.pop()
        else:
            while qualified_experts:
                reordered_remaining.append(qualified_experts[0][0])
                qualified_experts.pop(0)
        return reordered_remaining

    def load_expert(self, load_num: int, cyclic_load: bool = False) -> List[DataBlock]:
        """Return a window of expert blocks from the precomputed order.

        Args:
            load_num: Number of expert micro-slice blocks to load.
            cyclic_load: If True, wrap-around when reaching the end.

        Returns:
            A list of DataBlock in load order.

        Raises:
            ValueError: if load_num <= 0 or (not cyclic_load and load_num exceeds remaining blocks).
        """
        total_experts = len(self.sorted_expert_blocks)

        if load_num <= 0:
            raise ValueError("Load count must be positive")

        if total_experts == 0:
            return []

        # Non-cyclic mode must respect remaining capacity
        if not cyclic_load and (self.current_load_expert + load_num > total_experts):
            # Return whatever remains and advance to the end
            loaded_blocks = self.sorted_expert_blocks[self.current_load_expert:]
            self.current_load_expert = total_experts
            return loaded_blocks

        end_idx = self.current_load_expert + load_num

        if cyclic_load:
            if end_idx <= total_experts:
                loaded_blocks = self.sorted_expert_blocks[self.current_load_expert:end_idx]
            else:
                # Split across the tail and the head
                first_part = self.sorted_expert_blocks[self.current_load_expert:]
                remaining = end_idx - total_experts
                second_part = self.sorted_expert_blocks[:remaining]
                loaded_blocks = first_part + second_part
            self.current_load_expert = (self.current_load_expert + load_num) % total_experts
            return loaded_blocks

        # Simple contiguous slice in non-cyclic mode
        loaded_blocks = self.sorted_expert_blocks[self.current_load_expert:end_idx]
        self.current_load_expert += load_num
        return loaded_blocks


if __name__ == "__main__":
    """Test TaskEmulator functionality"""
    # Create TaskEmulator instance
    task_emulator = TaskEmulator()

    print(task_emulator.expert_blocks)
    print(task_emulator.token_blocks)
    for expert_id, token_list in task_emulator.bind_relations.items():
        print(
            f"Expert {expert_id} binds tokens: {token_list}, count: {len(token_list)}")

    print(task_emulator.get_expert_load_order())
    print(task_emulator.sorted_expert_blocks)

    print(task_emulator.load_expert(4))
    print(task_emulator.load_expert(9))
    print(task_emulator.load_expert(16))
