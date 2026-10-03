#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Token redispatch strategies for balancing per-expert token blocks across chips.

This module offers a few simple strategies (no-op, minimal-moves, simple) that
redistribute sequence/token blocks among chips to better balance compute load.
It also supports optional per-expert chip-group constraints so an expert's
tokens are balanced only within its allowed chip subset.
"""

from hardware_config import HardwareConfig
from task_emulator import TaskEmulator
from hardware_emulator import HardwareEmulator
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
from data_block import DataBlock
import copy


class RedispatchMode(Enum):
    """Redistribution modes.

    - NO_REDISPATCH: Do nothing.
    - MINIMAL_MOVES: Greedy rebalancing with minimal number of moves.
    - SIMPLE: A round-robin style balancing pass.
    """
    NO_REDISPATCH = 1  # No token redispatch
    MINIMAL_MOVES = 2  # Minimal moves redispatch
    SIMPLE = 3        # Simple redispatch strategy


class TokenRedispatcher:
    """Redistribute tokens among chips to improve balance for a given expert.

    Contract:
    - Inputs: expert_id for which we rebalance tokens; internal hardware/task state
    - Output: list of (token_block_name, (dest_chip_id, dest_index)) move actions
    - Error modes: raises ValueError for invalid mode; otherwise returns empty list when no work
    - Success: returned actions can be applied by scheduler/hardware to perform moves
    """

    def __init__(self, task_emulator: Optional[TaskEmulator] = None, hardware_emulator: Optional[HardwareEmulator] = None):
        """Initialize token redispatcher.

        Args:
            task_emulator: Task emulator
            hardware_emulator: Hardware emulator
        """
        self.task_emulator = task_emulator
        self.hardware_emulator = hardware_emulator

        self._redispatch_mode = RedispatchMode.SIMPLE
        # activated_tokens[expert_id][chip_id] = set(DataBlock)
        self.activated_tokens: Dict[int, Dict[str, Set[DataBlock]]] | None = None
        # Optional expert chip groups constraints: expert_id -> set(chip_id)
        self.expert_groups: Dict[int, Set[str]] = {}

    @property
    def redispatch_mode(self) -> RedispatchMode:
        return self._redispatch_mode

    @redispatch_mode.setter
    def redispatch_mode(self, mode: RedispatchMode):
        if not isinstance(mode, RedispatchMode):
            raise ValueError(
                f"Invalid redispatch mode: {mode}. Must use RedispatchMode enum type")
        self._redispatch_mode = mode

    def redispatch(self, expert_id: int) -> List[Tuple[str, Tuple[str, int]]]:
        """Select and execute token redistribution strategy based on mode.

        Args:
            expert_id: Expert ID

        Returns:
            List of actions: (token_block_name, (dest_chip_id, dest_index))
        """
        # Select implementation based on redispatch mode
        if self._redispatch_mode == RedispatchMode.NO_REDISPATCH:
            return []
        elif self._redispatch_mode == RedispatchMode.MINIMAL_MOVES:
            return self._minimal_moves_redispatch(expert_id)
        elif self._redispatch_mode == RedispatchMode.SIMPLE:
            return self._simple_redispatch(expert_id)
        else:
            raise ValueError(
                f"Unknown redispatch mode: {self._redispatch_mode}")

    # --------------------------
    # Internal helpers
    # --------------------------

    def _ensure_activated_tokens(self, expert_id: int, chips: List[str]) -> None:
        """Ensure activated_tokens dicts exist for the given expert and chips."""
        if self.activated_tokens is None:
            self.activated_tokens = {}
        if expert_id not in self.activated_tokens:
            self.activated_tokens[expert_id] = {chip: set() for chip in chips}
        else:
            # Ensure all chip keys exist
            for chip in chips:
                self.activated_tokens[expert_id].setdefault(chip, set())

    def _minimal_moves_redispatch(self, expert_id: int) -> List[Tuple[str, Tuple[str, int]]]:
        """Minimal moves redistribution strategy

        Args:
            expert_id: Expert ID

        Returns:
            List of actions
        """
        """Token redistribution"""
        actions: List[Tuple[str, Tuple[str, int]]] = []
        temp_hardware_model = HardwareEmulator()
        temp_hardware_model.chip_seq_data_blocks = copy.deepcopy(self.hardware_emulator.chip_seq_data_blocks)

        # Get token blocks bound to expert_id
        token_blocks = set()
        for block_name in self.task_emulator.bind_relations.get(expert_id, set()):
            token_blocks.add(self.task_emulator.token_blocks[block_name])

        if not token_blocks:
            return actions

        # Calculate target number of token blocks per chip
        total_blocks = len(token_blocks)
        allowed_chips = list(self.expert_groups.get(expert_id, set(temp_hardware_model.chips)))
        if not allowed_chips:
            allowed_chips = list(temp_hardware_model.chips)
        blocks_per_chip = total_blocks // len(allowed_chips)
        remaining_blocks = total_blocks % len(allowed_chips)

        # Count current token blocks on each chip
        current_distribution = {chip: 0 for chip in allowed_chips}
        block_locations = {}  # Track current chip location of each block

        # Ensure tracking structure exists
        self._ensure_activated_tokens(expert_id, list(temp_hardware_model.chips))

        for chip in temp_hardware_model.chips:
            self.activated_tokens[expert_id][chip] = set()
            for block in temp_hardware_model.chip_seq_data_blocks[chip]:
                if block in token_blocks:
                    if chip in allowed_chips:
                        current_distribution[chip] += 1
                    block_locations[block] = chip
                    # Add token block to activated_tokens
                    self.activated_tokens[expert_id][chip].add(block)

        # Calculate target distribution
        target_distribution = {chip: blocks_per_chip for chip in allowed_chips}
        for i, chip in enumerate(allowed_chips):
            if i < remaining_blocks:
                target_distribution[chip] += 1

        # Generate move actions until target distribution is reached
        while True:
            # Find chips with too many/few blocks
            source_chips = [chip for chip in allowed_chips if current_distribution[chip] > target_distribution[chip]]
            dest_chips = [chip for chip in allowed_chips if current_distribution[chip] < target_distribution[chip]]

            if not source_chips or not dest_chips:
                break

            source_chip = source_chips[0]
            dest_chip = dest_chips[0]

            # Find a block that can be moved
            move_block = None
            for block in temp_hardware_model.chip_seq_data_blocks[source_chip]:
                if block in token_blocks:
                    move_block = block
                    break

            if move_block:
                # Find empty buffer position on target chip
                dest_idx = temp_hardware_model.get_min_empty_seq_index(
                    dest_chip)

                # If target chip has no empty space
                if dest_idx == -1:
                    # Find another chip with empty space
                    alternative_chip = None
                    alternative_idx = -1
                    for chip in allowed_chips:
                        if chip == dest_chip:
                            continue
                        idx = temp_hardware_model.get_min_empty_seq_index(
                            chip)
                        if idx != -1:
                            alternative_chip = chip
                            alternative_idx = idx
                            break

                    if alternative_chip is None:
                        # Skip this token if no chip has space
                        continue

                    # Find an inactive token on target chip to move
                    inactive_block = None
                    for block in temp_hardware_model.chip_seq_data_blocks[dest_chip]:
                        if block not in token_blocks:
                            inactive_block = block
                            break

                    if inactive_block:
                        # First move the inactive token
                        actions.append(
                            (inactive_block.name, (alternative_chip, alternative_idx)))
                        # Then move the active token that needs to be moved
                        dest_idx = temp_hardware_model.get_min_empty_seq_index(
                            dest_chip)
                        actions.append((move_block.name, (dest_chip, dest_idx)))
                        temp_hardware_model.remove_data_block(inactive_block)
                        temp_hardware_model.add_new_block_in_location(
                            inactive_block, (alternative_chip, alternative_idx))
                        temp_hardware_model.remove_data_block(move_block)
                        temp_hardware_model.add_new_block_in_location(
                            move_block, (dest_chip, dest_idx))

                        # Update distribution counts
                        current_distribution[source_chip] -= 1
                        current_distribution[dest_chip] += 1
                        # Update activated_tokens
                        self.activated_tokens[expert_id][source_chip].remove(
                            move_block)
                        self.activated_tokens[expert_id][dest_chip].add(
                            move_block)
                else:
                    # Target chip has empty space, move directly
                    actions.append((move_block.name, (dest_chip, dest_idx)))
                    temp_hardware_model.remove_data_block(move_block)
                    temp_hardware_model.add_new_block_in_location(
                        move_block, (dest_chip, dest_idx))
                    # Update distribution counts
                    current_distribution[source_chip] -= 1
                    current_distribution[dest_chip] += 1
                    # Update activated_tokens
                    self.activated_tokens[expert_id][source_chip].remove(
                        move_block)
                    self.activated_tokens[expert_id][dest_chip].add(move_block)

        return actions

    def _simple_redispatch(self, expert_id: int) -> List[Tuple[str, Tuple[str, int]]]:
        """Simple redistribution strategy that balances token blocks across chips.

        This strategy aims to evenly distribute token blocks bound to an expert
        across all available chips. It first calculates target token count per chip,
        then iteratively moves tokens between chips to achieve the target distribution.

        Args:
            expert_id: Expert ID to redistribute tokens for

        Returns:
            List of actions, where each action is a tuple (token, (dest_chip, dest_idx))
            representing moving a token to destination chip at specified index
        """
        actions: List[Tuple[str, Tuple[str, int]]] = []

        temp_hardware_model = HardwareEmulator()
        temp_hardware_model.chip_seq_data_blocks = copy.deepcopy(self.hardware_emulator.chip_seq_data_blocks)

        # Get token blocks bound to expert_id
        token_blocks = set()
        for block_name in self.task_emulator.bind_relations.get(expert_id, set()):
            token_blocks.add(self.task_emulator.token_blocks[block_name])

        if not token_blocks:
            return actions

        # Calculate target token count per chip (respect group constraint)
        total_blocks = len(token_blocks)
        allowed_chips = list(self.expert_groups.get(expert_id, set(temp_hardware_model.chips)))
        if not allowed_chips:
            allowed_chips = list(temp_hardware_model.chips)
        blocks_per_chip = total_blocks // len(allowed_chips)
        remaining_blocks = total_blocks % len(allowed_chips)

        # Initialize activated token set for each chip
        self._ensure_activated_tokens(expert_id, list(temp_hardware_model.chips))
        # Reset per-chip sets
        self.activated_tokens[expert_id] = {chip: set() for chip in temp_hardware_model.chips}

        # Map tokens to their current chip locations
        token_locations = {}
        for chip in temp_hardware_model.chips:
            for block in temp_hardware_model.chip_seq_data_blocks[chip]:
                if block in token_blocks:
                    token_locations[block] = chip
                    self.activated_tokens[expert_id][chip].add(block)

        # Track final destination for tokens that need to be moved
        final_destinations = {}

        # Start from first chip and iterate through all chips
        current_idx = 0
        # Tokens currently on chips outside of allowed_chips should be reassigned
        remaining_tokens = set()
        for chip in temp_hardware_model.chips:
            if chip not in allowed_chips:
                remaining_tokens |= self.activated_tokens[expert_id][chip]

        # Iterate until all tokens are properly distributed
        while True:
            current_chip = allowed_chips[current_idx]
            
            # Calculate target token count for current chip
            target_count = blocks_per_chip + (1 if current_idx < remaining_blocks else 0)

            # Combine activated tokens on current chip with remaining tokens
            current_tokens = self.activated_tokens[expert_id][current_chip].union(
                remaining_tokens)
            self.activated_tokens[expert_id][current_chip] = current_tokens
            remaining_tokens = set()

            # If current chip has more tokens than target, redistribute excess
            if len(current_tokens) > target_count:
                tokens_to_keep = set(list(current_tokens)[:target_count])
                remaining_tokens = current_tokens - tokens_to_keep
                self.activated_tokens[expert_id][current_chip] = tokens_to_keep
    
                # Assign excess tokens to next chip
                next_idx = (current_idx + 1) % len(allowed_chips)
                next_chip = allowed_chips[next_idx]

                for token in remaining_tokens:
                    if token not in final_destinations:
                        final_destinations[token] = (
                            token_locations[token], next_chip)
                    else:
                        # Update destination if token was previously assigned
                        final_destinations[token] = (
                            final_destinations[token][0], next_chip)
            
            current_idx = (current_idx + 1) % len(allowed_chips)
            
            # Exit if we've processed all chips and no tokens remain
            if current_idx == 0 and not remaining_tokens:
                break

        # Generate move actions based on final destinations
        for token, (source_chip, dest_chip) in final_destinations.items():
            if source_chip != dest_chip:
                # Find empty slot on destination chip
                dest_idx = temp_hardware_model.get_min_empty_seq_index(
                    dest_chip)

                # If destination chip is full, move an inactive token first
                if dest_idx == -1:
                    for block in temp_hardware_model.chip_seq_data_blocks[dest_chip]:
                        if block not in token_blocks:
                            # Find another chip with empty space
                            for other_chip in allowed_chips:
                                if other_chip == dest_chip:
                                    continue
                                other_idx = temp_hardware_model.get_min_empty_seq_index(
                                    other_chip)
                                if other_idx != -1:
                                    # Move inactive token to make space
                                    actions.append(
                                        (block.name, (other_chip, other_idx)))
                                    temp_hardware_model.remove_data_block(block)
                                    temp_hardware_model.add_new_block_in_location(
                                        block, (other_chip, other_idx))
                                    # Update destination empty slot
                                    dest_idx = temp_hardware_model.get_min_empty_seq_index(
                                        dest_chip)
                                    break
                            if dest_idx != -1:
                                break

                # Move token if destination slot is available
                if dest_idx != -1:
                    actions.append((token.name, (dest_chip, dest_idx)))
                    temp_hardware_model.remove_data_block(token)
                    temp_hardware_model.add_new_block_in_location(
                        token, (dest_chip, dest_idx))

        return actions
