#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dynamic scheduling algorithm for chip task execution

This module defines the interface and implementation of the scheduling algorithm, including:
- Scheduler base class: handles generic operations such as data block registration, movement, computation, and release
- FSEDPScheduler: a simple scheduling flow based on pipeline/diffusion, implementing DDR loading, inter-chip copying, and computation

Note: this file is only responsible for "strategy orchestration" and does not involve visualization; see animation.py for animation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import copy

from data_block import DataBlock, DataBlockType, DataBlockState
from action import Action, ActionType
from hardware_config import HardwareConfig
from task_emulator import TaskEmulator
from hardware_emulator import HardwareEmulator
from strategy.path_generator import PathGenerator, PathMode
from strategy.init_distribution_strategy import InitDistributionStrategy, DistributionMode
from strategy.ddr_load_strategy import DDRLoadStrategy, DDRLoadMode, DestinationSelectMode
from strategy.token_redispatcher import TokenRedispatcher, RedispatchMode


class Scheduler(ABC):
    """Scheduler base class

    Responsible for:
    - Combining and holding sub-strategies (path generation, initial distribution, DDR loading, token redistribution)
    - Providing generic methods for data block addition/removal, movement, computation, and release
    - Maintaining global data structures (data_blocks, activated_tokens, etc.)
    """

    def __init__(self):
        # Sub-strategies and emulators
        self.path_generator = PathGenerator()
        self.task_emulator = TaskEmulator()
        self.hardware_emulator = HardwareEmulator()
        self.init_strategy = InitDistributionStrategy(
            self.task_emulator, self.hardware_emulator)
        self.ddr_load_strategy = DDRLoadStrategy(self.hardware_emulator, self.task_emulator)
        self.token_dispatcher = TokenRedispatcher(
            self.task_emulator, self.hardware_emulator)

        # Dictionary of all data blocks, key is name, value is the set of all data blocks with that name
        self.data_blocks: Dict[str, set] = {}

        # Scheduling state
        self.completed_actions: List[Action] = []  # Completed actions (reserved for extension)
        self.step_count: int = 0                   # Number of steps already scheduled
        self.activated_tokens: Dict[int, Dict[str, set]] = {}  # {expert_id: {chip_id: set(token_blocks)}}
        self.token_dispatcher.activated_tokens = self.activated_tokens

        # Expert -> allowed chip groups; empty means no restriction
        self.expert_groups: Dict[int, List[str]] = {}
        # Propagate group information to each sub-strategy
        self.path_generator.set_expert_groups(self.expert_groups)
        self.ddr_load_strategy.expert_groups = {k: set(v) for k, v in self.expert_groups.items()}

        # 当为 True 时：若某 token_block 将被本芯片 expert 缓冲中的其他 expert 绑定使用，则在计算完成后的释放阶段不释放该 token_block 的复制品
        # 默认关闭以保持历史行为一致
        self.retain_tokens_if_used_by_other_local_experts: bool = False
        self.token_dispatcher.expert_groups = {k: set(v) for k, v in self.expert_groups.items()}

        # 默认策略配置（两个子类共同使用的设置）
        self.init_strategy.token_distribution_mode = DistributionMode.EVEN
        self.init_strategy.expert_distribution_mode = DistributionMode.EVEN
        self.ddr_load_strategy.ddr_load_mode = DDRLoadMode.ALTERNATE
        self.ddr_load_strategy.destination_select_mode = DestinationSelectMode.MAX_BUFFER
        self.token_dispatcher.redispatch_mode = RedispatchMode.NO_REDISPATCH

        # 通用计算调度字段
        self.read_for_compute: Dict[str, List[DataBlock]] = {chip: [] for chip in self.hardware_emulator.chips}
        self.same_compute_number: int = 1
        self.current_compute_number: Dict[str, int] = {chip: 0 for chip in self.hardware_emulator.chips}

        # 通用完成状态与阶段控制
        self.expert_finish: Dict[int, bool] = {}
        # 0=预处理/再分发阶段（FSEDP 使用），1=常规调度阶段，2=结束
        self.schedule_phase: int = 0

        # 立即执行动作队列
        self._instant_actions: List[Action] = []

    @abstractmethod
    def init_data_blocks(self):
        """Initialize data blocks and their initial positions.

        Steps:
        1) Task side generates token/expert data blocks and binding relations
        2) Initial distribution strategy loads token/expert onto chips/DDR
        3) Register these data blocks in the scheduler
        """
        pass

    def _add_new_block(self, new_block: DataBlock):
        """Register a new data block and initialize activated_tokens if necessary."""
        if new_block.name not in self.data_blocks:
            self.data_blocks[new_block.name] = set()
        self.data_blocks[new_block.name].add(new_block)

        # Initialize activated_tokens view when encountering a block of a certain expert for the first time
        if new_block.block_type == DataBlockType.EXPERT and new_block.seq_or_expert_id not in self.activated_tokens:
            self.activated_tokens[new_block.seq_or_expert_id] = {
                chip: set() for chip in self.hardware_emulator.chips}
            for block_name in self.task_emulator.bind_relations.get(new_block.seq_or_expert_id, set()):
                if block_name in self.data_blocks:
                    for block in self.data_blocks[block_name]:
                        self.activated_tokens[new_block.seq_or_expert_id][block.location[0]].add(block)

    def _remove_block(self, data_block: DataBlock):
        """Remove data block from index (does not handle hardware-side removal)."""
        self.data_blocks[data_block.name].remove(data_block)
        if not self.data_blocks[data_block.name]:
            del self.data_blocks[data_block.name]
        # Note: activated_tokens is not updated here

    def move_data_block(self, data_block: DataBlock, destination: Tuple[str, int]) -> Action:
        """Move data block and return corresponding Action.

        - From DDR to chip: DDR_LOAD
        - Chip to chip: CHIP_MOVE
        """
        source = data_block.location
        if not source:
            raise ValueError(f"Data block {data_block.block_id} does not exist")

        self.hardware_emulator.remove_data_block(data_block)
        self.hardware_emulator.add_new_block_in_location(data_block, destination)

        if source[0].startswith("DDR"):
            self.hardware_emulator.ddr_bandwidth_used[source[0]] = True
            return Action(ActionType.DDR_LOAD, data_block, source, destination)
        elif destination[0].startswith("Chip"):  # source is Chip
            data_block.state = DataBlockState.MOVING
            return Action(ActionType.CHIP_MOVE, data_block, source, destination)
        else:
            raise ValueError(f"Not supported: {destination}")

    def copy_move_data_block(self, data_block: DataBlock, destination: Tuple[str, int]) -> List[Action]:
        """Copy and move data block (keep source copy), return [ALLOCATE, LOAD/MOVE] action list."""
        source = data_block.location
        if not source:
            raise ValueError(f"Data block {data_block.block_id} does not exist")

        copy_block = copy.copy(data_block)
        copy_block.copy_number = max(
            (b.copy_number for b in self.data_blocks.get(data_block.name, set())),
            default=0
        ) + 1
        copy_block.state = None

        self.hardware_emulator.add_new_block_in_location(copy_block, destination)
        self._add_new_block(copy_block)

        actions = [Action(ActionType.ALLOCATE, copy_block, destination=destination)]
        if source[0].startswith("DDR"):
            self.hardware_emulator.ddr_bandwidth_used[source[0]] = True
            actions.append(Action(ActionType.DDR_LOAD, copy_block, source, destination))
        elif destination[0].startswith("Chip"):  # source is Chip
            copy_block.state = DataBlockState.MOVING
            actions.append(Action(ActionType.CHIP_MOVE, copy_block, source, destination))
        else:
            raise ValueError(f"Not supported: {destination}")
        return actions

    def compute_data_block(self, data_block: DataBlock, other_blocks: List[DataBlock]) -> Action:
        """Perform computation on data block on current chip, other_blocks are bound tokens."""
        location = data_block.location
        if not location or not location[0].startswith("Chip"):
            raise ValueError(f"Data block is not on chip, cannot compute")

        data_block.state = DataBlockState.COMPUTING
        for other_block in other_blocks:
            if other_block.location[0] != location[0]:
                raise ValueError(f"Data block {other_block.block_id} is not on the same chip, cannot compute")
            other_block.state = DataBlockState.COMPUTING
        return Action(ActionType.COMPUTE, data_block, None, location, other_blocks=other_blocks)

    def release_data_block(self, data_block: DataBlock) -> Action:
        """Release data block and return release action."""
        location = data_block.location
        if not location:
            raise ValueError(f"Data block {data_block.block_id} does not exist")

        self.hardware_emulator.remove_data_block(data_block)
        self._remove_block(data_block)
        return Action(ActionType.RELEASE, data_block, location)

    @abstractmethod
    def step(self) -> List[Action]:
        """Execute one scheduling step, return list of scheduling actions"""
        pass

    @abstractmethod
    def handle_action_complete(self, action: Action):
        """Handle action completion"""
        pass

    def set_expert_groups(self, groups: dict):
        """设置 expert -> 允许的芯片组，并传递到子策略。"""
        self.expert_groups = groups or {}
        self.path_generator.set_expert_groups(self.expert_groups)
        self.ddr_load_strategy.expert_groups = {k: set(v) for k, v in self.expert_groups.items()}
        self.token_dispatcher.expert_groups = {k: set(v) for k, v in self.expert_groups.items()}

class FSEDPScheduler(Scheduler):
    """Simple FSEDP scheduler

    Process (per step):
    - Token redistribution phase (once): balance tokens bound to each expert across chips
    - Regular phase:
        * Load a batch of expert micro-fragments from DDR to chip
        * Schedule a batch of computable data blocks on each chip and copy to next chip
    """

    def __init__(self):
        super().__init__()
        # Set sub-strategy modes
        self.path_generator.path_mode = PathMode.VOTEX

        # Inter-chip transfer scheduling: queue of data blocks to be copied
        self.read_for_move_between_chips: Dict[str, List[DataBlock]] = {chip: [] for chip in self.hardware_emulator.chips}

        self.begin_token_redispatch: bool = False

        # expert_finish and _instant_actions already provided by parent class

    def init_data_blocks(self):
        """Initialize data blocks and their initial positions; must be run once before scheduling algorithm."""
        self.task_emulator.init_data()
        new_blocks = self.init_strategy.distribute()
        for block in new_blocks:
            self._add_new_block(block)

            if block.block_type == DataBlockType.EXPERT:
                self.expert_finish[(block.seq_or_expert_id, block.mico_slice)] = False

    def handle_action_complete(self, action: Action):
        """Handle action completion (called by animator to update internal state)."""
        if action.action_type in [ActionType.DDR_LOAD, ActionType.CHIP_MOVE]:
            if self.schedule_phase == 0:  # Do not process other actions during token redistribution phase
                return
            if action.action_type == ActionType.DDR_LOAD:
                self.hardware_emulator.ddr_bandwidth_used[action.source[0]] = False
            # Add data block to ready_to_compute
            self.read_for_compute[action.destination[0]].append(action.data_block)
        elif action.action_type == ActionType.DDR_SAVE:
            self.hardware_emulator.ddr_bandwidth_used[action.source[0]] = False
        elif action.action_type == ActionType.COMPUTE:
            self.current_compute_number[action.data_block.location[0]] -= 1
            # Release after computation and moving
            chip = action.data_block.location[0]
            if action.data_block not in self.read_for_move_between_chips[chip]:
                self._instant_actions.append(self.release_data_block(action.data_block))
            else:
                action.data_block.state = DataBlockState.AFTER_COMPUTE
            # Check if all experts are finished
            if all(self.expert_finish.values()):
                self.schedule_phase = 2
        elif action.action_type == ActionType.FLAG:
            if action.data_block.name == 'token-redispatch':
                self.schedule_phase = 1  # Enter regular scheduling phase

    def step(self) -> List[Action]:
        """Execute one scheduling step, return list of scheduling actions."""
        self.step_count += 1
        actions: List[Action] = [] + self._instant_actions
        self._instant_actions.clear()

        # 0) Token redistribution phase (execute only once)
        if self.schedule_phase == 0:
            if self.begin_token_redispatch:
                return actions
            self.begin_token_redispatch = True
            # Perform token redistribution for all required experts (by group if defined)
            expert_ids = list(self.expert_groups.keys()) if self.expert_groups else list(self.task_emulator.bind_relations.keys())
            for eid in expert_ids:
                move_lists = self.token_dispatcher.redispatch(eid)
                for token_block_id, destination in move_lists:
                    for token_block in self.data_blocks[token_block_id]:
                        actions.append(self.move_data_block(token_block, destination))
            actions.append(Action(ActionType.FLAG, DataBlock('token-redispatch')))
            return actions

        # 1) DDR -> chip loading
        for ddr in self.hardware_emulator.ddrs:
            if not self.hardware_emulator.ddr_bandwidth_used[ddr]:
                try:
                    data_block, destination = self.ddr_load_strategy.load_next_data_block(ddr)
                    actions.append(self.move_data_block(data_block, destination))
                except ValueError:
                    # No loadable or target has no space, skip to next step
                    continue

        # 2) Schedule computation on each chip and add copyable data blocks to copy queue
        for chip in self.hardware_emulator.chips:
            ready_blocks = self.read_for_compute[chip]

            # Take specified number of data blocks for computation
            compute_count = max(0, self.same_compute_number - self.current_compute_number[chip])
            compute_blocks = ready_blocks[:min(len(ready_blocks), compute_count)]

            for block in compute_blocks:
                # Use only bound tokens that actually exist on current chip to avoid cross-chip computation errors
                candidate_tokens = self.activated_tokens[block.seq_or_expert_id][chip]
                chip_seq_blocks = [b for b in candidate_tokens if b.location and b.location[0] == chip]
                actions.append(self.compute_data_block(block, chip_seq_blocks))
                self.current_compute_number[chip] += 1

                # Copy limit should be determined by expert group size (use full array if no group)
                total_allowed_chips = len(self.expert_groups.get(block.seq_or_expert_id, self.hardware_emulator.chips))
                if block.copy_number >= total_allowed_chips - 1:
                    # Mark this expert as finished
                    self.expert_finish[(block.seq_or_expert_id, block.mico_slice)] = True
                else:
                    self.read_for_move_between_chips[chip].append(block)

            # Remove scheduled compute data blocks from ready list
            self.read_for_compute[chip] = ready_blocks[len(compute_blocks):]

            # 3) Copy queued data blocks to next chip (subject to path/group constraints)
            blocks_to_retry: List[DataBlock] = []
            while self.read_for_move_between_chips[chip]:
                block = self.read_for_move_between_chips[chip].pop(0)

                # Determine next chip (limited by expert group)
                next_chip = self.path_generator.get_next_in_group(chip, block.seq_or_expert_id)

                # Find available expert buffer location on target chip
                index = self.hardware_emulator.get_min_empty_expert_index(next_chip)
                if index == -1:
                    blocks_to_retry.append(block)
                else:
                    if block.state == DataBlockState.AFTER_COMPUTE:
                        actions.append(self.move_data_block(block, (next_chip, index)))
                    else:
                        actions.extend(self.copy_move_data_block(block, (next_chip, index)))

            # Re-add unmoved data blocks to move list
            self.read_for_move_between_chips[chip].extend(blocks_to_retry)
        
        # If all experts are finished, return finish flag
        if self.schedule_phase == 2:
            actions.append(Action(ActionType.FLAG, DataBlock('finish')))

        return actions

class EPScheduler(Scheduler):

    """Expert-Parallel (EP) scheduler

    Design highlights:
    - In EP mode, each expert participates in scheduling as a whole (micro slice count is 1).
    - Before an expert starts computation, all tokens bound to that expert must be migrated to the chip where the expert resides.
    - No cross-chip expert copy diffusion; each expert is computed only once on a single chip.
    """

    def __init__(self):
        super().__init__()
        # Path generation strategy is not critical in EP (we do point-to-point migration directly), keep default
        # Default settings for DDR/distribution strategies already done in parent class

        # Compute scheduling fields provided by parent class

        # Expert -> chip assignment and completion status
        self.expert_assigned_chip: Dict[int, str] = {}

        # Scheduling phase:
        self.schedule_phase: int = 0

        # Each chip allows only one "active expert" for token migration and computation serialization gating
        # None means no expert is currently preparing/computing on this chip, next candidate can be selected
        self.chip_active_expert = {chip: None for chip in self.hardware_emulator.chips}
        # Per-chip "token sending" gating: a chip can only send one token at a time (CHIP_MOVE),
        # can continue sending next only after receiving completion callback for this CHIP_MOVE.
        self.chip_token_sending_locked: Dict[str, bool] = {chip: False for chip in self.hardware_emulator.chips}
        # Reverse index: token name -> set of experts that need this token
        self.token_used_by_experts: Dict[str, set] = {}

    def init_data_blocks(self):
        """Initialize data blocks"""
        # Generate token/expert data and binding relations
        self.task_emulator.init_data()

        # Perform initial distribution (tokens onto chips, experts into DDR)
        new_blocks = self.init_strategy.distribute()
        for block in new_blocks:
            self._add_new_block(block)
            if block.block_type == DataBlockType.EXPERT:
                self.expert_finish[(block.seq_or_expert_id, 1)] = False

        # Build reverse binding index for tokens: token -> experts
        self.token_used_by_experts = {}
        for eid, names in self.task_emulator.bind_relations.items():
            for name in names:
                self.token_used_by_experts.setdefault(name, set()).add(eid)

    def handle_action_complete(self, action: Action):
        """Animation completion callback: only add expert load/move to ready-to-compute queue."""
        if action.action_type in [ActionType.DDR_LOAD, ActionType.CHIP_MOVE]:
            if self.schedule_phase != 1:
                return
            if action.action_type == ActionType.DDR_LOAD:
                # Release DDR bandwidth usage
                self.hardware_emulator.ddr_bandwidth_used[action.source[0]] = False
            elif action.action_type == ActionType.CHIP_MOVE:
                action.data_block.state = None
                # After token sending completes, release source chip sending lock (only for tokens)
                if action.data_block.block_type == DataBlockType.SEQ and action.source and action.source[0].startswith("Chip"):
                    self.chip_token_sending_locked[action.source[0]] = False
            # Only when it is an expert data block, add to ready-to-compute queue and record assigned chip
            if action.data_block.block_type == DataBlockType.EXPERT:
                chip = action.destination[0]
                self.read_for_compute[chip].append(action.data_block)
                self.expert_assigned_chip[action.data_block.seq_or_expert_id] = chip
                # If this chip has no active expert yet, select this expert as active
                if self.chip_active_expert.get(chip) is None:
                    self.chip_active_expert[chip] = action.data_block.seq_or_expert_id
        elif action.action_type == ActionType.DDR_SAVE:
            self.hardware_emulator.ddr_bandwidth_used[action.source[0]] = False
        elif action.action_type == ActionType.COMPUTE:
            action.data_block.state = None
            chip = action.data_block.location[0]
            self.current_compute_number[chip] -= 1
            # Release expert after computation
            self._instant_actions.append(self.release_data_block(action.data_block))
            self.expert_finish[(action.data_block.seq_or_expert_id, action.data_block.mico_slice)] = True
            # If all experts are finished, enter end phase
            if all(self.expert_finish.values()) and self.expert_finish:
                self.schedule_phase = 2
            # Current expert computation completed, release active expert gating on chip, allow selecting next expert next step
            self.chip_active_expert[chip] = None
            # After computation, release corresponding token blocks for this expert on this chip, release immediately if copy_number > 0
            for tb in (action.other_blocks or []):
                if tb.copy_number > 0:
                    # If retention option is enabled and this token will be used by other experts in this chip's buffer, do not release copy of this token
                    if self.retain_tokens_if_used_by_other_local_experts:
                        if self._token_needed_by_other_local_experts(
                            tb,
                            chip,
                            exclude_expert_id=action.data_block.seq_or_expert_id
                        ):
                            continue
                    self._instant_actions.append(self.release_data_block(tb))
        elif action.action_type == ActionType.FLAG:
            if action.data_block.name == 'token-redispatch':
                self.schedule_phase = 1  # Enter regular scheduling phase


    def _get_bound_token_blocks(self, expert_id: int) -> List[DataBlock]:
        """Return all token data blocks bound to specified expert (across chips)."""
        names = self.task_emulator.bind_relations.get(expert_id, set())
        blocks: List[DataBlock] = []
        for name in names:
            for b in self.data_blocks.get(name, set()):
                if b.location:  # Only consider existing blocks
                    blocks.append(b)
        return blocks

    def _token_needed_by_other_local_experts(self, token_block: DataBlock, chip: str, exclude_expert_id: int) -> bool:
        """Determine if this token is used by other experts in this chip's expert buffer.

        Only check when other experts exist in this chip's expert buffer; return True if this token's name belongs to binding set of any of those experts.
        """
        # Collect all expert_ids on this chip except the currently finished expert
        other_expert_ids = set()
        for exp in self.hardware_emulator.chip_expert_data_blocks[chip]:
            if exp is None:
                continue
            if exp.seq_or_expert_id != exclude_expert_id:
                other_expert_ids.add(exp.seq_or_expert_id)

        if not other_expert_ids:
            return False

        # Check if this token name belongs to binding relations of these experts
        name = getattr(token_block, 'name', None)
        if not name:
            return False

        for eid in other_expert_ids:
            bound = self.task_emulator.bind_relations.get(eid, set())
            if name in bound:
                return True
        return False

    def _tokens_ready_on_chip(self, expert_id: int, chip: str) -> bool:
        """Determine if all tokens bound to this expert have at least one copy on specified chip."""
        names = self.task_emulator.bind_relations.get(expert_id, set())
        for name in names:
            present = False
            for b in self.data_blocks.get(name, set()):
                if b.location and b.location[0] == chip and b.state != DataBlockState.MOVING:
                    present = True
                    break
            if not present:
                return False
        return True

    def _token_is_present_on_chip(self, token_name: str, chip: str) -> bool:
        for b in self.data_blocks.get(token_name, set()):
            if b.location and b.location[0] == chip:
                return True
        return False

    def _find_evictable_tokens(self, chip: str, expert_id: int) -> List[DataBlock]:
        """Find evictable tokens on target chip.

        Evict in priority:
        1) Duplicate tokens in this expert's binding set (multiple with same name, keep one is enough)
        2) Tokens not in this expert's binding set (completely unrelated)
        """
        evict: List[DataBlock] = []
        bound_names = self.task_emulator.bind_relations.get(expert_id, set())
        seen: Dict[str, int] = {}
        # First count duplicates
        for tb in self.hardware_emulator.chip_seq_data_blocks[chip]:
            if tb is None:
                continue
            seen[tb.name] = seen.get(tb.name, 0) + 1
        # Evict duplicates (name count > 1) and belonging to this expert's set
        for tb in self.hardware_emulator.chip_seq_data_blocks[chip]:
            if tb is None:
                continue
            if tb.name in bound_names and seen.get(tb.name, 0) > 1:
                evict.append(tb)
                seen[tb.name] -= 1  # Decrement gradually to avoid selecting same copy repeatedly
        if evict:
            return evict
        # If no duplicates to evict, evict tokens not in this expert's binding set
        for tb in self.hardware_emulator.chip_seq_data_blocks[chip]:
            if tb is None:
                continue
            if tb.name not in bound_names:
                evict.append(tb)
        return evict

    def _schedule_token_moves_for_expert(self, expert_block: DataBlock, max_moves: int = 1) -> List[Action]:
        """Schedule token migration actions for specified expert, but arrange at most one migration or one eviction.

        - If target chip has no space, raise error directly.
        - If space available, migrate only one bound token to target chip (once).
        - Tokens already on target chip are not migrated.
        """
        actions: List[Action] = []
        if max_moves <= 0:
            return actions

        chip = expert_block.location[0]
        expert_id = expert_block.seq_or_expert_id

        bound_tokens = self._get_bound_token_blocks(expert_id)

        # Check bound tokens one by one, perform operation (move in) for first one not on target chip
        for tb in bound_tokens:
            # If a copy of this token already exists on target chip, skip (avoid filling buffer with duplicates)
            if self._token_is_present_on_chip(tb.name, chip):
                continue
            else:
                # Assertion: token block with same name should not already exist in target chip's seq slots
                if any(tb.name == existing.name for existing in self.hardware_emulator.chip_seq_data_blocks[chip] if existing is not None):
                    raise RuntimeError(f"token {tb.name} already exists in chip {chip} seq slot, duplicate migration prohibited")
            if not tb.location or tb.location[0] == chip:
                continue

            # If target chip has no space, raise error directly
            idx = self.hardware_emulator.get_min_empty_seq_index(chip)
            if idx == -1:
                raise RuntimeError(f"target chip {chip} has no space, cannot migrate token {tb.name}")

            # Target chip has space: migrate or copy one bound token to target chip
            # If the chip where tb currently resides also needs this token for computation, keep source copy (copy)
            should_copy = False
            src_chip = tb.location[0]
            # If source chip is currently in sending lock state (last CHIP_MOVE not completed), skip
            if self.chip_token_sending_locked.get(src_chip, False):
                continue

            src_active = self.chip_active_expert.get(src_chip)
            if src_active and tb.name in self.task_emulator.bind_relations.get(src_active, set()):
                should_copy = True

            if should_copy:
                actions.extend(self.copy_move_data_block(tb, (chip, idx)))
            else:
                actions.append(self.move_data_block(tb, (chip, idx)))
            return actions

        # All bound tokens already on target chip or no migration needed
        return actions

    def step(self) -> List[Action]:
        """Execute one scheduling step, return action list."""
        self.step_count += 1
        actions: List[Action] = [] + self._instant_actions
        self._instant_actions.clear()

        if self.schedule_phase == 0:
            actions.append(Action(ActionType.FLAG, DataBlock('token-redispatch')))
            return actions

        # 2 = all experts finished -> emit finish flag
        if self.schedule_phase == 2:
            actions.append(Action(ActionType.FLAG, DataBlock('finish')))
            return actions

        # 1) DDR -> Chip: load available expert blocks
        for ddr in self.hardware_emulator.ddrs:
            if not self.hardware_emulator.ddr_bandwidth_used[ddr]:
                try:
                    data_block, destination = self.ddr_load_strategy.load_next_data_block(ddr)
                    # Only load expert; if token (theoretically won't happen), handle normally
                    actions.append(self.move_data_block(data_block, destination))
                except ValueError:
                    continue

        # 2) Arrange token migration for "active expert" on each chip (one send per chip, multiple chips parallel)
        for chip in self.hardware_emulator.chips:
            active_id = self.chip_active_expert.get(chip)
            # If this chip currently has no active expert and has candidates, select first in queue as active expert
            if active_id is None and self.read_for_compute[chip]:
                self.chip_active_expert[chip] = self.read_for_compute[chip][0].seq_or_expert_id
                active_id = self.chip_active_expert[chip]

            if active_id is None:
                continue

            # Get data block for this active expert (EP: only one block per expert on chip) and arrange its bound token migration
            expert_blocks = [b for b in self.hardware_emulator.chip_expert_data_blocks[chip]
                             if b is not None and b.seq_or_expert_id == active_id]
            # One token-related move/copy/eviction per step per chip
            moves_budget = 1
            for exp in expert_blocks:
                if moves_budget <= 0:
                    break
                scheduled_actions = self._schedule_token_moves_for_expert(exp, max_moves=moves_budget)
                if scheduled_actions:
                    actions.extend(scheduled_actions)
                    moves_budget -= 1
                    # If CHIP_MOVE generated, lock its source chip, release after completion callback
                    for act in scheduled_actions:
                        if act.action_type == ActionType.CHIP_MOVE and act.source and act.source[0].startswith("Chip"):
                            self.chip_token_sending_locked[act.source[0]] = True
                    break

        # 3) Schedule computation: only compute current chip's active expert, ignore other ready blocks
        for chip in self.hardware_emulator.chips:
            ready_blocks = self.read_for_compute[chip]
            active_id = self.chip_active_expert.get(chip)
            # Concurrent quota (EP often 1)
            compute_quota = max(0, self.same_compute_number - self.current_compute_number[chip])

            if active_id is None or compute_quota <= 0:
                continue

            # Find block for active expert on this chip (EP: only one block per expert on chip)
            active_block: DataBlock = None
            for b in ready_blocks:
                if b.seq_or_expert_id == active_id:
                    active_block = b
                    break

            if not active_block:
                # No active block available for computation, skip
                continue

            # Collect arrived tokens for this active expert on current chip (keep only one copy per token name)
            bound_tokens_on_chip: List[DataBlock] = []
            for name in self.task_emulator.bind_relations.get(active_block.seq_or_expert_id, set()):
                for tb in self.data_blocks.get(name, set()):
                    if tb.location and tb.location[0] == chip:
                        bound_tokens_on_chip.append(tb)
                        break

            # Only compute when all bound token blocks for this active expert have arrived on this chip; do not wait for other ready blocks
            if self._tokens_ready_on_chip(active_block.seq_or_expert_id, chip):
                actions.append(self.compute_data_block(active_block, bound_tokens_on_chip))
                self.current_compute_number[chip] += 1
                # Remove this active block from ready queue, leave other non-active blocks unchanged
                self.read_for_compute[chip] = [b for b in ready_blocks if b is not active_block]
            else:
                # Bound tokens not all arrived, keep queue unchanged, wait for subsequent migration
                continue

        # If all experts finished, emit finish flag
        if self.schedule_phase == 2:
            actions.append(Action(ActionType.FLAG, DataBlock('finish')))

        return actions
