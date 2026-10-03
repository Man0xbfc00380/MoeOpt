#!/usr/bin/env python
# -*- coding: utf-8 -*-

from hardware_config import HardwareConfig
from task_emulator import TaskEmulator
from hardware_emulator import HardwareEmulator
from typing import List
from enum import Enum


class DistributionMode(Enum):
    """Distribution mode enumeration

    EVEN: Distribute blocks evenly across chips/DDRs
    CORRESPONDING: Distribute blocks to corresponding chips/DDRs based on block ID
    SEQUENTIAL: Fill each chip/DDR buffer sequentially before moving to next one
    """
    EVEN = "even"
    CORRESPONDING = "corresponding" 
    SEQUENTIAL = "sequential"


class InitDistributionStrategy():
    """Initialization Distribution Strategy Class"""

    def __init__(self, task_emulator: TaskEmulator = None, hardware_emulator: HardwareEmulator = None):
        """Initialize distribution strategy

        Args:
            task_emulator: Task emulator
            hardware_emulator: Hardware emulator
        """
        self.task_emulator = task_emulator
        self.hardware_emulator = hardware_emulator
        self.token_distribution_mode = DistributionMode.EVEN
        self.expert_distribution_mode = DistributionMode.EVEN

    def set_emulators(self, task_emulator: TaskEmulator, hardware_emulator: HardwareEmulator):
        """Set emulators

        Args:
            task_emulator: Task emulator
            hardware_emulator: Hardware emulator
        """
        self.task_emulator = task_emulator
        self.hardware_emulator = hardware_emulator

    def set_distribution_mode(self, token_mode: DistributionMode, expert_mode: DistributionMode):
        """Set distribution modes for token and expert blocks

        Args:
            token_mode: Distribution mode for token blocks
            expert_mode: Distribution mode for expert blocks
        """
        self.token_distribution_mode = token_mode
        self.expert_distribution_mode = expert_mode

    def distribute(self):
        """Execute distribution strategy based on set modes"""
        added_blocks = []

        # Execute token distribution
        if self.token_distribution_mode == DistributionMode.EVEN:
            added_blocks.extend(self._even_distribute_tokens())
        elif self.token_distribution_mode == DistributionMode.CORRESPONDING:
            added_blocks.extend(self._corresponding_distribute_tokens())
        elif self.token_distribution_mode == DistributionMode.SEQUENTIAL:
            added_blocks.extend(self._sequential_distribute_tokens())

        # Execute expert distribution
        if self.expert_distribution_mode == DistributionMode.EVEN:
            added_blocks.extend(self._even_distribute_experts())
        elif self.expert_distribution_mode == DistributionMode.CORRESPONDING:
            added_blocks.extend(self._corresponding_distribute_experts())
        elif self.expert_distribution_mode == DistributionMode.SEQUENTIAL:
            added_blocks.extend(self._sequential_distribute_experts())

        return added_blocks

    def _corresponding_distribute_tokens(self):
        """Distribute token blocks to chips by keeping all blocks with same seq_or_expert_id together

        Each chip stores complete token sequence rather than distributing across different chips.
        Raises error if a sequence cannot fit in one chip.
        Returns a list of added token blocks.
        """
        added_blocks = []
        token_blocks = self.task_emulator.current_alive_token

        if not token_blocks:
            return added_blocks

        # Group blocks by seq_or_expert_id
        block_groups = {}
        for block in token_blocks:
            if block.seq_or_expert_id not in block_groups:
                block_groups[block.seq_or_expert_id] = []
            block_groups[block.seq_or_expert_id].append(block)

        current_chip_idx = 0

        # Distribute each group to a chip
        for seq_id, blocks in block_groups.items():
            # Check if blocks can fit in one chip's token buffer
            if len(blocks) > HardwareConfig.token_buffer_size:
                raise ValueError(
                    f"Token sequence {seq_id} with {len(blocks)} blocks cannot fit in one chip with capacity {HardwareConfig.token_buffer_size}")

            chip_id = self.hardware_emulator.chips[current_chip_idx]

            # Add all blocks from this sequence to current chip
            for block in blocks:
                self.hardware_emulator.add_new_token_from_bottom(
                    block, chip_id)
                added_blocks.append(block)

            # Move to next chip
            current_chip_idx = (current_chip_idx +
                                1) % len(self.hardware_emulator.chips)

        return added_blocks

    def _even_distribute_tokens(self):
        """Distribute token blocks evenly across chips"""
        added_blocks = []

        total_chips = len(self.hardware_emulator.chips)
        token_blocks = self.task_emulator.current_alive_token
        blocks_per_chip = len(token_blocks) // total_chips
        remaining_blocks = len(token_blocks) % total_chips

        # Traverse all chips and token blocks for distribution
        block_idx = 0
        for chip_idx, chip_id in enumerate(self.hardware_emulator.chips):
            # Calculate number of blocks to be allocated to current chip
            curr_chip_blocks = blocks_per_chip + \
                (1 if chip_idx < remaining_blocks else 0)

            # Allocate specified number of blocks to current chip
            for _ in range(curr_chip_blocks):
                if block_idx >= len(token_blocks):
                    break

                token_block = token_blocks[block_idx]
                # Find first available buffer position on current chip
                self.hardware_emulator.add_new_token_from_bottom(token_block,
                                                                 chip_id)
                added_blocks.append(token_block)
                block_idx += 1

        return added_blocks

    def _sequential_distribute_tokens(self):
        """Distribute token blocks by filling each chip buffer sequentially"""
        added_blocks = []
        token_blocks = self.task_emulator.current_alive_token
        
        if not token_blocks:
            return added_blocks

        current_chip_idx = 0
        current_chip_count = 0

        for block in token_blocks:
            # Move to next chip if current chip token buffer is full
            if current_chip_count >= HardwareConfig.token_buffer_size:
                current_chip_idx = (current_chip_idx + 1) % len(self.hardware_emulator.chips)
                current_chip_count = 0

            chip_id = self.hardware_emulator.chips[current_chip_idx]
            self.hardware_emulator.add_new_token_from_bottom(block, chip_id)
            added_blocks.append(block)
            current_chip_count += 1

        return added_blocks

    def _even_distribute_experts(self):
        """Distribute expert blocks evenly across DDRs"""
        added_blocks = []

        expert_load_num = min(
            len(self.task_emulator.sorted_expert_blocks),
            len(self.hardware_emulator.ddrs) * HardwareConfig.ddr_buffer_size
        )
        expert_blocks = self.task_emulator.load_expert(expert_load_num)

        # Distribute expert blocks sequentially across DDRs
        current_ddr_idx = 0
        for expert_block in expert_blocks:
            assert expert_block.seq_or_expert_id > 0
            self.hardware_emulator.add_new_ddr_block_from_bottom(
                expert_block, self.hardware_emulator.ddrs[current_ddr_idx])
            # Move to next DDR
            current_ddr_idx = (current_ddr_idx +
                               1) % len(self.hardware_emulator.ddrs)
            added_blocks.append(expert_block)

        return added_blocks

    def _corresponding_distribute_experts(self):
        """Distribute expert blocks to DDRs by keeping all micro slices of the same expert together

        Each DDR stores complete experts rather than distributing micro slices across different DDRs.
        Returns a list of added expert blocks.
        """
        added_blocks = []

        # Calculate maximum number of expert blocks that can be loaded
        expert_load_num = min(
            len(self.task_emulator.sorted_expert_blocks),
            len(self.hardware_emulator.ddrs) * HardwareConfig.ddr_buffer_size
        )
        expert_blocks = self.task_emulator.load_expert(expert_load_num)

        if not expert_blocks:
            return added_blocks

        current_ddr_idx = 0
        current_expert_id = expert_blocks[0].seq_or_expert_id

        for expert_block in expert_blocks:
            # When encountering a new expert, move to next DDR
            if expert_block.seq_or_expert_id != current_expert_id:
                current_ddr_idx = (current_ddr_idx +
                                   1) % len(self.hardware_emulator.ddrs)
                current_expert_id = expert_block.seq_or_expert_id

            # Add block to current DDR
            self.hardware_emulator.add_new_ddr_block_from_bottom(
                expert_block,
                self.hardware_emulator.ddrs[current_ddr_idx]
            )
            added_blocks.append(expert_block)

        return added_blocks

    def _sequential_distribute_experts(self):
        """Distribute expert blocks by filling each DDR buffer sequentially
        
        Fills each DDR buffer to capacity before moving to the next DDR.
        Returns a list of added expert blocks.
        """
        added_blocks = []
        
        # Calculate maximum number of expert blocks that can be loaded
        expert_load_num = min(
            len(self.task_emulator.sorted_expert_blocks),
            len(self.hardware_emulator.ddrs) * HardwareConfig.ddr_buffer_size
        )
        expert_blocks = self.task_emulator.load_expert(expert_load_num)
        
        if not expert_blocks:
            return added_blocks

        current_ddr_idx = 0
        current_ddr_count = 0

        for block in expert_blocks:
            # Move to next DDR if current DDR buffer is full
            if current_ddr_count >= HardwareConfig.ddr_buffer_size:
                current_ddr_idx = (current_ddr_idx + 1) % len(self.hardware_emulator.ddrs)
                current_ddr_count = 0
                
            ddr_id = self.hardware_emulator.ddrs[current_ddr_idx]
            self.hardware_emulator.add_new_ddr_block_from_bottom(block, ddr_id)
            added_blocks.append(block)
            current_ddr_count += 1

        return added_blocks
