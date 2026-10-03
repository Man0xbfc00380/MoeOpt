#!/usr/bin/env python
# -*- coding: utf-8 -*-

from hardware_config import HardwareConfig
from task_emulator import TaskEmulator
from hardware_emulator import HardwareEmulator
from typing import List, Dict, Tuple, Optional
from enum import Enum
from data_block import DataBlock


class DDRLoadMode(Enum):
    """
    Defines two DDR loading modes:
    SEQUENTIAL - Sequential incremental loading mode, loading from low address to high address
    ALTERNATE - Alternating loading mode, alternately increased loading between two different regions
    """
    SEQUENTIAL = 1  # Sequential incremental loading
    ALTERNATE = 2   # Alternating loading


class DestinationSelectMode(Enum):
    """
    Defines two destination selection modes:
    NEAREST - Select the chip closest to DDR as destination
    MAX_BUFFER - Select the chip with maximum remaining buffer space as destination
    MAX_BONDED_TOKENS - Select the chip with maximum tokens bound to the expert
    """
    NEAREST = 1     # Select nearest chip
    MAX_BUFFER = 2  # Select chip with max buffer space
    MAX_BONDED_TOKENS = 3  # Select chip with most bonded tokens for expert


class DDRLoadStrategy():
    """A strategy class that manages DDR memory loading operations.

    This class implements different DDR loading strategies including sequential and alternate modes.
    It maintains the current loading state and provides methods to load data blocks from DDR memory.
    """

    def __init__(self, hardware_emulator: HardwareEmulator = None, task_emulator: Optional[TaskEmulator] = None):
        """Initialize the DDR loading strategy.

        Args:
            hardware_emulator (HardwareEmulator, optional): The hardware emulator instance that provides 
                access to DDR memory operations. Defaults to None.
            task_emulator (TaskEmulator, optional): The task emulator instance providing
                expert-token binding relations. Defaults to None.
        """
        self.task_emulator = task_emulator
        self.hardware_emulator = hardware_emulator
        self._ddr_load_mode = DDRLoadMode.SEQUENTIAL
        self._destination_select_mode = DestinationSelectMode.NEAREST

        self.is_ddr_pointer_1 = {
            ddr: True for ddr in self.hardware_emulator.ddrs}
        self.ddr_addr = {ddr: 0 for ddr in self.hardware_emulator.ddrs}
        # Optional expert chip groups constraints: expert_id -> set(chip_id)
        self.expert_groups: Dict[int, set] = {}

    @property
    def ddr_load_mode(self) -> DDRLoadMode:
        return self._ddr_load_mode

    @ddr_load_mode.setter
    def ddr_load_mode(self, mode: DDRLoadMode):
        if not isinstance(mode, DDRLoadMode):
            raise ValueError(
                f"Invalid DDR load mode: {mode}. Must use DDRLoadMode enum type")
        self._ddr_load_mode = mode

    @property
    def destination_select_mode(self) -> DestinationSelectMode:
        return self._destination_select_mode

    @destination_select_mode.setter
    def destination_select_mode(self, mode: DestinationSelectMode):
        if not isinstance(mode, DestinationSelectMode):
            raise ValueError(
                f"Invalid destination select mode: {mode}. Must use DestinationSelectMode enum type")
        self._destination_select_mode = mode

    def _select_nearest_destination(self, ddr_id: str) -> tuple:
        """Select the chip closest to the specified DDR as destination"""
        target_chip = HardwareConfig.ddr_chip_connections()[ddr_id]
        index = self.hardware_emulator.get_max_empty_expert_index(target_chip)
        if index == -1:
            raise ValueError(f"No available space in {target_chip}")
        return (target_chip, index)

    def _select_max_buffer_destination(self, ddr_id: str) -> tuple:
        """Select the chip with maximum remaining buffer space as destination"""
        max_space = -1
        target_chip = None
        nearest_chip = HardwareConfig.ddr_chip_connections()[ddr_id]
        nearest_chip_space = self.hardware_emulator.get_remaining_expert_space(
            nearest_chip)

        # Iterate through all chips to find the one with maximum space
        for chip in self.hardware_emulator.chips:
            space = self.hardware_emulator.get_remaining_expert_space(chip)
            if space > max_space:
                max_space = space
                target_chip = chip

        if target_chip is None or max_space == 0:
            raise ValueError("No available space in any chip")
            # return (nearest_chip, -1)

        # If the nearest chip has equal space as the maximum space found, select the nearest chip
        if nearest_chip_space == max_space:
            target_chip = nearest_chip

        index = self.hardware_emulator.get_max_empty_expert_index(target_chip)
        if index == -1:
            raise ValueError(f"No available space in {target_chip}")
        return (target_chip, index)

    def _select_max_bonded_tokens_destination(self, ddr_id: str, expert_id: int) -> tuple:
        """Select chip whose sequence buffer contains the most tokens bound to expert.

        Only considers chips with available expert buffer space. Falls back to
        nearest chip if bindings unavailable or no eligible chips.
        """
        # Guard: require task_emulator and binding relations
        if self.task_emulator is None:
            return self._select_nearest_destination(ddr_id)

        tokens_for_expert = self.task_emulator.bind_relations.get(expert_id, set())
        if not tokens_for_expert:
            return self._select_nearest_destination(ddr_id)

        # Tie-break by proximity to DDR if counts equal
        nearest_chip = HardwareConfig.ddr_chip_connections()[ddr_id]

        best_chip = None
        best_count = -1
        skip_full_chip = True

        for chip in self.hardware_emulator.chips:
            # Skip chips without expert buffer space
            idx = self.hardware_emulator.get_max_empty_expert_index(chip)
            if idx == -1:
                if skip_full_chip:
                    continue
                else:
                    raise ValueError(f"No available space in {chip}")
            
            seq_buf = self.hardware_emulator.chip_seq_data_blocks.get(chip, [])
            count = 0
            for block in seq_buf:
                if block is None:
                    continue
                # Count only sequence blocks whose name is bound to the expert
                try:
                    name = block.name
                except Exception:
                    name = None
                if name and name in tokens_for_expert:
                    count += 1

            if count > best_count:
                best_count = count
                best_chip = chip
            elif count == best_count and count >= 0:
                # Prefer nearest chip on tie
                if best_chip is None:
                    best_chip = chip
                elif chip == nearest_chip and best_chip != nearest_chip:
                    best_chip = chip

            # 如果芯片上没有 expert，则优先选择该芯片
            if self.hardware_emulator.is_chip_expert_buffer_empty(chip) and count > 0:
                return (chip, 0)

        if best_chip is None:
            return self._select_nearest_destination(ddr_id)

        index = self.hardware_emulator.get_max_empty_expert_index(best_chip)
        if index == -1:
            raise ValueError(f"No available space in {best_chip}")
        return (best_chip, index)

    def _select_max_buffer_destination_in_group(self, ddr_id: str, expert_id: int) -> tuple:
        """Select chip with max expert buffer space within the expert's group.

        Fallback to nearest if no space available in the group.
        """
        allowed = self.expert_groups.get(expert_id)
        if not allowed:
            return self._select_max_buffer_destination(ddr_id)

        max_space = -1
        target_chip = None
        for chip in allowed:
            if chip not in self.hardware_emulator.chips:
                continue
            space = self.hardware_emulator.get_remaining_expert_space(chip)
            if space > max_space:
                max_space = space
                target_chip = chip

        if target_chip is None or max_space <= 0:
            # Fallback to nearest
            return self._select_nearest_destination(ddr_id)

        index = self.hardware_emulator.get_max_empty_expert_index(target_chip)
        if index == -1:
            raise ValueError(f"No available space in {target_chip}")
        return (target_chip, index)

    def load_next_data_block(self, ddr_id: str) -> DataBlock:
        # Determine target location based on destination selection mode
        if self._destination_select_mode == DestinationSelectMode.NEAREST:
            destination = self._select_nearest_destination(ddr_id)
        elif self._destination_select_mode == DestinationSelectMode.MAX_BUFFER:
            # Try to respect expert group if the block belongs to an expert
            # We cannot know the expert before reading, so we peek by address
            # Note: This relies on DDR index mapping to expert blocks
            expert_block = self._peek_next_block(ddr_id)
            if expert_block is not None and expert_block.seq_or_expert_id in self.expert_groups:
                destination = self._select_max_buffer_destination_in_group(
                    ddr_id, expert_block.seq_or_expert_id)
            else:
                destination = self._select_max_buffer_destination(ddr_id)
        elif self._destination_select_mode == DestinationSelectMode.MAX_BONDED_TOKENS:
            # Need expert id to evaluate bonded tokens in chips
            expert_block = self._peek_next_block(ddr_id)
            if expert_block is not None:
                destination = self._select_max_bonded_tokens_destination(
                    ddr_id, expert_block.seq_or_expert_id)
            else:
                destination = self._select_nearest_destination(ddr_id)
        else:
            raise ValueError(
                f"Destination select mode {self.destination_select_mode} is not supported")

        if self.hardware_emulator.get_remaining_expert_space(destination[0]) < 2:
            raise ValueError(f"Destination chip {destination[0]} has insufficient expert buffer space (remaining < 2)")

        # Get data block based on loading mode
        if self._ddr_load_mode == DDRLoadMode.SEQUENTIAL:
            data_block = self._load_sequential(ddr_id)
        elif self._ddr_load_mode == DDRLoadMode.ALTERNATE:
            data_block = self._load_alternate(ddr_id)
        else:
            raise ValueError(
                f"DDR load mode {self.ddr_load_mode} is not supported")

        return data_block, destination

    def _peek_next_block(self, ddr_id: str) -> DataBlock:
        """Peek at the next block to be loaded without advancing pointers.

        Returns None if out of range or not available.
        """
        try:
            if self._ddr_load_mode == DDRLoadMode.SEQUENTIAL:
                addr = self.ddr_addr[ddr_id]
                if addr >= HardwareConfig.ddr_buffer_size:
                    return None
                return self.hardware_emulator.get_ddr_block_by_index(ddr_id, addr)
            elif self._ddr_load_mode == DDRLoadMode.ALTERNATE:
                # For peek, just use current address as approximation
                addr = self.ddr_addr[ddr_id]
                if addr >= HardwareConfig.ddr_buffer_size:
                    return None
                return self.hardware_emulator.get_ddr_block_by_index(ddr_id, addr)
        except Exception:
            return None

    def _load_sequential(self, ddr_id: str) -> DataBlock:
        """Sequential loading mode: loads from low address to high address, cycles when exceeding DDR size"""
        current_addr = self.ddr_addr[ddr_id]
        if current_addr >= HardwareConfig.ddr_buffer_size:
            raise ValueError("DDR address exceeded buffer size limit")
        self.ddr_addr[ddr_id] = current_addr + 1
        return self.hardware_emulator.get_ddr_block_by_index(ddr_id, current_addr)

    def _load_alternate(self, ddr_id: str, offset=4) -> DataBlock:
        """Alternate loading mode: alternates loading between two regions

        This mode divides DDR into two regions for alternate loading. The address difference between regions is offset.
        When loading from the first region, the address increases by (offset-1).
        When loading from the second region:
        - If current address%(2*offset)==(2*offset-1), increase by 1
        - Otherwise decrease by (offset-1)

        Args:
            ddr_id: DDR identifier
            offset: Address difference between two regions, defaults to 4

        Returns:
            DataBlock: Data block loaded from the specified DDR address
        """
        current_addr = self.ddr_addr[ddr_id]
        if current_addr >= HardwareConfig.ddr_buffer_size:
            raise ValueError("DDR address exceeded buffer size limit")
        if self.is_ddr_pointer_1[ddr_id]:
            self.ddr_addr[ddr_id] += offset
        else:
            # Load in second region
            if self.ddr_addr[ddr_id] % (2 * offset) == (2 * offset - 1):
                self.ddr_addr[ddr_id] += 1
            else:
                self.ddr_addr[ddr_id] -= (offset - 1)
        self.is_ddr_pointer_1[ddr_id] = not self.is_ddr_pointer_1[ddr_id]
        self.ddr_addr[ddr_id] = self.ddr_addr[ddr_id] % HardwareConfig.ddr_buffer_size
        return self.hardware_emulator.get_ddr_block_by_index(ddr_id, current_addr)
