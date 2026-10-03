#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hardware Emulator Module

This module provides a hardware emulator that simulates the behavior of a chip array system
with multiple DDR memory units. It handles data block management and expert loading operations.
"""

from hardware_config import HardwareConfig
from data_block import DataBlock, DataBlockType
from typing import List, Dict, Tuple, Optional


class HardwareEmulator:
    """Hardware Emulator Class - Simulates chip array and DDR memory system"""

    def __init__(self):
        self.row_number_of_chip = HardwareConfig.row_number_of_chip
        self.column_number_of_chip = HardwareConfig.column_number_of_chip
        self.chip_num = self.column_number_of_chip * self.row_number_of_chip

        # Create chip ID list based on chip rows and columns
        self.chips = []
        for row in range(self.row_number_of_chip):
            for col in range(self.column_number_of_chip):
                self.chips.append(f"Chip({row}, {col})")
        self.ddrs = [f"DDR{i}" for i in range(1, 5)]

        # Sequence data blocks in chips {chip_id: [data_blocks]}
        self.chip_seq_data_blocks = {
            chip: [None] * HardwareConfig.token_buffer_size for chip in self.chips}

        # Expert data blocks in chips {chip_id: [data_blocks]}
        self.chip_expert_data_blocks = {
            chip: [None] * HardwareConfig.expert_buffer_size for chip in self.chips}

        # Data blocks in DDR memory {ddr_id: [data_blocks]}
        self.ddr_data_blocks = {
            ddr: [None] * HardwareConfig.ddr_buffer_size for ddr in self.ddrs}
        # self.ddr_data_blocks = {ddr: [] for ddr in self.ddrs}

        # DDR bandwidth occupation status
        self.ddr_bandwidth_used = {ddr: False for ddr in self.ddrs}

    def add_new_block_in_location(self, new_block: DataBlock, location: Tuple[str, int], cover_existing: bool = False):
        """Add a new data block to the specified location

        Args:
            new_block (DataBlock): The data block to be added
            location (Tuple[str, int]): Target location tuple (device_id, buffer_index)
            cover_existing (bool, optional): Whether to allow overwriting existing data blocks. 
                If True, will overwrite any existing block at the target location.
                If False, will raise an assertion error if target location is occupied.
                Defaults to True.
        """
        target = location[0]  # Device ID (Chip or DDR)
        index = location[1]   # Buffer index
        new_block.location = location

        if target.startswith("Chip"):
            # Handle chip storage based on block type
            if new_block.block_type == DataBlockType.SEQ:
                assert location[1] >= 0 and location[1] < HardwareConfig.token_buffer_size, f"Invalid buffer index {location[1]}"
                if not cover_existing:
                    assert self.chip_seq_data_blocks[target][
                        index] is None, f"Chip buffer {index} is occupied"
                self.chip_seq_data_blocks[target][index] = new_block
            else:  # Expert block type
                assert location[1] >= 0 and location[1] < HardwareConfig.expert_buffer_size, f"Invalid buffer index {location[1]}"
                if not cover_existing:
                    assert self.chip_expert_data_blocks[target][
                        index] is None, f"Chip buffer {index} is occupied"
                self.chip_expert_data_blocks[target][index] = new_block
        elif target.startswith("DDR"):
            assert location[1] >= 0 and location[1] < HardwareConfig.ddr_buffer_size, f"Invalid buffer index {location[1]}"
            # Handle DDR storage
            if not cover_existing:
                assert self.ddr_data_blocks[target][
                    index] is None, f"DDR buffer {index} is occupied"
            self.ddr_data_blocks[target][index] = new_block

    def remove_data_block_from_location(self, location: Tuple[str, int], buffer_type: DataBlockType = None):
        """Remove a data block from the specified location

        Args:
            data_block (DataBlock): The data block to be removed
            location (Tuple[str, int]): Source location tuple (device_id, buffer_index)

        Raises:
            AssertionError: If buffer usage count becomes negative after removal
            ValueError: If no data block exists at the specified location
        """
        target = location[0]  # Device ID (Chip or DDR)
        index = location[1]   # Buffer index
        assert index >= 0, f"Invalid buffer index {index}"

        removed_block = None

        if target.startswith("Chip"):
            # Handle chip removal based on block type
            if buffer_type == DataBlockType.SEQ:
                removed_block = self.chip_seq_data_blocks[target][index]
                if removed_block is None:
                    raise ValueError(
                        f"No sequence block exists at location {location}")
                self.chip_seq_data_blocks[target][index] = None
            elif buffer_type == DataBlockType.EXPERT:  # Expert block type
                removed_block = self.chip_expert_data_blocks[target][index]
                if removed_block is None:
                    raise ValueError(
                        f"No expert block exists at location {location}")
                self.chip_expert_data_blocks[target][index] = None
            else:
                raise ValueError("Unknown buffer type for chip removal")
        elif target.startswith("DDR"):
            # Handle DDR removal
            removed_block = self.ddr_data_blocks[target][index]
            if removed_block is None:
                raise ValueError(
                    f"No data block exists at DDR location {location}")
            self.ddr_data_blocks[target][index] = None

        return removed_block

    def remove_data_block(self, data_block: DataBlock):
        """Remove a data block from the specified location"""
        if data_block.location is None:
            raise ValueError("Data block has no location assigned")
        self.remove_data_block_from_location(
            data_block.location, data_block.block_type)

    def get_min_empty_seq_index(self, chip_id: str) -> int:
        """Get the minimum empty index in the sequence buffer of the specified chip"""
        try:
            return self.chip_seq_data_blocks[chip_id].index(None)
        except ValueError:
            return -1  # Return -1 if no empty position is found

    def get_max_empty_seq_index(self, chip_id: str) -> int:
        """Get the maximum empty index in the sequence buffer of the specified chip"""
        for i in range(len(self.chip_seq_data_blocks[chip_id])-1, -1, -1):
            if self.chip_seq_data_blocks[chip_id][i] is None:
                return i
        return -1  # Return -1 if no empty position is found

    def get_min_empty_expert_index(self, chip_id: str) -> int:
        """Get the minimum empty index in the expert buffer of the specified chip"""
        try:
            return self.chip_expert_data_blocks[chip_id].index(None)
        except ValueError:
            return -1  # Return -1 if no empty position is found

    def get_max_empty_expert_index(self, chip_id: str) -> int:
        """Get the maximum empty index in the expert buffer of the specified chip"""
        for i in range(len(self.chip_expert_data_blocks[chip_id])-1, -1, -1):
            if self.chip_expert_data_blocks[chip_id][i] is None:
                return i
        return -1  # Return -1 if no empty position is found

    def get_min_empty_ddr_index(self, ddr_id: str) -> int:
        """Get the minimum empty index in the ddr buffer of the specified ddr"""
        try:
            return self.ddr_data_blocks[ddr_id].index(None)
        except ValueError:
            return -1  # Return -1 if no empty position is found

    def get_min_non_empty_ddr_index(self, ddr_id: str) -> int:
        """Get the minimum non-empty index in the DDR buffer of the specified DDR"""
        for i in range(len(self.ddr_data_blocks[ddr_id])):
            if self.ddr_data_blocks[ddr_id][i] is not None:
                return i
        return -1  # Return -1 if all positions are empty

    def add_new_token_from_bottom(self, token_block: DataBlock, chip_id: str):
        """Add a new token data block to the bottom of the sequence buffer of the specified chip"""
        min_empty_index = self.get_min_empty_seq_index(chip_id)
        if min_empty_index != -1:
            self.add_new_block_in_location(
                token_block, (chip_id, min_empty_index))
        else:
            raise ValueError(f"Chip {chip_id} sequence buffer is full")

    def add_new_token_from_top(self, token_block: DataBlock, chip_id: str):
        """Add a new token data block to the top of the sequence buffer of the specified chip"""
        max_empty_index = self.get_max_empty_seq_index(chip_id)
        if max_empty_index != -1:
            self.add_new_block_in_location(
                token_block, (chip_id, max_empty_index))
        else:
            raise ValueError(f"Chip {chip_id} sequence buffer is full")

    def add_new_expert_from_bottom(self, expert_block: DataBlock, chip_id: str):
        """Add a new expert data block to the bottom of the expert buffer of the specified chip"""
        min_empty_index = self.get_min_empty_expert_index(chip_id)
        if min_empty_index != -1:
            self.add_new_block_in_location(
                expert_block, (chip_id, min_empty_index))
        else:
            raise ValueError(f"Chip {chip_id} expert buffer is full")

    def add_new_expert_from_top(self, expert_block: DataBlock, chip_id: str):
        """Add a new expert data block to the top of the expert buffer of the specified chip"""
        max_empty_index = self.get_max_empty_expert_index(chip_id)
        if max_empty_index != -1:
            self.add_new_block_in_location(
                expert_block, (chip_id, max_empty_index))
        else:
            raise ValueError(f"Chip {chip_id} expert buffer is full")

    def add_new_ddr_block_from_bottom(self, ddr_block: DataBlock, ddr_id: str):
        """Add a new ddr data block to the bottom of the ddr buffer of the specified ddr"""
        min_empty_index = self.get_min_empty_ddr_index(ddr_id)
        if min_empty_index != -1:
            self.add_new_block_in_location(
                ddr_block, (ddr_id, min_empty_index))
        else:
            raise ValueError(f"DDR {ddr_id} buffer is full")

    def get_next_ddr_block(self, ddr_id, offset: int = 0) -> DataBlock:
        """Get the next data block from the specified DDR with an offset

        Args:
            ddr_id (str): The ID of the target DDR
            offset (int, optional): Offset from the minimum non-empty index. Defaults to 0.

        Returns:
            DataBlock: The data block at the target position

        Raises:
            AssertionError: If the calculated index is invalid or the target position is empty
        """
        index = self.get_min_non_empty_ddr_index(ddr_id) + offset
        assert index >= 0 and index < HardwareConfig.ddr_buffer_size, f"Invalid DDR index: {index}"
        assert self.ddr_data_blocks[ddr_id][index] is not None, "Target position is empty"
        return self.ddr_data_blocks[ddr_id][index]

    def get_ddr_block_by_index(self, ddr_id: str, index: int) -> DataBlock:
        """Get a data block from the specified DDR at the given index

        Args:
            ddr_id (str): The ID of the target DDR
            index (int): The buffer index to retrieve from

        Returns:
            DataBlock: The data block at the specified index

        Raises:
            AssertionError: If the index is invalid or the target position is empty
            ValueError: If the DDR ID is invalid
        """
        assert index >= 0 and index < HardwareConfig.ddr_buffer_size, f"Invalid DDR index: {index}"
        if ddr_id not in self.ddrs:
            raise ValueError(f"Invalid DDR ID: {ddr_id}")

        block = self.ddr_data_blocks[ddr_id][index]
        if block is None:
            raise ValueError(
                f"No data block exists at DDR {ddr_id} index {index}")

        return block

    def is_chip_expert_buffer_empty(self, chip_id: str) -> bool:
        """判断指定 chip 的 expert 缓冲区是否全为空（全为 None）"""
        if chip_id not in self.chips:
            raise ValueError(f"Invalid chip ID: {chip_id}")
        return all(block is None for block in self.chip_expert_data_blocks[chip_id])

    def is_chip_seq_buffer_empty(self, chip_id: str) -> bool:
        """判断指定 chip 的 sequence 缓冲区是否全为空（全为 None）"""
        if chip_id not in self.chips:
            raise ValueError(f"Invalid chip ID: {chip_id}")
        return all(block is None for block in self.chip_seq_data_blocks[chip_id])

    def get_remaining_expert_space(self, chip_id: str) -> int:
        """Get the remaining space in the expert buffer of the specified chip"""
        if chip_id not in self.chips:
            raise ValueError(f"Invalid chip ID: {chip_id}")
        return self.chip_expert_data_blocks[chip_id].count(None)

    def get_remaining_seq_space(self, chip_id: str) -> int:
        """Get the remaining space in the sequence buffer of the specified chip"""
        if chip_id not in self.chips:
            raise ValueError(f"Invalid chip ID: {chip_id}")
        return self.chip_seq_data_blocks[chip_id].count(None)

    def get_ramaining_on_chip_space(self, chip_id: str) -> int:
        """Get total remaining space (expert + sequence buffers) on the specified chip"""
        return self.get_remaining_expert_space(chip_id) + self.get_remaining_seq_space(chip_id)

    def get_remaining_ddr_space(self, ddr_id: str) -> int:
        """Get the remaining space in the buffer of the specified DDR"""
        if ddr_id not in self.ddrs:
            raise ValueError(f"Invalid DDR ID: {ddr_id}")
        return self.ddr_data_blocks[ddr_id].count(None)
