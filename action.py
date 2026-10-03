#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dynamic Scheduling Animation for Chip Task Execution - Action Class

This module defines the Action class to describe various operations during chip task execution,
including data loading/saving between DDR and chips, data movement between chips,
and actions like computation, allocation and release.
"""
from enum import Enum, auto
from typing import Tuple, List, Optional
from data_block import DataBlock, DataBlockType


class ActionType(Enum):
    """Action Types for Scheduling"""
    DDR_LOAD = auto()  # Load data from DDR to chip
    DDR_SAVE = auto()  # Save data from chip to DDR
    CHIP_MOVE = auto()  # Move data block between chips
    COMPUTE = auto()  # Compute data block
    RELEASE = auto()  # Release data block
    ALLOCATE = auto()  # Allocate storage space for data block

    FLAG = auto()  # Special event marker for scheduling phases


class Action:
    """Scheduling Action"""

    def __init__(self, action_type: ActionType, data_block: DataBlock,
                 source: Tuple[int, int] = None, destination: Tuple[int, int] = None, other_blocks: List[DataBlock] = None):
        self.action_type = action_type
        self.data_block = data_block
        self.source = source
        self.destination = destination
        # For actions involving multiple data blocks, like COMPUTE
        self.other_blocks = other_blocks

    def __str__(self):
        if self.action_type == ActionType.DDR_LOAD:
            return f"Load {self.data_block.block_id} from {self.source} to {self.destination}"
        elif self.action_type == ActionType.DDR_SAVE:
            return f"Save {self.data_block.block_id} from {self.source} to {self.destination}"
        elif self.action_type == ActionType.CHIP_MOVE:
            return f"Move {self.data_block.block_id} from {self.source} to {self.destination}"
        elif self.action_type == ActionType.COMPUTE:
            return f"Compute {self.data_block.block_id} with {self.other_blocks} on {self.destination}"
        elif self.action_type == ActionType.RELEASE:
            return f"Release {self.data_block.block_id} from {self.source}"
        elif self.action_type == ActionType.ALLOCATE:
            return f"Allocate space for {self.data_block.block_id} on {self.destination}"
        elif self.action_type == ActionType.FLAG:
            return f"Flag event: {self.data_block.block_id}"
        return "Unknown Action"

    def __repr__(self):
        return self.__str__()
