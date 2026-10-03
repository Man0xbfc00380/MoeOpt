#!/usr/bin/env python
# -*- coding: utf-8 -*-

from action import ActionType

class HardwareConfig:
    row_number_of_chip = 2
    column_number_of_chip = 2
    token_buffer_size = 16
    expert_buffer_size = 16
    ddr_buffer_size = 16

    
    @classmethod
    def ddr_chip_connections(cls):
        return {
            "DDR1": f"Chip(0, 0)",
            "DDR2": f"Chip(0, {cls.column_number_of_chip-1})",
            "DDR3": f"Chip({cls.row_number_of_chip-1}, {cls.column_number_of_chip-1})",
            "DDR4": f"Chip({cls.row_number_of_chip-1}, 0)"
        }

    action_durations = {
            ActionType.DDR_LOAD: 20,  # Number of frames for loading from DDR to chip
            ActionType.DDR_SAVE: 40,  # Number of frames for saving from chip to DDR
            ActionType.CHIP_MOVE: 5,  # Number of frames for inter-chip movement
            ActionType.COMPUTE: 5,  # Number of frames for computation
            ActionType.RELEASE: 2,  # Number of frames for release
            ActionType.ALLOCATE: 1,
        }
