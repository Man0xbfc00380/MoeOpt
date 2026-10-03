import sys
from scheduler import EPScheduler
from action import ActionType
from hardware_config import HardwareConfig
from strategy.init_distribution_strategy import DistributionMode
from strategy.ddr_load_strategy import DDRLoadMode, DestinationSelectMode

# Small hardware setup
HardwareConfig.row_number_of_chip = 2
HardwareConfig.column_number_of_chip = 2
HardwareConfig.ddr_buffer_size = 32
HardwareConfig.action_durations = {
    ActionType.DDR_LOAD: 1,
    ActionType.DDR_SAVE: 1,
    ActionType.CHIP_MOVE: 1,
    ActionType.COMPUTE: 1,
    ActionType.RELEASE: 1,
    ActionType.ALLOCATE: 1,
}

s = EPScheduler()
# Task params
s.task_emulator.expert_number = 4
s.task_emulator.request_num = 2
s.task_emulator.token_num_per_request = 8
s.task_emulator.min_activated_token = 0
s.task_emulator.token_weight_threshould = 1

# Distribute: tokens to chips, experts to DDR
s.init_strategy.expert_distribution_mode = DistributionMode.EVEN
s.init_strategy.token_distribution_mode = DistributionMode.EVEN
s.ddr_load_strategy.ddr_load_mode = DDRLoadMode.ALTERNATE
s.ddr_load_strategy.destination_select_mode = DestinationSelectMode.MAX_BUFFER

s.init_data_blocks()

finished = False
steps = 0
while not finished and steps < 50:
    actions = s.step()
    # Simulate animation completion callbacks
    for a in actions:
        if a.action_type in (ActionType.DDR_LOAD, ActionType.CHIP_MOVE, ActionType.DDR_SAVE, ActionType.COMPUTE):
            s.handle_action_complete(a)
        elif a.action_type == ActionType.FLAG and a.data_block.name == 'finish':
            finished = True
    steps += 1

print("EP smoke test done.", "steps=", steps, "finished=", finished)
