#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_emulator import TaskEmulator
from animation import ChipletAnimator
from scheduler import FSEDPScheduler
from hardware_config import HardwareConfig
from action import ActionType
from strategy.token_redispatcher import RedispatchMode
from strategy.ddr_load_strategy import DDRLoadMode, DestinationSelectMode
from strategy.init_distribution_strategy import DistributionMode
from strategy.path_generator import PathMode
import argparse
import matplotlib
if os.environ.get('DISPLAY', '') == '' and not sys.platform.startswith('win'):
    print('No display environment, using Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Spatial grouping demo: per-expert chip groups on 3x3 array')
    parser.add_argument('--duration', type=int, default=60, help='Animation duration (seconds)')
    parser.add_argument('--fps', type=int, default=15, help='Animation frame rate')
    parser.add_argument('--save', type=str, default='', help='Save animation to video file')
    return parser.parse_args()


def main():
    args = parse_args()

    # 3x3 chip array
    HardwareConfig.row_number_of_chip = 3
    HardwareConfig.column_number_of_chip = 3
    HardwareConfig.token_buffer_size = 12
    HardwareConfig.expert_buffer_size = 12
    HardwareConfig.ddr_buffer_size = 48
    HardwareConfig.action_durations = {
        ActionType.DDR_LOAD: 12,
        ActionType.DDR_SAVE: 30,
        ActionType.CHIP_MOVE: 8,
        ActionType.COMPUTE: 5,
        ActionType.RELEASE: 2,
        ActionType.ALLOCATE: 1,
    }

    scheduler = FSEDPScheduler()

    # Configure task: two experts to illustrate grouping
    scheduler.task_emulator.expert_number = 2
    scheduler.task_emulator.micro_slice_num_per_expert = 16
    scheduler.task_emulator.min_activated_token = 8
    scheduler.task_emulator.token_weight_threshould = 2
    
    # Init distribution: sequence tokens spread sequentially; experts loaded alternately
    scheduler.init_strategy.expert_distribution_mode = DistributionMode.CORRESPONDING
    scheduler.init_strategy.token_distribution_mode = DistributionMode.SEQUENTIAL
    scheduler.ddr_load_strategy.ddr_load_mode = DDRLoadMode.ALTERNATE
    scheduler.ddr_load_strategy.destination_select_mode = DestinationSelectMode.MAX_BUFFER
    scheduler.path_generator.path_mode = PathMode.VOTEX
    scheduler.token_dispatcher.redispatch_mode = RedispatchMode.SIMPLE

    # Define per-expert chip groups (two disjoint 2x2 clusters inside 3x3)
    group_expert_1 = [
        "Chip(0, 0)", "Chip(0, 1)",
        "Chip(1, 0)", "Chip(1, 1)",
    ]
    group_expert_2 = [
        "Chip(1, 1)", "Chip(1, 2)",
        "Chip(2, 1)", "Chip(2, 2)",
    ]
    # Note: they overlap on Chip(1,1) to show potential contention; adjust if strict disjointness needed.
    scheduler.set_expert_groups({
        1: group_expert_1,
        2: group_expert_2,
    })

    # Initialize data blocks
    scheduler.init_data_blocks()

    animator = ChipletAnimator(scheduler)

    frames = args.duration * args.fps
    interval = 1000 / args.fps

    fig = plt.figure(figsize=(12, 8))
    ani = FuncAnimation(fig, animator.update, frames=frames, init_func=animator.init, blit=True, interval=interval)

    if args.save:
        print(f'Saving animation to {args.save}...')
        animator.is_running = True
        try:
            writer = FFMpegWriter(fps=args.fps)
            ani.save(args.save, writer=writer)
            print('Save completed!')
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Please ensure FFmpeg is installed and added to PATH")
    else:
        print("Displaying animation...")
        plt.show()
        print("Animation display completed")


if __name__ == '__main__':
    main()
