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
from strategy.path_generator import PathGenerator, PathMode
import argparse
import time
import sys
import matplotlib
# If no display is available, use Agg backend
if os.environ.get('DISPLAY', '') == '' and not sys.platform.startswith('win'):
    print('No display environment, using Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Dynamic scheduling animation for chiplet task execution')
    parser.add_argument('--scheduler', type=str, default='fsdp')
    parser.add_argument('--duration', type=int, default=80,
                        help='Animation duration (seconds)')
    parser.add_argument('--fps', type=int, default=20,
                        help='Animation frame rate')
    parser.add_argument('--save', type=str, default='',
                        help='Save animation to video file')
    return parser.parse_args()


def main():
    """Main function"""
    try:
        args = parse_args()
        print(f"Using scheduler: {args.scheduler}")
        print(f"Animation duration: {args.duration} seconds")
        print(f"Animation frame rate: {args.fps} fps")

        # 用2×2阵列展示
        HardwareConfig.row_number_of_chip = 2
        HardwareConfig.column_number_of_chip = 2

        HardwareConfig.ddr_buffer_size = 32  # 放一个expert所有的micro slice
        HardwareConfig.action_durations = {
            ActionType.DDR_LOAD: 20,  # Number of frames for loading from DDR to chip
            ActionType.DDR_SAVE: 40,  # Number of frames for saving from chip to DDR
            ActionType.CHIP_MOVE: 10,  # Number of frames for inter-chip movement
            ActionType.COMPUTE: 5,  # Number of frames for computation
            ActionType.RELEASE: 2,  # Number of frames for release
            ActionType.ALLOCATE: 1,
        }

        # Create scheduler
        scheduler = FSEDPScheduler()

        # 只设置一个expert展示流水过程
        scheduler.task_emulator.expert_number = 1
        scheduler.task_emulator.micro_slice_num_per_expert = 16
        scheduler.task_emulator.min_activated_token = 8

        scheduler.init_strategy.expert_distribution_mode = DistributionMode.SEQUENTIAL  # 只放到一个DDR中
        scheduler.init_strategy.token_distribution_mode = DistributionMode.SEQUENTIAL

        scheduler.ddr_load_strategy.ddr_load_mode = DDRLoadMode.ALTERNATE
        scheduler.ddr_load_strategy.destination_select_mode = DestinationSelectMode.NEAREST
        scheduler.path_generator.path_mode = PathMode.VOTEX
        scheduler.token_dispatcher.redispatch_mode = RedispatchMode.SIMPLE  
        scheduler.init_data_blocks()

        # Create animator
        animator = ChipletAnimator(scheduler)

        # Set animation parameters
        frames = args.duration * args.fps
        interval = 1000 / args.fps  # milliseconds
        print(f"Total frames: {frames}, Frame interval: {interval} ms")

        # Create animation
        print("Creating animation...")
        fig = plt.figure(figsize=(12, 8))
        # Create animation object using FuncAnimation
        # fig: matplotlib figure object
        # animator.update: frame update function
        # frames: total number of frames
        # init_func: initialization function
        # blit=True: use buffer drawing to improve performance
        # interval: frame interval (milliseconds)
        ani = FuncAnimation(fig, animator.update, frames=frames,
                            init_func=animator.init, blit=True, interval=interval)

        # Save or display animation
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
    except Exception as e:
        print(f"Program error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
