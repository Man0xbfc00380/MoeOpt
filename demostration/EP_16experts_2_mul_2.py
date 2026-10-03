#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EPScheduler demo (2x2 chip array)

Functionality:
- Adopts Expert-Parallel (EP) strategy: each expert exclusively occupies one chip group (here, one chip per group).
- Configures hardware, scheduler, and expert grouping to showcase EPScheduler execution flow.

Run:
python demostration/EPScheduler_demo_EP_2x2.py --duration 50 --fps 15 [--save output.mp4]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animation import ChipletAnimator
from scheduler import EPScheduler
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
    parser = argparse.ArgumentParser(description='EPScheduler Demo (2x2): Experts assigned to disjoint chip groups')
    parser.add_argument('--duration', type=int, default=50)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--save', type=str, default='')
    return parser.parse_args()


def main():
    args = parse_args()

    # 2x2 chip array
    HardwareConfig.row_number_of_chip = 2
    HardwareConfig.column_number_of_chip = 2
    HardwareConfig.token_buffer_size = 16
    HardwareConfig.expert_buffer_size = 4
    HardwareConfig.ddr_buffer_size = 24
    # Shorten action durations for faster animation
    HardwareConfig.action_durations = {
        ActionType.DDR_LOAD: 30,
        ActionType.DDR_SAVE: 40,
        ActionType.CHIP_MOVE: 1,
        ActionType.COMPUTE: 3,
        ActionType.RELEASE: 1,
        ActionType.ALLOCATE: 1,
    }

    scheduler = EPScheduler()

    # Set task scale: 16 experts; under EP each expert has only 1 micro slice (EPScheduler internally enforces this)
    scheduler.task_emulator.expert_number = 16
    scheduler.task_emulator.micro_slice_num_per_expert = 1
    scheduler.task_emulator.token_threshold = 1
    scheduler.task_emulator.request_num = 1
    scheduler.task_emulator.token_num_per_request = 16

    # Scheduling/distribution strategy config (consistent with FSEDP EP demo style)
    scheduler.init_strategy.expert_distribution_mode = DistributionMode.EVEN
    scheduler.init_strategy.token_distribution_mode = DistributionMode.EVEN
    scheduler.ddr_load_strategy.ddr_load_mode = DDRLoadMode.SEQUENTIAL
    scheduler.ddr_load_strategy.destination_select_mode = DestinationSelectMode.MAX_BONDED_TOKENS
    scheduler.path_generator.path_mode = PathMode.VOTEX
    scheduler.token_dispatcher.redispatch_mode = RedispatchMode.NO_REDISPATCH
    scheduler.paired = False

    # Initialize data blocks (tokens distributed to chips, experts into DDR)
    scheduler.init_data_blocks()

    animator = ChipletAnimator(scheduler)
    frames = args.duration * args.fps
    interval = 1000 / args.fps

    fig = plt.figure(figsize=(12, 8))
    ani = FuncAnimation(fig, animator.update, frames=frames, init_func=animator.init, blit=True, interval=interval)

    if args.save:
        print(f'Saving animation to {args.save}...')
        animator.is_running = True
        writer = FFMpegWriter(fps=args.fps)
        ani.save(args.save, writer=writer)
    else:
        print('Displaying animation...')
        plt.show()


if __name__ == '__main__':
    import random
    random.seed(2025)
    main()