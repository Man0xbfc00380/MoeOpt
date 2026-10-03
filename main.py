#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dynamic Scheduling Animation for Chip Task Execution - Main Program

This program implements dynamic scheduling animation for chip task execution,
including scheduling algorithms and animation visualization.
"""

import argparse
import time
import os
import sys
import matplotlib
# Use Agg backend if no display is available
if os.environ.get('DISPLAY', '') == '' and not sys.platform.startswith('win'):
    print('No display environment detected, using Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from scheduler import FSEDPScheduler, EPScheduler
from animation import ChipletAnimator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Dynamic Scheduling Animation for Chip Task Execution')
    parser.add_argument('--scheduler', type=str, default='fsdp',
                        choices=['fsdp', 'ep', 'random', 'priority', 'round_robin'],
                        help='Scheduling algorithm: fsdp(FSDP), random(Random), priority(Priority-based), round_robin(Round Robin)')
    parser.add_argument('--duration', type=int, default=30,
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

        # Create scheduler
        if args.scheduler == 'fsdp':
            print("Creating FSDP scheduler...")
            scheduler = FSEDPScheduler()
        elif args.scheduler == 'ep':
            print("Creating EP scheduler...")
            scheduler = EPScheduler()
        else:
            raise ValueError(f'Unknown scheduling algorithm: {args.scheduler}')        # Create animator
        print("Creating animator...")
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
        # blit=True: use buffer drawing for better performance
        # interval: frame interval (milliseconds)
        ani = FuncAnimation(fig, animator.update, frames=frames,
                            init_func=animator.init, blit=True, interval=interval)

        # Save or display animation
        if args.save:
            animator.is_running = True
            print(f'Saving animation to {args.save}...')
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

