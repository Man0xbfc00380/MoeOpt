#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sweep.py
对硬件参数（COMPUTE 与 DDR_LOAD 持续时间）进行网格扫描，
生成吞吐与利用率热力图，用于快速评估不同配置下的系统性能。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple
import os

# 将项目根目录加入 Python 路径，确保能 import 自定义模块
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.runner import DiscreteSimulator, apply_hardware_overrides, apply_scheduler_overrides
from evaluation.plots import plot_throughput_over_time
from action import ActionType
from hardware_config import HardwareConfig

import csv
import math
import matplotlib.pyplot as plt


def run_one(cfg: Dict, frames: int, out_root: Path, label: str) -> Tuple[float, float, float, Path]:
    """
    单次仿真入口：应用硬件/调度配置 -> 运行仿真 -> 解析 trace.csv -> 返回平均指标
    :param cfg: 完整的硬件/调度配置字典
    :param frames: 仿真总帧数
    :param out_root: 结果根目录
    :param label: 本次实验子目录名
    :return: (平均吞吐, 平均 DDR 利用率, 平均计算利用率, trace.csv 路径)
    """
    # 根据配置覆盖硬件参数
    apply_hardware_overrides(cfg)
    import random
    random.seed(cfg.get('seed', 2025))
    sim = DiscreteSimulator(base_strategy = cfg.get('base_strategy', "FSEDP"))
    apply_scheduler_overrides(sim, cfg)
    sim.init_blocks()

    # 创建输出目录结构：实验子目录 /figures
    out_dir = out_root / label
    fig_dir = out_dir / 'figures'
    sim.run(frames=frames, out_dir=out_dir, fig_dir=fig_dir)

    # 读取 trace.csv，累加各帧指标
    csv_path = out_dir / 'trace.csv'
    total_frames = 0
    sum_completed_compute = 0
    sum_ddr_util = 0.0
    sum_cmp_util = 0.0

    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_frames += 1
            sum_completed_compute += int(row['completed_COMPUTE'])
            # DDR 链路利用率 = 活跃链路 / 总链路
            d_links = float(row['ddr_links_total'])
            if d_links > 0:
                sum_ddr_util += float(row['ddr_links_active']) / d_links
            # 计算单元利用率 = 正在计算的芯片 / 总芯片
            c_total = float(row['chips_total'])
            if c_total > 0:
                sum_cmp_util += float(row['chips_computing']) / c_total

    # 计算平均值，避免除零
    avg_throughput = sum_completed_compute / max(1, total_frames)
    avg_ddr_util = sum_ddr_util / max(1, total_frames)
    avg_cmp_util = sum_cmp_util / max(1, total_frames)
    return avg_throughput, avg_ddr_util, avg_cmp_util, csv_path


def heatmap(data, x_ticks, y_ticks, title: str, out_path: Path, cmap='viridis', vmin=None, vmax=None):
    """
    绘制二维热力图并保存
    :param data: 2D list/array，热力图数据
    :param x_ticks: X 轴刻度标签列表
    :param y_ticks: Y 轴刻度标签列表
    :param title: 图例/色条标题
    :param out_path: 输出图片路径
    :param cmap: 颜色映射
    :param vmin, vmax: 色条上下限
    """
    plt.figure(figsize=(5, 4))
    plt.imshow(data, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=title)
    plt.xticks(range(len(x_ticks)), x_ticks)
    plt.yticks(range(len(y_ticks)), y_ticks)
    plt.xlabel('COMPUTE duration')
    plt.ylabel('DDR_LOAD duration')
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    """
    主函数：解析命令行 -> 生成参数网格 -> 逐点仿真 -> 输出三张热力图
    """
    parser = argparse.ArgumentParser(description='网格扫描 COMPUTE 与 DDR_LOAD 持续时间')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent / 'configs/default.json'),
                        help='基准配置文件路径')
    parser.add_argument('--frames', type=int, default=200,
                        help='每轮仿真帧数')
    parser.add_argument('--compute_start', type=int, default=4,
                        help='COMPUTE 持续时间扫描起点（包含）')
    parser.add_argument('--compute_end', type=int, default=8,
                        help='COMPUTE 持续时间扫描终点（包含）')
    parser.add_argument('--compute_step', type=int, default=1,
                        help='COMPUTE 持续时间步长')
    parser.add_argument('--ddr_start', type=int, default=15,
                        help='DDR_LOAD 持续时间扫描起点（包含）')
    parser.add_argument('--ddr_end', type=int, default=30,
                        help='DDR_LOAD 持续时间扫描终点（包含）')
    parser.add_argument('--ddr_step', type=int, default=3,
                        help='DDR_LOAD 持续时间步长')
    parser.add_argument('--out', type=str, default=str(Path(__file__).parent / 'results/sweeps/ddr_vs_compute'),
                        help='结果输出根目录')
    args = parser.parse_args()

    # 读取基准配置
    base_cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))

    out_root = Path(args.out)

    # 生成扫描网格
    compute_vals = list(range(args.compute_start, args.compute_end + 1, args.compute_step))
    ddr_vals = list(range(args.ddr_start, args.ddr_end + 1, args.ddr_step))

    # 初始化结果网格：吞吐、DDR 利用率、计算利用率
    T = [[0.0 for _ in compute_vals] for _ in ddr_vals]
    U_ddr = [[0.0 for _ in compute_vals] for _ in ddr_vals]
    U_cmp = [[0.0 for _ in compute_vals] for _ in ddr_vals]

    # 双重循环：遍历 DDR_LOAD 时长 -> 遍历 COMPUTE 时长
    for yi, ddr_dur in enumerate(ddr_vals):
        for xi, cmp_dur in enumerate(compute_vals):
            # 深拷贝基准配置，避免污染
            cfg = json.loads(json.dumps(base_cfg))
            # 确保字典路径存在，再写入扫描值
            cfg.setdefault('hardware', {}).setdefault('action_durations', {})
            cfg['hardware']['action_durations']['COMPUTE'] = cmp_dur
            cfg['hardware']['action_durations']['DDR_LOAD'] = ddr_dur
            label = f"cmp{cmp_dur}_ddr{ddr_dur}"
            # 运行单次仿真并记录结果
            avg_t, avg_du, avg_cu, csvp = run_one(cfg, args.frames, out_root, label)
            T[yi][xi] = avg_t
            U_ddr[yi][xi] = avg_du
            U_cmp[yi][xi] = avg_cu

    # 保存三张热力图
    heatmap(T, compute_vals, ddr_vals,
            title='Avg throughput (compute/frame)',
            out_path=out_root / 'throughput_heatmap.png')
    heatmap(U_ddr, compute_vals, ddr_vals,
            title='Avg DDR utilization',
            out_path=out_root / 'ddr_util_heatmap.png')
    heatmap(U_cmp, compute_vals, ddr_vals,
            title='Avg compute utilization',
            out_path=out_root / 'compute_util_heatmap.png')


if __name__ == '__main__':
    main()
