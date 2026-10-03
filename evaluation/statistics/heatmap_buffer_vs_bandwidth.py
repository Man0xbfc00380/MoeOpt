#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Heatmap: STEADY-phase compute utilization vs total buffer usage (MB) and DDR bandwidth.

Generates a 2D heatmap where:
- X-axis: Actual total buffer usage (MB), descending with expert slice count
  Computed as: (token_num * 0.002 + expert_slice_num * 0.288) * 4
- Y-axis: DDR bandwidth descending from MAX_DDR_BW to MIN_DDR_BW (GB/s)
- Cell value: Average compute utilization during STEADY phase for the given run

Defaults:
- MODEL/DATASET fixed to Phi3.5-MoE × C4
- TOKEN_BUFFER_SIZE fixed to 5
- MAX_BUFFER_SIZE=128, MIN_BUFFER_SIZE=16, BUF_STEP=4
- MAX_DDR_BW=64, MIN_DDR_BW=16, BW_STEP=8

Saves figure to statistics/results/heatmap_buffer_vs_bandwidth.png with dpi=400.

Adds checkpointing via JSON: after each point is computed, the result
is stored to statistics/results/heatmap_buffer_vs_bandwidth_cache.json
and subsequent runs will read that file and skip points already computed.
The cache adapts to changes in MAX_BUFFER_SIZE/BUF_STEP by keying each
point with the exact parameters (model/dataset/config/layer/token/expert/bw).

Overlay (feasible region):
- Draws a closed outline for points satisfying both:
  1) utilization >= min_util (default 0.60)
  2) (alpha * DDR_bandwidth)^2 + (beta * chip_buffer_size)^2 <= weighted_sum_max
- Parameters are configurable via CLI: --alpha, --beta, --min_util, --weighted_sum_max
"""

import sys
import subprocess
import csv
from pathlib import Path
from typing import List, Tuple
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import argparse
import math


# Fixed model/dataset
MODEL = 'Qwen3-MoE'
DATASET = 'c4'
LAYER_INDEX = -1

# Buffer size sweep (expert slice count)
MAX_BUFFER_SIZE = 128 - 48
MIN_BUFFER_SIZE = 66 - 48
BUF_STEP = 2
TOKEN_BUFFER_SIZE = 16

# DDR bandwidth sweep (GB/s)
MAX_DDR_BW = 128
MIN_DDR_BW = 8
BW_STEP = 2

# Single strategy config (input length 16)
STRATEGY_CFG = 'configs/FSE-DP-paired_64_2_mul_2.json'

# Optional quick mode: reduce sweep size for faster verification
def _maybe_quick_mode() -> bool:
    return '--quick' in sys.argv

def _compute_steady_util_from_csv(csv_path: Path) -> float:
    """Compute average compute utilization during STEADY phase from aggregate trace.csv.

    Util per frame = chips_computing / chips_total, averaged over frames where phase == 2 (STEADY).
    Returns 0.0 if CSV missing or no STEADY frames.
    """
    if not csv_path.exists():
        return 0.0
    ratios: List[float] = []
    try:
        with csv_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    phase = int(row.get('phase', '0'))
                    if phase != 2:
                        continue
                    chips_total = int(row.get('chips_total', '0'))
                    chips_comp = int(row.get('chips_computing', '0'))
                    ratios.append((chips_comp / chips_total) if chips_total > 0 else 0.0)
                except Exception:
                    pass
        if not ratios:
            return 0.0
        return float(sum(ratios) / len(ratios))
    except Exception:
        return 0.0


def _run_once(evaluation_dir: Path, cfg_rel: str, expert_buf_size: int, ddr_bw: int) -> float:
    """Run runner_one_layer once and return STEADY-phase average compute utilization.

    Reads utilization from the generated trace.csv.
    """
    runner_path = evaluation_dir / 'runner_one_layer.py'
    cfg_path = evaluation_dir / cfg_rel
    out_dir = evaluation_dir / 'results'

    cmd = [
        sys.executable,
        str(runner_path),
        '--model', MODEL,
        '--dataset', DATASET,
        '--config', str(cfg_path),
        '--layer', str(LAYER_INDEX),
        '--write_csv',
        '--override_token_buffer_size', str(TOKEN_BUFFER_SIZE),
        '--override_expert_buffer_size', str(expert_buf_size),
        '--override_expert_bandwidth', str(ddr_bw),
    ]

    subprocess.run(
        cmd,
        cwd=str(evaluation_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    # Compute utilization from CSV
    csv_path = out_dir / 'trace.csv'
    util = _compute_steady_util_from_csv(csv_path)
    return util


def _point_key(model: str, dataset: str, cfg_rel: str, layer: int,
               token_buf: int, expert_buf: int, ddr_bw: int) -> str:
    """Create a stable key for a single sweep point.

    Includes all parameters that affect the run so cached results remain valid
    even when sweep ranges change across executions.
    """
    return (
        f"model={model}|dataset={dataset}|cfg={cfg_rel}|layer={layer}|"
        f"token={token_buf}|expert={expert_buf}|bw={ddr_bw}"
    )


def _load_cache(json_path: Path) -> dict:
    if not json_path.exists():
        return {"points": {}}
    try:
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'points' in data and isinstance(data['points'], dict):
                return data
            # normalize unexpected structure
            return {"points": {}}
    except Exception:
        return {"points": {}}


def _save_cache(json_path: Path, cache: dict) -> None:
    try:
        with json_path.open('w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        # best-effort; ignore write failures to avoid interrupting sweep
        pass


def _parse_cli_args():
    """Parse CLI arguments for overlay configuration.

    Returns an argparse.Namespace with:
    - alpha (float): weight for DDR bandwidth
    - beta (float): weight for chip buffer size
    - min_util (float): utilization threshold in [0,1]
    - weighted_sum_max (float|None): max allowed (alpha*DDR)^2 + (beta*chip_size)^2.
      If None, defaults to a conservative budget based on maxima.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--min_util', type=float, default=0.60)
    parser.add_argument('--weighted_sum_max', '--budget', dest='weighted_sum_max', type=float, default=None)
    args, _ = parser.parse_known_args()
    if args.weighted_sum_max is None:
        # Provide a default budget. Keep consistent scale with maxima.
        # Note: This default mirrors the reference script behavior style.
        args.weighted_sum_max = 8000
    args.min_util = max(0.0, min(1.0, args.min_util))
    return args


def main():
    args = _parse_cli_args()
    evaluation_dir = Path(__file__).resolve().parent.parent  # anim/evaluation
    stats_dir = evaluation_dir / 'statistics'
    results_dir = stats_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_path = results_dir / 'heatmap_buffer_vs_bandwidth_cache_-1.json'
    cache = _load_cache(cache_path)

    # expert_slice_num descending from (MAX_BUFFER_SIZE - TOKEN_BUFFER_SIZE) to (MIN_BUFFER_SIZE - TOKEN_BUFFER_SIZE)
    expert_sizes: List[int] = list(range(MAX_BUFFER_SIZE - TOKEN_BUFFER_SIZE, MIN_BUFFER_SIZE - TOKEN_BUFFER_SIZE - 1, -BUF_STEP))
    chip_sizes_desc: List[int] = [TOKEN_BUFFER_SIZE + e for e in expert_sizes]  # kept for reference/logs if needed
    # Actual total buffer usage (MB): (token_num*0.002 + expert_slice_num*0.288) * 4
    buffer_mb_desc: List[float] = [
        (TOKEN_BUFFER_SIZE * 0.002 + e * 0.288) * 4 for e in expert_sizes
    ]
    # DDR bandwidth descending so that from bottom to top it decreases
    ddr_bws_desc: List[int] = list(range(MAX_DDR_BW, MIN_DDR_BW - 1, -BW_STEP))

    # Quick mode narrows to a small subset
    if _maybe_quick_mode():
        # take every BUF_STEP-th chip size, and max/min bandwidths
        expert_sizes = expert_sizes[::max(1, BUF_STEP)]
        chip_sizes_desc = [TOKEN_BUFFER_SIZE + e for e in expert_sizes]
        buffer_mb_desc = [
            (TOKEN_BUFFER_SIZE * 0.002 + e * 0.288) * 4 for e in expert_sizes
        ]
        if len(ddr_bws_desc) > 2:
            ddr_bws_desc = [ddr_bws_desc[0], ddr_bws_desc[-1]]

    # Initialize matrix [len(ddr_bws) x len(chip_sizes)]
    H = np.zeros((len(ddr_bws_desc), len(buffer_mb_desc)), dtype=float)

    print(f"[Sweep] total buffer usage (MB): {[round(v, 3) for v in buffer_mb_desc]}")
    print(f"[Sweep] DDR bandwidths (GB/s): {ddr_bws_desc}")
    
    for bi, bw in enumerate(ddr_bws_desc):
        for ci, e_size in enumerate(expert_sizes):
            key = _point_key(MODEL, DATASET, STRATEGY_CFG, LAYER_INDEX,
                             TOKEN_BUFFER_SIZE, e_size, bw)
            cached = cache.get('points', {}).get(key)
            if cached is not None and 'util' in cached:
                util = float(cached['util'])
                used_cache = True
            else:
                assert False
                util = _run_once(evaluation_dir, STRATEGY_CFG, e_size, bw)
                cache.setdefault('points', {})[key] = {
                    'util': util,
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                }
                _save_cache(cache_path, cache)
                used_cache = False
            H[bi, ci] = util
            chip_size = TOKEN_BUFFER_SIZE + e_size
            buf_mb = (TOKEN_BUFFER_SIZE * 0.002 + e_size * 0.288) * 4
            if used_cache:
                print(f"[Skip] bw={bw} GB/s, buffer_MB={buf_mb:.3f}, util={util:.4f} (from cache)")
            else:
                print(f"[Done] bw={bw} GB/s, buffer_MB={buf_mb:.3f}, util={util:.4f}")

    # Build smooth contour surface via bilinear interpolation and plot
    plt.rcParams['font.size'] = 8
    fig, ax = plt.subplots(figsize=(4, 2.2), dpi=400)
    # Prepare ascending axes for interpolation
    x_old_desc = buffer_mb_desc
    y_old_desc = ddr_bws_desc
    x_old = sorted(x_old_desc)       # ascending
    y_old = sorted(y_old_desc)       # ascending
    Z_old = H[::-1, ::-1]            # reorder to match ascending axes
    # Resolution: fewer points in quick mode
    x_pts = 200
    y_pts = 200
    if _maybe_quick_mode():
        x_pts = 80
        y_pts = 80
    x_new = np.linspace(x_old[0], x_old[-1], x_pts)
    # interpolate along X for each Y
    Z_x = np.empty((len(y_old), len(x_new)), dtype=float)
    for i in range(len(y_old)):
        Z_x[i, :] = np.interp(x_new, x_old, Z_old[i, :])
    # interpolate along Y for each X
    y_new = np.linspace(y_old[0], y_old[-1], y_pts)
    Z_xy = np.empty((len(y_new), len(x_new)), dtype=float)
    for j in range(len(x_new)):
        Z_xy[:, j] = np.interp(y_new, y_old, Z_x[:, j])
    Xg, Yg = np.meshgrid(x_new, y_new)
    levels = np.linspace(0.0, 1.0, 21)
    cmap = 'turbo'
    cf = ax.contourf(Xg, Yg, Z_xy, levels=levels, cmap=cmap)
    cs = ax.contour(Xg, Yg, Z_xy, levels=levels, colors='white', linewidths=0.3, alpha=0.6)

    # X ticks: show only integers that are multiples of 4 (in MB)
    x_min, x_max = min(x_old_desc), max(x_old_desc)
    start_m4 = int(math.ceil(x_min / 8.0) * 8)
    end_m4 = int(math.floor(x_max / 8.0) * 8)
    x_ticks = list(range(start_m4, end_m4 + 1, 8)) if start_m4 <= end_m4 else []
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(v) for v in x_ticks], rotation=0)
    ax.set_xlim(x_max, x_min)
    ax.set_xlabel('Total Buffer Usage (MB)')

    # Y ticks: use actual DDR bandwidths and invert axis (MAX at bottom)
    tick_stride_y = max(1, BW_STEP)
    y_ticks_vals = ddr_bws_desc[::tick_stride_y * 4]
    ax.set_yticks(y_ticks_vals)
    ax.set_yticklabels([str(v) for v in y_ticks_vals])
    ax.set_ylabel('DDR Bandwidth (GB/s)')
    ax.set_ylim(max(ddr_bws_desc), min(ddr_bws_desc))
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('Compute Utilization')

    # 交集区域：满足 (util≥min_util) ∧ (成本公式1) ∧ (成本公式2)
    mask_util = (Z_xy >= args.min_util)
    # 成本公式1：3.2 * D2D/96 + 1.3 * buf_mb / 3.328 + 1.7 <= 30
    cost1 = 3.2 * (288 / 96.0) + 1.3 * (Xg / 3.328) + 1.7
    mask_cost1 = (cost1 <= 30.0)
    # 成本公式2：2.553 + 6.5 * D2D / 12.8 + 9.492 < 60
    cost2 = 2.553 + 6.5 * (Yg / 12.8) + 9.492
    mask_cost2 = (cost2 < 60.0)

    feasible_all = np.logical_and(mask_util, np.logical_and(mask_cost1, mask_cost2))

    # 用透明灰色背景与斜线填充交集区域
    if np.any(feasible_all):
        ax.contourf(Xg, Yg, feasible_all.astype(int), levels=[0.5, 1.5], colors=['#808080'], alpha=0.15)
        cs_hatch = ax.contourf(Xg, Yg, feasible_all.astype(int), levels=[0.5, 1.5], colors=['none'], hatches=['///'], alpha=0.0)
        try:
            for coll in cs_hatch.collections:
                coll.set_edgecolor('#808080')
        except Exception:
            pass
    else:
        assert False

    # 使用不同颜色绘制三条边界：成本1、成本2、利用率
    legend_handles = []
    # try:
        # 限定在交集上下文中绘制对应边界
    cost1_masked = np.where(mask_cost2 & mask_util, cost1, np.nan)
    cost2_masked = np.where(mask_cost1 & mask_util, cost2, np.nan)
    util_masked = np.where(mask_cost1 & mask_cost2, Z_xy, np.nan)

    cs_b1 = ax.contour(Xg, Yg, cost1_masked, levels=[30.0], colors='#d62728', linewidths=1.2)
    # cs_b1.set_label('边界：成本公式1')
    legend_handles.append(cs_b1)

    cs_b2 = ax.contour(Xg, Yg, cost2_masked, levels=[60.0], colors='#1f77b4', linewidths=1.2)
    # cs_b2.set_label('边界：成本公式2')
    legend_handles.append(cs_b2)

    cs_bu = ax.contour(Xg, Yg, util_masked, levels=[args.min_util], colors='#2ca02c', linewidths=1.2)
    # cs_bu.set_label(f'边界：利用率≥{int(args.min_util*100)}%')
    legend_handles.append(cs_bu)

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=6, frameon=False)
    # except Exception:
    #     from matplotlib.lines import Line2D
    #     proxy_handles = [
    #         Line2D([], [], color='#d62728', linewidth=1.2, label='边界：成本公式1'),
    #         Line2D([], [], color='#1f77b4', linewidth=1.2, label='边界：成本公式2'),
    #         Line2D([], [], color='#2ca02c', linewidth=1.2, label=f'边界：利用率≥{int(args.min_util*100)}%'),
    #     ]
    #     ax.legend(handles=proxy_handles, loc='upper right', fontsize=6, frameon=False)


    # cosmetic
    ax.grid(True, linestyle='--', linewidth=0.35, alpha=0.4)
    # ax.set_title('Utilization Contour (STEADY)')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    out_png = results_dir / 'heatmap_buffer_vs_bandwidth.png'
    fig.subplots_adjust(right=0.98)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=400, bbox_inches='tight')
    print(f"Saved figure to {out_png}")

if __name__ == '__main__':
    main()