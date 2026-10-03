#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Heatmap: STEADY-phase compute utilization vs CHIP_MOVE bandwidth and DDR bandwidth.

Generates a 2D heatmap where:
- X-axis: CHIP_MOVE_BANDWIDTH descending from MAX_CHIP_BW to MIN_CHIP_BW (GB/s)
- Y-axis: DDR_BANDWIDTH descending from MAX_DDR_BW to MIN_DDR_BW (GB/s)
- Cell value: Average compute utilization during STEADY phase for the given run

Defaults:
- MODEL/DATASET fixed to Qwen3-MoE × C4
- STRATEGY_CFG fixed to configs/FSE-DP-paired_256_2_mul_2.json
- MAX_CHIP_BW=2048, MIN_CHIP_BW=128, CHIP_BW_STEP=16
- MAX_DDR_BW=128, MIN_DDR_BW=8, BW_STEP=2

Caching:
Saves results to statistics/results/heatmap_move_vs_ddr_bandwidth_cache.json, keyed by
(model/dataset/config/layer/chip_bw/ddr_bw), to skip previously computed points.

Figure saved to statistics/results/heatmap_move_vs_ddr_bandwidth.png with dpi=400.

Overlay (feasible region):
- Draws a closed outline for points satisfying both:
  1) utilization >= min_util (default 0.60)
  2) alpha * DDR_bandwidth + beta * Chip_move_bandwidth <= weighted_sum_max
- Parameters are configurable via CLI: --alpha, --beta, --min_util, --weighted_sum_max
"""

import sys
import subprocess
import csv
from pathlib import Path
from typing import List
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import argparse


# Fixed model/dataset
MODEL = 'Qwen3-MoE'
DATASET = 'c4'
LAYER_INDEX = -1

# CHIP_MOVE bandwidth sweep (GB/s)
MAX_CHIP_BW = 2048
MIN_CHIP_BW = 128
CHIP_BW_STEP = 16

# DDR bandwidth sweep (GB/s)
MAX_DDR_BW = 128
MIN_DDR_BW = 8
BW_STEP = 2

# Single strategy config (input length inferred)
STRATEGY_CFG = 'configs/FSE-DP-paired_64_2_mul_2.json'


def _parse_cli_args():
    """Parse CLI arguments for overlay configuration.

    Returns an argparse.Namespace with:
    - alpha (float): weight for DDR bandwidth
    - beta (float): weight for CHIP_MOVE bandwidth
    - min_util (float): utilization threshold in [0,1]
    - weighted_sum_max (float|None): max allowed alpha*DDR + beta*Chip. If None,
      defaults to alpha*MAX_DDR_BW**2 + beta*MAX_CHIP_BW**2 (budget grows quadratically with maxima).
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--alpha', type=float, default=3)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--min_util', type=float, default=0.60)
    parser.add_argument('--weighted_sum_max', '--budget', dest='weighted_sum_max', type=float, default=None)
    # Keep unknown args for compatibility with other consumers
    args, _ = parser.parse_known_args()
    if args.weighted_sum_max is None:
        args.weighted_sum_max = 80000
    # Clamp min_util
    args.min_util = max(0.0, min(1.0, args.min_util))
    return args


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


def _run_once(evaluation_dir: Path, cfg_rel: str, chip_bw: int, ddr_bw: int) -> float:
    """Run runner_one_layer once and return STEADY-phase average compute utilization."""
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
        '--override_expert_bandwidth', str(ddr_bw),
        '--override_chip_move_bandwidth', str(chip_bw),
    ]

    subprocess.run(
        cmd,
        cwd=str(evaluation_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    csv_path = out_dir / 'trace.csv'
    util = _compute_steady_util_from_csv(csv_path)
    return util


def _point_key(model: str, dataset: str, cfg_rel: str, layer: int,
               chip_bw: int, ddr_bw: int) -> str:
    return (
        f"model={model}|dataset={dataset}|cfg={cfg_rel}|layer={layer}|"
        f"chip_bw={chip_bw}|ddr_bw={ddr_bw}"
    )


def _load_cache(json_path: Path) -> dict:
    if not json_path.exists():
        return {"points": {}}
    try:
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'points' in data and isinstance(data['points'], dict):
                return data
            return {"points": {}}
    except Exception:
        return {"points": {}}


def _save_cache(json_path: Path, cache: dict) -> None:
    try:
        with json_path.open('w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def main():
    args = _parse_cli_args()
    evaluation_dir = Path(__file__).resolve().parent.parent  # anim/evaluation
    stats_dir = evaluation_dir / 'statistics'
    results_dir = stats_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_path = results_dir / 'heatmap_move_vs_ddr_bandwidth_cache_-1.json'
    cache = _load_cache(cache_path)

    chip_bws_desc: List[int] = list(range(MAX_CHIP_BW, MIN_CHIP_BW - 1, -CHIP_BW_STEP))
    ddr_bws_desc: List[int] = list(range(MAX_DDR_BW, MIN_DDR_BW - 1, -BW_STEP))

    if _maybe_quick_mode():
        chip_bws_desc = chip_bws_desc[::max(1, CHIP_BW_STEP)]
        if len(ddr_bws_desc) > 2:
            ddr_bws_desc = [ddr_bws_desc[0], ddr_bws_desc[-1]]

    H = np.zeros((len(ddr_bws_desc), len(chip_bws_desc)), dtype=float)

    print(f"[Sweep] CHIP_MOVE_BW (GB/s): {chip_bws_desc}")
    print(f"[Sweep] DDR bandwidths (GB/s): {ddr_bws_desc}")

    for bi, bw in enumerate(ddr_bws_desc):
        for ci, chip_bw in enumerate(chip_bws_desc):
            key = _point_key(MODEL, DATASET, STRATEGY_CFG, LAYER_INDEX, chip_bw, bw)
            cached = cache.get('points', {}).get(key)
            if cached is not None and 'util' in cached:
                util = float(cached['util'])
                used_cache = True
            else:
                util = _run_once(evaluation_dir, STRATEGY_CFG, chip_bw, bw)
                cache.setdefault('points', {})[key] = {
                    'util': util,
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                }
                _save_cache(cache_path, cache)
                used_cache = False
            H[bi, ci] = util
            if used_cache:
                print(f"[Skip] bw={bw} GB/s, chip_move_bw={chip_bw} GB/s, util={util:.4f} (from cache)")
            else:
                print(f"[Done] bw={bw} GB/s, chip_move_bw={chip_bw} GB/s, util={util:.4f}")

    # Build smooth contour surface via bilinear interpolation and plot
    plt.rcParams['font.size'] = 8
    fig, ax = plt.subplots(figsize=(4, 2.2), dpi=400)

    x_old_desc = chip_bws_desc
    y_old_desc = ddr_bws_desc
    x_old = sorted(x_old_desc)
    y_old = sorted(y_old_desc)
    Z_old = H[::-1, ::-1]
    x_pts = 200
    y_pts = 200
    if _maybe_quick_mode():
        x_pts = 80
        y_pts = 80
    x_new = np.linspace(x_old[0], x_old[-1], x_pts)
    Z_x = np.empty((len(y_old), len(x_new)), dtype=float)
    for i in range(len(y_old)):
        Z_x[i, :] = np.interp(x_new, x_old, Z_old[i, :])
    y_new = np.linspace(y_old[0], y_old[-1], y_pts)
    Z_xy = np.empty((len(y_new), len(x_new)), dtype=float)
    for j in range(len(x_new)):
        Z_xy[:, j] = np.interp(y_new, y_old, Z_x[:, j])
    Xg, Yg = np.meshgrid(x_new, y_new)
    levels = np.linspace(0.0, 1.0, 21)
    cmap = 'turbo'
    cf = ax.contourf(Xg, Yg, Z_xy, levels=levels, cmap=cmap)
    ax.contour(Xg, Yg, Z_xy, levels=levels, colors='white', linewidths=0.3, alpha=0.6)

    # X ticks and axis (MAX→MIN left-to-right)
    tick_stride_x = max(1, CHIP_BW_STEP)
    x_ticks = x_old_desc[::tick_stride_x]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(v) for v in x_ticks])
    ax.set_xlim(max(x_old_desc), min(x_old_desc))
    ax.set_xlabel('D2D Bandwidth (GB/s)')

    # Y ticks and axis (MAX at bottom)
    tick_stride_y = max(1, BW_STEP)
    y_ticks_vals = y_old_desc[::tick_stride_y * 4]
    ax.set_yticks(y_ticks_vals)
    ax.set_yticklabels([str(v) for v in y_ticks_vals])
    ax.set_ylabel('DDR Bandwidth (GB/s)')
    ax.set_ylim(max(y_old_desc), min(y_old_desc))
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('Compute Utilization')

    # Intersection region: (util ≥ min_util) ∧ (cost1) ∧ (cost2)
    mask_util = (Z_xy >= args.min_util)
    # Cost formula 1: 3.2 * D2D/96 + 1.3 * 13.952/3.328 + 1.7 ≤ 30
    cost1 = 3.2 * (Xg / 96.0) + 1.3 * (13.952 / 3.328) + 1.7
    mask_cost1 = (cost1 <= 30.0)
    # Cost formula 2: D2D * 8.864 / 1000 + 6.5 * DDR Bandwidth / 12.8 + 9.492 < 60
    cost2 = (Xg * 8.864 / 1000.0) + 6.5 * (Yg / 12.8) + 9.492
    mask_cost2 = (cost2 < 60.0)

    feasible_all = np.logical_and(mask_util, np.logical_and(mask_cost1, mask_cost2))

    # Fill intersection with transparent gray and diagonal hatches
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

    # Draw boundaries with distinct colors: cost1, cost2, utilization
    cost1_masked = np.where(mask_cost2 & mask_util, cost1, np.nan)
    cost2_masked = np.where(mask_cost1 & mask_util, cost2, np.nan)
    util_masked = np.where(mask_cost1 & mask_cost2, Z_xy, np.nan)

    ax.contour(Xg, Yg, cost1_masked, levels=[30.0], colors='#d62728', linewidths=1.2)
    ax.contour(Xg, Yg, cost2_masked, levels=[60.0], colors='#1f77b4', linewidths=1.2)
    ax.contour(Xg, Yg, util_masked, levels=[args.min_util], colors='#2ca02c', linewidths=1.2)

    ax.grid(True, linestyle='--', linewidth=0.35, alpha=0.4)
    # ax.set_title('Utilization Contour (CHIP_MOVE vs DDR)')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    out_png = results_dir / 'heatmap_move_vs_ddr_bandwidth.png'
    fig.subplots_adjust(right=0.98)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=400, bbox_inches='tight')
    print(f"Saved figure to {out_png}")


if __name__ == '__main__':
    main()