#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a 4-chip activity timeline plot from a single run:
- Configuration: Phi3.5-MoE × C4 using configs/FSE-DP_16_2_mul_2.json
- Activities per chip: receive, ddr_load, compute, send
- Output image: statistics/activity_timeline.png

The script:
1) Runs runner_one_layer.py once with --write_csv to produce CSVs
2) Reads per-chip CSV and builds a 4×T×4 (chip, frame, activity_type) tensor
3) Plots discontinuous line segments for activity presence
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List
import csv
import numpy as np
import matplotlib.pyplot as plt
# Qwen3-MoE
MODEL = 'Qwen3-MoE'
DATASET = 'C4'
CFG_REL = 'configs/FSE-DP-paired_256_2_mul_2.json'
INPUT_LENGTH = 256
LAYER_INDEX = 16

# Plot range configuration (None means use full range)
# Set START_FRAME/END_FRAME to integers to slice timeline [START_FRAME, END_FRAME)
START_FRAME = None
END_FRAME = None

# Global font configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# Buffer utilization polyline styling
# Vertical position offset relative to chip index (negative moves downward)
BUFFER_LINE_BASE_OFFSET = -1
# Amplitude (height range within the chip band)
BUFFER_LINE_AMPLITUDE = 0.5
# Line width for buffer polyline
BUFFER_LINE_WIDTH = 1

CHIP_SPACING_FACTOR = 1.5


def run_once(evaluation_dir: Path) -> Path:
    """Run runner_one_layer for the single configuration and return path to per-chip CSV."""
    runner_path = evaluation_dir / 'runner_one_layer.py'
    cfg_path = evaluation_dir / CFG_REL
    out_dir = evaluation_dir / 'results'

    cmd = [
        sys.executable,
        str(runner_path),
        '--model', MODEL,
        '--dataset', DATASET,
        '--config', str(cfg_path),
        '--layer', str(LAYER_INDEX),
        '--input_length', str(INPUT_LENGTH),
        '--write_csv',
    ]

    # proc = subprocess.run(
    #     cmd,
    #     cwd=str(evaluation_dir),
    #     capture_output=True,
    #     text=False,
    #     check=False,
    # )
    # Return per-chip CSV path irrespective of stdout/stderr decoding
    return out_dir / 'trace_per_chip4.csv'


def read_timeline(per_chip_csv: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Read per-chip activity CSV and return (timeline, buffer_mat, T).

    timeline shape: (4, T, 4) where activities order is [receive, ddr_load, compute, send].
    buffer_mat shape: (4, T) with values in [0,1] per chip per frame.
    """
    if not per_chip_csv.exists():
        raise FileNotFoundError(f"per-chip CSV not found: {per_chip_csv}")

    rows: List[Dict] = []
    with per_chip_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Determine T by counting unique frames
    frames = sorted({int(r['frame']) for r in rows})
    T = (frames[-1] + 1) if frames else 0
    chips = sorted({int(r['chip_index']) for r in rows})
    C = max(chips) + 1 if chips else 0
    if C < 4:
        # pad to 4 chips if fewer
        C = 4

    # activity indices: 0=receive, 1=ddr_load, 2=compute, 3=send
    timeline = np.zeros((C, T, 4), dtype=np.int32)
    buffer_mat = np.zeros((C, T), dtype=np.float64)
    for r in rows:
        ci = int(r.get('chip_index', 0))
        fi = int(r.get('frame', 0))
        # handle missing fields by fallback
        recv = int(r.get('receive', 0) or 0)
        ddr = int(r.get('ddr_load', 0) or 0)
        cmp_flag = int(r.get('compute_active', 0) or 0)
        send = int(r.get('send', 0) or 0)
        try:
            buf = float(r.get('buffer_util', 0.0) or 0.0)
        except Exception:
            buf = 0.0
        # If compute_active missing, infer from compute_util > 0
        if 'compute_active' not in r:
            try:
                cmp_flag = 1 if float(r.get('compute_util', 0.0) or 0.0) > 0.0 else 0
            except Exception:
                cmp_flag = 0
        if ci < C and fi < T:
            timeline[ci, fi, 0] = recv
            timeline[ci, fi, 1] = ddr
            timeline[ci, fi, 2] = cmp_flag
            timeline[ci, fi, 3] = send
            buffer_mat[ci, fi] = buf
    return timeline[:4], buffer_mat[:4], T


def plot_timeline(timeline: np.ndarray, buffer_mat: np.ndarray, T: int, out_path: Path) -> None:
    """Plot the activity timeline and save to out_path."""
    if timeline.shape[0] < 4:
        raise ValueError("timeline must have 4 chips on axis 0")

    # Styling
    colors = {
        0: ('receive', '#2c5aa0'),  # low-saturation dark blue
        1: ('ddr_load', '#2e8b57'),  # low-saturation dark green
        2: ('compute', '#b22222'),  # low-saturation dark red
        3: ('send', '#d2691e'),  # low-saturation brown
    }
    offsets = {
        0: -0.3,
        1: -0.1,
        2: 0.1,
        3: 0.3,
    }

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=400)

    # Draw lines per chip and per activity
    for chip in range(4):
        base_y = float(chip) * CHIP_SPACING_FACTOR + CHIP_SPACING_FACTOR/4
        for act_idx in range(4):
            label, color = colors[act_idx]
            y = base_y + offsets[act_idx]
            # Find contiguous segments of activity=1
            active = timeline[chip, :, act_idx]
            in_seg = False
            seg_start = 0
            for f in range(T):
                if active[f] and not in_seg:
                    in_seg = True
                    seg_start = f
                elif (not active[f]) and in_seg:
                    in_seg = False
                    ax.hlines(y=y, xmin=seg_start, xmax=f, colors=color, linewidth=2.0)
            if in_seg:
                ax.hlines(y=y, xmin=seg_start, xmax=T, colors=color, linewidth=2.0)

        # Buffer utilization polyline under the activity lines
        buf = buffer_mat[chip, :T]
        y0 = base_y + BUFFER_LINE_BASE_OFFSET
        amp = BUFFER_LINE_AMPLITUDE
        ybuf = y0 + buf * amp
        xbuf = np.arange(T)
        ax.plot(xbuf, ybuf, color='#4d4d4d', linewidth=BUFFER_LINE_WIDTH, alpha=0.9)

    # Axes and labels
    ax.set_ylim(-0.6, 3.8 * CHIP_SPACING_FACTOR)
    ax.set_xlim(0, max(1, T))
    ax.set_yticks([i * CHIP_SPACING_FACTOR for i in range(4)])
    ax.set_yticklabels([f"Chip-{i}" for i in range(4)])
    ax.set_xlabel('Frame')
    # ax.set_title('Activity Timeline (Phi3.5-MoE × C4, FSE-DP)')

    # Legend
    handles = [plt.Line2D([0], [0], color=colors[i][1], lw=2, label=colors[i][0]) for i in range(4)]
    # Add buffer utilization line to legend
    handles.append(plt.Line2D([0], [0], color='#4d4d4d', lw=BUFFER_LINE_WIDTH, label='buffer size'))
    ax.legend(handles=handles, loc='upper center', ncol=5, fontsize=11, bbox_to_anchor=(0.5, 1.05))

    # Light grid and horizontal separators between chips
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    for i in range(1, 4):  # separators between chip bands
        ax.axhline(y=(i - 0.4) * CHIP_SPACING_FACTOR , color='#dddddd', lw=1.0, alpha=0.7)

    # 去掉右侧与上侧的边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.show()  # 显示图片
    plt.close(fig)

def main():
    evaluation_dir = Path(__file__).resolve().parent.parent
    per_chip_csv = run_once(evaluation_dir)
    timeline, buffer_mat, T = read_timeline(per_chip_csv)
    # Apply optional frame slicing
    s = 10000 if START_FRAME is None else max(0, int(START_FRAME))
    e = 20000 if END_FRAME is None else min(T, int(END_FRAME))
    if e <= s:
        e = min(T, s + 1)
    timeline = timeline[:, s:e, :]
    buffer_mat = buffer_mat[:, s:e]
    T = e - s
    cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    cache[(MODEL, DATASET)] = {'timeline': timeline, 'buffer': buffer_mat}
    out_path = Path(__file__).resolve().parent / 'results' / 'activity_timeline.png'
    plot_timeline(timeline, buffer_mat, T, out_path)
    print(f"Saved timeline to: {out_path}")


if __name__ == '__main__':
    main()