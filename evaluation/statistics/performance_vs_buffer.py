#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot: completion frames vs chip buffer size for four strategies.

Generates a single figure with one subplot containing four lines:
- EP
- Hydra
- FSE-DP (no paired-load)
- FSE-DP (paired-load)

For each strategy, it runs runner_one_layer.py once per buffer size where:
- token_buffer_size is fixed to 5
- expert_buffer_size varies from (MAX_BUFFER_SIZE-5) down to (MIN_BUFFER_SIZE-5) (inclusive)
Thus chip buffer size = token_buffer_size + expert_buffer_size varies from MAX_BUFFER_SIZE down to MIN_BUFFER_SIZE.

Model/Dataset is fixed to Phi3.5-MoE × C4.
Saves figure to statistics/performance_vs_buffer.png with dpi=400.
"""

import sys
import subprocess
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


# Fixed model/dataset
MODEL = 'Phi3.5-MoE'
DATASET = 'c4'

# 可参数化的芯片缓冲区大小范围
MAX_BUFFER_SIZE = 64
MIN_BUFFER_SIZE = 8
TOKEN_BUFFER_SIZE = 5
LAYER_INDEX = 16

# Strategy configs (input length 16)
STRATEGIES = [
    ('EP', 'configs/EP_16_2_mul_2.json'),
    ('Hydra', 'configs/Hydra_16_2_mul_2.json'),
    ('FSE-DP (no paired-load)', 'configs/FSE-DP_16_2_mul_2.json'),
    ('FSE-DP (paired-load)', 'configs/FSE-DP-paired_16_2_mul_2.json'),
]

# 设置颜色
COLORS = {
    "EP": "#2c5aa0",                    # low-saturation dark blue
    "Hydra": "#2e8b57",                 # low-saturation dark green
    "FSE-DP (no paired-load)": "#b22222",  # low-saturation dark red
    "FSE-DP (paired-load)": "#d2691e",     # low-saturation dark orange
}

# Regex to parse finish frame from stdout
FINISH_RE = re.compile(r"FINISH occurred at frame:\s*([0-9]+)")


def _run_once(evaluation_dir: Path, cfg_rel: str, expert_buf_size: int) -> Tuple[int, Path]:
    """Run runner_one_layer once for given config and expert buffer size.

    Returns (finish_frame, out_dir).
    If parsing fails, finish_frame = -1.
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
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(evaluation_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    # Try to parse finish frame from stdout
    finish_frame = -1
    stdout = proc.stdout or ''
    match = FINISH_RE.search(stdout)
    if match:
        try:
            finish_frame = int(match.group(1))
        except Exception:
            finish_frame = -1

    # Fallback: if CSV exists, count total frames by row count
    if finish_frame < 0:
        csv_path = out_dir / 'trace.csv'
        if csv_path.exists():
            try:
                with csv_path.open('r', encoding='utf-8') as f:
                    # subtract header
                    line_count = sum(1 for _ in f) - 1
                finish_frame = max(line_count - 1, 0)
            except Exception:
                finish_frame = -1

    return finish_frame, out_dir


def main():
    evaluation_dir = Path(__file__).resolve().parent.parent  # anim/evaluation
    stats_dir = evaluation_dir / 'statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)

    # expert_buffer_size from (MAX_BUFFER_SIZE-TOKEN_BUFFER_SIZE) down to (MIN_BUFFER_SIZE-TOKEN_BUFFER_SIZE)
    expert_sizes: List[int] = list(range(MAX_BUFFER_SIZE - TOKEN_BUFFER_SIZE, MIN_BUFFER_SIZE - TOKEN_BUFFER_SIZE - 1, -1))
    chip_sizes: List[int] = [TOKEN_BUFFER_SIZE + e for e in expert_sizes]

    # Collect frames for each strategy
    frames_by_strategy = {}
    for label, cfg_rel in STRATEGIES:
        frames_list: List[int] = []
        for e_size in expert_sizes:
            finish_frame, _ = _run_once(evaluation_dir, cfg_rel, e_size)
            frames_list.append(finish_frame)
            # Optional progress print to help long sweeps
            print(f"[Done] {label}, chip={TOKEN_BUFFER_SIZE+e_size}, frames={finish_frame}")
        frames_by_strategy[label] = frames_list

    # Plotting
    plt.rcParams['font.size'] = 6
    fig, ax = plt.subplots(figsize=(3.4, 1.6), dpi=400)

    # To show left-to-right descending (MAX_BUFFER_SIZE→MIN_BUFFER_SIZE), set xlim inverted
    # Plot using ascending x values for better line rendering, then invert axis
    chip_sizes_asc = sorted(chip_sizes)

    for label, _ in STRATEGIES:
        data = frames_by_strategy[label]
        # reorder data to ascending chip sizes
        # expert_sizes descending maps to chip_sizes descending; reverse for ascending
        data_asc = list(reversed(data))
        ax.plot(chip_sizes_asc, data_asc, label=label, linewidth=0.7, marker='o', markersize=1, color=COLORS[label])

    ax.set_xlabel('Chip buffer size')
    ax.set_ylabel('Completion Time')
    # Major ticks: MAX_BUFFER_SIZE→MIN_BUFFER_SIZE descending
    ax.set_xticks([x * 4 for x in range(MAX_BUFFER_SIZE, MIN_BUFFER_SIZE - 1, -8)])
    ax.set_xlim(MAX_BUFFER_SIZE, MIN_BUFFER_SIZE)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    legend = ax.legend(loc="upper right", fontsize=6, ncol=2)
    legend.get_frame().set_facecolor((1, 1, 1, 0.3))
    # Remove top and right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_png = stats_dir / 'results' / 'performance_vs_buffer.png'

    fig.subplots_adjust(right=0.98)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=400, bbox_inches='tight')
    print(f"Saved figure to {out_png}")


if __name__ == '__main__':
    main()