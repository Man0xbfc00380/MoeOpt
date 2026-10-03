#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot utilization line charts (half-column size for papers):
- Default groups: (DeepSeek-MoE, Qwen3-MoE) × c4
- For each group, with input length 256, run 4 configurations in order:
  EP_256_2_mul_2.json, Hydra_256_2_mul_2.json, FSE-DP_256_2_mul_2.json, FSE-DP-paired_256_2_mul_2.json
- Run evaluation/runner_one_layer.py for each config, write CSV; read per-frame chip computing utilization chips_computing / chips_total from CSV
- Smooth curves using Savitzky–Golay (polyorder=3) when SciPy is available;
  otherwise fall back to Gaussian convolution with reflect padding. Target
  window length is 2048 and shrinks if the sequence is shorter, avoiding
  abrupt turns while preserving trend.
- Same strategy uses fixed colors; groups are differentiated by linestyle:
  first group solid, second group dashed.

Usage example:
python anim/evaluation/statistics/plot_utilization_lines.py \
  --save anim/evaluation/statistics/results/utilization_lines_64_4groups.png

Optional arguments:
- --models: comma-separated, default "DeepSeek-MoE,Qwen3-MoE"
- --datasets: comma-separated, default "c4" (internally forced to "c4")
- --model: specify a single model to plot (overrides --models)
"""

from pathlib import Path
import subprocess
import sys
import csv
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import numpy as np
try:
    from scipy.signal import savgol_filter  # type: ignore
except Exception:
    savgol_filter = None  # type: ignore


# Fixed config file names for length 256 (4 per group)
ORDERED_CONFIGS = [
    "EP_256_2_mul_2.json",
    "Hydra_256_2_mul_2.json",
    "FSE-DP_256_2_mul_2.json",
    "FSE-DP-paired_256_2_mul_2.json",
]
LAYER_INDEX = 16
# "expert_buffer_size": 64,

def parse_config_name(cfg_name: str) -> Tuple[str, int]:
    """Parse strategy and length, return (strategy_label, length)."""
    try:
        parts = cfg_name.split("_")
        length = int(parts[1])
    except Exception:
        length = 0

    if cfg_name.startswith("EP_"):
        strategy = "EP"
    elif cfg_name.startswith("Hydra_"):
        strategy = "Hydra"
    elif cfg_name.startswith("FSE-DP-paired_"):
        strategy = "FSE-DP (paired-load)"
    elif cfg_name.startswith("FSE-DP_"):
        strategy = "FSE-DP (no paired-load)"
    else:
        strategy = cfg_name
    return strategy, length


def _read_csv(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _compute_util_curve(rows: List[Dict[str, float]]) -> Tuple[List[int], List[float]]:
    """Compute utilization curve chips_computing/chips_total from CSV rows.
    Smooth the sequence using a curve-friendly method to avoid abrupt turns:
    - Prefer Savitzky–Golay filtering (polyorder=3) when SciPy is available.
    - Otherwise use Gaussian convolution with reflect padding.
    """
    x: List[int] = []
    util: List[float] = []
    for r in rows:
        frame = int(r.get("frame", 0))
        chips_total = float(r.get("chips_total", 0))
        chips_computing = float(r.get("chips_computing", 0))
        x.append(frame)
        util.append(chips_computing / chips_total if chips_total else 0.0)

    n = len(util)
    if n == 0:
        return x, []

    # Fixed target window length: 2048
    desired_w = 2048

    # Prefer Savitzky–Golay: requires odd window length and window_length <= n
    if savgol_filter is not None:
        w_sg = desired_w if (desired_w % 2 == 1) else desired_w + 1
        max_odd = n if (n % 2 == 1) else max(1, n - 1)
        w_sg = min(w_sg, max_odd)
        if w_sg >= 5:
            # Savitzky–Golay produces smooth, curve-like transitions while preserving trends
            y = savgol_filter(np.array(util, dtype=np.float64), window_length=w_sg, polyorder=3, mode="interp")
            y = np.clip(y, 0.0, 1.0)
            return x, y.tolist()

    # Fallback: Gaussian smoothing with reflect padding to avoid edge artifacts
    util_arr = np.array(util, dtype=np.float64)
    # Fallback Gaussian: shrink window if series shorter than desired
    w = min(desired_w, n) if n > 0 else desired_w
    sigma = w / 6.0  # ~99.7% mass within the window
    half = w // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(util_arr, (half, half), mode="reflect")
    y = np.convolve(padded, kernel, mode="valid")  # same length as util
    y = np.clip(y, 0.0, 1.0)
    return x, y.tolist()


def _mix_with_white(hex_color: str, amount: float) -> str:
    """Mix color with white to lighten (amount ∈ [0,1], larger is lighter)."""
    rgb = mcolors.to_rgb(hex_color)
    r = rgb[0] + amount * (1.0 - rgb[0])
    g = rgb[1] + amount * (1.0 - rgb[1])
    b = rgb[2] + amount * (1.0 - rgb[2])
    return mcolors.to_hex((r, g, b))


def run_once_and_get_curve(model: str, dataset: str, cfg_path: Path, evaluation_dir: Path) -> Tuple[List[int], List[float]]:
    """Run runner_one_layer.py once (with --write_csv), read and return (x, y_smooth) from trace.csv in out_dir."""
    runner_path = evaluation_dir / "runner_one_layer.py"
    cmd = [
        sys.executable,
        str(runner_path),
        "--model", model,
        "--dataset", dataset,
        "--config", str(cfg_path),
        '--layer', str(LAYER_INDEX),
        "--write_csv",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(evaluation_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    # runner default out_dir: evaluation/results/{model}_{dataset}
    # out_dir = evaluation_dir / "results" / f"{model}_{dataset}"
    out_dir = evaluation_dir / "results"
    csv_path = out_dir / "trace.csv"
    if not csv_path.exists():
        # fallback: print output for debugging
        print("[WARN] trace.csv not generated:", csv_path)
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print("[STDERR]", proc.stderr)
        return [], []

    rows = _read_csv(csv_path)
    return _compute_util_curve(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", type=str, default="DeepSeek-MoE,Qwen3-MoE", help="comma-separated model list")
    parser.add_argument(
        "--model", type=str, default="DeepSeek-MoE", help="single model to plot; overrides --models")
    parser.add_argument("--datasets", type=str,
                        default="c4", help="comma-separated dataset list")
    parser.add_argument("--save", type=str, default=str(Path(__file__).parent /
                        "results" / "utilization_lines_4groups.png"), help="output image path")
    args = parser.parse_args()

    evaluation_dir = Path(__file__).parent.parent  # evaluation/
    configs_dir = evaluation_dir / "configs"

    # Fix output font and style
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5

    # Group definition order (now only c4)
    models = [s for s in args.models.split(",") if s]
    # If --model is provided, override to single model
    if args.model:
        models = [args.model]
    # Force using only c4 dataset regardless of CLI input
    datasets = ["c4"]

    # Construct groups: only first two models paired with c4
    ordered_groups: List[Tuple[str, str]] = []
    if len(models) >= 2:
        ordered_groups = [
            (models[0], datasets[0]),
            (models[1], datasets[0]),
        ]
    else:
        # If fewer than 2 models provided, pair all with c4
        for m in models:
            ordered_groups.append((m, datasets[0]))

    # Strategy base colors (soft palette, refer to stat bar charts)
    base_colors: Dict[str, str] = {
        "EP": "#2c5aa0",                    # low-saturation dark blue
        "Hydra": "#2e8b57",                 # low-saturation dark green
        "FSE-DP (no paired-load)": "#b22222",  # low-saturation dark red
        "FSE-DP (paired-load)": "#d2691e",     # low-saturation dark orange
    }
    # Linestyles for groups: first solid, second dashed
    group_linestyles = ["-", "--"]

    fig, ax = plt.subplots(figsize=(3.4, 1.6))  # paper half-column size
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Utilization")
    ax.set_ylim(0, 1.0)
    # 将 y 轴刻度格式化为百分比
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    line_handles = []
    line_labels = []

    # 用于记录每个 strategy 是否已经添加过 legend
    strategy_legend_added: Dict[str, bool] = {}

    for gi, (model, dataset) in enumerate(ordered_groups):
        for cfg_name in ORDERED_CONFIGS:
            strategy, length = parse_config_name(cfg_name)
            cfg_path = configs_dir / cfg_name
            x, y = run_once_and_get_curve(
                model, dataset, cfg_path, evaluation_dir)
            if not x:
                print(f"[WARN] No curve data: {model}-{dataset} | {cfg_name}")
                continue

            # Use same color per strategy; differentiate groups by linestyle
            color = base_colors.get(strategy, "#808080")
            linestyle = group_linestyles[gi] if gi < len(group_linestyles) else "-"
            label = f"{model}-{dataset} | {strategy}"
            lh, = ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=0.6)
            # 只添加一次 legend
            if strategy not in strategy_legend_added:
                line_handles.append(lh)
                line_labels.append(strategy)
                strategy_legend_added[strategy] = True

    # 添加 legend，只显示四个 strategy
    legend = ax.legend(handles=line_handles, labels=line_labels, loc="upper right", fontsize=5, ncol=1,
                       bbox_to_anchor=(1, 1.1))
    legend.get_frame().set_facecolor((1, 1, 1, 0.3))
    legend.get_frame().set_linewidth(0)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    print(f"[Saved] Figure saved to: {save_path}")


if __name__ == "__main__":
    main()
    
