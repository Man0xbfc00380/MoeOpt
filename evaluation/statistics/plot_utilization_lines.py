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
- 四根线画成 2×2 子图：每个子图对应一种策略；
  同一策略使用固定颜色，不同组用线型区分（第一组实线，第二组虚线）。
 - 每得到一个组合的结果即写入 JSON 缓存（默认路径：results/utilization_lines_cache.json），
   下次运行优先读取缓存，可用 --refresh 强制重算并覆盖缓存。

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
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import json
from datetime import datetime
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
    desired_w = 512

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

def _cache_key(model: str, dataset: str, cfg_name: str, layer: int) -> str:
    return f"{model}|{dataset}|{cfg_name}|L{layer}"

def _load_cache(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load cache {path}: {e}")
    return {}

def _save_cache(path: Path, cache: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"[WARN] Failed to save cache {path}: {e}")


def run_once_and_get_curve(model: str, dataset: str, cfg_path: Path, evaluation_dir: Path,
                           cache: Dict[str, Any], cache_path: Path, refresh: bool) -> Tuple[List[int], List[float], float]:
    """Get (x, y_smooth, util_mean) for a (model, dataset, config).
    - If cached and not refresh, return from JSON cache.
    - Otherwise run and append to cache immediately.
    """
    key = _cache_key(model, dataset, cfg_path.name, LAYER_INDEX)

    if (not refresh) and key in cache:
        entry = cache.get(key) or {}
        x_cached = entry.get("x")
        y_cached = entry.get("y")
        util_mean_cached = entry.get("util_mean")
        if isinstance(x_cached, list) and isinstance(y_cached, list):
            # 允许老缓存没有 util_mean 的情况，使用 y 的均值近似
            if isinstance(util_mean_cached, (int, float)):
                util_mean = float(util_mean_cached)
            else:
                try:
                    util_mean = float(np.mean(np.array(y_cached, dtype=np.float64))) if len(y_cached) > 0 else 0.0
                except Exception:
                    util_mean = 0.0
            return x_cached, y_cached, util_mean
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
    # 计算未平滑的逐帧利用率用于总利用率统计
    util_values: List[float] = []
    x, y = _compute_util_curve(rows)
    for r in rows:
        try:
            chips_total = float(r.get("chips_total", 0))
            chips_computing = float(r.get("chips_computing", 0))
            util_values.append(chips_computing / chips_total if chips_total else 0.0)
        except Exception:
            util_values.append(0.0)
    util_mean = float(np.mean(np.array(util_values, dtype=np.float64))) if len(util_values) > 0 else 0.0
    cache[key] = {
        "x": x,
        "y": y,
        "model": model,
        "dataset": dataset,
        "config": cfg_path.name,
        "layer": LAYER_INDEX,
        "util_mean": util_mean,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_cache(cache_path, cache)
    return x, y, util_mean


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", type=str, default="DeepSeek-MoE,Qwen3-MoE", help="comma-separated model list")
    parser.add_argument(
        "--model", type=str, default=None, help="single model to plot; overrides --models")
    parser.add_argument("--datasets", type=str,
                        default="c4", help="comma-separated dataset list")
    parser.add_argument("--save", type=str, default=str(Path(__file__).parent /
                        "results" / "utilization_lines_4groups.png"), help="output image path")
    parser.add_argument("--cache", type=str, default=str(Path(__file__).parent /
                        "results" / "utilization_lines_cache.json"), help="cache JSON path")
    parser.add_argument("--refresh", action="store_true", help="recompute curves even if cached")
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

    # 2×2 子图，每个子图一个策略
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 1.6), sharex=True, sharey=True)
    axes_flat = np.array(axes).flatten()
    cache_path = Path(args.cache)
    cache: Dict[str, Any] = _load_cache(cache_path)

    for idx, cfg_name in enumerate(ORDERED_CONFIGS):
        ax = axes_flat[idx]
        strategy, length = parse_config_name(cfg_name)
        cfg_path = configs_dir / cfg_name
        color = base_colors.get(strategy, "#808080")

        # 子图样式
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.set_title(strategy, fontsize=6, pad=2)

        # 在各组上绘制该策略
        # 收集每条线的总利用率与线型，以便子图右上角标注
        per_line_means: List[Tuple[str, float]] = []

        for gi, (model, dataset) in enumerate(ordered_groups):
            x, y, util_mean = run_once_and_get_curve(model, dataset, cfg_path, evaluation_dir, cache, cache_path, args.refresh)
            if not x:
                print(f"[WARN] No curve data: {model}-{dataset} | {cfg_name}")
                continue
            # X坐标乘以40，并以K显示
            x_scaled = (np.array(x, dtype=np.float64) * 60.0).tolist()
            linestyle = group_linestyles[gi] if gi < len(group_linestyles) else "-"
            label = f"{model}-{dataset}"
            ax.plot(x_scaled, y, label=label, color=color, linestyle=linestyle, linewidth=0.6)
            # 记录用于标注的线型与总利用率（不显示文字名）
            per_line_means.append((linestyle, util_mean))

        # 仅在底行显示 x 标签，左列显示 y 标签
        row = idx // 2
        col = idx % 2
        if row == 1:
            ax.set_xlabel("Clock")
        if col == 0:
            ax.set_ylabel("Utilization")

        # X 轴刻度以 K 显示
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, pos: f"{int(round(v / (1024.0 * 1024)))}M"))

        # 在子图右上角标注每条线的总利用率（百分比，保留1位小数），左侧用对应线型表示
        if per_line_means:
            y0 = 0.95
            gap = 0.24
            for i, (style, umean) in enumerate(per_line_means):
                y = y0 - i * gap
                x_text = 0.98
                # 在文本左侧绘制一个短横线，使用与曲线相同的颜色与线型
                x2 = x_text - 0.03
                x1 = x2 - 0.08
                line = Line2D([x1, x2], [y + 0.03, y + 0.03], transform=ax.transAxes,
                               color=color, linestyle=style, linewidth=0.6)
                ax.add_line(line)
                txt = f"{umean*100:.1f}%"
                ax.text(x_text, y, txt,
                        transform=ax.transAxes,
                        ha="right", va="top", fontsize=5, color="#333333")

    # 共享 legend：仅展示组区分（线型），居中放置在整图顶部
    legend_elements: List[Line2D] = []
    if len(ordered_groups) >= 1:
        legend_elements.append(Line2D([0], [0], color="#333333", linestyle=group_linestyles[0], linewidth=0.8,
                                      label=f"{ordered_groups[0][0]}"))
    if len(ordered_groups) >= 2:
        legend_elements.append(Line2D([0], [0], color="#333333", linestyle=group_linestyles[1], linewidth=0.8,
                                      label=f"{ordered_groups[1][0]}"))
    if legend_elements:
        fig.legend(handles=legend_elements, loc="upper center", fontsize=5, ncol=len(legend_elements),
                   bbox_to_anchor=(0.5, 1.02), frameon=False)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # 为顶部共享图例预留空间
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=400)
    print(f"[Saved] Figure saved to: {save_path}")


if __name__ == "__main__":
    main()
    
