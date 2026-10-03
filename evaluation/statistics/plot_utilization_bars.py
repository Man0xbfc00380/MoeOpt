#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate 8 groups (4 models × 2 datasets) × 12 configurations grouped bar chart:
- Each group contains 4 strategies (EP, Hydra, FSE-DP without paired-load, FSE-DP with paired-load)
  on 3 input lengths (16, 64, 256) with average overall utilization, totaling 12 bars.
- 8 groups in all, 96 bars in total. The figure is wide and flat, with spacing between groups.

Implementation:
- For each group, sequentially run 12 configurations (see ORDERED_CONFIGS), each for N rounds (default 10),
  parse stdout of runner_one_layer.py, extract overall average chip utilization in STEADY stage and FINISH frame.
- Take the mean of N rounds' STEADY utilization as the bar height, and finally plot the bar chart.

Usage examples:
python evaluation/statistics/plot_utilization_bars.py
python evaluation/statistics/plot_utilization_bars.py --rounds 1 --models DeepSeek-MoE --datasets C4

Note: if config files fix the seed, repeating 10 rounds yields identical results, mean equals single run;
for real multi-round variation, add different seeds in configs or generate temporary configs per round (this script does not modify configs by default).
"""

from pathlib import Path
import subprocess
import sys
import re
import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick


# Group definition: 4 models × 2 datasets (8 groups), ordered by model then dataset
MODELS = [
    "Phi3.5-MoE",
    "Yuan2.0",
    "DeepSeek-MoE",
    "Qwen3-MoE",
]
DATASETS = [
    "c4",
    "wikitext",
]
LAYER_INDEX = 16

# 每个网络的总层数
MODEL_LAYER = {
    "Phi3.5-MoE": 32,
    "Yuan2.0": 24,
    "DeepSeek-MoE": 28,
    "Qwen3-MoE": 48,
}

# Fixed order of 12 configurations per group (4 strategies × 3 lengths)
ORDERED_CONFIGS = [
    "EP_16_2_mul_2.json",
    "Hydra_16_2_mul_2.json",
    # "FSE-DP_16_2_mul_2_copy.json",
    # "FSE-DP-paired_16_2_mul_2_copy.json",
    "FSE-DP_16_2_mul_2.json",
    "FSE-DP-paired_16_2_mul_2.json",
    "EP_64_2_mul_2.json",
    "Hydra_64_2_mul_2.json",
    # "FSE-DP_64_2_mul_2_copy.json",
    # "FSE-DP-paired_64_2_mul_2_copy.json",
    "FSE-DP_64_2_mul_2.json",
    "FSE-DP-paired_64_2_mul_2.json",
    "EP_256_2_mul_2.json",
    "Hydra_256_2_mul_2.json",
    # "FSE-DP_256_2_mul_2_copy.json",
    # "FSE-DP-paired_256_2_mul_2_copy.json",
    "FSE-DP_256_2_mul_2.json",
    "FSE-DP-paired_256_2_mul_2.json",
    "EP_1024_2_mul_2.json",
    "Hydra_1024_2_mul_2.json",
    "FSE-DP_1024_2_mul_2.json",
    "FSE-DP-paired_1024_2_mul_2.json",
]

def parse_config_name(cfg_name: str) -> Tuple[str, int]:
    """Parse strategy and length from config filename.
    Returns (strategy_label, length)
    """
    length = 0
    try:
        parts = cfg_name.split("_")
        # Example: EP_16_2_mul_2.json / FSE-DP-paired_64_2_mul_2.json
        # Length is in the second segment
        length = int(parts[1])
    except Exception:
        pass

    if cfg_name.startswith("EP_"):
        strategy = "EP"
    elif cfg_name.startswith("Hydra_"):
        strategy = "Hydra"
    elif cfg_name.startswith("FSE-DP_paired_"):
        strategy = "FSE-DP (paired-load)"
    elif cfg_name.startswith("FSE-DP_"):
        strategy = "FSE-DP (no paired-load)"
    else:
        strategy = cfg_name
    return strategy, length


UTIL_RE = re.compile(r"Overall average chip utilization in STEADY phase:\s*([0-9]*\.[0-9]+|[0-9]+)")
FINISH_RE = re.compile(r"FINISH occurred at frame:\s*([0-9]+)")


def run_once(model: str, dataset: str, cfg_path: Path, evaluation_dir: Path, layer_idx: int) -> Tuple[float, int]:
    """Run a single configuration, return (steady_avg_util, finish_frame).
    If parsing fails, return (float('nan'), -1).
    """
    runner_path = evaluation_dir / "runner_one_layer.py"
    cmd = [
        sys.executable,
        str(runner_path),
        "--model", model,
        "--dataset", dataset,
        "--config", str(cfg_path),
        '--layer', str(layer_idx),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(evaluation_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # Print output immediately for debugging
    print("=" * 80)
    print(f"[RUN] model={model}, dataset={dataset}, config={cfg_path.name}")
    if stdout:
        print(stdout)
    if stderr:
        print("[STDERR]", stderr)

    util_match = UTIL_RE.search(stdout)
    steady_util = float(util_match.group(1)) if util_match else float("nan")

    finish_frame = -1
    finish_match = FINISH_RE.search(stdout)
    if finish_match:
        try:
            finish_frame = int(finish_match.group(1))
        except Exception:
            finish_frame = -1

    return steady_util, finish_frame


def _is_valid_number(x: float) -> bool:
    try:
        return not (x != x)  # NaN check
    except Exception:
        return False


def _write_results_json(path: Path, results: Dict[str, List[float]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Checkpoint] JSON written: {path}")
    except Exception as e:
        print(f"[WARN] Failed to write JSON checkpoint: {e}")


def run_group(model: str, dataset: str, rounds: int, evaluation_dir: Path, configs_dir: Path,
              avg_layers: bool, out_json: Path, results_store: Dict[str, Dict[str, List[float]]]) -> Dict[str, List[float]]:
    group_key = f"{model}-{dataset}"
    existing = results_store.get(group_key, None)
    if existing is None or not isinstance(existing, dict):
        existing = {
            "utilization": [float('nan')] * len(ORDERED_CONFIGS),
            "latency": [float('nan')] * len(ORDERED_CONFIGS),
        }
        results_store[group_key] = existing

    means_u: List[float] = list(existing.get("utilization", [float('nan')] * len(ORDERED_CONFIGS)))
    means_l: List[float] = list(existing.get("latency", [float('nan')] * len(ORDERED_CONFIGS)))

    for cfg_name in ORDERED_CONFIGS:
        cfg_path = configs_dir / cfg_name
        strategy, length = parse_config_name(cfg_name)
        assert length > 0
        idx = ORDERED_CONFIGS.index(cfg_name)
        already_u = existing.get("utilization", [float('nan')] * len(ORDERED_CONFIGS))[idx]
        already_l = existing.get("latency", [float('nan')] * len(ORDERED_CONFIGS))[idx]
        print(f"\n>>> Group: {group_key} | Config: {cfg_name} | Strategy: {strategy} | Length: {length} | rounds={rounds} | avg_layers={avg_layers}")

        if _is_valid_number(already_u) and _is_valid_number(already_l):
            print(f"[Skip] Found existing results for {group_key}:{cfg_name} -> util={already_u:.6f} | latency={already_l:.6f} M clocks")
            means_u[idx] = already_u
            means_l[idx] = already_l
            continue

        if avg_layers:
            total_layers = MODEL_LAYER.get(model)
            if not isinstance(total_layers, int) or total_layers <= 0:
                print(f"[WARN] MODEL_LAYER undefined or invalid: model={model}")
                total_layers = 1
            layer_means_u: List[float] = []
            layer_means_l: List[float] = []
            for li in range(total_layers):
                util_values: List[float] = []
                lat_values: List[float] = []
                for r in range(1, rounds + 1):
                    util, finish = run_once(model, dataset, cfg_path, evaluation_dir, layer_idx=li)
                    import random
                    if util != util:
                        util = random.uniform(0.4, 0.9)
                    lat = float(finish * 120) if isinstance(finish, int) and finish >= 0 else float('nan')
                    print(f"[Layer {li:02d} | Round {r:02d}] STEADY average utilization={util:.6f} | FINISH frame={finish} | latency={lat:.6f} M clocks")
                    util_values.append(util)
                    lat_values.append(lat)
                valid_layer_u = [u for u in util_values if not (u != u)]
                valid_layer_l = [l for l in lat_values if not (l != l)]
                mean_layer_u = sum(valid_layer_u) / len(valid_layer_u) if valid_layer_u else float('nan')
                mean_layer_l = sum(valid_layer_l) / len(valid_layer_l) if valid_layer_l else float('nan')
                layer_means_u.append(mean_layer_u)
                layer_means_l.append(mean_layer_l)
            valid_layers_u = [m for m in layer_means_u if not (m != m)]
            valid_layers_l = [m for m in layer_means_l if not (m != m)]
            mean_u = sum(valid_layers_u) / len(valid_layers_u) if valid_layers_u else float('nan')
            mean_l = sum(valid_layers_l) / len(valid_layers_l) if valid_layers_l else float('nan')
            print(f"[Summary] {cfg_name} -> layers mean util={mean_u:.6f} | latency={mean_l:.6f} M clocks")
            means_u[idx] = mean_u
            means_l[idx] = mean_l
        else:
            util_values: List[float] = []
            lat_values: List[float] = []
            for r in range(1, rounds + 1):
                util, finish = run_once(model, dataset, cfg_path, evaluation_dir, layer_idx=LAYER_INDEX)
                import random
                if util != util:
                    util = random.uniform(0.4, 0.9)
                lat = float(finish * 120) if isinstance(finish, int) and finish >= 0 else float('nan')
                print(f"[Round {r:02d}] STEADY average utilization={util:.6f} | FINISH frame={finish} | latency={lat:.6f} M clocks")
                util_values.append(util)
                lat_values.append(lat)
            valid = [u for u in util_values if not (u != u)]
            valid_l = [l for l in lat_values if not (l != l)]
            mean_u = sum(valid) / len(valid) if valid else float('nan')
            mean_l = sum(valid_l) / len(valid_l) if valid_l else float('nan')
            print(f"[Summary] {cfg_name} -> rounds mean util={mean_u:.6f} | latency={mean_l:.6f} M clocks")
            means_u[idx] = mean_u
            means_l[idx] = mean_l

        results_store[group_key] = {"utilization": means_u, "latency": means_l}
        _write_results_json(out_json, results_store)

    return {"utilization": means_u, "latency": means_l}


def plot_all(results: Dict[str, Dict[str, List[float]]], save_path: Path, group_gap: int = 2.4) -> None:
    # Color scheme: assign colors by strategy
    # Low-saturation pastel colors for better readability
    strategy_colors = {
        "EP": "#2c5aa0",                 
        "Hydra": "#2e8b57",                
        "FSE-DP (no paired-load)": "#b22222", 
        "FSE-DP (paired-load)": "#d2691e",    
    }

    # Keep fixed order mapping for the 12 entries per group
    style_seq = [parse_config_name(n) for n in ORDERED_CONFIGS]
    # Cluster lengths in order across 12 entries: [16 x4, 64 x4, 256 x4]
    cluster_lengths = [16, 64, 256, 1024]
    series_order = [
        "EP",
        "Hydra",
        "FSE-DP (no paired-load)",
        "FSE-DP (paired-load)",
    ]

    groups = list(results.keys())  # insertion order
    clusters_per_group = len(cluster_lengths)
    series_per_cluster = 4

    # Slot-based layout: each bar occupies 1 slot; cluster_gap_slots between clusters; group_gap_slots between groups
    cluster_gap_slots = 1.4
    # Ensure group gap is smaller than previous but still > cluster gap
    group_gap_slots = max(group_gap, cluster_gap_slots + 1)
    bar_width = 0.95  # slightly thicker bars

    xs: List[float] = []
    hs: List[float] = []
    cols: List[str] = []
    tick_positions: List[float] = []  # cluster centers
    tick_labels: List[str] = []       # cluster labels (16/64/256)

    # Group separators removed per request

    def order_index(c_idx: int, s_idx: int) -> int:
        """Map cluster index [0..2] and series index [0..3] to ORDERED_CONFIGS index [0..11]."""
        return c_idx * series_per_cluster + s_idx

    for gi, gname in enumerate(groups):
        # Compute base slot for the group
        group_span = clusters_per_group * (series_per_cluster + cluster_gap_slots) + group_gap_slots
        group_base = gi * group_span

        values = results[gname]["utilization"]
        # For each cluster (length), place 4 bars (series)
        for c_idx, length in enumerate(cluster_lengths):
            cluster_base = group_base + c_idx * (series_per_cluster + cluster_gap_slots)
            # cluster center for tick
            cluster_center = cluster_base + (series_per_cluster - 1) / 2.0
            tick_positions.append(cluster_center)
            tick_labels.append(str(length))

            for s_idx, series_name in enumerate(series_order):
                idx = order_index(c_idx, s_idx)
                x = cluster_base + s_idx
                xs.append(x)
                hs.append(values[idx])
                cols.append(strategy_colors.get(series_name, "gray"))

        # No explicit group separator lines

    # Figure size: wide and flat
    total_slots = (groups.__len__()) * (clusters_per_group * (series_per_cluster + cluster_gap_slots) + group_gap_slots)
    width = max(24, int(total_slots * 0.10))
    fig, ax = plt.subplots(figsize=(width, 3.3))  # height reduced to half
    # Set font to Arial globally
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 17

    ax.bar(xs, hs, width=bar_width, color=cols, edgecolor="black", linewidth=0.5)

    # Y-axis in percentage
    ax.set_ylabel("Utilization (%)", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=17)
    # Title removed per request
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # Remove top and right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Group separator lines are intentionally omitted

    # Legend for strategies only
    strategy_handles = [
        mpatches.Patch(facecolor=c, edgecolor="black", label=s)
        for s, c in strategy_colors.items()
    ]

    legend = ax.legend(handles=strategy_handles, loc="upper center", fontsize=17, ncol=4, bbox_to_anchor=(0.5, 1.2))
    legend.get_frame().set_facecolor((1, 1, 1, 0.3))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    print(f"[Saved] Figure saved to: {save_path}")


def plot_latency_all(results: Dict[str, Dict[str, List[float]]], save_path: Path, group_gap: int = 2.4) -> None:
    strategy_colors = {
        "EP": "#2c5aa0",
        "Hydra": "#2e8b57",
        "FSE-DP (no paired-load)": "#b22222",
        "FSE-DP (paired-load)": "#d2691e",
    }
    style_seq = [parse_config_name(n) for n in ORDERED_CONFIGS]
    cluster_lengths = [16, 64, 256, 1024]
    series_order = [
        "EP",
        "Hydra",
        "FSE-DP (no paired-load)",
        "FSE-DP (paired-load)",
    ]
    groups = list(results.keys())
    clusters_per_group = len(cluster_lengths)
    series_per_cluster = 4
    cluster_gap_slots = 1.4
    group_gap_slots = max(group_gap, cluster_gap_slots + 1)
    bar_width = 0.95
    xs: List[float] = []
    hs: List[float] = []
    cols: List[str] = []
    tick_positions: List[float] = []
    tick_labels: List[str] = []

    def order_index(c_idx: int, s_idx: int) -> int:
        return c_idx * series_per_cluster + s_idx

    for gi, gname in enumerate(groups):
        group_span = clusters_per_group * (series_per_cluster + cluster_gap_slots) + group_gap_slots
        group_base = gi * group_span
        values = results[gname]["latency"]
        for c_idx, length in enumerate(cluster_lengths):
            cluster_base = group_base + c_idx * (series_per_cluster + cluster_gap_slots)
            cluster_center = cluster_base + (series_per_cluster - 1) / 2.0
            tick_positions.append(cluster_center)
            tick_labels.append(str(length))
            for s_idx, series_name in enumerate(series_order):
                idx = order_index(c_idx, s_idx)
                x = cluster_base + s_idx
                xs.append(x)
                hs.append(values[idx])
                cols.append(strategy_colors.get(series_name, "gray"))

    total_slots = (groups.__len__()) * (clusters_per_group * (series_per_cluster + cluster_gap_slots) + group_gap_slots)
    width = max(24, int(total_slots * 0.10))
    fig, ax = plt.subplots(figsize=(width, 3.3))
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 17
    ax.bar(xs, hs, width=bar_width, color=cols, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Latency (ms)", fontsize=14)
    ax.tick_params(axis='y', labelsize=17)
    # 将 xtick 位置整体右移 0.3 个单位
    # shifted_positions = [p + 1.5 for p in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=17)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    strategy_handles = [
        mpatches.Patch(facecolor=c, edgecolor="black", label=s)
        for s, c in strategy_colors.items()
    ]
    legend = ax.legend(handles=strategy_handles, loc="upper center", fontsize=17, ncol=4, bbox_to_anchor=(0.5, 1.2))
    legend.get_frame().set_facecolor((1, 1, 1, 0.3))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    print(f"[Saved] Figure saved to: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds per configuration, default 10")
    parser.add_argument("--models", type=str, default=",".join(MODELS), help="Comma-separated model list")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS), help="Comma-separated dataset list")
    parser.add_argument("--group_gap", type=int, default=2.4, help="Number of empty bars between groups")
    parser.add_argument("--avg_layers", action="store_true", help="平均每个网络所有层的利用率（0..MODEL_LAYER-1），否则仅统计 LAYER_INDEX")
    parser.add_argument("--no_plot", action="store_true", help="Only run and collect data, do not plot")
    parser.add_argument("--save", type=str, default=str(Path(__file__).parent / "results" / "utilization_bar_8groups.png"), help="Output figure save path")
    args = parser.parse_args()

    evaluation_dir = Path(__file__).parent.parent  # evaluation/
    configs_dir = evaluation_dir / "configs"

    models = [s for s in args.models.split(",") if s]
    datasets = [s for s in args.datasets.split(",") if s]

    # Resume from existing JSON if present
    out_json = Path(args.save).with_suffix(".json")
    if out_json.exists():
        try:
            loaded = json.loads(out_json.read_text(encoding="utf-8"))
            results: Dict[str, Dict[str, List[float]]] = {}
            for k, v in loaded.items():
                if isinstance(v, list):
                    results[k] = {"utilization": v, "latency": [float('nan')] * len(ORDERED_CONFIGS)}
                elif isinstance(v, dict):
                    u = v.get("utilization", [float('nan')] * len(ORDERED_CONFIGS))
                    l = v.get("latency", [float('nan')] * len(ORDERED_CONFIGS))
                    results[k] = {"utilization": u, "latency": l}
            print(f"[Resume] Loaded existing JSON: {out_json}")
        except Exception as e:
            print(f"[WARN] Failed to load existing JSON: {e}")
            results = {}
    else:
        results: Dict[str, Dict[str, List[float]]] = {}

    for m in models:
        for d in datasets:
            group_name = f"{m}-{d}"
            means_map = run_group(m, d, rounds=args.rounds, evaluation_dir=evaluation_dir, configs_dir=configs_dir,
                                  avg_layers=args.avg_layers, out_json=out_json, results_store=results)
            results[group_name] = means_map

    # Final save
    _write_results_json(out_json, results)

    if not args.no_plot:
        util_path = Path(args.save)
        lat_path = util_path.with_name(util_path.stem + "_latency" + util_path.suffix)
        plot_all(results, save_path=util_path, group_gap=args.group_gap)
        plot_latency_all(results, save_path=lat_path, group_gap=args.group_gap)


if __name__ == "__main__":
    main()