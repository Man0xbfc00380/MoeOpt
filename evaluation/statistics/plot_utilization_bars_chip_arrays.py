#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
绘制 Qwen3-MoE 在 C4 数据集上的 3 组（芯片阵列 2×2、3×3、4×4）× 16 根柱状图：
 - 每组包含 4 个策略（EP, Hydra, FSE-DP without paired-load, FSE-DP with paired-load）
  在 4 种输入长度（16, 64, 256, 1024）下的整体平均利用率，共 16 根柱子。
 - 3 组一起绘制在一个宽而扁的柱状图上，组与组之间留空隙，风格参考 plot_utilization_bars.py。

实现要点：
 - 只运行 Qwen3-MoE 网络，在 C4 数据集上。
 - 每个数据点分别运行层 [16, 20, 24, 28]，取这 4 层的平均利用率作为该柱子的高度。
 - 每次跑完一层后立即写入 JSON 断点文件，避免中断后从头跑；下次运行会跳过已有结果。
 - 对于 3×3、4×4 组，需要在运行时覆盖 HardwareConfig.row_number_of_chip 和 column_number_of_chip。
 - 随着阵列增大，DDR 带宽通过 runner_one_layer 的 `--override_expert_bandwidth` 进行覆盖：
   2×2 为 25.6；3×3 为 2×2 的 1.5 倍（38.4）；4×4 为 2×2 的 2 倍（51.2）。

用法示例：
python evaluation/statistics/plot_utilization_bars_chip_arrays.py
python evaluation/statistics/plot_utilization_bars_chip_arrays.py --dry_run  # 仅生成示例图（不调用仿真）
python evaluation/statistics/plot_utilization_bars_chip_arrays.py --no_plot  # 只跑并更新JSON
"""

from pathlib import Path
import subprocess
import sys
import re
import json
from typing import Dict, List, Tuple, Optional

import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick


# 固定模型与数据集
MODEL = "Qwen3-MoE"
DATASET = "c4"

# 需要平均的层索引
LAYERS_TO_AVG = [16, 20, 24, 28]

# 芯片阵列分组（行×列）——移除 2×3 组
CHIP_ARRAYS: List[Tuple[int, int]] = [
    (2, 2),
    (3, 3),
    (4, 4),
]

# 每组固定顺序的 16 个配置（4策略×4长度）
ORDERED_CONFIGS = [
    "EP_16_2_mul_2.json",
    "Hydra_16_2_mul_2.json",
    "FSE-DP_16_2_mul_2.json",
    "FSE-DP-paired_16_2_mul_2.json",
    "EP_64_2_mul_2.json",
    "Hydra_64_2_mul_2.json",
    "FSE-DP_64_2_mul_2.json",
    "FSE-DP-paired_64_2_mul_2.json",
    "EP_256_2_mul_2.json",
    "Hydra_256_2_mul_2.json",
    "FSE-DP_256_2_mul_2.json",
    "FSE-DP-paired_256_2_mul_2.json",
    "EP_1024_2_mul_2.json",
    "Hydra_1024_2_mul_2.json",
    "FSE-DP_1024_2_mul_2.json",
    "FSE-DP-paired_1024_2_mul_2.json",
]


def parse_config_name(cfg_name: str) -> Tuple[str, int]:
    """从配置文件名解析策略与输入长度，返回 (strategy_label, length)。"""
    length = 0
    try:
        parts = cfg_name.split("_")
        length = int(parts[1])
    except Exception:
        pass

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


UTIL_RE = re.compile(r"Overall average chip utilization in STEADY phase:\s*([0-9]*\.[0-9]+|[0-9]+)")
FINISH_RE = re.compile(r"FINISH occurred at frame:\s*([0-9]+)")


def _is_valid_number(x: float) -> bool:
    try:
        return not (x != x)  # NaN check
    except Exception:
        return False


def _write_results_json(path: Path, results: Dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Checkpoint] JSON written: {path}")
    except Exception as e:
        print(f"[WARN] Failed to write JSON checkpoint: {e}")


def _group_key(row: int, col: int) -> str:
    return f"{MODEL}-{DATASET}-{row}x{col}"


def _prepare_temp_config(base_cfg: Path, evaluation_dir: Path, row: int, col: int) -> Path:
    """基于原始配置生成带芯片阵列覆盖的临时配置文件，返回其路径。"""
    try:
        cfg = json.loads(base_cfg.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to load config {base_cfg}: {e}")

    hw = cfg.get('hardware', {})
    hw['row_number_of_chip'] = int(row)
    hw['column_number_of_chip'] = int(col)
    cfg['hardware'] = hw

    # 将模型与数据集写入（可被CLI再次覆盖，但这里也保持一致）
    cfg['model'] = MODEL
    cfg['dataset'] = DATASET

    temp_dir = evaluation_dir / "configs" / "__temp_chip_arrays__"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{base_cfg.stem}_{row}x{col}.json"
    temp_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return temp_path


def _bandwidth_for_array(row: int, col: int) -> Optional[float]:
    """根据阵列规模返回覆盖的 DDR 带宽（GB/s）。未定义返回 None。"""
    base = 25.6
    if (row, col) == (2, 2):
        return base
    if (row, col) == (3, 3):
        return base * 1.5
    if (row, col) == (4, 4):
        return base * 2.0
    return None


def run_once(cfg_path: Path, evaluation_dir: Path, layer_idx: int, dry_run: bool = False,
             override_bandwidth: Optional[float] = None) -> Tuple[float, int]:
    """运行单次仿真，返回 (steady_avg_util, finish_frame)。dry_run 时返回随机示例值。"""
    if dry_run:
        util = random.uniform(0.55, 0.88)
        return util, -1

    runner_path = evaluation_dir / "runner_one_layer.py"
    cmd = [
        sys.executable,
        str(runner_path),
        "--model", MODEL,
        "--dataset", DATASET,
        "--config", str(cfg_path),
        "--layer", str(layer_idx),
    ]

    # 随阵列规模覆盖 DDR 带宽
    if override_bandwidth is not None:
        cmd += ["--override_expert_bandwidth", str(override_bandwidth)]

    proc = subprocess.run(
        cmd,
        cwd=str(evaluation_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    print("=" * 80)
    print(f"[RUN] group-config={cfg_path.name} | layer={layer_idx}")
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


def _ensure_group_entry(store: Dict, row: int, col: int) -> None:
    gk = _group_key(row, col)
    if gk not in store:
        store[gk] = {}
    # 为每个配置预创建结构
    for cfg_name in ORDERED_CONFIGS:
        if cfg_name not in store[gk]:
            store[gk][cfg_name] = {"layers": {}, "mean": float('nan')}


def run_group(row: int, col: int, evaluation_dir: Path, configs_dir: Path,
              out_json: Path, results_store: Dict, dry_run: bool = False) -> None:
    """对一个芯片阵列组（row×col）运行 16 个配置，每个配置按 LAYERS_TO_AVG 平均。每层完成即写 JSON。"""
    gk = _group_key(row, col)
    _ensure_group_entry(results_store, row, col)

    # 计算该组对应的 DDR 带宽覆盖
    bw_override = _bandwidth_for_array(row, col)

    for cfg_name in ORDERED_CONFIGS:
        base_cfg = configs_dir / cfg_name
        if not base_cfg.exists():
            print(f"[WARN] Config not found: {base_cfg}")
            continue
        # 为当前组生成临时配置（覆盖行列）
        temp_cfg = _prepare_temp_config(base_cfg, evaluation_dir, row, col)
        entry = results_store[gk][cfg_name]

        util_values: List[float] = []
        for li in LAYERS_TO_AVG:
            # 若该层已有结果则跳过
            key = str(li)
            already = entry["layers"].get(key, None)
            if isinstance(already, (int, float)) and _is_valid_number(float(already)):
                util_values.append(float(already))
                print(f"[Skip] {gk}:{cfg_name} | layer={li} -> {float(already):.6f}")
                continue

            util, finish = run_once(
                temp_cfg,
                evaluation_dir,
                layer_idx=li,
                dry_run=dry_run,
                override_bandwidth=bw_override,
            )
            # 仅在 dry_run 时做示例填充；真实运行保留 NaN 以便后续重跑
            if dry_run and not _is_valid_number(util):
                util = random.uniform(0.55, 0.88)
            print(f"[Layer {li:02d}] STEADY average utilization={util:.6f} | FINISH frame={finish}")
            entry["layers"][key] = util
            util_values.append(util)
            # 每层写一次断点
            _write_results_json(out_json, results_store)

        # 计算该配置的层平均（过滤 NaN）
        valid = [u for u in util_values if _is_valid_number(u)]
        mean_u = sum(valid) / len(valid) if valid else float('nan')
        entry["mean"] = mean_u
        print(f"[Summary] {gk}:{cfg_name} -> mean of layers={mean_u:.6f}")
        _write_results_json(out_json, results_store)


def _select_config_name(length: int, strategy: str) -> Optional[str]:
    """根据长度与策略名称选择配置文件名。"""
    for name in ORDERED_CONFIGS:
        s, l = parse_config_name(name)
        if l == length and s == strategy:
            return name
    return None


def plot_all(results: Dict, save_path: Path, group_gap: float = 1.2) -> None:
    """绘制 4 组，每组包含 4 个长度的 4 策略（共 16 根）柱状图。"""
    # 策略配色（与参考脚本保持一致风格）
    strategy_colors = {
        "EP": "#2c5aa0",
        "Hydra": "#2e8b57",
        "FSE-DP (no paired-load)": "#b22222",
        "FSE-DP (paired-load)": "#d2691e",
    }

    cluster_lengths = [16, 64, 256, 1024]
    series_order = [
        "EP",
        "Hydra",
        "FSE-DP (no paired-load)",
        "FSE-DP (paired-load)",
    ]

    groups = [
        _group_key(r, c) for (r, c) in CHIP_ARRAYS
    ]
    clusters_per_group = len(cluster_lengths)
    series_per_cluster = len(series_order)

    cluster_gap_slots = 0.6
    group_gap_slots = max(group_gap, cluster_gap_slots + 0.6)
    bar_width = 0.98

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

        # 组中按长度×策略取均值
        for c_idx, length in enumerate(cluster_lengths):
            cluster_base = group_base + c_idx * (series_per_cluster + cluster_gap_slots)
            cluster_center = cluster_base + (series_per_cluster - 1) / 2.0
            tick_positions.append(cluster_center)
            tick_labels.append(str(length))

            for s_idx, series_name in enumerate(series_order):
                cfg_name = _select_config_name(length, series_name)
                mean_val = float('nan')
                try:
                    mean_val = results[gname][cfg_name]["mean"]
                except Exception:
                    mean_val = float('nan')

                x = cluster_base + s_idx
                xs.append(x)
                hs.append(mean_val)
                cols.append(strategy_colors.get(series_name, "gray"))

    total_slots = len(groups) * (clusters_per_group * (series_per_cluster + cluster_gap_slots) + group_gap_slots)
    width = max(16, int(total_slots * 0.10))
    fig, ax = plt.subplots(figsize=(width, 3.3))
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20

    ax.bar(xs, hs, width=bar_width, color=cols, edgecolor="black", linewidth=0.5)

    # ax.set_ylabel("Utilization (%)", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xticks([x + 1.5 for x in tick_positions])
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=20)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 图例
    strategy_handles = [
        mpatches.Patch(facecolor=c, edgecolor="black", label=s)
        for s, c in strategy_colors.items()
    ]
    legend = ax.legend(handles=strategy_handles, loc="upper center", fontsize=20, ncol=4, bbox_to_anchor=(0.5, 1.2))
    legend.get_frame().set_facecolor((1, 1, 1, 0.3))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    print(f"[Saved] Figure saved to: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_gap", type=float, default=1.2, help="组之间的空槽数量（影响整体宽度）")
    parser.add_argument("--dry_run", action="store_true", help="不调用仿真，生成示例随机数据用于预览")
    parser.add_argument("--no_plot", action="store_true", help="只收集数据并保存 JSON，不绘图")
    parser.add_argument("--save", type=str, default=str(Path(__file__).parent / "results" / "utilization_chiparrays_qwen_c4.png"), help="输出图片保存路径")
    args = parser.parse_args()

    evaluation_dir = Path(__file__).parent.parent  # evaluation/
    configs_dir = evaluation_dir / "configs"

    # 断点 JSON 与输出图片同名（不同后缀）
    out_json = Path(args.save).with_suffix(".json")
    if out_json.exists():
        try:
            results: Dict = json.loads(out_json.read_text(encoding="utf-8"))
            print(f"[Resume] Loaded existing JSON: {out_json}")
        except Exception as e:
            print(f"[WARN] Failed to load existing JSON: {e}")
            results = {}
    else:
        results: Dict = {}

    # 逐组运行并写入断点
    for (r, c) in CHIP_ARRAYS:
        print(f"\n>>> Group: {MODEL}-{DATASET} | Chip Array: {r}x{c} | dry_run={args.dry_run}")
        run_group(r, c, evaluation_dir=evaluation_dir, configs_dir=configs_dir,
                  out_json=out_json, results_store=results, dry_run=args.dry_run)

    # 最终保存 JSON
    _write_results_json(out_json, results)

    if not args.no_plot:
        plot_all(results, save_path=Path(args.save), group_gap=args.group_gap)


if __name__ == "__main__":
    main()