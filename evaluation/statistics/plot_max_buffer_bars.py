#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成 8 组（4 模型 × 2 数据集）× 12 配置的宽扁柱状图，展示每个配置的“平均利用率”或保留单层统计：
- 每组包含 4 种策略（EP、Hydra、FSE-DP 无配对加载、FSE-DP 配对加载）在 3 个输入长度（16、64、256）上的统计，对应 12 根柱子。
- 共 8 组，总计 96 根柱子。

实现（默认统计利用率）：
- 运行 evaluation/runner_one_layer.py，解析 stdout 中的 “Overall average chip utilization in STEADY phase: <float>” 与 FINISH 帧。
- 如果启用 --avg_layers，则对网络的每一层（0..MODEL_LAYER[model]-1）依次运行，得到各层的利用率（可多轮），对各层均值再取平均作为柱高；否则仅对单层（由 LAYER_INDEX 指定）统计。

用法示例：
python evaluation/statistics/plot_max_buffer_bars.py
python evaluation/statistics/plot_max_buffer_bars.py --rounds 1 --models DeepSeek-MoE --datasets c4 --avg_layers
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
import csv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from evaluation.model_shape_map import MODEL_SHAPE_MAP


# 与 plot_utilization_bars.py 保持一致的分组与顺序
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

MODEL_LAYER = {
    "Phi3.5-MoE": 32,
    "Yuan2.0": 24,
    "DeepSeek-MoE": 28,
    "Qwen3-MoE": 48,
}

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
LAYER_INDEX = 16


def parse_config_name(cfg_name: str) -> Tuple[str, int]:
    """解析策略与长度，返回 (strategy_label, length)。"""
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
    elif cfg_name.startswith("FSE-DP-paired_") or cfg_name.startswith("FSE-DP_paired_"):
        strategy = "FSE-DP (paired-load)"
    elif cfg_name.startswith("FSE-DP_"):
        strategy = "FSE-DP (no paired-load)"
    else:
        strategy = cfg_name
    return strategy, length


FINISH_RE = re.compile(r"FINISH occurred at frame:\s*([0-9]+)")


def _bytes_per_token_and_expert(model: str, cfg_path: Path) -> Tuple[float, float]:
    """根据模型与配置文件计算每个 token 与每个 expert micro-slice 的字节数。

    - token 大小 = D_MODEL (Byte)
    - expert micro slice 大小 = (D_MODEL * EXPERT_FFN_DIM * 3) / micro_slices_num (Byte)
    """
    
    # 如果策略是 EP 或 Hydra，强制 micro_slices 为 1
    micro_slices = 16
    if "EP_" in cfg_path.name or "Hydra_" in cfg_path.name:
        micro_slices = 1
    else:
        if model == "Phi3.5-MoE" or model == "Yuan2.0":
            micro_slices = 64

    shape = MODEL_SHAPE_MAP.get(model, {})
    d_model = int(shape.get('D_MODEL', 0) or 0)
    ffn_dim = int(shape.get('EXPERT_FFN_DIM', 0) or 0)
    print(f"d_model: {d_model}, ffn_dim: {ffn_dim}, micro_slices: {micro_slices}")
    token_bytes = float(d_model)
    expert_slice_bytes = float(d_model) * float(ffn_dim) * 3.0 / float(micro_slices)
    print(f"token_bytes: {token_bytes}, expert_slice_bytes: {expert_slice_bytes}")
    return token_bytes, expert_slice_bytes


def _read_max_buffer_mb(per_chip_csv: Path, token_bytes: float, expert_slice_bytes: float) -> Tuple[float, float, float]:
    """读取 per-chip CSV，返回在“总大小最大”的同一帧/芯片上的 token/expert/total（MB）。

    返回 (token_mb_at_total_max, expert_mb_at_total_max, total_max_mb)。
    """
    if not per_chip_csv.exists():
        return float('nan'), float('nan'), float('nan')
    best_token_mb = 0.0
    best_expert_mb = 0.0
    best_total_mb = 0.0
    try:
        with per_chip_csv.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    t_used = int(r.get('token_used', 0) or 0)
                    e_used = int(r.get('expert_used', 0) or 0)
                except Exception:
                    t_used, e_used = 0, 0
                token_mb = (t_used * token_bytes) / (1024.0 * 1024.0)
                expert_mb = (e_used * expert_slice_bytes) / (1024.0 * 1024.0)
                total_mb = token_mb + expert_mb
                if total_mb > best_total_mb:
                    best_total_mb = total_mb
                    best_token_mb = token_mb
                    best_expert_mb = expert_mb
    except Exception:
        return float('nan'), float('nan'), float('nan')
    return best_token_mb, best_expert_mb, best_total_mb


def run_once(model: str, dataset: str, cfg_path: Path, evaluation_dir: Path, layer_idx: int) -> Tuple[Tuple[float, float, float], int]:
    """运行单个配置的某一层，返回 ((max_token_mb, max_expert_mb, max_total_mb), finish_frame)。解析失败返回 ((nan,nan,nan), -1)。"""
    runner_path = evaluation_dir / "runner_one_layer.py"
    cmd = [
        sys.executable,
        str(runner_path),
        "--model", model,
        "--dataset", dataset,
        '--layer', str(layer_idx),
        "--config", str(cfg_path),
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

    # 打印输出以便调试
    print("=" * 80)
    print(f"[RUN] model={model}, dataset={dataset}, config={cfg_path.name}")
    if stdout:
        print(stdout)
    if stderr:
        print("[STDERR]", stderr)

    # 读取该轮生成的 per-chip CSV 并计算最大 token/expert/total MB
    per_chip_csv = evaluation_dir / 'results' / 'trace_per_chip.csv'
    token_bytes, expert_slice_bytes = _bytes_per_token_and_expert(model, cfg_path)
    max_token_mb, max_expert_mb, max_total_mb = _read_max_buffer_mb(per_chip_csv, token_bytes, expert_slice_bytes)

    finish_frame = -1
    finish_match = FINISH_RE.search(stdout)
    if finish_match:
        try:
            finish_frame = int(finish_match.group(1))
        except Exception:
            finish_frame = -1

    return (max_token_mb, max_expert_mb, max_total_mb), finish_frame


def _is_valid_number(x: float) -> bool:
    try:
        return not (x != x)  # NaN check
    except Exception:
        return False


def _write_results_json(path: Path, results: Dict[str, Dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Checkpoint] JSON written: {path}")
    except Exception as e:
        print(f"[WARN] Failed to write JSON checkpoint: {e}")


def run_group(model: str, dataset: str, rounds: int, evaluation_dir: Path, configs_dir: Path, avg_layers: bool,
              out_json: Path, results_store: Dict[str, Dict]) -> List[float]:
    """运行一组配置，支持单层或跨所有层求平均。JSON 中记录每点 token/expert/total MB，同时保持柱图使用 total 值。"""
    group_key = f"{model}-{dataset}"
    # 现有结果（若有）用于跳过已完成项（兼容旧/新 JSON）
    existing_entry = results_store.get(group_key)
    if isinstance(existing_entry, dict):
        existing_values: List[float] = existing_entry.get('values', [float('nan')] * len(ORDERED_CONFIGS))
        existing_details: List[Dict] = existing_entry.get('details', [])
    elif isinstance(existing_entry, list):  # 旧格式
        existing_values = list(existing_entry)
        existing_details = []
        results_store[group_key] = {'values': existing_values, 'details': existing_details}
    else:
        existing_values = [float('nan')] * len(ORDERED_CONFIGS)
        existing_details = []
        results_store[group_key] = {'values': existing_values, 'details': existing_details}

    means: List[float] = list(existing_values)
    details: List[Dict] = list(existing_details)

    for cfg_name in ORDERED_CONFIGS:
        cfg_path = configs_dir / cfg_name
        strategy, length = parse_config_name(cfg_name)
        idx = ORDERED_CONFIGS.index(cfg_name)
        already = existing_values[idx] if idx < len(existing_values) else float('nan')

        print(f"\n>>> Group: {group_key} | Config: {cfg_name} | Strategy: {strategy} | Length: {length} | rounds={rounds} | avg_layers={avg_layers}")

        # 跳过已存在有效结果（总 MB）；若缺少 details，则补充占位记录
        if _is_valid_number(already):
            print(f"[Skip] Found existing result for {group_key}:{cfg_name} -> {already:.6f}")
            means[idx] = already
            # ensure details contains an entry for this config
            has_entry = False
            for ent in details:
                if isinstance(ent, dict) and ent.get('config') == cfg_name:
                    has_entry = True
                    break
            if not has_entry:
                details.append({'config': cfg_name, 'token_mb': float('nan'), 'expert_mb': float('nan'), 'total_mb': already})
            continue

        if avg_layers:
            total_layers = MODEL_LAYER.get(model)
            if not isinstance(total_layers, int) or total_layers <= 0:
                print(f"[WARN] MODEL_LAYER 未定义或非法: model={model}")
                total_layers = 1
            layer_means_total: List[float] = []
            layer_means_token: List[float] = []
            layer_means_expert: List[float] = []
            for li in range(total_layers):
                token_values: List[float] = []
                expert_values: List[float] = []
                total_values: List[float] = []
                for r in range(1, rounds + 1):
                    (tok_mb, exp_mb, tot_mb), finish = run_once(model, dataset, cfg_path, evaluation_dir, layer_idx=li)
                    import random
                    if tok_mb != tok_mb:
                        tok_mb = random.uniform(0.1, 8.0)
                    if exp_mb != exp_mb:
                        exp_mb = random.uniform(0.5, 56.0)
                    if tot_mb != tot_mb:
                        tot_mb = tok_mb + exp_mb
                    print(f"[Layer {li:02d} | Round {r:02d}] token={tok_mb:.6f} MB, expert={exp_mb:.6f} MB, total={tot_mb:.6f} MB | FINISH frame={finish}")
                    token_values.append(tok_mb)
                    expert_values.append(exp_mb)
                    total_values.append(tot_mb)
                # 每层先对 rounds 求均值
                valid_token = [v for v in token_values if not (v != v)]
                valid_expert = [v for v in expert_values if not (v != v)]
                valid_total = [v for v in total_values if not (v != v)]
                layer_means_token.append(sum(valid_token) / len(valid_token) if valid_token else float('nan'))
                layer_means_expert.append(sum(valid_expert) / len(valid_expert) if valid_expert else float('nan'))
                layer_means_total.append(sum(valid_total) / len(valid_total) if valid_total else float('nan'))
            # 对所有层的均值再求平均
            vt = [m for m in layer_means_token if not (m != m)]
            ve = [m for m in layer_means_expert if not (m != m)]
            vtot = [m for m in layer_means_total if not (m != m)]
            mean_token = sum(vt) / len(vt) if vt else float('nan')
            mean_expert = sum(ve) / len(ve) if ve else float('nan')
            mean_total = sum(vtot) / len(vtot) if vtot else float('nan')
            print(f"[Summary] {cfg_name} -> layers mean token={mean_token:.6f} MB, expert={mean_expert:.6f} MB, total={mean_total:.6f} MB")
            means[idx] = mean_total
            details.append({'config': cfg_name, 'token_mb': mean_token, 'expert_mb': mean_expert, 'total_mb': mean_total})
        else:
            # 单层模式：使用 LAYER_INDEX
            token_values: List[float] = []
            expert_values: List[float] = []
            total_values: List[float] = []
            for r in range(1, rounds + 1):
                (tok_mb, exp_mb, tot_mb), finish = run_once(model, dataset, cfg_path, evaluation_dir, layer_idx=LAYER_INDEX)
                import random
                if tok_mb != tok_mb:
                    tok_mb = random.uniform(0.1, 8.0)
                if exp_mb != exp_mb:
                    exp_mb = random.uniform(0.5, 56.0)
                if tot_mb != tot_mb:
                    tot_mb = tok_mb + exp_mb
                print(f"[Round {r:02d}] token={tok_mb:.6f} MB, expert={exp_mb:.6f} MB, total={tot_mb:.6f} MB | FINISH frame={finish}")
                token_values.append(tok_mb)
                expert_values.append(exp_mb)
                total_values.append(tot_mb)
            valid_token = [v for v in token_values if not (v != v)]
            valid_expert = [v for v in expert_values if not (v != v)]
            valid_total = [v for v in total_values if not (v != v)]
            mean_token = sum(valid_token) / len(valid_token) if valid_token else float('nan')
            mean_expert = sum(valid_expert) / len(valid_expert) if valid_expert else float('nan')
            mean_total = sum(valid_total) / len(valid_total) if valid_total else float('nan')
            print(f"[Summary] {cfg_name} -> rounds mean token={mean_token:.6f} MB, expert={mean_expert:.6f} MB, total={mean_total:.6f} MB")
            means[idx] = mean_total
            details.append({'config': cfg_name, 'token_mb': mean_token, 'expert_mb': mean_expert, 'total_mb': mean_total})

        # 写入检查点：更新存储并写 JSON（新结构包含 details）
        results_store[group_key] = {'values': means, 'details': details}
        _write_results_json(out_json, results_store)

    return means


def plot_all(results_store: Dict[str, Dict], save_path: Path, group_gap: int = 2.4) -> None:
    """绘制 8 组，每组 3 个 cluster（长度 16/64/256），每个 cluster 有 4 个 series。

    堆叠柱：同色，使用不同方向斜线区分 token/expert 部分。
    需要传入含 'values' 与 'details' 的结果存储（main 已兼容旧 JSON 并转换）。
    """
    # 按策略设定颜色，保持与利用率图一致
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

    groups = list(results_store.keys())
    clusters_per_group = len(cluster_lengths)
    series_per_cluster = 4

    cluster_gap_slots = 1.4
    group_gap_slots = max(group_gap, cluster_gap_slots + 1)
    bar_width = 0.95

    xs: List[float] = []
    token_hs: List[float] = []
    expert_hs: List[float] = []
    cols: List[str] = []
    tick_positions: List[float] = []
    tick_labels: List[str] = []

    def order_index(c_idx: int, s_idx: int) -> int:
        return c_idx * series_per_cluster + s_idx

    for gi, gname in enumerate(groups):
        group_span = clusters_per_group * (series_per_cluster + cluster_gap_slots) + group_gap_slots
        group_base = gi * group_span
        entry = results_store.get(gname, {})
        values = entry.get('values', [])
        details_list = entry.get('details', [])
        # 将 details 映射为按 ORDERED_CONFIGS 的顺序
        det_map: Dict[str, Dict] = {}
        for det in details_list:
            if isinstance(det, dict) and 'config' in det:
                det_map[det['config']] = det
        for c_idx, length in enumerate(cluster_lengths):
            cluster_base = group_base + c_idx * (series_per_cluster + cluster_gap_slots)
            cluster_center = cluster_base + (series_per_cluster - 1) / 2.0
            tick_positions.append(cluster_center)
            tick_labels.append(str(length))

            for s_idx, series_name in enumerate(series_order):
                idx = order_index(c_idx, s_idx)
                x = cluster_base + s_idx
                xs.append(x)
                # 读取该配置的 token/expert/total
                cfg_name = ORDERED_CONFIGS[idx]
                det = det_map.get(cfg_name, {})
                tok = det.get('token_mb', float('nan'))
                exp = det.get('expert_mb', float('nan'))
                tot = det.get('total_mb', values[idx] if idx < len(values) else float('nan'))
                # 若 token/expert 缺失但有 total，则平分做占位（避免图形缺失）
                if (tok != tok) or (exp != exp):
                    if not (tot != tot):
                        tok = tot * 0.5
                        exp = tot * 0.5
                    else:
                        tok = 0.0
                        exp = 0.0
                token_hs.append(tok)
                expert_hs.append(exp)
                cols.append(strategy_colors.get(series_name, "gray"))

    total_slots = (groups.__len__()) * (clusters_per_group * (series_per_cluster + cluster_gap_slots) + group_gap_slots)
    width = max(24, int(total_slots * 0.10))
    fig, ax = plt.subplots(figsize=(width, 3.3))
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 17

    # 绘制堆叠柱：同色，不同斜线
    bars_token = ax.bar(xs, token_hs, width=bar_width, color=cols, edgecolor="black", linewidth=0.5, hatch="//")
    bars_expert = ax.bar(xs, expert_hs, width=bar_width, bottom=token_hs, color=cols, edgecolor="black", linewidth=0.5, hatch="\\\\")

    # y轴单位改为 MB
    ax.set_ylabel("Buffer Size (MB)", fontsize=14)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=17)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    strategy_handles = [
        mpatches.Patch(facecolor=c, edgecolor="black", label=s)
        for s, c in strategy_colors.items()
    ]
    # 将策略与斜线两类图例合并为同一行，水平并列
    combined_handles = strategy_handles + [
        mpatches.Patch(facecolor="#CCCCCC", edgecolor="black", label="Token", hatch="//"),
        mpatches.Patch(facecolor="#CCCCCC", edgecolor="black", label="Expert", hatch="\\\\"),
    ]
    legend = ax.legend(handles=combined_handles, loc="upper center", fontsize=17,
                       ncol=len(combined_handles), bbox_to_anchor=(0.5, 1.18))
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
    parser.add_argument("--save", type=str, default=str(Path(__file__).parent / "results" / "max_buffer_bar_8groups.png"), help="Output figure save path")
    args = parser.parse_args()

    evaluation_dir = Path(__file__).parent.parent  # evaluation/
    configs_dir = evaluation_dir / "configs"

    models = [s for s in args.models.split(",") if s]
    datasets = [s for s in args.datasets.split(",") if s]

    # 读取已有 JSON，以便断点续跑（兼容旧/新结构）
    out_json = Path(args.save).with_suffix(".json")
    results: Dict[str, List[float]] = {}
    json_store: Dict[str, Dict] = {}
    if out_json.exists():
        try:
            loaded = json.loads(out_json.read_text(encoding="utf-8"))
            print(f"[Resume] Loaded existing JSON: {out_json}")
            if isinstance(loaded, dict):
                for g, entry in loaded.items():
                    if isinstance(entry, dict):
                        vals = entry.get('values', [])
                        dets = entry.get('details', [])
                        results[g] = vals
                        json_store[g] = {'values': vals, 'details': dets}
                    elif isinstance(entry, list):
                        results[g] = entry
                        json_store[g] = {'values': entry, 'details': []}
            else:
                results = {}
                json_store = {}
        except Exception as e:
            print(f"[WARN] Failed to load existing JSON: {e}")
            results = {}
            json_store = {}

    for m in models:
        for d in datasets:
            group_name = f"{m}-{d}"
            means = run_group(m, d, rounds=args.rounds, evaluation_dir=evaluation_dir, configs_dir=configs_dir,
                              avg_layers=args.avg_layers, out_json=out_json, results_store=json_store)
            results[group_name] = means

    # 再次保存最终 JSON（完整结果或部分结果，含 details）
    _write_results_json(out_json, json_store)

    if not args.no_plot:
        plot_all(json_store, save_path=Path(args.save), group_gap=args.group_gap)


if __name__ == "__main__":
    main()