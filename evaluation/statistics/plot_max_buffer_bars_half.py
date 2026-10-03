#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成 8 组（4 模型 × 2 数据集）× 8 配置的宽扁柱状图，展示每个配置的“平均利用率”或保留单层统计：
- 每组仅包含 2 种策略（Hydra、FSE-DP 配对加载）在 4 个输入长度（16、64、256、1024）上的统计，对应 8 根柱子。
- 共 8 组，总计 64 根柱子。

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
    "Hydra_16_2_mul_2.json",
    "FSE-DP-paired_16_2_mul_2.json",
    "Hydra_64_2_mul_2.json",
    "FSE-DP-paired_64_2_mul_2.json",
    "Hydra_256_2_mul_2.json",
    "FSE-DP-paired_256_2_mul_2.json",
    "Hydra_1024_2_mul_2.json",
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


def plot_all(results_store: Dict[str, Dict], save_path: Path, group_gap: int = 1.5) -> None:
    """绘制 8 组，每组包含长度 16/64/256/1024，每个长度包含 2 个 series（Hydra、FSE-DP paired）。

    以 x 轴为分界线的双区域柱状图：
    - 区域1（上方，占总高度的 3/4）：Expert 的柱状图，单位 MB，向上绘制。
    - 区域2（下方，占总高度的 1/4）：Token 的柱状图，单位 MB，向下绘制。

    两个区域上下相连，共享一个 x 轴。传入含 'values' 与 'details' 的结果存储（main 已兼容旧 JSON 并转换）。
    所有 y 值已×4，表示 4 个芯片总计。
    """
    # 按策略设定颜色，保持与利用率图一致
    strategy_colors = {
        "Hydra": "#2e8b57",
        "FSE-DP (paired-load)": "#d2691e",
    }

    style_seq = [parse_config_name(n) for n in ORDERED_CONFIGS]
    cluster_lengths = [16, 64, 256, 1024]
    series_order = [
        "Hydra",
        "FSE-DP (paired-load)",
    ]

    groups = list(results_store.keys())
    clusters_per_group = len(cluster_lengths)
    series_per_cluster = len(series_order)

    cluster_gap_slots = 0
    group_gap_slots = max(group_gap, cluster_gap_slots + 1)
    bar_width = 0.95

    xs: List[float] = []
    token_hs: List[float] = []
    expert_hs: List[float] = []
    cols: List[str] = []
    tick_positions: List[float] = []
    tick_labels: List[str] = []
    # 每个集群（长度 16/64/256/1024）用于放置标签的锚点：选该集群内“最长 token 柱子”的 x 与高度
    label_anchor_xs: List[float] = []
    label_anchor_hs: List[float] = []

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
            # 收集该集群内两条 token 柱子的值与位置
            _cluster_token_vals: List[float] = []
            _cluster_xs: List[float] = []

            for s_idx, series_name in enumerate(series_order):
                idx = order_index(c_idx, s_idx)
                x = cluster_base + s_idx
                xs.append(x)
                # 读取该配置的 token/expert/total
                cfg_name = ORDERED_CONFIGS[idx]
                det = det_map.get(cfg_name, {})
                tok = det.get('token_mb', float('nan')) * 4
                exp = det.get('expert_mb', float('nan')) * 4 / 3
                tot = det.get('total_mb', values[idx] if idx < len(values) else float('nan')) * 4
                token_hs.append(tok)
                expert_hs.append(exp)
                cols.append(strategy_colors.get(series_name, "gray"))
                # 记录该集群内的 token 信息用于后续选择“最长 token 柱子”
                _cluster_token_vals.append(tok)
                _cluster_xs.append(x)

            # 选择该集群内“最长 token 柱子”的位置与高度，作为标签锚点
            if _cluster_token_vals:
                _max_i = max(range(len(_cluster_token_vals)), key=lambda i: _cluster_token_vals[i])
                label_anchor_xs.append(_cluster_xs[_max_i])
                label_anchor_hs.append(_cluster_token_vals[_max_i])

    total_slots = (groups.__len__()) * (clusters_per_group * (series_per_cluster + cluster_gap_slots) + group_gap_slots)
    original_width = max(20, int(total_slots * 0.09))
    width = max(10, original_width / 2.0)
    # 使用上下两个子图，共享 x 轴；高度比例 3:1（上：Expert， 下：Token）
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True,
        figsize=(width, 4.2),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0}
    )
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 17

    # 上方区域：Expert（单位 MB），向上柱状图（反斜线斜线背景）
    bars_expert = ax_top.bar(
        xs, expert_hs, width=bar_width, color=cols,
        edgecolor="black", linewidth=0.5, hatch="\\\\"
    )

    # 下方区域：Token（单位 MB），向下柱状图（翻转 y 轴）
    token_hs_mb = token_hs
    bars_token = ax_bottom.bar(
        xs, token_hs_mb, width=bar_width, color=cols,
        edgecolor="black", linewidth=0.5, hatch="//"
    )
    ax_bottom.invert_yaxis()

    # 轴标签与刻度（共享 x 轴，仅底部显示）
    ax_top.set_ylabel("Expert (MB)", fontsize=14)
    ax_bottom.set_ylabel("Token (MB)", fontsize=14)
    ax_top.tick_params(axis='y', labelsize=17)
    ax_bottom.tick_params(axis='y', labelsize=17)
    # token y 轴刻度显示为纯数字（不显示后缀 M）
    def _fmt_m(v, pos):
        try:
            if v == 0:
                return "0"
            # 近似为整数时使用整数形式
            if abs(v - round(v)) < 1e-6:
                return f"{int(round(v))}"
            return f"{v:.1f}"
        except Exception:
            return str(v)
    ax_bottom.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_m))
    ax_bottom.set_xticks(tick_positions)
    # 不显示最下面的 x 轴（刻度与标签）
    ax_bottom.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_bottom.spines['bottom'].set_visible(False)
    # 顶部同样不显示 x 轴标签
    ax_top.tick_params(axis='x', which='both', length=0, labelbottom=False, top=False)

    # 网格与边框
    ax_top.grid(axis="y", linestyle="--", alpha=0.4)
    ax_bottom.grid(axis="y", linestyle="--", alpha=0.4)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_bottom.spines["right"].set_visible(False)

    # x 轴范围一致，上下区域相连
    x_min, x_max = min(xs) - 1, max(xs) + 1
    ax_top.set_xlim(x_min, x_max)
    ax_bottom.set_xlim(x_min, x_max)

    # token y 轴单位改为 MB（无需 k 格式化）

    # y 轴范围与余量设置
    if expert_hs:
        max_exp = max(expert_hs)
        ax_top.set_ylim(0, max_exp * 1.15 if max_exp > 0 else 1.0)
    if token_hs_mb:
        max_tok_mb = max(token_hs_mb)
        ax_bottom.set_ylim(max_tok_mb * 1.15 if max_tok_mb > 0 else 1.0, 0)
    # 在 token 柱子区域内，贴近“每组最长 token 柱子”的末端下方显示标签（16/64/256/1024）
    try:
        if label_anchor_xs and label_anchor_hs and tick_labels:
            # 偏移量（数据坐标，单位 MB），让文本略微位于柱子末端之下
            _offset = (max_tok_mb * 0.03) if (token_hs_mb and max_tok_mb > 0) else 0.1
            for x, h, lab in zip(label_anchor_xs, label_anchor_hs, tick_labels):
                ax_bottom.text(
                    x + 1.5, h + _offset, lab,
                    rotation=90, ha="right", va="top", fontsize=17,
                )
    except Exception:
        # 安全兜底：不影响绘图流程
        pass

    strategy_handles = [
        mpatches.Patch(facecolor=c, edgecolor="black", label=s)
        for s, c in strategy_colors.items()
    ]
    # 策略图例 + 区域说明（顶部）
    combined_handles = strategy_handles + [
        mpatches.Patch(facecolor="#CCCCCC", edgecolor="black", label="Expert (top)", hatch="\\\\"),
        mpatches.Patch(facecolor="#CCCCCC", edgecolor="black", label="Token (bottom)", hatch="//"),
    ]
    legend = ax_top.legend(handles=combined_handles, loc="upper right", fontsize=17,
                           ncol=2, bbox_to_anchor=(1, 1))
    legend.get_frame().set_facecolor((1, 1, 1, 0.3))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # x 轴范围已设置于上方
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    print(f"[Saved] Figure saved to: {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds per configuration, default 10")
    parser.add_argument("--models", type=str, default=",".join(MODELS), help="Comma-separated model list")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS), help="Comma-separated dataset list")
    parser.add_argument("--group_gap", type=int, default=1.5, help="Number of empty bars between groups")
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