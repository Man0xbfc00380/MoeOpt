from pathlib import Path
import sys
import json
import subprocess
import re
from typing import Dict, List, Tuple, Set

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches


EVAL_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = EVAL_DIR / 'configs' / 'ablation_config'
STATS_DIR = EVAL_DIR / 'statistics'
RESULTS_DIR = STATS_DIR / 'results'
CACHE_DIR = RESULTS_DIR / 'cache_ablation'
TMP_DIR = STATS_DIR / 'tmp'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


MODELS = [
    'DeepSeek-MoE',
    'Phi3.5-MoE',
    'Yuan2.0',
    'Qwen3-MoE',
]

LAYER_INDEX = 6
INPUT_LENGTHS = [64, 256]
ROUNDS = 50


SERIES_64 = [
    # ('A1', 'A1_64_2_mul_2.json', 0),
    ('A2', 'A2_64_2_mul_2.json', 0),
    ('A3', 'A3_64_2_mul_2.json', 0),
    ('A4', 'A4_64_2_mul_2.json', 0),
    ('A5', 'A5_64_2_mul_2.json', 20),
]

SERIES_256 = [
    # ('A1', 'A1_256_2_mul_2.json', 0),
    # ('A2', 'A2_256_2_mul_2.json', 0),
    # ('A3', 'A3_256_2_mul_2.json', 0),
    ('A4', 'A4_256_2_mul_2.json', 0),
    # ('A5', 'A5_256_2_mul_2.json', 10),
]


UTIL_RE = re.compile(r"Overall average chip utilization in STEADY phase:\s*([0-9]*\.[0-9]+|[0-9]+)")
FINISH_RE = re.compile(r"FINISH occurred at frame:\s*([0-9]+)")


def model_to_folder(model_name: str) -> str:
    mapping = {
        'DeepSeek-MoE': 'deepseek_outputs',
        'Qwen3-MoE': 'qwen_outputs',
        'Phi3.5-MoE': 'phi_outputs',
        'Yuan2.0': 'yuan_outputs',
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown model: {model_name}")
    return mapping[model_name]


def dataset_to_folder(dataset_name: str) -> str:
    name = (dataset_name or '').lower()
    if name in ('wikitext', 'wikitext-2', 'wikitext2', 'wikitext_2'):
        return 'wikitext'
    if name in ('c4', 'C4'):
        return 'c4'
    return name


def resolve_base_trace_json(model: str, dataset: str, layer: int, input_len: int, run_idx: int) -> Path:
    base = EVAL_DIR / 'network_test' / model_to_folder(model)
    ds_folder = dataset_to_folder(dataset)
    return base / f'{ds_folder}_{input_len}_100' / f'run_{run_idx}' / f'layer_{layer}_token_experts.json'


def load_mapping(json_path: Path) -> Dict[int, List[int]]:
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            result: Dict[int, List[int]] = {}
            for k, v in data.items():
                try:
                    tk = int(k)
                except Exception:
                    continue
                lst = v if isinstance(v, list) else []
                result[tk] = [int(e) for e in lst]
            return result
    return {}


def compute_expert_token_counts(mapping: Dict[int, List[int]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for tok, experts in mapping.items():
        if not isinstance(experts, list):
            continue
        for e in experts:
            counts[e] = counts.get(e, 0) + 1
    return counts


def bottom_experts(counts: Dict[int, int], percent: int, total_tokens: int) -> Set[int]:
    if percent <= 0:
        return set()
    sorted_items = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    limit = max(1, int(round(total_tokens * (percent / 100.0))))
    acc = 0
    chosen: Set[int] = set()
    for e, c in sorted_items:
        if acc + c > limit:
            break
        chosen.add(e)
        acc += c
    return chosen


def apply_pool_and_buffer(base: Dict[int, List[int]], pool: Set[Tuple[int, int]], percent: int,
                          final_run: bool, total_tokens: int) -> Tuple[Dict[int, List[int]], Set[Tuple[int, int]], Set[int], int]:
    mapping = {int(k): list(v) if isinstance(v, list) else [] for k, v in base.items()}
    added = 0
    max_key = max(mapping.keys()) if mapping else -1
    unique_tokens = sorted({int(t) for (t, _) in pool})
    reindex: Dict[int, int] = {}
    next_id = max_key + 1
    for orig in unique_tokens:
        reindex[orig] = next_id
        next_id += 1
    for t, e in pool:
        key = reindex[int(t)]
        if key not in mapping:
            mapping[key] = []
        ee = int(e)
        if ee not in mapping[key]:
            mapping[key].append(ee)
            added += 1
    if final_run or percent <= 0:
        return mapping, set(), set(), added
    counts = compute_expert_token_counts(mapping)
    low = bottom_experts(counts, percent, total_tokens)
    new_pool: Set[Tuple[int, int]] = set()
    for t, experts in mapping.items():
        keep: List[int] = []
        for e in experts:
            if e in low:
                new_pool.add((int(t), int(e)))
            else:
                keep.append(e)
        mapping[t] = keep
    return mapping, new_pool, low, added


def write_tmp_mapping(tmp_path: Path, mapping: Dict[int, List[int]]) -> None:
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    to_dump = {str(k): list(v) if isinstance(v, list) else [] for k, v in mapping.items()}
    tmp_path.write_text(json.dumps(to_dump, ensure_ascii=False), encoding='utf-8')


def run_once(model: str, dataset: str, cfg_rel: str, layer: int, trace_json: Path) -> Tuple[float, int, str]:
    runner_path = EVAL_DIR / 'runner_one_layer.py'
    cfg_path = CONFIGS_DIR / cfg_rel
    cmd = [
        sys.executable,
        str(runner_path),
        '--model', model,
        '--dataset', dataset,
        '--config', str(cfg_path),
        '--layer', str(layer),
        '--write_csv',
        '--trace_json', str(trace_json),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(EVAL_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout or ''
    util_match = UTIL_RE.search(stdout)
    steady_util = float(util_match.group(1)) if util_match else float('nan')
    finish_frame = -1
    m2 = FINISH_RE.search(stdout)
    if m2:
        try:
            finish_frame = int(m2.group(1))
        except Exception:
            finish_frame = -1
    return steady_util, finish_frame, stdout


def load_cache(cache_path: Path) -> Dict:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {
        'runs': [],
        'pool': [],
        'slackness': 0,
        'last_run_index': 0,
    }


def save_cache(cache_path: Path, data: Dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def run_series_with_buffering(model: str, dataset: str, cfg_name: str, slack_percent: int,
                              layer: int, rounds: int, quick: bool, input_len: int) -> float:
    if quick:
        rounds = min(rounds, 1)
    cache_path = CACHE_DIR / f"{model}-{dataset}" / f"{cfg_name}.json"
    cache = load_cache(cache_path)
    cache['slackness'] = slack_percent
    pool: Set[Tuple[int, int]] = set(tuple(p) for p in cache.get('pool', []))
    util_values: List[float] = []
    for i in range(1, rounds + 1):
        if any(r.get('index') == i for r in cache.get('runs', [])):
            u = next(r['steady_util'] for r in cache['runs'] if r['index'] == i)
            util_values.append(u)
            continue
        final_run = (i == rounds)
        base_trace = resolve_base_trace_json(model, dataset, layer, input_len, i)
        base_mapping = load_mapping(base_trace)
        mapping_for_run, new_pool, low_exps, added_units = apply_pool_and_buffer(
            base_mapping, pool, slack_percent, final_run, input_len)
        tmp_map_path = TMP_DIR / f"{model}-L{input_len}" / cfg_name / f"run_{i}.json"
        write_tmp_mapping(tmp_map_path, mapping_for_run)
        util, finish_frame, stdout = run_once(model, dataset, cfg_name, layer, tmp_map_path)
        pool = new_pool
        cache['pool'] = [[t, e] for (t, e) in sorted(pool)]
        cache['runs'].append({
            'index': i,
            'steady_util': util,
            'finish_frame': finish_frame,
            'pool_size': len(pool),
            'low_experts_count': len(low_exps),
            'added_units': added_units,
        })
        cache['last_run_index'] = i
        save_cache(cache_path, cache)
        print(f"[Run {i:03d}] model={model}, dataset={dataset}, cfg={cfg_name}, util={util:.6f}, finish={finish_frame}, pool_size={len(pool)}")
        util_values.append(util)
    valid = [u for u in util_values if not (u != u)]
    mean_util = (sum(valid) / len(valid)) if valid else float('nan')
    return mean_util


def compute_throughput(mean_util: float, input_len: int) -> float:
    if mean_util != mean_util or mean_util <= 0.0:
        return float('nan')
    effective_compute = mean_util * 2048 * 4 * 800 * 10**6
    operations = 3 * 2048 * 1024 * 40 * 100 * input_len
    if operations == 0:
        return float('nan')
    return effective_compute / operations


def plot_groups(means: Dict[str, List[float]], save_path: Path) -> None:
    strategy_colors = {
        'A1': '#2c5aa0',
        'A2': '#2e8b57',
        'A3': '#d2691e',
        'A4': '#8b4513',
        'A5': '#a0522d',
    }
    group_count = len(MODELS) * len(INPUT_LENGTHS)
    series_count = len(SERIES_256)
    bar_width = 0.22
    group_gap = 0.4
    series_gap = 0.05
    positions: List[List[float]] = []
    cursor = 0.0
    for gi in range(group_count):
        ser_pos = []
        start = cursor
        for si in range(series_count):
            ser_pos.append(start + si * (bar_width + series_gap))
        positions.append(ser_pos)
        cursor = ser_pos[-1] + bar_width + group_gap
    total_slots = cursor
    width = max(16, int(total_slots * 0.10))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots(figsize=(width, 3.3))
    all_x: List[float] = []
    for gi, (m, il) in enumerate([(m, il) for m in MODELS for il in INPUT_LENGTHS]):
        group_key = f"{m}-L{il}"
        vals = means.get(group_key, [float('nan')] * series_count)
        for si, (series_label, _, _) in enumerate(SERIES_256):
            x = positions[gi][si]
            y = vals[si] if si < len(vals) else float('nan')
            all_x.append(x)
            ax.bar(
                x, y,
                width=bar_width,
                color=strategy_colors.get(series_label, 'gray'),
                edgecolor='black',
                linewidth=0.5,
                label=None,
            )
    tick_positions = [sum(positions[gi]) / len(positions[gi]) for gi in range(group_count)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([])
    ax.set_ylabel('Utilization', fontsize=20)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend_handles = [
        mpatches.Patch(facecolor=strategy_colors.get(lbl, 'gray'), edgecolor='black', label=lbl)
        for (lbl, _, _) in SERIES_256
    ]
    legend = ax.legend(handles=legend_handles, loc='upper center', fontsize=20, ncol=series_count, bbox_to_anchor=(0.5, 1.2))
    legend.get_frame().set_facecolor((1, 1, 1, 0.3))
    if all_x:
        left_edge = min(all_x) - bar_width / 2.0 - 0.1
        right_edge = max(all_x) + bar_width / 2.0 + 0.1
        ax.set_xlim(left_edge, right_edge)
        ax.margins(x=0)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=400)
    print(f"[Saved] {save_path}")


def plot_groups_throughput(means: Dict[str, List[float]], save_path: Path) -> None:
    strategy_colors = {
        'A1': '#2c5aa0',
        'A2': '#2e8b57',
        'A3': '#d2691e',
        'A4': '#8b4513',
        'A5': '#a0522d',
    }
    group_count = len(MODELS) * len(INPUT_LENGTHS)
    series_count = len(SERIES_256)
    bar_width = 0.22
    group_gap = 0.4
    series_gap = 0.05
    positions: List[List[float]] = []
    cursor = 0.0
    for gi in range(group_count):
        ser_pos = []
        start = cursor
        for si in range(series_count):
            ser_pos.append(start + si * (bar_width + series_gap))
        positions.append(ser_pos)
        cursor = ser_pos[-1] + bar_width + group_gap
    total_slots = cursor
    width = max(16, int(total_slots * 0.10))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots(figsize=(width, 3.3))
    all_x: List[float] = []
    for gi, (m, il) in enumerate([(m, il) for m in MODELS for il in INPUT_LENGTHS]):
        group_key = f"{m}-L{il}"
        vals = means.get(group_key, [float('nan')] * series_count)
        for si, (series_label, _, _) in enumerate(SERIES_256):
            x = positions[gi][si]
            y = vals[si] if si < len(vals) else float('nan')
            all_x.append(x)
            ax.bar(
                x, y,
                width=bar_width,
                color=strategy_colors.get(series_label, 'gray'),
                edgecolor='black',
                linewidth=0.5,
                label=None,
            )
    tick_positions = [sum(positions[gi]) / len(positions[gi]) for gi in range(group_count)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([])
    ax.set_ylabel('Throughput', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend_handles = [
        mpatches.Patch(facecolor=strategy_colors.get(lbl, 'gray'), edgecolor='black', label=lbl)
        for (lbl, _, _) in SERIES_256
    ]
    legend = ax.legend(handles=legend_handles, loc='upper center', fontsize=20, ncol=series_count, bbox_to_anchor=(0.5, 1.2))
    legend.get_frame().set_facecolor((1, 1, 1, 0.3))
    if all_x:
        left_edge = min(all_x) - bar_width / 2.0 - 0.1
        right_edge = max(all_x) + bar_width / 2.0 + 0.1
        ax.set_xlim(left_edge, right_edge)
        ax.margins(x=0)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=400)
    print(f"[Saved] {save_path}")


def main():
    quick = '--quick' in sys.argv
    means: Dict[str, List[float]] = {}
    thrs: Dict[str, List[float]] = {}
    for model in MODELS:
        dataset = 'C4'
        for il in INPUT_LENGTHS:
            group_key = f"{model}-L{il}"
            group_means: List[float] = []
            group_thrs: List[float] = []
            series_list = SERIES_64 if il == 64 else SERIES_256
            for series_label, cfg_name, slack in series_list:
                mean_u = run_series_with_buffering(model, dataset, cfg_name, slack, LAYER_INDEX, ROUNDS, quick, il)
                group_means.append(mean_u)
                thr = compute_throughput(mean_u, il)
                group_thrs.append(thr)
            means[group_key] = group_means
            thrs[group_key] = group_thrs
    save_path_u = RESULTS_DIR / 'ablation_groups_bars.png'
    plot_groups(means, save_path_u)
    save_path_t = RESULTS_DIR / 'ablation_groups_throughput_bars.png'
    plot_groups_throughput(thrs, save_path_t)


if __name__ == '__main__':
    main()