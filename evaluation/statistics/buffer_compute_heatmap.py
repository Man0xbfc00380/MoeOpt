#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate two heatmaps (Computation Utilization and Buffer Utilization) across 4 chips over time.
- Runs a single configuration: DeepSeek-MoE × C4 using configs/FSE-DP_64_2_mul_2.json
- Reads per-frame, per-chip metrics from `trace_per_chip.csv`
- Caches results to a dict keyed by (model, dataset) with two 4×T numpy arrays
- Saves figure to `statistics/buffer_compute_heatmap.png` with high DPI
Usage:
    python buffer_compute_heatmap.py
"""
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
MODEL = 'DeepSeek-MoE'
DATASET = 'c4'
CFG_REL = 'configs/FSE-DP_64_2_mul_2.json'
def _run_once(evaluation_dir: Path) -> Tuple[Path, Path]:
    """Run runner_one_layer for the single configuration and return paths to per-chip CSV and output dir."""
    runner_path = evaluation_dir / 'runner_one_layer.py'
    cfg_path = evaluation_dir / CFG_REL
    # Runner default out_dir is taken from config; when 'output_dir' is set in cfg,
    # results are directly under evaluation_dir / output_dir without model/dataset subdir.
    base_out_dir = evaluation_dir / 'results'
    out_dir = base_out_dir
    cmd = [
        sys.executable,
        str(runner_path),
        '--model', MODEL,
        '--dataset', DATASET,
        '--config', str(cfg_path),
        '--write_csv',
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(evaluation_dir),
        capture_output=True,
        text=False,  # Avoid Windows console decoding issues
        check=False,
    )
    # Do not attempt to decode stdout/stderr to prevent UnicodeDecodeError in cp1252 consoles
    per_chip_csv = out_dir / 'trace_per_chip.csv'
    return per_chip_csv, out_dir
def _read_per_chip_csv(per_chip_csv: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read per-chip metrics CSV and return matrices (chips×frames) and frames array.
    Returns: (buffer_util_4xT, compute_util_4xT, frames_1xT)
    """
    rows = []
    with per_chip_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                frame = int(r['frame'])
                chip_idx = int(r['chip_index'])
                buf = float(r['buffer_util'])
                cmpu = float(r['compute_util'])
            except Exception:
                # Skip malformed rows
                continue
            rows.append((frame, chip_idx, buf, cmpu))
    if not rows:
        raise RuntimeError(f'No rows read from {per_chip_csv}')
    # Group by frame and chip index
    frames_sorted = sorted(set(r[0] for r in rows))
    frame_to_col = {f: i for i, f in enumerate(frames_sorted)}
    T = len(frames_sorted)
    num_chips = 4
    buffer_mat = np.zeros((num_chips, T), dtype=float)
    compute_mat = np.zeros((num_chips, T), dtype=float)
    for frame, chip_idx, buf, cmpu in rows:
        if 0 <= chip_idx < num_chips:
            col = frame_to_col.get(frame)
            if col is not None:
                buffer_mat[chip_idx, col] = buf
                compute_mat[chip_idx, col] = cmpu
    return buffer_mat, compute_mat, np.array(frames_sorted, dtype=int)
def _smooth_along_time(mat: np.ndarray, window: int) -> np.ndarray:
    """Apply simple moving average smoothing along the time axis (axis=1)."""
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / float(window)
    out = np.empty_like(mat)
    for i in range(mat.shape[0]):
        out[i] = np.convolve(mat[i], kernel, mode='same')
    return np.clip(out, 0.0, 1.0)
def _plot_heatmaps(buffer_mat: np.ndarray, compute_mat: np.ndarray, frames: np.ndarray, save_path: Path) -> None:
    """Plot two 4×T heatmaps in a 2×1 layout and save to `save_path`.
    Updates:
    - Make figure wider and flatter to avoid x tick overlap
    - Keep nearest-neighbor interpolation to avoid unintended vertical smoothing
    - Use a warm/cool color map: low=blue, high=red
    - Set sparse x ticks
    """
    # Wider and flatter figure
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.0, 3.0), constrained_layout=True, sharex=True)

    # Smoothing window based on total frames: ~T/200 capped at 501, min 7
    T = frames.size
    win = min(501, max(7, T // 200))
    buffer_sm = _smooth_along_time(buffer_mat, win)
    compute_sm = _smooth_along_time(compute_mat, win)

    # Common settings
    y_labels = [f'Chiplet {i}' for i in range(buffer_mat.shape[0])]
    num_ticks = 8
    xticks = np.linspace(0, T - 1, num=num_ticks, dtype=int)

    # Top: Computation (warm for high, cool for low)
    vmax_compute = compute_sm.max()
    vmin_compute = compute_sm.min()
    im1 = axes[0].imshow(compute_sm, aspect='auto', interpolation='nearest', cmap='coolwarm', vmin=vmin_compute, vmax=vmax_compute)
    axes[0].set_title('Computation Utilization Heatmap')
    # axes[0].set_ylabel('Chip')
    axes[0].set_yticks(np.arange(len(y_labels)))
    axes[0].set_yticklabels(y_labels)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(frames[xticks])
    axes[0].tick_params(axis='x', labelsize=8)
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.set_label('Utilization')

    # Bottom: Buffer
    vmax_buffer = buffer_sm.max()
    vmin_buffer = buffer_sm.min()
    im2 = axes[1].imshow(buffer_sm, aspect='auto', interpolation='nearest', cmap='coolwarm', vmin=vmin_buffer, vmax=vmax_buffer)
    axes[1].set_title('Buffer Utilization Heatmap')
    # axes[1].set_ylabel('Chip')
    axes[1].set_yticks(np.arange(len(y_labels)))
    axes[1].set_yticklabels(y_labels)
    # axes[1].set_xlabel('Frame')
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(frames[xticks])
    axes[1].tick_params(axis='x', labelsize=8)
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.set_label('Utilization')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=400)
    print(f'[SAVE] {save_path} (dpi=400)')

def main():
    evaluation_dir = Path(__file__).parent.parent
    per_chip_csv, out_dir = _run_once(evaluation_dir)
    if not per_chip_csv.exists():
        raise FileNotFoundError(f'Expected per-chip CSV not found: {per_chip_csv}')
    buffer_mat, compute_mat, frames = _read_per_chip_csv(per_chip_csv)

    # Cache results
    results: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    results[(MODEL, DATASET)] = {
        'buffer': buffer_mat,
        'compute': compute_mat,
    }
    # Plot
    save_path = Path(__file__).parent / "results" / 'buffer_compute_heatmap.png'
    _plot_heatmaps(buffer_mat, compute_mat, frames, save_path)


if __name__ == '__main__':
    main()