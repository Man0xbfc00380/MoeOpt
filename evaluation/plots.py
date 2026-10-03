#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from pathlib import Path
import matplotlib.pyplot as plt


def _read_csv(csv_path: Path):
    rows = []
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (int(v) if v.isdigit() else float(v) if v.replace('.', '', 1).isdigit() else v) for k, v in r.items()})
    return rows


def plot_throughput_over_time(csv_path: Path, out_path: Path):
    rows = _read_csv(csv_path)
    x = [int(r['frame']) for r in rows]
    y = [int(r['completed_COMPUTE']) for r in rows]
    plt.figure(figsize=(6, 3))
    # simple moving average over window 5
    ma = []
    w = 5
    for i in range(len(y)):
        s = max(0, i - w + 1)
        ma.append(sum(y[s:i+1]) / (i - s + 1))
    plt.plot(x, ma, label='compute/frame (5-frame MA)')
    plt.xlabel('frame')
    plt.ylabel('throughput (compute/frame)')
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_utilizations(csv_path: Path, out_path: Path):
    rows = _read_csv(csv_path)
    x = [int(r['frame']) for r in rows]
    plt.figure(figsize=(6, 3))
    ddr_util = [float(r['ddr_links_active'])/float(r['ddr_links_total']) if float(r['ddr_links_total']) else 0 for r in rows]
    cmp_util = [float(r['chips_computing'])/float(r['chips_total']) if float(r['chips_total']) else 0 for r in rows]
    # 计算平滑曲线
    w = 512
    cmp_util_smooth = []
    for i in range(len(cmp_util)):
        s = max(0, i - w + 1)
        cmp_util_smooth.append(sum(cmp_util[s:i+1]) / (i - s + 1))
    # plt.plot(x, ddr_util, label='DDR util')
    plt.plot(x, cmp_util, label='Compute util')
    plt.plot(x, cmp_util_smooth, label='Compute util (smooth)')
    plt.xlabel('frame')
    plt.ylabel('utilization')
    plt.ylim(0, 1)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_buffer_free(csv_path: Path, out_path: Path):
    rows = _read_csv(csv_path)
    x = [int(r['frame']) for r in rows]
    plt.figure(figsize=(6, 3))
    plt.plot(x, [float(r['avg_seq_free']) for r in rows], label='avg seq free')
    plt.plot(x, [float(r['avg_expert_free']) for r in rows], label='avg expert free')
    plt.xlabel('frame')
    plt.ylabel('slots')
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
