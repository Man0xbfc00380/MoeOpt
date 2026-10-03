#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick preview for the half plot (expert top in MB, token bottom in KB).

Builds a minimal synthetic `results` dict (one group) and calls
`plot_all` from plot_max_buffer_bars_half to generate a figure.

Output: evaluation/statistics/results/max_buffer_bar_half_test.png
"""

from pathlib import Path
from typing import Dict, List
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from evaluation.statistics.plot_max_buffer_bars_half import plot_all


def main():
    # One group only for preview
    group_name = "DeepSeek-MoE-c4"
    # totals (MB) for reference; details provide token/expert split
    values: List[float] = [
        8.0, 11.5,    # 16: Hydra, FSE-DP paired
        22.0, 24.5,   # 64
        42.0, 46.0,   # 256
        68.0, 72.0,   # 1024
    ]

    # ORDERED_CONFIGS in half file: Hydra / FSE-DP paired × 16/64/256/1024
    configs = [
        "Hydra_16_2_mul_2.json", "FSE-DP-paired_16_2_mul_2.json",
        "Hydra_64_2_mul_2.json", "FSE-DP-paired_64_2_mul_2.json",
        "Hydra_256_2_mul_2.json", "FSE-DP-paired_256_2_mul_2.json",
        "Hydra_1024_2_mul_2.json", "FSE-DP-paired_1024_2_mul_2.json",
    ]

    def make_details(vals: List[float]) -> List[Dict]:
        out: List[Dict] = []
        for i, tot in enumerate(vals):
            tok_mb = tot * 0.25
            exp_mb = tot * 0.75
            out.append({"config": configs[i], "token_mb": tok_mb, "expert_mb": exp_mb, "total_mb": tot})
        return out

    results = {
        group_name: {
            "values": values,
            "details": make_details(values),
        }
    }

    save_path = Path(__file__).parent / "results" / "max_buffer_bar_half_test.png"
    plot_all(results, save_path=save_path, group_gap=2)


if __name__ == "__main__":
    main()