#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick preview script to validate plotting changes for max buffer bars.

It builds a minimal synthetic `results` dict (one group) and calls
`plot_all` to generate a figure without running heavy simulations.

Output: evaluation/statistics/results/max_buffer_bar_8groups_test.png
"""

from pathlib import Path
from typing import Dict, List
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from evaluation.statistics.plot_max_buffer_bars import plot_all


def main():
    # One group only for preview (with stacked token/expert parts)
    group_name = "DeepSeek-MoE-c4"
    # totals for plotting reference
    values: List[float] = [
        8.0, 10.5, 12.0, 11.2,   # 16
        20.0, 22.5, 24.0, 23.2,  # 64
        40.0, 42.0, 48.0, 46.0,  # 256
        64.0, 68.0, 72.0, 70.0,  # 1024
    ]
    # Create details matching ORDERED_CONFIGS in plot file: 16,64,256,1024 × EP/Hydra/FSE-DP no/paired
    # Split totals roughly 30% token, 70% expert for visibility
    def make_details(vals: List[float]) -> List[Dict]:
        configs = [
            "EP_16_2_mul_2.json", "Hydra_16_2_mul_2.json", "FSE-DP_16_2_mul_2.json", "FSE-DP-paired_16_2_mul_2.json",
            "EP_64_2_mul_2.json", "Hydra_64_2_mul_2.json", "FSE-DP_64_2_mul_2.json", "FSE-DP-paired_64_2_mul_2.json",
            "EP_256_2_mul_2.json", "Hydra_256_2_mul_2.json", "FSE-DP_256_2_mul_2.json", "FSE-DP-paired_256_2_mul_2.json",
            "EP_1024_2_mul_2.json", "Hydra_1024_2_mul_2.json", "FSE-DP_1024_2_mul_2.json", "FSE-DP-paired_1024_2_mul_2.json",
        ]
        out: List[Dict] = []
        for i, tot in enumerate(vals):
            tok = tot * 0.3
            exp = tot * 0.7
            out.append({"config": configs[i], "token_mb": tok, "expert_mb": exp, "total_mb": tot})
        return out

    results = {
        group_name: {
            "values": values,
            "details": make_details(values)
        }
    }

    save_path = Path(__file__).parent / "results" / "max_buffer_bar_8groups_test.png"
    plot_all(results, save_path=save_path, group_gap=2)


if __name__ == "__main__":
    main()