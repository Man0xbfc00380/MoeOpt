#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluation and metrics collection module.

This module defines the simulation-phase enum `Phase`, per-frame statistics
structure `StepStats`, and the `MetricsLogger` for recording and exporting
simulation metrics. During discrete simulation it samples hardware state at
the start of every frame, updates counters when actions start/finish, and
persists a statistics record at frame end. After simulation the data can
be written to CSV for plotting.
"""

from __future__ import annotations
import csv

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path

from action import ActionType
from hardware_config import HardwareConfig


class Phase(Enum):
    """Simulation-phase tags.

    - INIT: initialization (load/prepare)
    - REDISPATCH: token redispatch phase
    - STEADY: steady computation phase
    - FINISH: scheduling finished
    """
    INIT = 0
    REDISPATCH = 1
    STEADY = 2
    FINISH = 3


@dataclass
class StepStats:
    """Per-frame statistics, serialized to CSV.

    Contains per-frame action start/completion counts, DDR/chip utilization
    snapshots, average buffer free space, and phase tag.
    """
    frame: int
    # actions started this frame by type
    started_DDR_LOAD: int = 0
    started_DDR_SAVE: int = 0
    started_CHIP_MOVE: int = 0
    started_COMPUTE: int = 0
    started_RELEASE: int = 0
    started_ALLOCATE: int = 0
    started_FLAG: int = 0
    # actions completed this frame by type
    completed_DDR_LOAD: int = 0
    completed_DDR_SAVE: int = 0
    completed_CHIP_MOVE: int = 0
    completed_COMPUTE: int = 0
    completed_RELEASE: int = 0
    completed_ALLOCATE: int = 0
    completed_FLAG: int = 0
    # instantaneous utilization/state
    ddr_links_active: int = 0  # number of DDR links busy
    ddr_links_total: int = 4
    chips_computing: int = 0
    chips_total: int = 0
    # buffer occupancy snapshots (averaged across chips)
    avg_seq_free: float = 0.0
    avg_expert_free: float = 0.0
    # current frame's maximum buffer used (sum of token+expert used) across chips
    max_buffer_used: int = 0
    # phase marker
    phase: int = Phase.INIT.value


class MetricsLogger:
    """Frame-by-frame logger for recording and exporting metrics.

    Usage:
    - call `start_frame` at the beginning of each frame
    - call `inc_started` / `inc_completed` when actions start/finish
    - call `end_frame` at the end of each frame
    - call `write_csv` after simulation ends
    """
    def __init__(self, chips: List[str], ddrs: List[str]) -> None:
        self.chips = chips
        self.ddrs = ddrs
        self.rows: List[StepStats] = []
        self._current: Optional[StepStats] = None

        # throughput over time (number of compute completions per frame)
        self.compute_throughput: List[int] = []

        # Per-chip metrics captured per frame (for heatmap plotting)
        # Each entry corresponds to a frame and contains a list of length len(self.chips)
        self._per_chip_buffer_util_current: Optional[List[float]] = None
        self._per_chip_compute_util_current: Optional[List[float]] = None
        self.per_chip_buffer_utils_over_time: List[List[float]] = []
        self.per_chip_compute_utils_over_time: List[List[float]] = []
        # per-chip buffer used (token+expert) per frame
        self._per_chip_buffer_used_current: Optional[List[int]] = None
        self.per_chip_buffer_used_over_time: List[List[int]] = []
        # per-chip token/expert used counts per frame (for MB conversion later)
        self._per_chip_token_used_current: Optional[List[int]] = None
        self._per_chip_expert_used_current: Optional[List[int]] = None
        self.per_chip_token_used_over_time: List[List[int]] = []
        self.per_chip_expert_used_over_time: List[List[int]] = []
        # global maximum buffer used on any chip across all frames
        self._global_max_buffer_used: int = 0

        # Per-chip activity flags (receive, ddr_load, compute, send) captured per frame
        # Each is a list of length len(self.chips) with 0/1 flags
        self._per_chip_activity_current: Optional[Dict[str, List[int]]] = None
        self.per_chip_activity_over_time: List[Dict[str, List[int]]] = []

    def start_frame(self, frame: int, hardware) -> None:
        """Start frame statistics: sample DDR/chip state and buffer free space."""
        self._current = StepStats(frame=frame)
        # DDR busy count
        self._current.ddr_links_active = sum(1 for d in self.ddrs if hardware.ddr_bandwidth_used[d])
        self._current.ddr_links_total = len(self.ddrs)
        # chips computing approximated by actions in progress tracked externally
        self._current.chips_total = len(self.chips)
        # buffer free spaces snapshot
        seq_frees, expert_frees = [], []
        per_chip_buffer_util: List[float] = []
        per_chip_buffer_used: List[int] = []
        per_chip_token_used: List[int] = []
        per_chip_expert_used: List[int] = []
        for chip in self.chips:
            seq_frees.append(hardware.get_remaining_seq_space(chip))
            expert_frees.append(hardware.get_remaining_expert_space(chip))
            # Compute overall buffer utilization ratio on this chip
            seq_cap = HardwareConfig.token_buffer_size
            exp_cap = HardwareConfig.expert_buffer_size
            token_used = seq_cap - hardware.get_remaining_seq_space(chip)
            expert_used = exp_cap - hardware.get_remaining_expert_space(chip)
            used = token_used + expert_used
            total = seq_cap + exp_cap
            per_chip_buffer_util.append(used / total if total > 0 else 0.0)
            per_chip_buffer_used.append(int(used))
            per_chip_token_used.append(int(token_used))
            per_chip_expert_used.append(int(expert_used))
        if seq_frees:
            self._current.avg_seq_free = sum(seq_frees) / len(seq_frees)
        if expert_frees:
            self._current.avg_expert_free = sum(expert_frees) / len(expert_frees)
        # Store per-chip buffer utilization for current frame
        self._per_chip_buffer_util_current = per_chip_buffer_util
        # Store per-chip buffer used counts for current frame
        self._per_chip_buffer_used_current = per_chip_buffer_used
        # Store per-chip token/expert used counts for current frame
        self._per_chip_token_used_current = per_chip_token_used
        self._per_chip_expert_used_current = per_chip_expert_used
        # Update current frame's max buffer used
        if per_chip_buffer_used:
            frame_max = max(per_chip_buffer_used)
            self._current.max_buffer_used = int(frame_max)
            if frame_max > self._global_max_buffer_used:
                self._global_max_buffer_used = int(frame_max)

    def mark_phase(self, phase: Phase) -> None:
        """Tag the current frame with the corresponding simulation phase."""
        if self._current:
            self._current.phase = phase.value

    def inc_started(self, at: ActionType) -> None:
        """Increment the start counter for the given action type in the current frame."""
        if not self._current:
            return
        setattr(self._current, f"started_{at.name}", getattr(self._current, f"started_{at.name}") + 1)

    def inc_completed(self, at: ActionType) -> None:
        """Increment the completion counter for the given action type in the current frame; log throughput if COMPUTE."""
        if not self._current:
            return
        setattr(self._current, f"completed_{at.name}", getattr(self._current, f"completed_{at.name}") + 1)
        if at == ActionType.COMPUTE:
            self.compute_throughput.append(1)

    def set_per_chip_compute_util(self, values: List[float]) -> None:
        """Set per-chip compute utilization for the current frame.

        The input list length should equal the number of chips; values are ratios in [0,1].
        """
        self._per_chip_compute_util_current = list(values) if values is not None else None

    def set_per_chip_activity_flags(self, receive: List[int], ddr_load: List[int], compute: List[int], send: List[int]) -> None:
        """Set per-chip activity flags for the current frame.

        Inputs should be lists of length `len(self.chips)` with integer 0/1 values indicating
        whether the corresponding activity is active on that chip in this frame.
        """
        self._per_chip_activity_current = {
            'receive': list(receive) if receive is not None else [0] * len(self.chips),
            'ddr_load': list(ddr_load) if ddr_load is not None else [0] * len(self.chips),
            'compute': list(compute) if compute is not None else [0] * len(self.chips),
            'send': list(send) if send is not None else [0] * len(self.chips),
        }

    def end_frame(self, chips_computing: int) -> None:
        """Finish the current frame and append to the log. `chips_computing` is the number of chips currently computing."""
        if not self._current:
            return
        self._current.chips_computing = chips_computing
        self.rows.append(self._current)
        # Persist per-chip metrics for this frame
        if self._per_chip_buffer_util_current is not None:
            self.per_chip_buffer_utils_over_time.append(self._per_chip_buffer_util_current)
        else:
            # Fallback: zeros
            self.per_chip_buffer_utils_over_time.append([0.0] * len(self.chips))
        if self._per_chip_compute_util_current is not None:
            self.per_chip_compute_utils_over_time.append(self._per_chip_compute_util_current)
        else:
            self.per_chip_compute_utils_over_time.append([0.0] * len(self.chips))
        # Persist per-chip buffer used counts
        if self._per_chip_buffer_used_current is not None:
            self.per_chip_buffer_used_over_time.append(self._per_chip_buffer_used_current)
        else:
            self.per_chip_buffer_used_over_time.append([0] * len(self.chips))
        # Persist per-chip token/expert used counts
        if self._per_chip_token_used_current is not None:
            self.per_chip_token_used_over_time.append(self._per_chip_token_used_current)
        else:
            self.per_chip_token_used_over_time.append([0] * len(self.chips))
        if self._per_chip_expert_used_current is not None:
            self.per_chip_expert_used_over_time.append(self._per_chip_expert_used_current)
        else:
            self.per_chip_expert_used_over_time.append([0] * len(self.chips))
        # Persist per-chip activity flags
        if self._per_chip_activity_current is not None:
            self.per_chip_activity_over_time.append(self._per_chip_activity_current)
        else:
            self.per_chip_activity_over_time.append({
                'receive': [0] * len(self.chips),
                'ddr_load': [0] * len(self.chips),
                'compute': [0] * len(self.chips),
                'send': [0] * len(self.chips),
            })
        # Reset current frame holders
        self._per_chip_buffer_util_current = None
        self._per_chip_compute_util_current = None
        self._per_chip_buffer_used_current = None
        self._per_chip_token_used_current = None
        self._per_chip_expert_used_current = None
        self._per_chip_activity_current = None
        self._current = None

    def write_csv(self, out_path: Path) -> None:
        """Write accumulated per-frame statistics to CSV, creating directories if necessary.

        In addition to the aggregate `trace.csv`, also write `trace_per_chip.csv` containing
        per-frame, per-chip buffer and compute utilization for heatmap plotting.
        Also include absolute buffer used counts.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 1) Aggregate per-frame CSV
        with out_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(StepStats(0)).keys()))
            writer.writeheader()
            for r in self.rows:
                writer.writerow(asdict(r))

        # 2) Per-chip CSV for heatmaps
        per_chip_path = out_path.parent / 'trace_per_chip.csv'
        with per_chip_path.open('w', newline='', encoding='utf-8') as f:
            fieldnames = ['frame', 'chip_index', 'chip_id', 'buffer_util', 'compute_util', 'buffer_used', 'token_used', 'expert_used', 'receive', 'ddr_load', 'compute_active', 'send']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            total_frames = len(self.rows)
            for fi in range(total_frames):
                frame_id = self.rows[fi].frame
                buf_list = self.per_chip_buffer_utils_over_time[fi] if fi < len(self.per_chip_buffer_utils_over_time) else [0.0] * len(self.chips)
                cmp_list = self.per_chip_compute_utils_over_time[fi] if fi < len(self.per_chip_compute_utils_over_time) else [0.0] * len(self.chips)
                used_list = self.per_chip_buffer_used_over_time[fi] if fi < len(self.per_chip_buffer_used_over_time) else [0] * len(self.chips)
                token_list = self.per_chip_token_used_over_time[fi] if fi < len(self.per_chip_token_used_over_time) else [0] * len(self.chips)
                expert_list = self.per_chip_expert_used_over_time[fi] if fi < len(self.per_chip_expert_used_over_time) else [0] * len(self.chips)
                act_map = self.per_chip_activity_over_time[fi] if fi < len(self.per_chip_activity_over_time) else {
                    'receive': [0] * len(self.chips),
                    'ddr_load': [0] * len(self.chips),
                    'compute': [0] * len(self.chips),
                    'send': [0] * len(self.chips),
                }
                for ci, chip in enumerate(self.chips):
                    writer.writerow({
                        'frame': frame_id,
                        'chip_index': ci,
                        'chip_id': chip,
                        'buffer_util': float(buf_list[ci] if ci < len(buf_list) else 0.0),
                        'compute_util': float(cmp_list[ci] if ci < len(cmp_list) else 0.0),
                        'buffer_used': int(used_list[ci] if ci < len(used_list) else 0),
                        'token_used': int(token_list[ci] if ci < len(token_list) else 0),
                        'expert_used': int(expert_list[ci] if ci < len(expert_list) else 0),
                        'receive': int(act_map['receive'][ci] if ci < len(act_map.get('receive', [])) else 0),
                        'ddr_load': int(act_map['ddr_load'][ci] if ci < len(act_map.get('ddr_load', [])) else 0),
                        'compute_active': int(act_map['compute'][ci] if ci < len(act_map.get('compute', [])) else 0),
                        'send': int(act_map['send'][ci] if ci < len(act_map.get('send', [])) else 0),
                    })

    def average_chip_utilization_for_phase(self, phase: Phase) -> float:
        """Return the overall average chip compute utilization for frames in the given phase.

        Utilization per frame is `chips_computing / chips_total` (0 if `chips_total` is 0).
        The returned value is the arithmetic mean of per-frame utilizations across all
        frames whose `phase` equals the given phase.
        """
        ratios: List[float] = []
        phase_val = phase.value
        for r in self.rows:
            if r.phase != phase_val:
                continue
            if r.chips_total > 0:
                ratios.append(r.chips_computing / r.chips_total)
            else:
                ratios.append(0.0)
        if not ratios:
            return 0.0
        return sum(ratios) / len(ratios)

    def finish_frame(self) -> Optional[int]:
        """Return the frame index where FINISH phase is recorded (first occurrence)."""
        fin = Phase.FINISH.value
        for r in self.rows:
            if r.phase == fin:
                return r.frame
        return None

    def get_global_max_buffer_used(self) -> int:
        """Return the global maximum buffer used (sum of token+expert) on any chip across all frames."""
        return int(self._global_max_buffer_used)
