#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Discrete simulation entry point and runner.

Responsible for:
- Parsing and applying configuration to hardware and scheduler
- Driving simulation frame-by-frame, logging metrics and outputting CSVs and plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
# Set the directory of the current script as the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure Windows console can print UTF-8 (Chinese) without UnicodeEncodeError
# try:
#     # Python 3.7+: reconfigure is available
#     import sys as _sys
#     _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
#     _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
# except Exception:
#     try:
#         import io as _io
#         _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
#         _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding='utf-8', errors='replace')
#     except Exception:
#         pass

from data_block import DataBlock, DataBlockType
from scheduler import FSEDPScheduler, EPScheduler  # noqa: E402
from action import ActionType, Action  # noqa: E402
from hardware_config import HardwareConfig  # noqa: E402

from evaluation.metrics import MetricsLogger, Phase  # noqa: E402
from evaluation.plots import plot_throughput_over_time, plot_utilizations, plot_buffer_free  # noqa: E402
from evaluation.model_shape_map import apply_model_overrides  # noqa: E402


class DiscreteSimulator:
    """Lightweight discrete-event simulator advancing frame-by-frame and completing actions according to `HardwareConfig.action_durations`.

    Conventions:
    - `step()`: requests schedulable actions each frame
    - Maintains remaining duration for in-flight actions, decrementing each frame
    - Calls `scheduler.handle_action_complete(action)` upon action completion
    - Produces per-frame metrics for evaluation and plotting
    """

    def __init__(self, base_strategy: str = "FSEDP") -> None:
        self.scheduler = None
        if base_strategy == "EP":
            self.scheduler = EPScheduler()
        else:
            self.scheduler = FSEDPScheduler()
        
        self.inflight: List[Tuple[Action, int]] = []  # (action, remaining)
        self.frame = 0
        self.metrics = MetricsLogger(self.scheduler.hardware_emulator.chips, self.scheduler.hardware_emulator.ddrs)
        self.draw_image = False
        # phase
        self.phase = Phase.INIT

    def init_blocks(self):
        """Initialize task/data blocks for later scheduling."""
        self.scheduler.init_data_blocks()

    def _duration(self, action_type: ActionType) -> int:
        """Return duration (frames) for an action type. FLAG actions are instantaneous (0)."""
        # FLAG is instantaneous
        if action_type.name == 'FLAG':
            return 0
        # Fallback to 1 frame if not specified
        return int(HardwareConfig.action_durations.get(action_type, 1))

    def run(self, frames: int, out_dir: Path, fig_dir: Optional[Path] = None):
        """Main simulation loop.

        Executes the following steps each frame:
        1) Sample hardware state and record phase
        2) Query scheduler and start actions; instantaneous actions finish in the same frame
        3) Advance remaining time of in-flight actions and record completions
        4) Estimate compute utilization and write per-frame statistics
        After simulation, output CSV and plots.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / 'trace.csv'
        # advance frames
        finished = False
        for f in range(frames):
            self.frame = f
            hw = self.scheduler.hardware_emulator
            self.metrics.start_frame(f, hw)
            self.metrics.mark_phase(self.phase)

            # Query scheduler: set of actions that can start this frame
            new_actions = self.scheduler.step()
            # start actions; complete immediately if duration == 0
            immediate_complete: List[Action] = []
            for act in new_actions:
                dur = self._duration(act.action_type)
                # Compute duration based on action type; COMPUTE actions scale with length of other_blocks
                if act.action_type == ActionType.COMPUTE:
                    dur = max(
                        dur * len(act.other_blocks), 1)
                
                self.metrics.inc_started(act.action_type)
                if dur <= 0:
                    # Instantaneous actions (e.g. FLAG) complete in this frame
                    immediate_complete.append(act)
                else:
                    self.inflight.append((act, dur))

            # Advance in-flight actions: remaining time minus 1, finish when reaching 0
            completed: List[Action] = []
            new_inflight: List[Tuple[Action, int]] = []
            for act, t in self.inflight:
                t2 = t - 1
                if t2 <= 0:
                    completed.append(act)
                else:
                    new_inflight.append((act, t2))
            self.inflight = new_inflight

            # Handle instantaneous completions first (e.g. FLAG)
            chips_computing = 0
            for act in immediate_complete:
                self.scheduler.handle_action_complete(act)
                self.metrics.inc_completed(act.action_type)
                if act.action_type == ActionType.FLAG and getattr(act.data_block, 'name', '') == 'token-redispatch':
                    self.phase = Phase.STEADY
                    self.metrics.mark_phase(self.phase)
                if act.action_type == ActionType.FLAG and getattr(act.data_block, 'name', '') == 'finish':
                    self.phase = Phase.FINISH
                    self.metrics.mark_phase(self.phase)
                    finished = True

            # Then handle completions of in-flight actions
            chips_computing = 0
            for act in completed:
                self.scheduler.handle_action_complete(act)
                self.metrics.inc_completed(act.action_type)
                if act.action_type == ActionType.COMPUTE:
                    chips_computing += 1
                if act.action_type == ActionType.FLAG and getattr(act.data_block, 'name', '') == 'token-redispatch':
                    self.phase = Phase.STEADY
                    self.metrics.mark_phase(self.phase)
                if act.action_type == ActionType.FLAG and getattr(act.data_block, 'name', '') == 'finish':
                    self.phase = Phase.FINISH
                    self.metrics.mark_phase(self.phase)
                    finished = True

            # Compute utilization: in-flight COMPUTE count + estimated completions this frame
            active_compute = sum(1 for act, _ in self.inflight if act.action_type == ActionType.COMPUTE)
            chips_computing += active_compute
            # Per-chip compute utilization from active in-flight COMPUTE actions
            per_chip_counts = {chip: 0 for chip in self.scheduler.hardware_emulator.chips}
            for act, _ in self.inflight:
                if act.action_type == ActionType.COMPUTE:
                    chip_id = None
                    try:
                        chip_id = act.destination[0] if act.destination else None
                    except Exception:
                        chip_id = None
                    if not chip_id and act.data_block and act.data_block.location:
                        chip_id = act.data_block.location[0]
                    if chip_id in per_chip_counts:
                        per_chip_counts[chip_id] += 1
            max_concurrent = max(1, getattr(self.scheduler, 'same_compute_number', 1))
            per_chip_util = [min(1.0, per_chip_counts[c] / max_concurrent) for c in self.scheduler.hardware_emulator.chips]
            self.metrics.set_per_chip_compute_util(per_chip_util)

            # Per-chip activity flags for timeline (receive, ddr_load, compute, send)
            chips = list(self.scheduler.hardware_emulator.chips)
            idx_map = {chip: i for i, chip in enumerate(chips)}
            recv_flags = [0] * len(chips)
            ddr_flags = [0] * len(chips)
            cmp_flags = [0] * len(chips)
            send_flags = [0] * len(chips)
            # compute flags from per_chip_counts
            for chip, cnt in per_chip_counts.items():
                if chip in idx_map and cnt > 0:
                    cmp_flags[idx_map[chip]] = 1
            # scan inflight actions for DDR_LOAD and CHIP_MOVE
            for act, _ in self.inflight:
                try:
                    if act.action_type == ActionType.DDR_LOAD:
                        dest_chip = act.destination[0] if act.destination else None
                        if dest_chip in idx_map:
                            ddr_flags[idx_map[dest_chip]] = 1
                    elif act.action_type == ActionType.CHIP_MOVE:
                        src_chip = act.source[0] if act.source else None
                        dst_chip = act.destination[0] if act.destination else None
                        if src_chip in idx_map:
                            send_flags[idx_map[src_chip]] = 1
                        if dst_chip in idx_map:
                            recv_flags[idx_map[dst_chip]] = 1
                except Exception:
                    pass
            self.metrics.set_per_chip_activity_flags(recv_flags, ddr_flags, cmp_flags, send_flags)

            self.metrics.end_frame(chips_computing)

            if finished:
                break

        # Compute and output average utilization for STEADY phase
        steady_avg_util = self.metrics.average_chip_utilization_for_phase(Phase.STEADY)
        print(f"Overall average chip utilization in STEADY phase: {steady_avg_util:.4f}")
        # Output global maximum buffer used across chips over all frames
        try:
            max_buf_used = self.metrics.get_global_max_buffer_used()
            print(f"Overall maximum buffer used: {max_buf_used}")
        except Exception:
            pass
        finish_frame = self.metrics.finish_frame()
        if finish_frame is not None:
            print(f"FINISH occurred at frame: {finish_frame}")
        else:
            print("FINISH phase not recorded")

        self.metrics.write_csv(csv_path)
        # Write CSV and generate plots
        if self.draw_image:
            # Generate plots
            target_fig_dir = fig_dir if fig_dir is not None else out_dir
            target_fig_dir.mkdir(parents=True, exist_ok=True)
            plot_throughput_over_time(csv_path, target_fig_dir / 'throughput.png')
            plot_utilizations(csv_path, target_fig_dir / 'utilization.png')
            plot_buffer_free(csv_path, target_fig_dir / 'buffer_free.png')
        assert finish_frame > 0


# Removed legacy HuggingFace mapping and dataset loading helpers (no longer used).

def _model_to_trace_folder(model_name: str) -> str:
    """Map logical model to the corresponding trace folder name."""
    mapping = {
        'DeepSeek-MoE': 'deepseek_outputs',
        'Qwen3-MoE': 'qwen_outputs',
        'Phi3.5-MoE': 'phi_outputs',
        'Yuan2.0': 'yuan_outputs',
    }
    folder = mapping.get(model_name)
    if not folder:
        raise Exception(f"Unknown model '{model_name}'. Supported: {list(mapping.keys())}")
    return folder


def _dataset_to_folder(dataset_name: str) -> str:
    """Normalize dataset name to the folder prefix used by trace outputs."""
    name = (dataset_name or '').lower()
    if name in ('wikitext', 'wikitext-2', 'wikitext2', 'wikitext_2'):
        return 'wikitext'
    if name in ('c4','C4'):
        return 'c4'
    if name in ('winogrande',):
        return 'winogrande'
    # Fallback: use lowercased name directly
    return name


def _build_bind_relations_from_trace(sim: DiscreteSimulator, model_name: str, dataset_name: str, input_length: int, layer_index: int) -> bool:
    """Read expert-token activation trace JSON and configure TaskEmulator accordingly.

    This function:
    - Locates JSON under evaluation/network_test/<model_folder>/<dataset>_<input_length>/layer_<layer>_token_experts.json
    - Builds expert->token block bindings across all requests using 1-based expert IDs
    - Initializes TaskEmulator's tokens, experts, and bindings, then distributes to hardware
    """
    base_dir = Path(__file__).parent / 'network_test'
    model_dir = base_dir / _model_to_trace_folder(model_name)
    dataset_folder = _dataset_to_folder(dataset_name)
    trace_dir = model_dir / f"{dataset_folder}_{int(input_length)}"
    json_path = trace_dir / f"layer_{int(layer_index)}_token_experts.json"

    if not trace_dir.exists():
        raise Exception(f"Trace directory not found: {trace_dir}")
    if not json_path.exists():
        raise Exception(f"Trace JSON not found: {json_path}")

    # Load JSON: { token_id: [expert_ids...] }
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    te = sim.scheduler.task_emulator
    te.token_num_per_request = int(input_length)

    # Prepare blocks and relations
    te.token_blocks = {}
    te.expert_blocks = {}
    te.bind_relations = {}
    te.sorted_expert_blocks = []
    te.current_alive_token = []

    # Initialize token & expert blocks
    te._init_token_blocks()
    te._init_expert_blocks()

    # Build bindings: replicate the per-sequence activations across all requests
    # JSON keys are token indices [0..input_length-1]
    request_num = te.request_num
    for r in range(1, request_num + 1):
        for t_zero in range(0, int(input_length)):
            token_key = str(t_zero)
            if token_key not in data:
                # If sparse, skip missing tokens
                continue
            expert_list = data[token_key]
            if not isinstance(expert_list, list):
                expert_list = [expert_list]
            block_name = f"R{r}-T{t_zero + 1}"
            for e_zero in expert_list:
                e_id = int(e_zero) + 1  # convert to 1-based
                if e_id not in te.bind_relations:
                    te.bind_relations[e_id] = set()
                te.bind_relations[e_id].add(block_name)

    # Resort expert blocks per configured policy
    te.resort_expert_blocks()

    # Distribute to hardware and register with scheduler
    new_blocks = sim.scheduler.init_strategy.distribute()
    for block in new_blocks:
        sim.scheduler._add_new_block(block)
        if block.block_type == DataBlockType.EXPERT:
            sim.scheduler.expert_finish[(block.seq_or_expert_id, block.mico_slice)] = False
    return True

def apply_hardware_overrides(cfg: Dict):
    """Override hardware parameters and action durations from configuration."""
    # Set HardwareConfig from cfg
    hw = cfg.get('hardware', {})
    if 'row_number_of_chip' in hw:
        HardwareConfig.row_number_of_chip = int(hw['row_number_of_chip'])
    if 'column_number_of_chip' in hw:
        HardwareConfig.column_number_of_chip = int(hw['column_number_of_chip'])
    # New buffer size keys; maintain backward compatibility with legacy 'chip_buffer_size'
    if 'token_buffer_size' in hw:
        HardwareConfig.token_buffer_size = int(hw['token_buffer_size'])
    if 'expert_buffer_size' in hw:
        HardwareConfig.expert_buffer_size = int(hw['expert_buffer_size'])
    if 'chip_buffer_size' in hw:
        # Fallback: if only legacy is provided, apply to both new sizes
        val = int(hw['chip_buffer_size'])
        HardwareConfig.token_buffer_size = val
        HardwareConfig.expert_buffer_size = val
    if 'ddr_buffer_size' in hw:
        HardwareConfig.ddr_buffer_size = int(hw['ddr_buffer_size'])
    # action durations
    action_map = hw.get('action_durations', {})
    for k, v in action_map.items():
        at = getattr(ActionType, k)
        HardwareConfig.action_durations[at] = int(v)


def apply_scheduler_overrides(sim: DiscreteSimulator, cfg: Dict):
    """Override scheduler policies and task parameters from configuration."""
    s_cfg = cfg.get('scheduler', {})
    if not s_cfg:
        assert False
    # Path mode
    try:
        from strategy.path_generator import PathMode
        pm = s_cfg.get('path_mode')
        if pm:
            sim.scheduler.path_generator.path_mode = getattr(PathMode, pm)
    except Exception:
        assert False

    # Init distribution
    try:
        from strategy.init_distribution_strategy import DistributionMode
        td = s_cfg.get('token_distribution_mode')
        ed = s_cfg.get('expert_distribution_mode')
        if td:
            sim.scheduler.init_strategy.token_distribution_mode = getattr(DistributionMode, td)
        if ed:
            sim.scheduler.init_strategy.expert_distribution_mode = getattr(DistributionMode, ed)
    except Exception:
        assert False

    # DDR load strategy
    try:
        from strategy.ddr_load_strategy import DDRLoadMode, DestinationSelectMode
        dlm = s_cfg.get('ddr_load_mode')
        dsm = s_cfg.get('destination_select_mode')
        if dlm:
            sim.scheduler.ddr_load_strategy.ddr_load_mode = getattr(DDRLoadMode, dlm)
        if dsm:
            sim.scheduler.ddr_load_strategy.destination_select_mode = getattr(DestinationSelectMode, dsm)
    except Exception:
        assert False

    # Token redispatch mode
    try:
        from strategy.token_redispatcher import RedispatchMode
        rm = s_cfg.get('redispatch_mode')
        if rm:
            sim.scheduler.token_dispatcher.redispatch_mode = getattr(RedispatchMode, rm)
    except Exception:
        assert False

    # paied-load experts
    try:
        paired = s_cfg.get('paired', False)
        sim.scheduler.task_emulator.enable_expert_load_sort = paired
    except Exception:
        assert False

    # Set task_emulator parameters via config
    if 'expert_number' in s_cfg:
        sim.scheduler.task_emulator.expert_number = int(s_cfg['expert_number'])
    if 'micro_slice_num_per_expert' in s_cfg:
        sim.scheduler.task_emulator.micro_slice_num_per_expert = int(s_cfg['micro_slice_num_per_expert'])
    if 'token_number' in s_cfg:
        sim.scheduler.task_emulator.request_num = 1
        sim.scheduler.task_emulator.token_num_per_request = int(s_cfg['token_number'])
    if 'min_activated_token' in s_cfg:
        sim.scheduler.task_emulator.min_activated_token = int(s_cfg['min_activated_token'])
    if 'token_weight_threshould' in s_cfg:
        sim.scheduler.task_emulator.token_weight_threshould = int(s_cfg['token_weight_threshould'])

    # if sim.scheduler.task_emulator.micro_slice_num_per_expert > 1 and sim.scheduler.task_emulator.micro_slice_num_per_expert !=4:
    #     if cfg['model'] == 'DeepSeek-MoE' or cfg['model'] == 'Qwen3-MoE':
    #         sim.scheduler.task_emulator.micro_slice_num_per_expert = 16
    #     elif cfg['model'] == 'Phi3.5-MoE' or cfg['model'] == 'Yuan2.0':
    #         sim.scheduler.task_emulator.micro_slice_num_per_expert = 64

def _build_cli_parser() -> argparse.ArgumentParser:
    """Create and return the CLI parser for the runner."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent / 'configs/FSE-DP_64_2_mul_2.json'))
    parser.add_argument('--model', type=str, default='DeepSeek-MoE',
                        help='Model name used to locate trace folder: DeepSeek-MoE, Qwen3-MoE, Phi3.5-MoE, Yuan2.0')
    parser.add_argument('--dataset', type=str, default='wikitext',
                        help='Dataset name used in trace folder prefix: wikitext, c4, winogrande')
    parser.add_argument('--layer', type=int, default=16,
                        help='Layer index to load from trace JSON (e.g., 3)')
    parser.add_argument('--input_length', type=int,
                        help='Input token length used when generating the trace (e.g., 128)')
    parser.add_argument('--write_csv', action='store_true',
                        help='Write per-frame trace CSV and plots to output/figure dir')
    parser.add_argument('--override_token_buffer_size', type=int, default=None,
                        help='Override HardwareConfig.token_buffer_size for this run (integer)')
    parser.add_argument('--override_expert_buffer_size', type=int, default=None,
                        help='Override HardwareConfig.expert_buffer_size for this run (integer)')
    parser.add_argument('--override_expert_bandwidth', type=float, default=None,
                        help='Override DDR bandwidth (GB/s); applied in model_shape_map to derive DDR_LOAD/DDR_SAVE durations directly')
    parser.add_argument('--override_chip_move_bandwidth', type=float, default=None,
                        help='Override inter-chip bandwidth (GB/s) affecting CHIP_MOVE durations; applied via model_shape_map during duration derivation')
    parser.add_argument('--trace_json', type=str, default=None,
                        help='Override path to expert-token binding JSON (token_index -> [expert_ids...]). If provided, runner uses this file instead of auto-resolved network_test path.')
    return parser


def _merge_cli_into_cfg(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Merge CLI args into loaded config and apply basic overrides."""
    if args.model is not None:
        cfg['model'] = args.model
    if args.dataset is not None:
        cfg['dataset'] = args.dataset
    if args.layer is not None:
        cfg['layer'] = int(args.layer)
    if args.input_length is not None:
        cfg['input_length'] = int(args.input_length)
    # Apply CLI overrides for buffer sizes if provided
    if args.override_token_buffer_size is not None or args.override_expert_buffer_size is not None:
        hw = cfg.get('hardware', {})
        if args.override_token_buffer_size is not None:
            hw['token_buffer_size'] = int(args.override_token_buffer_size)
        if args.override_expert_buffer_size is not None:
            hw['expert_buffer_size'] = int(args.override_expert_buffer_size)
        cfg['hardware'] = hw
    # Apply CLI override for DDR expert bandwidth (GB/s)
    if getattr(args, 'override_expert_bandwidth', None) is not None:
        hw = cfg.get('hardware', {})
        hw['override_expert_bandwidth'] = float(args.override_expert_bandwidth)
        cfg['hardware'] = hw
    # Apply CLI override for inter-chip move bandwidth (GB/s)
    if getattr(args, 'override_chip_move_bandwidth', None) is not None:
        hw = cfg.get('hardware', {})
        hw['override_chip_move_bandwidth'] = float(args.override_chip_move_bandwidth)
        cfg['hardware'] = hw
    return cfg


def _resolve_output_dirs(cfg: Dict, args: argparse.Namespace) -> Tuple[Path, Path]:
    """Resolve output and figure directories based on config and CLI args."""
    base_out_dir = Path(cfg.get('output_dir', Path(__file__).parent / 'results'))
    if 'output_dir' not in cfg and (cfg.get('model') or cfg.get('dataset') or cfg.get('layer') is not None):
        subdir = '_'.join([x for x in [cfg.get('model'), cfg.get('dataset'),
                                       (f"L{cfg.get('layer')}" if cfg.get('layer') is not None else None)] if x])
        out_dir = base_out_dir / subdir if subdir else base_out_dir
    else:
        out_dir = base_out_dir
    fig_dir = Path(cfg.get('figure_dir', str(out_dir)))
    return out_dir, fig_dir


def main():
    """Parse CLI, load config, initialize simulator, and run."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    cfg = _merge_cli_into_cfg(cfg, args)

    # Apply hardware overrides and seed
    apply_hardware_overrides(cfg)
    # DDR bandwidth override is applied in model_shape_map; no scaling here
    import random
    random.seed(cfg.get('seed', 2025))

    # Resolve output dirs and frames
    out_dir, fig_dir = _resolve_output_dirs(cfg, args)
    frames = int(cfg.get('frames', 200))

    # Build simulator and apply scheduler overrides
    sim = DiscreteSimulator(base_strategy = cfg.get('base_strategy', "FSEDP"))
    apply_scheduler_overrides(sim, cfg)
    # Further adjust expert count and key action durations based on model shape and hardware parameters
    apply_model_overrides(sim, cfg, cfg.get('base_strategy', "FSEDP") == "EP")

    # DDR bandwidth override already considered in model_shape_map; no re-scaling

    if args.write_csv or bool(cfg.get('write_csv', False)) or bool(cfg.get('draw_image', False)):
        sim.draw_image = True

    # Trace-based initialization
    used_custom_init = False
    model_name = cfg.get('model')
    dataset_name = cfg.get('dataset')
    layer_index = cfg.get('layer')
    input_length = cfg.get('input_length')
    if input_length is None:
        # Default to the task emulator's token count if not provided
        try:
            input_length = int(sim.scheduler.task_emulator.token_num_per_request)
        except Exception:
            input_length = None
    print(f"model_name: {model_name}, dataset_name: {dataset_name}, layer_index: {layer_index}, input_length: {input_length}")
    # If an override trace JSON is provided, use it directly
    if getattr(args, 'trace_json', None):
        try:
            override_path = Path(getattr(args, 'trace_json'))
            if not override_path.exists():
                raise Exception(f"Override trace JSON not found: {override_path}")
            with override_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            te = sim.scheduler.task_emulator
            if input_length is None:
                try:
                    input_length = int(sim.scheduler.task_emulator.token_num_per_request)
                except Exception:
                    try:
                        input_length = max(0, max(int(k) for k in data.keys()) + 1)
                    except Exception:
                        input_length = 0
            te.token_num_per_request = int(input_length)
            te.token_blocks = {}
            te.expert_blocks = {}
            te.bind_relations = {}
            te.sorted_expert_blocks = []
            te.current_alive_token = []
            te._init_token_blocks()
            te._init_expert_blocks()
            request_num = te.request_num
            for r in range(1, request_num + 1):
                for t_zero in range(0, int(input_length)):
                    token_key = str(t_zero)
                    if token_key not in data:
                        continue
                    expert_list = data[token_key]
                    if not isinstance(expert_list, list):
                        expert_list = [expert_list]
                    block_name = f"R{r}-T{t_zero + 1}"
                    for e_zero in expert_list:
                        e_id = int(e_zero) + 1
                        if e_id not in te.bind_relations:
                            te.bind_relations[e_id] = set()
                        te.bind_relations[e_id].add(block_name)
            te.resort_expert_blocks()
            new_blocks = sim.scheduler.init_strategy.distribute()
            for block in new_blocks:
                sim.scheduler._add_new_block(block)
                if block.block_type == DataBlockType.EXPERT:
                    sim.scheduler.expert_finish[(block.seq_or_expert_id, block.mico_slice)] = False
            used_custom_init = True
        except Exception as e:
            raise Exception(f"Failed to initialize from override trace JSON: {e}")
    elif model_name and dataset_name and layer_index is not None and input_length is not None:
        used_custom_init = _build_bind_relations_from_trace(sim, model_name, dataset_name,
                                                            int(input_length), int(layer_index))
    elif model_name or dataset_name or layer_index is not None:
        raise Exception("Trace-based init requires specifying model, dataset, layer, and input_length.")

    assert used_custom_init

    # Fallback to default init if not using trace-based init
    if not used_custom_init:
        sim.init_blocks()
    sim.run(frames=frames, out_dir=out_dir, fig_dir=fig_dir)

if __name__ == '__main__':
    main()
