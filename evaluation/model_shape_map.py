import math
from typing import Dict, Optional

MODEL_SHAPE_MAP = {
    "DeepSeek-MoE": {
        "D_MODEL": 2048,          # model hidden dimension
        "D_KEY": 2048,            # model hidden dimension
        "D_VALUE": 2048,          # model hidden dimension
        "EXPERT_FFN_DIM": 1408,   # FFN dimension of a single expert
        "NUM_EXPERTS": 64,        # total number of experts
        "ACTIVATED_EXPERTS": 6,   # 6 experts activated per token
        "SHARED_EXPERTS": 2,      # 2 shared experts (always activated in addition to the 6)
        "HEAD": 16,
        "LAYER": 28
    },
    "Qwen3-MoE": {
        "D_MODEL": 2048,          # model hidden dimension
        "D_KEY": 2048,            # key vector dimension
        "D_VALUE": 2048,          # value vector dimension
        "EXPERT_FFN_DIM": 768,    # FFN dimension of a single expert
        "NUM_EXPERTS": 128,       # total number of experts
        "ACTIVATED_EXPERTS": 8,   # 8 experts activated per token
        "HEAD": 32,               # number of attention heads
        "KV_HEAD": 4,             # number of key-value heads (grouped-query attention)
        "LAYER": 48
    },
    "Phi3.5-MoE": {
        "D_MODEL": 4096,          # model hidden dimension
        "D_KEY": 4096,            # key vector dimension
        "D_VALUE": 4096,          # value vector dimension
        "EXPERT_FFN_DIM": 3200,   # FFN dimension of a single expert
        "NUM_EXPERTS": 16,        # total number of experts
        "ACTIVATED_EXPERTS": 2,   # 2 experts activated per token
        "HEAD": 32,               # number of attention heads
        "KV_HEAD": 4,             # number of key-value heads (grouped-query attention)
        "LAYER": 32
    },
    "Yuan2.0": {
        "D_MODEL": 2048,          # model hidden dimension
        "ATTENTION_PROJECTION": 4096,  # attention projection dimension
        "D_KEY": 2048,            # key vector dimension
        "D_VALUE": 2048,          # value vector dimension
        "EXPERT_FFN_DIM": 4096,   # FFN dimension of a single expert
        "NUM_EXPERTS": 32,        # total number of experts
        "ACTIVATED_EXPERTS": 2,   # 2 experts activated per token
        "HEAD": 16,               # number of attention heads
        "LAYER": 24
    }
}

HARDWARE_CONFIG = {
    "FREQUENCY": 0.8,              # clock frequency, GHz
    "CHIP_MOVE_BANDWIDTH": 288,  # inter-chip bandwidth, GB/s
    "DDR_BANDWIDTH": 25.6,        # DDR memory bandwidth, GB/s
    "PARALLELISM": 2048         # per-chip compute parallelism
}


# Derive and update actual simulation configuration based on model shape and hardware parameters:
# - Set task_emulator.expert_number (DeepSeek-MoE = ACTIVATED_EXPERTS + SHARED_EXPERTS, others = ACTIVATED_EXPERTS)
# - Compute and overwrite COMPUTE / DDR_LOAD / CHIP_MOVE / DDR_SAVE durations in HardwareConfig.action_durations
#   Rules (in cycles):
#   compute per micro-slice = D_MODEL * EXPERT_FFN_DIM / micro_slice_num_per_expert
#   COMPUTE duration (cycles) = ceil(compute per micro-slice / PARALLELISM)
#   DDR_LOAD duration (cycles) = ceil(bytes per micro-slice / DDR_BANDWIDTH)
#   CHIP_MOVE duration (cycles) = ceil(bytes per micro-slice / CHIP_MOVE_BANDWIDTH)
# Note: To keep units consistent and comparable to compute duration, both "bytes" and "compute amount"
#       are treated as element counts; bandwidths are used in abstract "elements per cycle" units
#       instead of real GB/s -> seconds -> cycles conversion.


def apply_model_overrides(sim, cfg: Dict, is_ep = False) -> None:
    """Update expert count and key action durations according to model shape mapping and hardware parameters.

    Args:
    - sim: DiscreteSimulator instance to access scheduler/task_emulator and HardwareConfig
    - cfg: merged CLI config dict (must contain 'model' and 'scheduler.micro_slice_num_per_expert')
    """

    # deferred import to avoid circular dependency
    from hardware_config import HardwareConfig
    from action import ActionType

    model_key = cfg.get('model')
    if not model_key or model_key not in MODEL_SHAPE_MAP:
        # skip if unspecified or not in mapping
        assert False, f"Model {model_key} not found in MODEL_SHAPE_MAP"

    shape = MODEL_SHAPE_MAP[model_key]

    # Allow runtime overrides of hardware approximation parameters from cfg
    try:
        hw_cfg = cfg.get('hardware', {})
        if hw_cfg and 'override_chip_move_bandwidth' in hw_cfg:
            HARDWARE_CONFIG['CHIP_MOVE_BANDWIDTH'] = float(hw_cfg['override_chip_move_bandwidth'])
        if hw_cfg and 'override_expert_bandwidth' in hw_cfg:
            HARDWARE_CONFIG['DDR_BANDWIDTH'] = float(hw_cfg['override_expert_bandwidth'])
    except Exception:
        pass

    # 1) expert_number
    sim.scheduler.task_emulator.expert_number = shape.get('NUM_EXPERTS')

    # 2) Estimate micro-slice workload and traffic based on D_MODEL and EXPERT_FFN_DIM
    d_model = int(shape.get('D_MODEL', 0) or 0)
    ffn_dim = int(shape.get('EXPERT_FFN_DIM', 0) or 0)
    # number of micro-slices: prefer runtime task_emulator value, fallback to cfg.scheduler, default 1
    try:
        micro_slices = int(getattr(sim.scheduler.task_emulator, 'micro_slice_num_per_expert', 1) or 1)
    except Exception:
        micro_slices = int(cfg.get('scheduler', {}).get('micro_slice_num_per_expert', 1) or 1)
    micro_slices = max(1, micro_slices)

    # compute amount and "byte amount" (abstract uniform unit)
    ops_per_slice = (d_model * ffn_dim) / micro_slices

    # hardware approximation parameters (elements per cycle scale)
    # * 40 to accelerate the simulation.
    parallelism = int(HARDWARE_CONFIG.get('PARALLELISM', 2048) or 2048) * 40
    ddr_bw = float(HARDWARE_CONFIG.get('DDR_BANDWIDTH', 12) or 12.0) / HARDWARE_CONFIG.get('FREQUENCY', 1.2) * 40
    chip_bw = float(HARDWARE_CONFIG.get('CHIP_MOVE_BANDWIDTH', 64) or 64.0) / HARDWARE_CONFIG.get('FREQUENCY', 1.2) * 40

    # duration estimation (in cycles): ceil, at least 1 cycle
    compute_dur = max(1, math.ceil(ops_per_slice * 2 / max(1, parallelism)))
    ddr_load_dur = max(5, math.ceil(ops_per_slice / ddr_bw))
    chip_move_dur = max(5, math.ceil(d_model * ffn_dim / chip_bw))
    if is_ep:
        chip_move_dur = max(5, math.ceil(d_model / chip_bw))

    # apply to global HardwareConfig.action_durations
    HardwareConfig.action_durations[ActionType.COMPUTE] = int(compute_dur)
    HardwareConfig.action_durations[ActionType.DDR_LOAD] = int(ddr_load_dur)
    HardwareConfig.action_durations[ActionType.CHIP_MOVE] = int(chip_move_dur)
    # DDR_SAVE approximately reuses DDR_LOAD duration (refine if differentiation needed)
    HardwareConfig.action_durations[ActionType.DDR_SAVE] = int(ddr_load_dur)
    HardwareConfig.ddr_buffer_size = int(sim.scheduler.task_emulator.expert_number * micro_slices / 4)

    # debug print to confirm override correctness
    print(f"[DEBUG] apply_model_overrides: model={model_key}, expert_number={sim.scheduler.task_emulator.expert_number}, "
          f"micro_slices={micro_slices}, ops_per_slice={ops_per_slice:.0f}, "
          f"compute_dur={compute_dur}, ddr_load_dur={ddr_load_dur}, chip_move_dur={chip_move_dur}")
