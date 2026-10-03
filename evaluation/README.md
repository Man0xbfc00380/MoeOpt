Evaluation harness for MoE scheduling experiments

This package runs the simulator (FSE-DP scheduler + hardware/task emulators), collects metrics, and generates Matplotlib plots.

Structure
- runner.py: main simulation entry. Loads config, runs frames, logs CSV.
- metrics.py: metrics recorder and CSV writer.
- plots.py: static plotting utilities (reads CSVs -> images).
- configs/default.json: default simulation settings.
- results/: CSVs and figures.

Quick start
- From this workspace in VS Code, run the short sanity sim via the Run task or calling runner.py with the default config. Figures will be saved under evaluation/results/figures.

Notes
- This harness is intentionally light-weight; it leverages the animation scheduler API but runs without interactive plotting.
- We avoid changing core scheduler logic; instead we observe actions per step and complete them using HardwareConfig.action_durations to approximate time.
