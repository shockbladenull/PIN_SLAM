# Repository Guidelines

## Project Structure & Module Organization
`pin_slam.py` is the main CLI entrypoint for offline SLAM runs, `pin_slam_ros.py` handles ROS input, and `vis_pin_map.py` reconstructs meshes from saved results. Core logic lives in `model/` (decoder, neural points), `utils/` (tracking, mapping, meshing, loop closure, config), and `dataset/` (dataset indexing plus `dataloaders/` and format `converter/` scripts). Runtime configs are under `config/lidar_slam/` and `config/rgbd_slam/`; GUI code is in `gui/`; example CAD assets are in `cad/`; evaluation notebooks and helpers are in `eval/`.

## Build, Test, and Development Commands
Prefer the managed Pixi environment: `pixi install`, then run commands with `pixi run ...`. The manifest pins Python 3.10, PyTorch 2.5.1, and CUDA 11.8 for Linux. If Pixi is unavailable, fall back to the README’s conda plus `pip3 install -r requirements.txt` flow.

Useful commands:
- `pixi run python pin_slam.py -h`: show all runtime flags and dataset loader options.
- `pixi run download-kitti-example`: fetch the small KITTI sanity-check dataset.
- `pixi run demo`: run the standard sanity test with viewer, map save, and mesh save.
- `pixi run oxford-convert-help`: inspect the Oxford-to-PIN conversion CLI.
- `pixi run vis-help`: inspect mesh reconstruction options for saved runs.
- `pixi run test`: run the lightweight regression tests under `tests/`.
- `cd docker && ./build_docker.sh && ./start_docker.sh`: build and start the provided Docker workflow.

## Coding Style & Naming Conventions
This repository is Python-first. Follow existing style: 4-space indentation, snake_case for functions, variables, modules, and YAML configs such as `run_kitti.yaml`; PascalCase for classes such as `SLAMDataset` or `PoseGraphManager`. Prefer small, focused helpers in `utils/` or `dataset/dataloaders/` rather than adding logic to the top-level scripts. No formatter or linter config is checked in, so keep imports, type hints, and inline comments consistent with nearby files.

## Testing Guidelines
Automated coverage is minimal but now includes small regression tests under `tests/`. Run `pixi run smoke-test`, then `pixi run test`, and finally the smallest relevant workflow, usually the KITTI demo above or a dataset-specific command that exercises the modified loader, mapper, or GUI path. For Oxford conversions, also verify that `pc_path` contains only point clouds and that `meta/config_all.yaml` shows `track_on: true` and `pgo_on: true`. Do not commit generated `data/`, `experiments/`, screenshots, or other large outputs.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects, sometimes prefixed with tags like `[MINOR]`, for example `[MINOR] update readme`. Keep commits narrow, mention affected dataset/config when relevant, and reference issue numbers in the subject or body when applicable. Pull requests should explain the behavioral change, list the commands or datasets used for validation, and attach screenshots only when GUI, visualization, or mesh output changed.
