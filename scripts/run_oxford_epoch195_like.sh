#!/usr/bin/env bash

set -euo pipefail

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
  shift
fi

if [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/run_oxford_epoch195_like.sh [--check-only]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_IDS=(1)
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"

OXFORD_ROOT="${OXFORD_ROOT:-/Localize/ljc/Dataset/Oxford}"
POSE_ROOT="${POSE_ROOT:-/home/ljc/Downloads/QEOxford}"
FULL_H5_ROOT="${FULL_H5_ROOT:-/home/ljc/Downloads/2-h5data}"
OXFORD_0226_MASK_ROOT="${OXFORD_0226_MASK_ROOT:-/home/ljc/Downloads/h5filewithruenandstrait_swapped_turning/2-h5data}"
OXFORD_0300_MASK_ROOT="${OXFORD_0300_MASK_ROOT:-/home/ljc/Downloads/2-h5data}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/oxford_pin}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_ROOT}/experiments/oxford_epoch195_like}"
RUN_STAMP="${RUN_STAMP:-$(date '+%Y-%m-%d_%H-%M-%S')}"
RUN_ROOT="${EXPERIMENT_ROOT}/${RUN_STAMP}"

TEMPLATE_CONFIG="${REPO_ROOT}/config/lidar_slam/run_oxford.yaml"
CONVERTER="${REPO_ROOT}/dataset/converter/oxford_to_pin_format.py"
STITCHER="${REPO_ROOT}/eval/stitch_segment_runs.py"
SEGMENT_HELPER="${REPO_ROOT}/eval/build_oxford_segment_manifest.py"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

require_tool() {
  local tool_name="$1"
  command -v "${tool_name}" >/dev/null 2>&1 || die "required tool not found: ${tool_name}"
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "required file not found: ${path}"
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || die "required directory not found: ${path}"
}

pose_txt_path() {
  local sequence_short="$1"
  printf '%s/Oxford_SLAM_result_%s/gicp_Oxford%s_050_v1.txt' "${POSE_ROOT}" "${sequence_short}" "${sequence_short}"
}

check_gpu_ids() {
  require_tool "nvidia-smi"
  local available
  available="$(nvidia-smi --query-gpu=index --format=csv,noheader)"
  for gpu_id in "${GPU_IDS[@]}"; do
    if ! printf '%s\n' "${available}" | grep -qx "[[:space:]]*${gpu_id}"; then
      die "GPU ${gpu_id} is not available according to nvidia-smi"
    fi
  done
}

check_common_prereqs() {
  require_tool "pixi"
  require_file "${TEMPLATE_CONFIG}"
  require_file "${CONVERTER}"
  require_file "${STITCHER}"
  require_file "${SEGMENT_HELPER}"
  [[ "${PROCS_PER_GPU}" =~ ^[0-9]+$ ]] || die "PROCS_PER_GPU must be a positive integer"
  [[ "${PROCS_PER_GPU}" -ge 1 ]] || die "PROCS_PER_GPU must be >= 1"
  check_gpu_ids
}

report_conversion_state() {
  local sequence_name="$1"
  local output_dir="${DATA_ROOT}/${sequence_name}"

  if [[ -d "${output_dir}/ply" ]] \
    && [[ -f "${output_dir}/poses.txt" ]] \
    && [[ -f "${output_dir}/timestamps.txt" ]] \
    && [[ -f "${output_dir}/aligned_poses.txt" ]] \
    && [[ -f "${output_dir}/aligned_timestamps.txt" ]]; then
    log "conversion ready: ${output_dir}"
  else
    log "conversion missing or stale: ${output_dir}"
    log "expected files: ply/ poses.txt timestamps.txt aligned_poses.txt aligned_timestamps.txt"
  fi
}

check_route_inputs() {
  local route_tag="$1"
  local sequence_name="$2"
  local sequence_short="$3"
  local mask_root="$4"
  local mask_name="$5"

  require_dir "${OXFORD_ROOT}/${sequence_name}"
  require_dir "${OXFORD_ROOT}/${sequence_name}/velodyne_left"
  require_dir "${FULL_H5_ROOT}/${sequence_short}"
  require_file "${FULL_H5_ROOT}/${sequence_short}/velodyne_left_calibrateFalse.h5"
  require_dir "${mask_root}/${sequence_short}"
  require_file "${mask_root}/${sequence_short}/${mask_name}"
  require_file "$(pose_txt_path "${sequence_short}")"

  log "${route_tag}: raw Oxford prerequisites are present"
  report_conversion_state "${sequence_name}"
}

ensure_converted() {
  local route_tag="$1"
  local sequence_name="$2"
  local sequence_short="$3"
  local mask_root="$4"
  local mask_name="$5"

  local output_dir="${DATA_ROOT}/${sequence_name}"
  local pointcloud_dir="${output_dir}/ply"
  local poses_path="${output_dir}/poses.txt"
  local timestamps_path="${output_dir}/timestamps.txt"
  local aligned_poses_path="${output_dir}/aligned_poses.txt"
  local aligned_timestamps_path="${output_dir}/aligned_timestamps.txt"

  if [[ -d "${pointcloud_dir}" ]] \
    && [[ -f "${poses_path}" ]] \
    && [[ -f "${timestamps_path}" ]] \
    && [[ -f "${aligned_poses_path}" ]] \
    && [[ -f "${aligned_timestamps_path}" ]]; then
    log "${route_tag}: reusing converted PIN-SLAM input at ${output_dir}"
    return
  fi

  if [[ ${CHECK_ONLY} -eq 1 ]]; then
    log "${route_tag}: would run Oxford conversion because converted outputs are missing or stale"
    return
  fi

  log "${route_tag}: converting Oxford route into PIN-SLAM input (resume-safe)"
  pixi run python "${CONVERTER}" \
    --oxford-root "${OXFORD_ROOT}" \
    --sequence "${sequence_name}" \
    --output-root "${DATA_ROOT}" \
    --mask-h5-root "${mask_root}" \
    --mask-h5-name "${mask_name}" \
    --full-h5-root "${FULL_H5_ROOT}" \
    --full-h5-name velodyne_left_calibrateFalse.h5 \
    --pose-root "${POSE_ROOT}" \
    --pose-skip-start 5 \
    --pose-skip-end 5 \
    --resume
}

write_route_config() {
  local config_out="$1"
  local route_name="$2"
  local sequence_name="$3"

  mkdir -p "$(dirname "${config_out}")"
  pixi run python - "${TEMPLATE_CONFIG}" "${config_out}" "${route_name}" "${DATA_ROOT}/${sequence_name}/ply" "${DATA_ROOT}/${sequence_name}/poses.txt" <<'PY'
import sys
import yaml

template_path, output_path, route_name, pc_path, pose_path = sys.argv[1:]

with open(template_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

setting = config.setdefault("setting", {})
setting["name"] = route_name
setting["pc_path"] = pc_path
setting["pose_path"] = pose_path

with open(output_path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
}

run_route() {
  local route_tag="$1"
  local route_name="$2"
  local sequence_name="$3"

  local route_root="${RUN_ROOT}/${route_tag}"
  local split_root="${route_root}/split_runs"
  local stitch_root="${route_root}/stitched"
  local config_path="${route_root}/config/${route_tag}.yaml"
  local timestamps_path="${DATA_ROOT}/${sequence_name}/timestamps.txt"
  local aligned_timestamps_path="${DATA_ROOT}/${sequence_name}/aligned_timestamps.txt"
  local segment_manifest_path="${route_root}/segments.json"

  write_route_config "${config_path}" "${route_name}" "${sequence_name}"

  mkdir -p "${route_root}"
  pixi run python "${SEGMENT_HELPER}" \
    --selected-timestamps "${timestamps_path}" \
    --aligned-timestamps "${aligned_timestamps_path}" \
    -o "${segment_manifest_path}" >/dev/null

  local segment_summary
  segment_summary="$(pixi run python - "${segment_manifest_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
print(
    "{selected}|{segments}|{runnable}|{skipped}|{seg_min}|{seg_max}|{seg_median}|{run_min}|{run_max}|{run_median}".format(
        selected=manifest["selected_frame_count"],
        segments=manifest["segment_count"],
        runnable=manifest["runnable_segment_count"],
        skipped=manifest["skipped_segment_count"],
        seg_min=manifest["segment_frame_count_min"],
        seg_max=manifest["segment_frame_count_max"],
        seg_median=manifest["segment_frame_count_median"],
        run_min=manifest["runnable_frame_count_min"],
        run_max=manifest["runnable_frame_count_max"],
        run_median=manifest["runnable_frame_count_median"],
    )
)
PY
)"
  IFS='|' read -r selected_frame_count segment_count runnable_segment_count skipped_segment_count segment_min segment_max segment_median runnable_min runnable_max runnable_median <<< "${segment_summary}"

  [[ "${selected_frame_count}" -gt 0 ]] || die "${route_tag}: timestamps.txt is empty: ${timestamps_path}"
  [[ "${runnable_segment_count}" -gt 0 ]] || die "${route_tag}: no runnable discontinuity segments were found"

  if [[ ${CHECK_ONLY} -eq 1 ]]; then
    log "${route_tag}: check-only mode, discovered ${segment_count} segments (${runnable_segment_count} runnable, ${skipped_segment_count} skipped) from ${selected_frame_count} selected frames"
    log "${route_tag}: segment lengths frames min=${segment_min} median=${segment_median} max=${segment_max}; runnable min=${runnable_min} median=${runnable_median} max=${runnable_max}"
    return
  fi

  mkdir -p "${split_root}" "${stitch_root}"

  local gpu_count="${#GPU_IDS[@]}"
  local worker_count=$(( gpu_count * PROCS_PER_GPU ))
  log "${route_tag}: launching ${runnable_segment_count} runnable discontinuity segments (${segment_count} discovered, ${skipped_segment_count} skipped) on GPUs ${GPU_IDS[*]} (PROCS_PER_GPU=${PROCS_PER_GPU})"
  local pids=()
  local segment_queue
  segment_queue="$(pixi run python - "${segment_manifest_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
segments = [segment for segment in manifest["segments"] if segment["runnable"]]
segments.sort(key=lambda segment: (-int(segment["frame_count"]), int(segment["begin_frame"]), int(segment["segment_index"])))
for segment in segments:
    print(
        "{segment_index}|{begin_frame}|{end_frame}|{frame_count}".format(
            segment_index=int(segment["segment_index"]),
            begin_frame=int(segment["begin_frame"]),
            end_frame=int(segment["end_frame"]),
            frame_count=int(segment["frame_count"]),
        )
    )
PY
)"

  local worker_slot
  for (( worker_slot=0; worker_slot<worker_count; worker_slot++ )); do
    local gpu_slot=$(( worker_slot % gpu_count ))
    local gpu_id="${GPU_IDS[${gpu_slot}]}"
    {
      local line segment_index begin_frame end_frame frame_count segment_root
      while IFS= read -r line; do
        [[ -n "${line}" ]] || continue
        IFS='|' read -r segment_index begin_frame end_frame frame_count <<< "${line}"
        segment_root="${split_root}/seg_${segment_index}_${begin_frame}_${end_frame}"
        mkdir -p "${segment_root}"
        printf '[%s] %s worker=%s GPU=%s segment=%s range=[%s,%s) frames=%s\n' \
          "$(date '+%F %T')" "${route_tag}" "${worker_slot}" "${gpu_id}" "${segment_index}" "${begin_frame}" "${end_frame}" "${frame_count}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
          pixi run python pin_slam.py "${config_path}" -l --range "${begin_frame}" "${end_frame}" 1 -o "${segment_root}" \
          |& tee "${segment_root}/run.log"
      done
    } < <(printf '%s\n' "${segment_queue}" | awk -F'|' -v workers="${worker_count}" -v slot="${worker_slot}" '((NR-1) % workers) == slot') &
    pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  [[ "${failed}" -eq 0 ]] || die "${route_tag}: at least one segment run failed"

  local run_dirs=()
  local run_dir
  for segment_root in "${split_root}"/seg_*; do
    [[ -d "${segment_root}" ]] || continue
    run_dir="$(find "${segment_root}" -mindepth 1 -maxdepth 1 -type d | LC_ALL=C sort | tail -n 1)"
    [[ -n "${run_dir}" ]] || die "${route_tag}: missing PIN-SLAM run output under ${segment_root}"
    run_dirs+=("${run_dir}")
  done
  [[ "${#run_dirs[@]}" -gt 0 ]] || die "${route_tag}: no segment run directories found"

  log "${route_tag}: stitching ${#run_dirs[@]} segment runs"
  pixi run python "${STITCHER}" \
    "${run_dirs[@]}" \
    -o "${stitch_root}" \
    --trajectory-source slam

  log "${route_tag}: full_route_path.png -> ${stitch_root}/full_route_path.png"
  log "${route_tag}: full_route_path_3D.png -> ${stitch_root}/full_route_path_3D.png"
}

main() {
  check_common_prereqs

  check_route_inputs \
    "0226_turning_straight" \
    "2019-01-11-14-02-26-radar-oxford-10k" \
    "0226" \
    "${OXFORD_0226_MASK_ROOT}" \
    "velodyne_left_calibrateFalse_SCR_turning_straight.h5"

  check_route_inputs \
    "0300_lo300" \
    "2019-01-17-14-03-00-radar-oxford-10k" \
    "0300" \
    "${OXFORD_0300_MASK_ROOT}" \
    "velodyne_left_calibrateFalse_LO300m.h5"

  ensure_converted \
    "0226_turning_straight" \
    "2019-01-11-14-02-26-radar-oxford-10k" \
    "0226" \
    "${OXFORD_0226_MASK_ROOT}" \
    "velodyne_left_calibrateFalse_SCR_turning_straight.h5"

  ensure_converted \
    "0300_lo300" \
    "2019-01-17-14-03-00-radar-oxford-10k" \
    "0300" \
    "${OXFORD_0300_MASK_ROOT}" \
    "velodyne_left_calibrateFalse_LO300m.h5"

  mkdir -p "${RUN_ROOT}"

  run_route \
    "0226_turning_straight" \
    "oxford_0226_turning_straight_pin" \
    "2019-01-11-14-02-26-radar-oxford-10k"

  run_route \
    "0300_lo300" \
    "oxford_0300_lo300_pin" \
    "2019-01-17-14-03-00-radar-oxford-10k"

  if [[ ${CHECK_ONLY} -eq 1 ]]; then
    log "check-only mode finished"
    exit 0
  fi

  log "all routes finished"
  log "outputs root: ${RUN_ROOT}"
}

main "$@"
