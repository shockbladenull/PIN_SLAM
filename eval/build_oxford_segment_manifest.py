#!/usr/bin/env python3
# @file      build_oxford_segment_manifest.py
# @author    OpenAI Codex

import argparse
import json
from pathlib import Path

import numpy as np


def load_timestamp_array(path: str | Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64)
    if values.ndim == 0:
        values = values.reshape(1)
    return np.asarray(values, dtype=np.int64)


def build_segment_manifest(
    selected_timestamps_path: str | Path,
    aligned_timestamps_path: str | Path,
) -> dict:
    selected_timestamps_path = Path(selected_timestamps_path).resolve()
    aligned_timestamps_path = Path(aligned_timestamps_path).resolve()

    selected_timestamps = load_timestamp_array(selected_timestamps_path)
    aligned_timestamps = load_timestamp_array(aligned_timestamps_path)
    aligned_lookup = {int(timestamp): idx for idx, timestamp in enumerate(aligned_timestamps.tolist())}

    aligned_indices = []
    missing_timestamps = []
    for timestamp in selected_timestamps.tolist():
        aligned_index = aligned_lookup.get(int(timestamp))
        if aligned_index is None:
            missing_timestamps.append(int(timestamp))
            continue
        aligned_indices.append(aligned_index)

    if missing_timestamps:
        raise ValueError(
            "Selected timestamps contain {} values that do not exist in aligned timestamps".format(
                len(missing_timestamps)
            )
        )

    aligned_indices = np.asarray(aligned_indices, dtype=np.int64)
    segments = []
    start_idx = 0
    for end_idx in range(1, len(selected_timestamps) + 1):
        reached_end = end_idx == len(selected_timestamps)
        is_break = (
            not reached_end
            and int(aligned_indices[end_idx]) != (int(aligned_indices[end_idx - 1]) + 1)
        )
        if not reached_end and not is_break:
            continue

        frame_count = int(end_idx - start_idx)
        segment = {
            "segment_index": int(len(segments) + 1),
            "begin_frame": int(start_idx),
            "end_frame": int(end_idx),
            "frame_count": frame_count,
            "start_timestamp": int(selected_timestamps[start_idx]),
            "end_timestamp": int(selected_timestamps[end_idx - 1]),
            "start_aligned_index": int(aligned_indices[start_idx]),
            "end_aligned_index": int(aligned_indices[end_idx - 1]),
            "runnable": bool(frame_count >= 2),
        }
        segments.append(segment)
        start_idx = end_idx

    frame_counts = [segment["frame_count"] for segment in segments]
    runnable_frame_counts = [segment["frame_count"] for segment in segments if segment["runnable"]]
    return {
        "selected_timestamps_path": str(selected_timestamps_path),
        "aligned_timestamps_path": str(aligned_timestamps_path),
        "selected_frame_count": int(len(selected_timestamps)),
        "aligned_frame_count": int(len(aligned_timestamps)),
        "segment_count": int(len(segments)),
        "runnable_segment_count": int(sum(1 for segment in segments if segment["runnable"])),
        "skipped_segment_count": int(sum(1 for segment in segments if not segment["runnable"])),
        "segment_frame_count_min": int(min(frame_counts)) if frame_counts else 0,
        "segment_frame_count_max": int(max(frame_counts)) if frame_counts else 0,
        "segment_frame_count_median": int(np.median(np.asarray(frame_counts, dtype=np.int64))) if frame_counts else 0,
        "runnable_frame_count_min": int(min(runnable_frame_counts)) if runnable_frame_counts else 0,
        "runnable_frame_count_max": int(max(runnable_frame_counts)) if runnable_frame_counts else 0,
        "runnable_frame_count_median": int(np.median(np.asarray(runnable_frame_counts, dtype=np.int64)))
        if runnable_frame_counts
        else 0,
        "segments": segments,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build TransLO-style Oxford discontinuity segments from selected/aligned timestamp lists."
    )
    parser.add_argument("--selected-timestamps", required=True, help="Path to selected timestamps.txt")
    parser.add_argument("--aligned-timestamps", required=True, help="Path to aligned_timestamps.txt")
    parser.add_argument("-o", "--output", required=True, help="Output JSON manifest path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_segment_manifest(args.selected_timestamps, args.aligned_timestamps)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Oxford segment manifest:", output_path)
    print("Segments:", manifest["segment_count"])
    print("Runnable:", manifest["runnable_segment_count"])
    print("Skipped:", manifest["skipped_segment_count"])


if __name__ == "__main__":
    main()
