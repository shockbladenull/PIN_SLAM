#!/usr/bin/env python3
# @file      oxford_to_pin_format.py
# @author    OpenAI Codex

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

try:
    import h5py
except ImportError:  # pragma: no cover - exercised via runtime import check
    h5py = None


DEFAULT_MASK_H5_NAME = "velodyne_left_calibrateFalse.h5"
DEFAULT_FULL_H5_NAME = "velodyne_left_calibrateFalse.h5"
DEFAULT_POSE_TXT_TEMPLATE = "Oxford_SLAM_result_{sequence_short}/gicp_Oxford{sequence_short}_050_v1.txt"


def oxford_sequence_short_name(sequence_name: str) -> str:
    parts = sequence_name.split("-")
    if len(parts) >= 6 and all(part.isdigit() for part in parts[:6]):
        return "{}{}".format(parts[4], parts[5])
    return sequence_name


def resolve_oxford_sequence_file(
    sequence_name: str,
    seq_dir: str,
    filename: str,
    root_override: Optional[str] = None,
) -> str:
    if root_override is None:
        return os.path.join(seq_dir, filename)

    sequence_short = oxford_sequence_short_name(sequence_name)
    candidates = [
        os.path.join(root_override, sequence_name, filename),
        os.path.join(root_override, sequence_short, filename),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def load_valid_timestamps(h5_path: str) -> np.ndarray:
    if h5py is None:
        raise ImportError("h5py is required for Oxford conversion. Install it with Pixi or requirements.txt.")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError("Oxford H5 not found: {}".format(h5_path))

    with h5py.File(h5_path, "r") as h5_file:
        if "valid_timestamps" not in h5_file:
            raise KeyError("Oxford H5 is missing 'valid_timestamps': {}".format(h5_path))
        return np.asarray(h5_file["valid_timestamps"], dtype=np.int64)


def load_kitti_pose_rows(txt_path: str) -> np.ndarray:
    if not os.path.isfile(txt_path):
        raise FileNotFoundError("Oxford TXT pose file not found: {}".format(txt_path))

    pose_rows = np.loadtxt(txt_path, dtype=np.float32)
    if pose_rows.ndim == 1:
        pose_rows = pose_rows.reshape(1, -1)
    if pose_rows.shape[1] != 12:
        raise ValueError("Expected Oxford TXT poses with shape [N, 12], got {}".format(pose_rows.shape))
    return pose_rows


def align_txt_pose_rows_to_full_timestamps(
    pose_rows: np.ndarray,
    full_timestamps: np.ndarray,
    skip_start: int,
    skip_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    skip_start = max(int(skip_start), 0)
    skip_end = max(int(skip_end), 0)
    stop = len(full_timestamps) - skip_end if skip_end > 0 else len(full_timestamps)
    if stop <= skip_start:
        raise ValueError("Invalid Oxford TXT skip range: start={}, end={}".format(skip_start, skip_end))

    aligned_timestamps = np.asarray(full_timestamps[skip_start:stop], dtype=np.int64)
    if len(aligned_timestamps) != len(pose_rows):
        raise ValueError(
            "Oxford TXT/full-H5 length mismatch: {} poses vs {} timestamps after skipping {} front and {} back".format(
                len(pose_rows), len(aligned_timestamps), skip_start, skip_end
            )
        )
    return aligned_timestamps, np.asarray(pose_rows, dtype=np.float32)


def select_masked_pose_rows(
    mask_timestamps: np.ndarray,
    full_timestamps: np.ndarray,
    aligned_timestamps: np.ndarray,
    aligned_pose_rows: np.ndarray,
    skip_start: int,
    skip_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    aligned_lookup = {
        int(timestamp): aligned_pose_rows[idx]
        for idx, timestamp in enumerate(np.asarray(aligned_timestamps, dtype=np.int64))
    }

    trimmed_out = set(int(timestamp) for timestamp in np.asarray(full_timestamps[: max(skip_start, 0)], dtype=np.int64))
    if skip_end > 0:
        trimmed_out.update(int(timestamp) for timestamp in np.asarray(full_timestamps[-skip_end:], dtype=np.int64))

    selected_timestamps = []
    selected_rows = []
    missing_timestamps = []
    for timestamp in np.asarray(mask_timestamps, dtype=np.int64):
        timestamp = int(timestamp)
        pose_row = aligned_lookup.get(timestamp)
        if pose_row is None:
            missing_timestamps.append(timestamp)
            continue
        selected_timestamps.append(timestamp)
        selected_rows.append(pose_row)

    unexpected_missing = [timestamp for timestamp in missing_timestamps if timestamp not in trimmed_out]
    if unexpected_missing:
        raise ValueError(
            "Oxford TXT is missing {} masked timestamps outside the configured front/back skip window".format(
                len(unexpected_missing)
            )
        )
    if not selected_timestamps:
        raise ValueError("Oxford TXT alignment removed every masked timestamp")

    return np.asarray(selected_timestamps, dtype=np.int64), np.asarray(selected_rows, dtype=np.float32)


def read_oxford_scan_bin(scan_path: str) -> np.ndarray:
    scan = np.fromfile(scan_path, dtype=np.float32)
    if scan.size == 0 or scan.size % 4 != 0:
        raise ValueError("Invalid Oxford scan file: {}".format(scan_path))
    scan = scan.reshape(4, -1).transpose()[:, :3]
    scan[:, 2] *= -1.0
    return np.ascontiguousarray(scan.astype(np.float64))


def write_ply(points: np.ndarray, ply_path: str) -> None:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if not o3d.io.write_point_cloud(ply_path, point_cloud):
        raise RuntimeError("Failed to write Oxford point cloud as PLY: {}".format(ply_path))


def write_kitti_pose_rows(output_path: str, pose_rows: np.ndarray) -> None:
    np.savetxt(output_path, np.asarray(pose_rows, dtype=np.float32), fmt="%.9f")


def load_oxford_pose_selection(
    oxford_root: str,
    sequence_name: str,
    mask_h5_name: str = DEFAULT_MASK_H5_NAME,
    mask_h5_root: Optional[str] = None,
    full_h5_name: str = DEFAULT_FULL_H5_NAME,
    full_h5_root: Optional[str] = None,
    pose_root: Optional[str] = None,
    pose_txt_template: str = DEFAULT_POSE_TXT_TEMPLATE,
    pose_skip_start: int = 5,
    pose_skip_end: int = 5,
    trim_edges: int = 0,
) -> dict:
    seq_dir = os.path.join(oxford_root, sequence_name)
    scan_dir = os.path.join(seq_dir, "velodyne_left")
    if not os.path.isdir(scan_dir):
        raise FileNotFoundError("Oxford scan directory not found: {}".format(scan_dir))

    mask_h5_path = resolve_oxford_sequence_file(
        sequence_name=sequence_name,
        seq_dir=seq_dir,
        filename=mask_h5_name,
        root_override=mask_h5_root,
    )
    full_h5_path = resolve_oxford_sequence_file(
        sequence_name=sequence_name,
        seq_dir=seq_dir,
        filename=full_h5_name,
        root_override=full_h5_root,
    )

    txt_root = pose_root if pose_root is not None else oxford_root
    sequence_short = oxford_sequence_short_name(sequence_name)
    pose_txt_path = os.path.join(
        txt_root,
        pose_txt_template.format(sequence=sequence_name, sequence_short=sequence_short),
    )

    mask_timestamps = load_valid_timestamps(mask_h5_path)
    full_timestamps = load_valid_timestamps(full_h5_path)
    pose_rows = load_kitti_pose_rows(pose_txt_path)
    aligned_timestamps, aligned_pose_rows = align_txt_pose_rows_to_full_timestamps(
        pose_rows=pose_rows,
        full_timestamps=full_timestamps,
        skip_start=pose_skip_start,
        skip_end=pose_skip_end,
    )
    selected_timestamps, selected_pose_rows = select_masked_pose_rows(
        mask_timestamps=mask_timestamps,
        full_timestamps=full_timestamps,
        aligned_timestamps=aligned_timestamps,
        aligned_pose_rows=aligned_pose_rows,
        skip_start=pose_skip_start,
        skip_end=pose_skip_end,
    )

    trim_edges = max(int(trim_edges), 0)
    if trim_edges > 0:
        if len(selected_timestamps) <= (2 * trim_edges):
            raise ValueError(
                "Oxford sequence {} is too short after trimming {} masked frames on each side".format(
                    sequence_name, trim_edges
                )
            )
        selected_timestamps = selected_timestamps[trim_edges:-trim_edges]
        selected_pose_rows = selected_pose_rows[trim_edges:-trim_edges]

    return {
        "sequence_name": sequence_name,
        "sequence_short": sequence_short,
        "scan_dir": scan_dir,
        "mask_h5_path": mask_h5_path,
        "full_h5_path": full_h5_path,
        "pose_txt_path": pose_txt_path,
        "aligned_timestamps": aligned_timestamps,
        "aligned_pose_rows": aligned_pose_rows,
        "selected_timestamps": selected_timestamps,
        "selected_pose_rows": selected_pose_rows,
    }


def prepare_output_dirs(
    sequence_output_dir: str,
    pointcloud_dir_name: str,
    overwrite: bool,
    resume: bool,
) -> tuple[str, str]:
    if overwrite and resume:
        raise ValueError("Oxford conversion cannot use overwrite and resume at the same time")

    if overwrite and os.path.isdir(sequence_output_dir):
        shutil.rmtree(sequence_output_dir)

    os.makedirs(sequence_output_dir, exist_ok=True)
    pointcloud_dir = os.path.join(sequence_output_dir, pointcloud_dir_name)
    os.makedirs(pointcloud_dir, exist_ok=True)
    if (not resume) and os.listdir(pointcloud_dir):
        raise FileExistsError(
            "Output point cloud directory is not empty. Use --overwrite or a fresh output root: {}".format(pointcloud_dir)
        )
    return sequence_output_dir, pointcloud_dir


def validate_existing_pointcloud_dir(pointcloud_dir: str, expected_filenames: set[str]) -> int:
    existing_filenames = os.listdir(pointcloud_dir)
    if any(not filename.endswith(".ply") for filename in existing_filenames):
        raise RuntimeError("Output point cloud directory contains non-PLY files: {}".format(pointcloud_dir))

    unexpected_files = sorted(set(existing_filenames) - expected_filenames)
    if unexpected_files:
        raise RuntimeError(
            "Output point cloud directory contains {} unexpected PLY files for the requested Oxford selection, "
            "use --overwrite to regenerate cleanly: {}".format(len(unexpected_files), pointcloud_dir)
        )
    return len(existing_filenames)


def convert_sequence(
    oxford_root: str,
    sequence_name: str,
    output_root: str,
    mask_h5_name: str = DEFAULT_MASK_H5_NAME,
    mask_h5_root: Optional[str] = None,
    full_h5_name: str = DEFAULT_FULL_H5_NAME,
    full_h5_root: Optional[str] = None,
    pose_root: Optional[str] = None,
    pose_txt_template: str = DEFAULT_POSE_TXT_TEMPLATE,
    pose_skip_start: int = 5,
    pose_skip_end: int = 5,
    trim_edges: int = 0,
    pointcloud_dir_name: str = "ply",
    overwrite: bool = False,
    resume: bool = False,
) -> dict:
    selection = load_oxford_pose_selection(
        oxford_root=oxford_root,
        sequence_name=sequence_name,
        mask_h5_name=mask_h5_name,
        mask_h5_root=mask_h5_root,
        full_h5_name=full_h5_name,
        full_h5_root=full_h5_root,
        pose_root=pose_root,
        pose_txt_template=pose_txt_template,
        pose_skip_start=pose_skip_start,
        pose_skip_end=pose_skip_end,
        trim_edges=trim_edges,
    )

    sequence_output_dir, pointcloud_dir = prepare_output_dirs(
        sequence_output_dir=os.path.join(output_root, sequence_name),
        pointcloud_dir_name=pointcloud_dir_name,
        overwrite=overwrite,
        resume=resume,
    )

    expected_filenames = {"{}.ply".format(int(timestamp)) for timestamp in selection["selected_timestamps"]}
    existing_count = validate_existing_pointcloud_dir(pointcloud_dir, expected_filenames) if resume else 0
    total_count = len(selection["selected_timestamps"])
    print(
        "Oxford conversion {}: {} selected scans ({} already present)".format(
            sequence_name, total_count, existing_count
        )
    )

    for idx, timestamp in enumerate(selection["selected_timestamps"], start=1):
        ply_name = "{}.ply".format(int(timestamp))
        ply_path = os.path.join(pointcloud_dir, ply_name)
        if resume and os.path.isfile(ply_path):
            if idx == 1 or idx == total_count or idx % 1000 == 0:
                print("Oxford conversion {}: reuse {}/{}".format(sequence_name, idx, total_count))
            continue

        scan_path = os.path.join(selection["scan_dir"], "{}.bin".format(int(timestamp)))
        if not os.path.isfile(scan_path):
            raise FileNotFoundError("Oxford scan file not found: {}".format(scan_path))
        points = read_oxford_scan_bin(scan_path)
        write_ply(points, ply_path)
        if idx == 1 or idx == total_count or idx % 1000 == 0:
            print("Oxford conversion {}: wrote {}/{}".format(sequence_name, idx, total_count))

    pose_output_path = os.path.join(sequence_output_dir, "poses.txt")
    timestamps_output_path = os.path.join(sequence_output_dir, "timestamps.txt")
    aligned_pose_output_path = os.path.join(sequence_output_dir, "aligned_poses.txt")
    aligned_timestamps_output_path = os.path.join(sequence_output_dir, "aligned_timestamps.txt")
    write_kitti_pose_rows(pose_output_path, selection["selected_pose_rows"])
    np.savetxt(timestamps_output_path, selection["selected_timestamps"], fmt="%d")
    write_kitti_pose_rows(aligned_pose_output_path, selection["aligned_pose_rows"])
    np.savetxt(aligned_timestamps_output_path, selection["aligned_timestamps"], fmt="%d")

    pointcloud_filenames = os.listdir(pointcloud_dir)
    if not pointcloud_filenames:
        raise RuntimeError("Oxford conversion produced zero point clouds")
    if any(not filename.endswith(".ply") for filename in pointcloud_filenames):
        raise RuntimeError("Output point cloud directory contains non-PLY files: {}".format(pointcloud_dir))

    return {
        "sequence_output_dir": sequence_output_dir,
        "pointcloud_dir": pointcloud_dir,
        "pose_output_path": pose_output_path,
        "timestamps_output_path": timestamps_output_path,
        "aligned_pose_output_path": aligned_pose_output_path,
        "aligned_timestamps_output_path": aligned_timestamps_output_path,
        "frame_count": int(len(selection["selected_timestamps"])),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Oxford/RobotCar data in the TransLO layout into PIN-SLAM's generic pointcloud + poses format."
    )
    parser.add_argument("--oxford-root", required=True, help="Oxford root that contains <sequence>/velodyne_left")
    parser.add_argument("--sequence", required=True, help="Sequence name, e.g. 2019-01-11-14-02-26-radar-oxford-10k")
    parser.add_argument("--output-root", required=True, help="Output root; results are written to <output-root>/<sequence>")
    parser.add_argument("--mask-h5-name", default=DEFAULT_MASK_H5_NAME, help="Mask H5 filename used to select frames")
    parser.add_argument("--mask-h5-root", default=None, help="Optional alternate root for the mask H5 files")
    parser.add_argument("--full-h5-name", default=DEFAULT_FULL_H5_NAME, help="Full-route H5 filename used to align TXT poses")
    parser.add_argument("--full-h5-root", default=None, help="Optional alternate root for the full H5 files")
    parser.add_argument("--pose-root", default=None, help="Optional alternate root for the Oxford TXT pose exports")
    parser.add_argument(
        "--pose-txt-template",
        default=DEFAULT_POSE_TXT_TEMPLATE,
        help="Pose TXT template relative to pose root; supports {sequence} and {sequence_short}",
    )
    parser.add_argument("--pose-skip-start", type=int, default=5, help="Skip N timestamps from the front before TXT alignment")
    parser.add_argument("--pose-skip-end", type=int, default=5, help="Skip N timestamps from the back before TXT alignment")
    parser.add_argument("--trim-edges", type=int, default=0, help="Trim N masked frames from both ends after alignment")
    parser.add_argument("--pointcloud-dir-name", default="ply", help="Subdirectory name for exported point clouds")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output sequence directory")
    parser.add_argument("--resume", action="store_true", help="Reuse existing matching PLY files and only fill in missing outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    conversion = convert_sequence(
        oxford_root=args.oxford_root,
        sequence_name=args.sequence,
        output_root=args.output_root,
        mask_h5_name=args.mask_h5_name,
        mask_h5_root=args.mask_h5_root,
        full_h5_name=args.full_h5_name,
        full_h5_root=args.full_h5_root,
        pose_root=args.pose_root,
        pose_txt_template=args.pose_txt_template,
        pose_skip_start=args.pose_skip_start,
        pose_skip_end=args.pose_skip_end,
        trim_edges=args.trim_edges,
        pointcloud_dir_name=args.pointcloud_dir_name,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    print("Oxford conversion complete")
    print("Sequence output:", conversion["sequence_output_dir"])
    print("Point clouds:", conversion["pointcloud_dir"])
    print("Poses:", conversion["pose_output_path"])
    print("Aligned poses:", conversion["aligned_pose_output_path"])
    print("Frames:", conversion["frame_count"])


if __name__ == "__main__":
    main()
