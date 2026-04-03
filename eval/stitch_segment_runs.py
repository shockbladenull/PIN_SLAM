#!/usr/bin/env python3
# @file      stitch_segment_runs.py
# @author    OpenAI Codex

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import yaml


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    qw, qx, qy, qz = q / q_norm

    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def read_pose_file(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)

    if rows.shape[1] == 12:
        poses = np.zeros((rows.shape[0], 4, 4), dtype=np.float64)
        poses[:, 3, 3] = 1.0
        poses[:, :3, :] = rows.reshape(rows.shape[0], 3, 4)
        return poses

    if rows.shape[1] == 8:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], rows.shape[0], axis=0)
        for idx, row in enumerate(rows):
            _, tx, ty, tz, qx, qy, qz, qw = row
            poses[idx, :3, :3] = quaternion_to_matrix(qx, qy, qz, qw)
            poses[idx, :3, 3] = np.asarray([tx, ty, tz], dtype=np.float64)
        return poses

    raise ValueError("Unsupported pose format {} for {}".format(rows.shape, path))


def resolve_existing_path(raw_path: str, run_dir: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    search_roots = [Path.cwd(), run_dir, Path(__file__).resolve().parents[1]]
    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError("Referenced path does not exist: {}".format(candidate))

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError("Could not resolve path '{}' from {}".format(raw_path, run_dir))


def set_equal_axis_2d(ax, x_values: np.ndarray, y_values: np.ndarray) -> None:
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if x_values.size == 0 or y_values.size == 0:
        return

    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    radius = 0.5 * max(x_max - x_min, y_max - y_min, 1e-6)
    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)


def stack_positions(trajectories: list[np.ndarray]) -> np.ndarray:
    if not trajectories:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate([trajectory[:, :3, 3] for trajectory in trajectories], axis=0)


def align_segment_start(predicted_poses: np.ndarray, gt_start_pose: np.ndarray) -> np.ndarray:
    if predicted_poses.shape[0] == 0:
        return predicted_poses
    anchor = gt_start_pose @ np.linalg.inv(predicted_poses[0])
    return anchor[None, :, :] @ predicted_poses


@dataclass
class SegmentRun:
    run_dir: Path
    begin_frame: int
    end_frame: int
    step_frame: int
    pose_path: Path
    trajectory_label: str
    trajectory_file: Path
    gt_poses: np.ndarray
    pred_poses: np.ndarray


def resolve_trajectory_file(run_dir: Path, trajectory_source: str) -> tuple[str, Path]:
    candidates = []
    if trajectory_source in ("auto", "slam"):
        candidates.extend([("SLAM", run_dir / "slam_poses_kitti.txt"), ("SLAM", run_dir / "slam_poses_tum.txt")])
    if trajectory_source in ("auto", "odom"):
        candidates.extend([("Odometry", run_dir / "odom_poses_kitti.txt"), ("Odometry", run_dir / "odom_poses_tum.txt")])

    for label, candidate in candidates:
        if candidate.is_file():
            return label, candidate

    raise FileNotFoundError(
        "No trajectory file found in {} for source '{}'".format(run_dir, trajectory_source)
    )


def load_segment_run(run_dir: Path, trajectory_source: str, align_start_to_gt: bool) -> SegmentRun:
    config_path = run_dir / "meta" / "config_all.yaml"
    if not config_path.is_file():
        raise FileNotFoundError("Missing config snapshot: {}".format(config_path))

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    pose_path_raw = config.get("pose_path", "")
    if not pose_path_raw:
        raise ValueError("Run {} does not contain pose_path; cannot stitch a global route".format(run_dir))
    if bool(config.get("first_frame_ref", False)):
        raise ValueError("Run {} used first_frame_ref=True; stitched global plotting requires False".format(run_dir))

    begin_frame = int(config.get("begin_frame", 0))
    end_frame = int(config.get("end_frame", 100000))
    step_frame = int(config.get("step_frame", 1))
    pose_path = resolve_existing_path(str(pose_path_raw), run_dir)
    full_gt_poses = read_pose_file(pose_path)
    gt_segment_poses = np.asarray(full_gt_poses[begin_frame:end_frame:step_frame], dtype=np.float64)
    if gt_segment_poses.shape[0] == 0:
        raise ValueError("Run {} resolved to an empty GT slice".format(run_dir))

    trajectory_label, trajectory_file = resolve_trajectory_file(run_dir, trajectory_source)
    predicted_poses = read_pose_file(trajectory_file)
    if predicted_poses.shape[0] < gt_segment_poses.shape[0]:
        raise ValueError(
            "Run {} has fewer predicted poses ({}) than GT poses ({})".format(
                run_dir, predicted_poses.shape[0], gt_segment_poses.shape[0]
            )
        )
    if predicted_poses.shape[0] > gt_segment_poses.shape[0]:
        predicted_poses = predicted_poses[: gt_segment_poses.shape[0]]
    if align_start_to_gt:
        predicted_poses = align_segment_start(predicted_poses, gt_segment_poses[0])

    return SegmentRun(
        run_dir=run_dir,
        begin_frame=begin_frame,
        end_frame=end_frame,
        step_frame=step_frame,
        pose_path=pose_path,
        trajectory_label=trajectory_label,
        trajectory_file=trajectory_file,
        gt_poses=gt_segment_poses,
        pred_poses=predicted_poses,
    )


def resolve_background_pose_path(
    segments: list[SegmentRun],
    background_pose_path: str | None,
) -> Path | None:
    if background_pose_path:
        return resolve_existing_path(background_pose_path, segments[0].run_dir)

    pose_paths = {segment.pose_path for segment in segments}
    if len(pose_paths) != 1:
        return None

    common_pose_path = next(iter(pose_paths))
    aligned_candidate = common_pose_path.parent / "aligned_poses.txt"
    if aligned_candidate.is_file():
        return aligned_candidate
    return common_pose_path


def save_full_route_path_plot(
    gt_segments: list[np.ndarray],
    pred_segments: list[np.ndarray],
    output_dir: Path,
    pred_label: str,
    background_trajectory: np.ndarray | None = None,
) -> None:
    gt_positions = [segment[:, :3, 3] for segment in gt_segments]
    pred_positions = [segment[:, :3, 3] for segment in pred_segments]
    background_positions = None if background_trajectory is None else background_trajectory[:, :3, 3]

    all_positions = [stack_positions(gt_segments), stack_positions(pred_segments)]
    if background_positions is not None:
        all_positions.append(background_positions)
    all_positions = np.concatenate(all_positions, axis=0)

    fig = plt.figure(figsize=(20, 6), dpi=110)
    axes = [fig.add_subplot(1, 3, idx + 1) for idx in range(3)]
    projections = (
        (0, 2, "x (m)", "z (m)"),
        (0, 1, "x (m)", "y (m)"),
        (1, 2, "y (m)", "z (m)"),
    )

    for axis, (x_idx, y_idx, x_label, y_label) in zip(axes, projections):
        if background_positions is not None:
            axis.plot(
                background_positions[:, x_idx],
                background_positions[:, y_idx],
                color="#888888",
                linewidth=1.0,
                label="Background GT",
            )
        for segment_idx, points in enumerate(gt_positions):
            axis.plot(points[:, x_idx], points[:, y_idx], "r-", label="GT" if segment_idx == 0 else None)
            axis.plot([points[0, x_idx]], [points[0, y_idx]], "ko", label="Start" if segment_idx == 0 else None)
        for segment_idx, points in enumerate(pred_positions):
            axis.plot(points[:, x_idx], points[:, y_idx], "b-", label=pred_label if segment_idx == 0 else None)

        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        set_equal_axis_2d(axis, all_positions[:, x_idx], all_positions[:, y_idx])
        axis.legend(loc="upper right")

    png_path = output_dir / "full_route_path.png"
    pdf_path = output_dir / "full_route_path.pdf"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def save_full_route_path_3d_plot(
    gt_segments: list[np.ndarray],
    pred_segments: list[np.ndarray],
    output_dir: Path,
    pred_label: str,
    background_trajectory: np.ndarray | None = None,
) -> None:
    gt_points = [segment[:, :3, 3] for segment in gt_segments]
    pred_points = [segment[:, :3, 3] for segment in pred_segments]
    background_points = None if background_trajectory is None else background_trajectory[:, :3, 3]

    fig = plt.figure(figsize=(8, 8), dpi=110)
    axis = fig.add_subplot(111, projection="3d")
    if background_points is not None:
        axis.plot(
            background_points[:, 0],
            background_points[:, 2],
            background_points[:, 1],
            color="#888888",
            linewidth=1.0,
            label="Background GT",
        )

    for segment_idx, points in enumerate(pred_points):
        axis.plot(points[:, 0], points[:, 2], points[:, 1], "b-", label=pred_label if segment_idx == 0 else None)
    for segment_idx, points in enumerate(gt_points):
        axis.plot(points[:, 0], points[:, 2], points[:, 1], "r-", label="GT" if segment_idx == 0 else None)
        axis.plot([points[0, 0]], [points[0, 2]], [points[0, 1]], "ko", label="Start" if segment_idx == 0 else None)

    axis.set_xlabel("x (m)")
    axis.set_ylabel("z (m)")
    axis.set_zlabel("y (m)")
    axis.view_init(elev=20.0, azim=-35.0)
    axis.legend(loc="upper right")

    point_sets = [
        np.stack([stack_positions(pred_segments)[:, 0], stack_positions(pred_segments)[:, 2], stack_positions(pred_segments)[:, 1]], axis=1),
        np.stack([stack_positions(gt_segments)[:, 0], stack_positions(gt_segments)[:, 2], stack_positions(gt_segments)[:, 1]], axis=1),
    ]
    if background_points is not None:
        point_sets.append(np.stack([background_points[:, 0], background_points[:, 2], background_points[:, 1]], axis=1))
    all_points = np.concatenate(point_sets, axis=0)
    center = np.mean(all_points, axis=0)
    radius = max(np.max(np.abs(all_points - center), axis=0).max(), 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)

    png_path = output_dir / "full_route_path_3D.png"
    pdf_path = output_dir / "full_route_path_3D.pdf"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def stitch_segment_runs(
    run_dirs: list[str],
    output_dir: str,
    trajectory_source: str = "auto",
    background_pose_path: str | None = None,
    align_start_to_gt: bool = True,
) -> dict:
    segments = [
        load_segment_run(Path(run_dir).resolve(), trajectory_source, align_start_to_gt)
        for run_dir in run_dirs
    ]
    segments.sort(key=lambda segment: (segment.pose_path.as_posix(), segment.begin_frame, segment.end_frame, segment.run_dir.as_posix()))

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    background_path = resolve_background_pose_path(segments, background_pose_path)
    background_trajectory = None if background_path is None else read_pose_file(background_path)

    gt_segments = [segment.gt_poses for segment in segments]
    pred_segments = [segment.pred_poses for segment in segments]
    pred_label = segments[0].trajectory_label

    save_full_route_path_plot(gt_segments, pred_segments, output_path, pred_label, background_trajectory)
    save_full_route_path_3d_plot(gt_segments, pred_segments, output_path, pred_label, background_trajectory)

    manifest = {
        "output_dir": str(output_path),
        "segment_count": len(segments),
        "trajectory_source": trajectory_source,
        "align_start_to_gt": bool(align_start_to_gt),
        "background_pose_path": None if background_path is None else str(background_path),
        "segments": [
            {
                "run_dir": str(segment.run_dir),
                "trajectory_file": str(segment.trajectory_file),
                "pose_path": str(segment.pose_path),
                "begin_frame": int(segment.begin_frame),
                "end_frame": int(segment.end_frame),
                "step_frame": int(segment.step_frame),
                "frame_count": int(segment.gt_poses.shape[0]),
            }
            for segment in segments
        ],
        "artifacts": {
            "full_route_path_png": "full_route_path.png",
            "full_route_path_pdf": "full_route_path.pdf",
            "full_route_path_3D_png": "full_route_path_3D.png",
            "full_route_path_3D_pdf": "full_route_path_3D.pdf",
        },
    }
    with open(output_path / "stitch_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stitch multiple PIN-SLAM segment runs into TransLO-style full-route path plots."
    )
    parser.add_argument("run_dirs", nargs="+", help="PIN-SLAM run directories to stitch")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory for stitched plots")
    parser.add_argument(
        "--trajectory-source",
        choices=("auto", "slam", "odom"),
        default="auto",
        help="Prefer SLAM or odometry trajectory files; auto prefers SLAM first",
    )
    parser.add_argument(
        "--background-poses",
        default=None,
        help="Optional KITTI/TUM pose file for the gray background route; auto-detected when omitted",
    )
    parser.add_argument(
        "--keep-pred-start",
        action="store_true",
        help="Do not re-anchor each predicted segment to the GT start pose",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = stitch_segment_runs(
        run_dirs=args.run_dirs,
        output_dir=args.output_dir,
        trajectory_source=args.trajectory_source,
        background_pose_path=args.background_poses,
        align_start_to_gt=not args.keep_pred_start,
    )
    print("Stitched {} segments".format(manifest["segment_count"]))
    print("Output:", manifest["output_dir"])
    print("2D plot:", Path(manifest["output_dir"]) / manifest["artifacts"]["full_route_path_png"])
    print("3D plot:", Path(manifest["output_dir"]) / manifest["artifacts"]["full_route_path_3D_png"])


if __name__ == "__main__":
    main()
