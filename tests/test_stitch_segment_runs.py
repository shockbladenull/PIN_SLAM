import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "eval" / "stitch_segment_runs.py"
MODULE_SPEC = importlib.util.spec_from_file_location("stitch_segment_runs", MODULE_PATH)
stitch_segment_runs = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(stitch_segment_runs)


def make_pose(tx: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = tx
    return pose


def write_kitti_poses(path: Path, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, poses[:, :3, :].reshape(poses.shape[0], 12), fmt="%.6f")


def write_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


class StitchSegmentRunsTests(unittest.TestCase):
    def test_stitch_segment_runs_writes_full_route_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sequence_dir = temp_path / "sequence"
            full_pose_path = sequence_dir / "poses.txt"
            aligned_pose_path = sequence_dir / "aligned_poses.txt"

            full_poses = np.asarray([make_pose(float(x)) for x in range(6)], dtype=np.float64)
            aligned_poses = np.asarray([make_pose(float(x) - 0.25) for x in range(6)], dtype=np.float64)
            write_kitti_poses(full_pose_path, full_poses)
            write_kitti_poses(aligned_pose_path, aligned_poses)

            run_a = temp_path / "run_a"
            write_config(
                run_a / "meta" / "config_all.yaml",
                {
                    "pose_path": str(full_pose_path),
                    "begin_frame": 0,
                    "end_frame": 3,
                    "step_frame": 1,
                    "first_frame_ref": False,
                },
            )
            pred_a = np.asarray([make_pose(0.5), make_pose(1.5), make_pose(2.5), make_pose(999.0)], dtype=np.float64)
            write_kitti_poses(run_a / "slam_poses_kitti.txt", pred_a)

            run_b = temp_path / "run_b"
            write_config(
                run_b / "meta" / "config_all.yaml",
                {
                    "pose_path": str(full_pose_path),
                    "begin_frame": 3,
                    "end_frame": 6,
                    "step_frame": 1,
                    "first_frame_ref": False,
                },
            )
            pred_b = np.asarray([make_pose(3.5), make_pose(4.5), make_pose(5.5), make_pose(999.0)], dtype=np.float64)
            write_kitti_poses(run_b / "slam_poses_kitti.txt", pred_b)

            output_dir = temp_path / "stitched"
            manifest = stitch_segment_runs.stitch_segment_runs(
                run_dirs=[str(run_b), str(run_a)],
                output_dir=str(output_dir),
            )

            self.assertEqual(manifest["segment_count"], 2)
            self.assertEqual(Path(manifest["background_pose_path"]), aligned_pose_path)
            self.assertTrue((output_dir / "full_route_path.png").is_file())
            self.assertTrue((output_dir / "full_route_path.pdf").is_file())
            self.assertTrue((output_dir / "full_route_path_3D.png").is_file())
            self.assertTrue((output_dir / "full_route_path_3D.pdf").is_file())
            self.assertTrue((output_dir / "stitch_manifest.json").is_file())
            self.assertEqual(manifest["segments"][0]["begin_frame"], 0)
            self.assertEqual(manifest["segments"][1]["begin_frame"], 3)


if __name__ == "__main__":
    unittest.main()
