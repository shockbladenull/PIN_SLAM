import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d
from natsort import natsorted


MODULE_PATH = Path(__file__).resolve().parents[1] / "dataset" / "converter" / "oxford_to_pin_format.py"
MODULE_SPEC = importlib.util.spec_from_file_location("oxford_to_pin_format", MODULE_PATH)
oxford_to_pin_format = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(oxford_to_pin_format)


def write_h5(path: Path, timestamps: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("valid_timestamps", data=np.asarray(timestamps, dtype=np.int64))


def write_oxford_scan(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
    raw = np.concatenate([points.astype(np.float32), intensity], axis=1).transpose()
    raw.tofile(path)


class OxfordToPinFormatTests(unittest.TestCase):
    def test_resolve_sequence_file_falls_back_to_short_name(self) -> None:
        sequence_name = "2019-01-14-12-05-52-radar-oxford-10k"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            short_dir = temp_path / "0552"
            short_dir.mkdir(parents=True)
            target = short_dir / "velodyne_left_calibrateFalse_SCR300m.h5"
            target.write_bytes(b"")

            resolved = oxford_to_pin_format.resolve_oxford_sequence_file(
                sequence_name=sequence_name,
                seq_dir=str(temp_path / sequence_name),
                filename="velodyne_left_calibrateFalse_SCR300m.h5",
                root_override=str(temp_path),
            )

            self.assertEqual(resolved, str(target))

    def test_convert_sequence_exports_selected_ply_and_poses(self) -> None:
        sequence_name = "2019-01-14-12-05-52-radar-oxford-10k"
        sequence_short = oxford_to_pin_format.oxford_sequence_short_name(sequence_name)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            oxford_root = temp_path / "Oxford"
            h5_root = temp_path / "h5"
            pose_root = temp_path / "poses"
            output_root = temp_path / "output"

            scan_dir = oxford_root / sequence_name / "velodyne_left"
            scan_points = {
                20: np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
                40: np.asarray([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=np.float32),
                50: np.asarray([[7.0, 8.0, 9.0], [9.0, 8.0, 7.0]], dtype=np.float32),
            }
            for timestamp, points in scan_points.items():
                write_oxford_scan(scan_dir / "{}.bin".format(timestamp), points)

            full_timestamps = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.int64)
            mask_timestamps = np.asarray([10, 20, 40, 50, 60], dtype=np.int64)
            write_h5(h5_root / sequence_short / "velodyne_left_calibrateFalse.h5", full_timestamps)
            write_h5(h5_root / sequence_short / "velodyne_left_calibrateFalse_SCR300m.h5", mask_timestamps)

            pose_dir = pose_root / "Oxford_SLAM_result_{}".format(sequence_short)
            pose_dir.mkdir(parents=True, exist_ok=True)
            pose_rows = np.asarray(
                [
                    [1, 0, 0, 20, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 30, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 40, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 50, 0, 1, 0, 0, 0, 0, 1, 0],
                ],
                dtype=np.float32,
            )
            np.savetxt(
                pose_dir / "gicp_Oxford{}_050_v1.txt".format(sequence_short),
                pose_rows,
                fmt="%.1f",
            )

            conversion = oxford_to_pin_format.convert_sequence(
                oxford_root=str(oxford_root),
                sequence_name=sequence_name,
                output_root=str(output_root),
                mask_h5_name="velodyne_left_calibrateFalse_SCR300m.h5",
                mask_h5_root=str(h5_root),
                full_h5_name="velodyne_left_calibrateFalse.h5",
                full_h5_root=str(h5_root),
                pose_root=str(pose_root),
                pose_skip_start=1,
                pose_skip_end=1,
            )

            pointcloud_dir = Path(conversion["pointcloud_dir"])
            self.assertEqual(natsorted(os.listdir(pointcloud_dir)), ["20.ply", "40.ply", "50.ply"])
            self.assertEqual(sorted(os.listdir(pointcloud_dir)), ["20.ply", "40.ply", "50.ply"])

            exported_poses = np.loadtxt(Path(conversion["pose_output_path"]), dtype=np.float32)
            expected_poses = pose_rows[[0, 2, 3]]
            np.testing.assert_allclose(exported_poses, expected_poses)

            exported_timestamps = np.loadtxt(Path(conversion["timestamps_output_path"]), dtype=np.int64)
            np.testing.assert_array_equal(exported_timestamps, np.asarray([20, 40, 50], dtype=np.int64))

            exported_cloud = o3d.io.read_point_cloud(str(pointcloud_dir / "20.ply"))
            exported_points = np.asarray(exported_cloud.points)
            expected_points = scan_points[20].copy().astype(np.float64)
            expected_points[:, 2] *= -1.0
            np.testing.assert_allclose(exported_points, expected_points)

            self.assertEqual(sorted(os.listdir(pointcloud_dir)), ["20.ply", "40.ply", "50.ply"])
            self.assertTrue((Path(conversion["sequence_output_dir"]) / "timestamps.txt").is_file())
            self.assertNotIn("timestamps.txt", os.listdir(pointcloud_dir))


if __name__ == "__main__":
    unittest.main()
