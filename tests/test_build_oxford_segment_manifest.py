import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "eval" / "build_oxford_segment_manifest.py"
MODULE_SPEC = importlib.util.spec_from_file_location("build_oxford_segment_manifest", MODULE_PATH)
build_oxford_segment_manifest = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(build_oxford_segment_manifest)


def write_timestamps(path: Path, timestamps: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(timestamps, dtype=np.int64), fmt="%d")


class BuildOxfordSegmentManifestTests(unittest.TestCase):
    def test_build_segment_manifest_splits_on_aligned_index_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            selected_path = temp_path / "timestamps.txt"
            aligned_path = temp_path / "aligned_timestamps.txt"
            output_path = temp_path / "segments.json"
            write_timestamps(selected_path, np.asarray([20, 30, 60, 80, 90, 120], dtype=np.int64))
            write_timestamps(aligned_path, np.asarray([20, 30, 40, 50, 60, 80, 90, 110, 120], dtype=np.int64))

            manifest = build_oxford_segment_manifest.build_segment_manifest(selected_path, aligned_path)
            self.assertEqual(manifest["segment_count"], 3)
            self.assertEqual(manifest["runnable_segment_count"], 2)
            self.assertEqual(manifest["skipped_segment_count"], 1)
            self.assertEqual(
                [(segment["begin_frame"], segment["end_frame"], segment["frame_count"], segment["runnable"]) for segment in manifest["segments"]],
                [(0, 2, 2, True), (2, 5, 3, True), (5, 6, 1, False)],
            )

            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
            self.assertTrue(output_path.is_file())

    def test_build_segment_manifest_rejects_missing_selected_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            selected_path = temp_path / "timestamps.txt"
            aligned_path = temp_path / "aligned_timestamps.txt"
            write_timestamps(selected_path, np.asarray([20, 30, 999], dtype=np.int64))
            write_timestamps(aligned_path, np.asarray([20, 30, 40], dtype=np.int64))

            with self.assertRaises(ValueError):
                build_oxford_segment_manifest.build_segment_manifest(selected_path, aligned_path)


if __name__ == "__main__":
    unittest.main()
