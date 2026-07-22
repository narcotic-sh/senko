from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

try:
    from .pack import ALIGNMENT, build_package, pack_matrix_i_o4, unpack_matrix_i_o4
except ImportError:
    from pack import ALIGNMENT, build_package, pack_matrix_i_o4, unpack_matrix_i_o4


REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL = REPO_ROOT / "web/public/models/pyannote-segmentation-3.0-tail-b8.onnx"


class LayoutTests(unittest.TestCase):
    def test_matrix_round_trip_with_classifier_padding(self) -> None:
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype):
                matrix = np.arange(5 * 7, dtype=dtype).reshape(5, 7)
                packed, shape = pack_matrix_i_o4(matrix)
                self.assertEqual(shape, [5, 2, 4])
                np.testing.assert_array_equal(
                    unpack_matrix_i_o4(packed, matrix.shape), matrix
                )
                self.assertEqual(np.count_nonzero(packed[:, 1, 3]), 0)


@unittest.skipUnless(MODEL.exists(), "tail ONNX artifact is unavailable")
class RealModelTests(unittest.TestCase):
    def test_b8_package_is_complete_reproducible_and_small(self) -> None:
        for dtype, maximum_bytes in ((np.float16, 104 * 1024), (np.float32, 204 * 1024)):
            with self.subTest(dtype=dtype):
                first_binary, first = build_package(MODEL, "tail.bin", dtype)
                second_binary, second = build_package(MODEL, "tail.bin", dtype)
                self.assertEqual(first_binary, second_binary)
                self.assertEqual(first, second)
                self.assertEqual(first["binary"]["section_count"], 6)
                self.assertEqual(first["compute"]["macs_per_batch_item"], 29_478_272)
                self.assertEqual(first["compute"]["macs_full_batch"], 235_826_176)
                self.assertLess(first["binary"]["byte_length"], maximum_bytes)
                self.assertEqual(
                    first["contract"]["weight_dtype"],
                    "float16" if dtype == np.float16 else "float32",
                )
                for section in first["sections"]:
                    self.assertEqual(section["byte_offset"] % ALIGNMENT, 0)


if __name__ == "__main__":
    unittest.main()
