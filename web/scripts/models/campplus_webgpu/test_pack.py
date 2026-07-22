from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pack import (
    BinaryBuilder,
    FORMAT_MAGIC,
    HEADER_BYTES,
    SECTION_ALIGNMENT,
    align_up,
    apply_compiled_batch_norm,
    build_package,
    compile_batch_norm_affine,
    pack_conv_oihw4,
    parse_header,
    sha256_bytes,
    unpack_conv_oihw4,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_MODEL = REPO_ROOT / "web/public/models/campplus-t150-b32-fp16.onnx"


class LayoutTests(unittest.TestCase):
    def test_align_up_rejects_non_power_of_two(self) -> None:
        self.assertEqual(align_up(0), 0)
        self.assertEqual(align_up(1), 256)
        self.assertEqual(align_up(256), 256)
        self.assertEqual(align_up(257), 512)
        with self.assertRaises(ValueError):
            align_up(4, 12)

    def test_conv_vec4_layout_round_trips_padding(self) -> None:
        logical = np.arange(5 * 3 * 2, dtype=np.float16).reshape(5, 3, 2)
        packed, packed_shape = pack_conv_oihw4(logical)
        self.assertEqual(packed_shape, [2, 2, 1, 4, 4])
        for output_channel in range(5):
            for input_channel in range(3):
                for kernel in range(2):
                    self.assertEqual(
                        packed[
                            kernel,
                            output_channel // 4,
                            input_channel // 4,
                            input_channel % 4,
                            output_channel % 4,
                        ],
                        logical[output_channel, input_channel, kernel],
                    )
        np.testing.assert_array_equal(unpack_conv_oihw4(packed, logical.shape), logical)
        self.assertEqual(np.count_nonzero(packed) - np.count_nonzero(logical), 0)

    def test_batch_norm_affine_matches_inference_equation(self) -> None:
        rng = np.random.default_rng(7)
        channels = 7
        gamma = rng.normal(size=channels).astype(np.float16)
        beta = rng.normal(size=channels).astype(np.float16)
        mean = rng.normal(size=channels).astype(np.float16)
        variance = np.exp(rng.normal(size=channels)).astype(np.float16)
        values = rng.normal(size=(3, 5, channels)).astype(np.float16)
        epsilon = 1e-5
        packed, actual_channels = compile_batch_norm_affine(
            gamma, beta, mean, variance, epsilon
        )
        self.assertEqual(actual_channels, channels)
        self.assertEqual(list(packed.shape), [2, 2, 4])
        self.assertEqual(packed.dtype, np.float32)
        expected = (
            gamma.astype(np.float32)
            * (values.astype(np.float32) - mean.astype(np.float32))
            / np.sqrt(variance.astype(np.float32) + epsilon)
            + beta.astype(np.float32)
        )
        actual = apply_compiled_batch_norm(values, packed, channels)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_binary_header_and_sections_are_deterministic_and_aligned(self) -> None:
        builder = BinaryBuilder.create()
        builder.add(
            section_id="test",
            kind="conv_bias",
            array=np.array([1, 2, 3, 4], dtype=np.float16),
            logical_shape=[4],
            packed_shape=[1, 4],
            layout="O4",
            source_tensors=["bias"],
        )
        kwargs = {
            "source_sha256": "ab" * 32,
            "batch": 32,
            "frames": 150,
            "feature_dim": 80,
            "embedding_dim": 192,
        }
        first = builder.finish(**kwargs)
        header = parse_header(first)
        self.assertEqual(first[:8], FORMAT_MAGIC)
        self.assertEqual(header["header_bytes"], HEADER_BYTES)
        self.assertEqual(header["section_alignment"], SECTION_ALIGNMENT)
        self.assertEqual(header["section_count"], 1)
        self.assertEqual(header["total_bytes"], len(first))
        self.assertEqual(header["payload_sha256"], sha256_bytes(first[HEADER_BYTES:]))
        self.assertEqual(builder.sections[0]["byte_offset"] % SECTION_ALIGNMENT, 0)


@unittest.skipUnless(REAL_MODEL.exists(), "ignored CAM++ artifact has not been exported")
class RealModelContractTests(unittest.TestCase):
    def test_real_model_pack_is_complete_and_reproducible(self) -> None:
        first_binary, first_metadata = build_package(REAL_MODEL, "campplus.bin")
        second_binary, second_metadata = build_package(REAL_MODEL, "campplus.bin")
        self.assertEqual(first_binary, second_binary)
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_metadata["inventory"]["onnx_nodes"], 845)
        self.assertEqual(first_metadata["inventory"]["lowered_convolutions"], 225)
        self.assertEqual(first_metadata["inventory"]["compiled_batch_norms"], 56)
        self.assertEqual(first_metadata["inventory"]["dense_cam_layers"], 52)
        self.assertEqual(first_metadata["binary"]["section_count"], 506)
        self.assertLess(
            first_metadata["binary"]["byte_length"],
            first_metadata["source"]["byte_length"],
        )
        self.assertLess(
            first_metadata["binary"]["byte_length"]
            - first_metadata["inventory"]["source_initializer_bytes"],
            64 * 1024,
        )
        section_ends = []
        for section in first_metadata["sections"]:
            self.assertEqual(section["byte_offset"] % SECTION_ALIGNMENT, 0)
            section_ends.append(section["byte_offset"] + section["byte_length"])
        self.assertLessEqual(max(section_ends), len(first_binary))
        encoded = json.dumps(first_metadata, sort_keys=True)
        self.assertNotIn(str(REPO_ROOT), encoded)


if __name__ == "__main__":
    unittest.main()
