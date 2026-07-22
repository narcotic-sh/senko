from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

try:
    from .pack import (
        BinaryBuilder,
        FORMAT_MAGIC,
        HEADER_BYTES,
        SECTION_ALIGNMENT,
        align_up,
        build_package,
        pack_conv_ki_o4,
        pack_instance_norm_affine,
        parse_header,
        sha256_bytes,
        unpack_conv_ki_o4,
    )
except ImportError:  # Support direct execution from this directory.
    from pack import (  # type: ignore[no-redef]
        BinaryBuilder,
        FORMAT_MAGIC,
        HEADER_BYTES,
        SECTION_ALIGNMENT,
        align_up,
        build_package,
        pack_conv_ki_o4,
        pack_instance_norm_affine,
        parse_header,
        sha256_bytes,
        unpack_conv_ki_o4,
    )


REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_MODEL_CANDIDATES = (
    REPO_ROOT / "web/public/models/pyannote-segmentation-3.0-frontend-b8.onnx",
    REPO_ROOT
    / ".research/.typed-umap-dist/models/pyannote-segmentation-3.0-frontend-b8.onnx",
)
REAL_MODEL = next((path for path in REAL_MODEL_CANDIDATES if path.exists()), REAL_MODEL_CANDIDATES[0])


def _instance_norm(
    value: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float
) -> np.ndarray:
    mean = value.mean(axis=2, keepdims=True, dtype=np.float32)
    centered = value - mean
    variance = np.mean(centered * centered, axis=2, keepdims=True, dtype=np.float32)
    return (
        centered
        / np.sqrt(variance + np.float32(epsilon))
        * gamma.reshape(1, -1, 1)
        + beta.reshape(1, -1, 1)
    ).astype(np.float32)


def _fused_sinc_reference(
    waveform: np.ndarray,
    packed_weight: np.ndarray,
    logical_weight_shape: tuple[int, int, int],
    gamma: np.ndarray,
    beta: np.ndarray,
    *,
    epsilon: float,
    stride: int,
    pool: int,
) -> np.ndarray:
    normalized = _instance_norm(waveform, gamma, beta, epsilon)
    weight = unpack_conv_ki_o4(packed_weight, logical_weight_shape)
    kernel = logical_weight_shape[2]
    windows = np.lib.stride_tricks.sliding_window_view(
        normalized, kernel, axis=2
    )[:, :, ::stride, :]
    convolved = np.einsum("bitk,oik->bot", windows, weight, optimize=True)
    absolute = np.abs(convolved)
    pooled_frames = absolute.shape[2] // pool
    return absolute[:, :, : pooled_frames * pool].reshape(
        absolute.shape[0], absolute.shape[1], pooled_frames, pool
    ).max(axis=3)


def _synthetic_sinc_onnx(
    waveform_shape: tuple[int, int, int],
    weight: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    *,
    epsilon: float,
    stride: int,
    pool: int,
) -> bytes:
    batch, channels, samples = waveform_shape
    output_channels, _, kernel = weight.shape
    conv_frames = (samples - kernel) // stride + 1
    pool_frames = (conv_frames - pool) // pool + 1
    graph = helper.make_graph(
        [
            helper.make_node(
                "InstanceNormalization",
                ["waveform", "gamma", "beta"],
                ["normalized"],
                epsilon=epsilon,
            ),
            helper.make_node(
                "Conv",
                ["normalized", "weight"],
                ["convolved"],
                kernel_shape=[kernel],
                strides=[stride],
            ),
            helper.make_node("Abs", ["convolved"], ["absolute"]),
            helper.make_node(
                "MaxPool",
                ["absolute"],
                ["pooled"],
                kernel_shape=[pool],
                strides=[pool],
            ),
        ],
        "synthetic-sinc-stage",
        [helper.make_tensor_value_info("waveform", TensorProto.FLOAT, waveform_shape)],
        [
            helper.make_tensor_value_info(
                "pooled", TensorProto.FLOAT, [batch, output_channels, pool_frames]
            )
        ],
        [
            numpy_helper.from_array(gamma, "gamma"),
            numpy_helper.from_array(beta, "beta"),
            numpy_helper.from_array(weight, "weight"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=8,
    )
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


class LayoutTests(unittest.TestCase):
    def test_align_up_rejects_non_power_of_two(self) -> None:
        self.assertEqual(align_up(0), 0)
        self.assertEqual(align_up(1), 256)
        self.assertEqual(align_up(257), 512)
        with self.assertRaises(ValueError):
            align_up(4, 12)

    def test_conv_layout_round_trips_output_padding(self) -> None:
        logical = np.arange(5 * 3 * 7, dtype=np.float32).reshape(5, 3, 7)
        packed, packed_shape = pack_conv_ki_o4(logical)
        self.assertEqual(packed_shape, [7, 3, 2, 4])
        for output_channel in range(5):
            for input_channel in range(3):
                for kernel in range(7):
                    self.assertEqual(
                        packed[
                            kernel,
                            input_channel,
                            output_channel // 4,
                            output_channel % 4,
                        ],
                        logical[output_channel, input_channel, kernel],
                    )
        np.testing.assert_array_equal(
            unpack_conv_ki_o4(packed, logical.shape), logical
        )

    def test_instance_norm_affine_layout(self) -> None:
        gamma = np.arange(1, 7, dtype=np.float32)
        beta = -gamma
        packed = pack_instance_norm_affine(gamma, beta)
        self.assertEqual(packed.shape, (2, 2, 4))
        np.testing.assert_array_equal(packed[:, 0].reshape(-1)[:6], gamma)
        np.testing.assert_array_equal(packed[:, 1].reshape(-1)[:6], beta)

    def test_binary_header_is_deterministic_and_aligned(self) -> None:
        builder = BinaryBuilder.create()
        builder.add(
            section_id="test",
            kind="conv_bias",
            array=np.array([[1, 2, 3, 4]], dtype=np.float32),
            logical_shape=[4],
            packed_shape=[1, 4],
            layout="O4",
            source_tensors=["bias"],
        )
        binary = builder.finish(source_sha256="ab" * 32, batch=8)
        header = parse_header(binary)
        self.assertEqual(binary[:8], FORMAT_MAGIC)
        self.assertEqual(header["header_bytes"], HEADER_BYTES)
        self.assertEqual(header["section_alignment"], SECTION_ALIGNMENT)
        self.assertEqual(header["section_count"], 1)
        self.assertEqual(header["total_bytes"], len(binary))
        self.assertEqual(header["payload_sha256"], sha256_bytes(binary[HEADER_BYTES:]))
        self.assertEqual(builder.sections[0]["byte_offset"] % SECTION_ALIGNMENT, 0)


class FusedSincParityTests(unittest.TestCase):
    def test_packed_cpu_fusion_matches_onnx(self) -> None:
        rng = np.random.default_rng(0x5E4B0)
        waveform = rng.normal(0.0, 0.2, size=(2, 1, 83)).astype(np.float32)
        weight = rng.normal(0.0, 0.1, size=(7, 1, 11)).astype(np.float32)
        gamma = np.array([1.125], dtype=np.float32)
        beta = np.array([-0.03125], dtype=np.float32)
        epsilon = 1e-5
        packed, _ = pack_conv_ki_o4(weight)
        expected = ort.InferenceSession(
            _synthetic_sinc_onnx(
                waveform.shape,
                weight,
                gamma,
                beta,
                epsilon=epsilon,
                stride=2,
                pool=3,
            ),
            providers=["CPUExecutionProvider"],
        ).run(["pooled"], {"waveform": waveform})[0]
        actual = _fused_sinc_reference(
            waveform,
            packed,
            weight.shape,
            gamma,
            beta,
            epsilon=epsilon,
            stride=2,
            pool=3,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@unittest.skipUnless(REAL_MODEL.exists(), "ignored pyannote frontend artifact is unavailable")
class RealModelContractTests(unittest.TestCase):
    def test_real_b8_package_is_complete_reproducible_and_memory_bounded(self) -> None:
        first_binary, first_metadata = build_package(REAL_MODEL, "frontend.bin")
        second_binary, second_metadata = build_package(REAL_MODEL, "frontend.bin")
        self.assertEqual(first_binary, second_binary)
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_metadata["inventory"]["onnx_nodes"], 15)
        self.assertEqual(first_metadata["inventory"]["source_initializers"], 13)
        self.assertEqual(first_metadata["inventory"]["lowered_convolutions"], 3)
        self.assertEqual(first_metadata["inventory"]["lowered_instance_normalizations"], 4)
        self.assertEqual(first_metadata["binary"]["section_count"], 10)
        self.assertEqual(first_metadata["compute"]["macs_per_batch_item"], 480_324_000)
        self.assertEqual(first_metadata["compute"]["macs_full_batch"], 3_842_592_000)
        plan = first_metadata["memory"]["planned_webgpu"]
        self.assertEqual(plan["unfused_first_conv_activation_bytes"], 40_896_000)
        self.assertEqual(plan["fused_first_pool_activation_bytes"], 13_632_000)
        self.assertEqual(plan["saved_first_stage_activation_bytes"], 27_264_000)
        self.assertLess(plan["aliased_arena"]["activation_arena_bytes"], 19 * 1024 * 1024)
        self.assertLess(first_metadata["binary"]["byte_length"], 256 * 1024)
        for section in first_metadata["sections"]:
            self.assertEqual(section["byte_offset"] % SECTION_ALIGNMENT, 0)


if __name__ == "__main__":
    unittest.main()
