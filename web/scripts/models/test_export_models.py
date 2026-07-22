from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


SCRIPT_PATH = Path(__file__).with_name("export_models.py")
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("export_models", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
export_models = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = export_models
SPEC.loader.exec_module(export_models)

PUBLIC_MODELS = export_models.REPO_ROOT / "web" / "public" / "models"
DIRECT_FRONTEND = PUBLIC_MODELS / "pyannote-segmentation-3.0-frontend-b8.onnx"
DIRECT_TAIL = PUBLIC_MODELS / "pyannote-segmentation-3.0-tail-b8.onnx"
DIRECT_CAM_FP16 = PUBLIC_MODELS / "campplus-t150-b32-fp16.onnx"
DIRECT_CAM_FP32 = PUBLIC_MODELS / "campplus-t150-b32.onnx"


class ExportModelToolsTest(unittest.TestCase):
    def test_lstm_package_preserves_pytorch_weights_in_gpu_tiled_layout(self) -> None:
        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=60,
                    hidden_size=128,
                    num_layers=4,
                    batch_first=True,
                    bidirectional=True,
                )

        torch.manual_seed(17)
        model = FakeModel()
        payload, metadata = export_models.pack_pyannote_lstm(model)
        self.assertEqual(len(payload), 5_521_408)
        self.assertEqual(metadata["gate_order"], ["input", "forget", "cell", "output"])
        self.assertEqual(len(metadata["layers"]), 4)

        first = metadata["layers"][0]["directions"][0]
        self.assertEqual(first["direction"], "forward")
        self.assertEqual(first["tensors"]["matrix"]["shape"], [512, 188])
        self.assertEqual(
            first["tensors"]["matrix"]["packed_shape"], [4, 47, 128, 4]
        )
        self.assertEqual(
            first["tensors"]["matrix"]["layout"],
            "gate-column4-hidden-input4",
        )
        for layer in metadata["layers"]:
            for direction in layer["directions"]:
                for tensor in direction["tensors"].values():
                    self.assertEqual(tensor["offset_bytes"] % 256, 0)

        matrix_record = first["tensors"]["matrix"]
        matrix = torch.frombuffer(
            bytearray(payload),
            dtype=torch.float32,
            count=512 * 188,
            offset=matrix_record["offset_bytes"],
        ).reshape(4, 47, 128, 4).permute(0, 2, 1, 3).reshape(512, 188)
        expected = torch.cat(
            (model.lstm.weight_ih_l0, model.lstm.weight_hh_l0),
            dim=1,
        )
        torch.testing.assert_close(matrix, expected, rtol=0.0, atol=0.0)

    def test_fp16_lstm_package_is_deterministic_tiled_and_half_size(self) -> None:
        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=60,
                    hidden_size=128,
                    num_layers=4,
                    batch_first=True,
                    bidirectional=True,
                )

        torch.manual_seed(17)
        model = FakeModel()
        payload, metadata = export_models.pack_pyannote_lstm(model, "float16")
        repeated_payload, repeated_metadata = export_models.pack_pyannote_lstm(
            model, "float16"
        )

        self.assertEqual(payload, repeated_payload)
        self.assertEqual(metadata, repeated_metadata)
        self.assertEqual(len(payload), 2_760_704)
        self.assertEqual(metadata["version"], 3)
        self.assertEqual(metadata["format"], "senko-persistent-lstm-f16-gc4h")
        self.assertEqual(metadata["storage_dtype"], "float16")
        self.assertEqual(metadata["accumulator_dtype"], "float32")
        self.assertEqual(metadata["required_webgpu_features"], ["shader-f16"])
        self.assertEqual(
            metadata["weights"]["sha256"], hashlib.sha256(payload).hexdigest()
        )

        first = metadata["layers"][0]["directions"][0]
        matrix_record = first["tensors"]["matrix"]
        self.assertEqual(matrix_record["dtype"], "float16-le")
        matrix = torch.frombuffer(
            bytearray(payload),
            dtype=torch.float16,
            count=512 * 188,
            offset=matrix_record["offset_bytes"],
        ).reshape(4, 47, 128, 4).permute(0, 2, 1, 3).reshape(512, 188)
        expected = torch.cat(
            (model.lstm.weight_ih_l0, model.lstm.weight_hh_l0),
            dim=1,
        ).to(torch.float16)
        torch.testing.assert_close(matrix, expected, rtol=0.0, atol=0.0)

        for layer in metadata["layers"]:
            for direction in layer["directions"]:
                for tensor in direction["tensors"].values():
                    self.assertEqual(tensor["dtype"], "float16-le")
                    self.assertEqual(tensor["offset_bytes"] % 256, 0)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            record = export_models.write_or_verify_pyannote_lstm_package(
                model, output_dir, False, "float16"
            )
            self.assertEqual(record["weights"]["bytes"], 2_760_704)
            export_models.write_or_verify_pyannote_lstm_package(
                model, output_dir, True, "float16"
            )

    def test_lstm_package_rejects_unknown_precision(self) -> None:
        model = nn.Module()
        with self.assertRaisesRegex(ValueError, "Unsupported.*precision"):
            export_models.pack_pyannote_lstm(model, "float8")

    @unittest.skipUnless(
        DIRECT_FRONTEND.is_file() and DIRECT_TAIL.is_file(),
        "direct VAD split ONNX artifacts are unavailable",
    )
    def test_direct_vad_export_emits_both_precision_variants(self) -> None:
        packages = {
            "float16": {
                "format": "senko-persistent-lstm-f16-gc4h",
                "weights": {"bytes": 2_760_704},
            },
            "float32": {
                "format": "senko-persistent-lstm-f32-gc4h",
                "weights": {"bytes": 5_521_408},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            shutil.copyfile(DIRECT_FRONTEND, output_dir / DIRECT_FRONTEND.name)
            shutil.copyfile(DIRECT_TAIL, output_dir / DIRECT_TAIL.name)
            result = export_models.write_or_verify_direct_webgpu_vad(
                output_dir,
                False,
                packages,
            )
            variants = result["precision_variants"]
            self.assertEqual(set(variants), {"float16", "float32"})
            self.assertEqual(
                variants["float16"]["batches"]["8"]["explicit_gpu_bytes"],
                44_145_664,
            )
            self.assertEqual(
                variants["float32"]["batches"]["8"]["explicit_gpu_bytes"],
                53_947_648,
            )
            self.assertTrue(
                (output_dir / "pyannote-segmentation-3.0-tail-webgpu-f32.bin").is_file()
            )
            self.assertEqual(
                export_models.write_or_verify_direct_webgpu_vad(
                    output_dir,
                    True,
                    packages,
                ),
                result,
            )

    def test_partial_export_preserves_current_direct_vad_precision_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            variants = {}
            for precision, suffix in (("float16", "f16"), ("float32", "f32")):
                weights = f"lstm-{suffix}.bin"
                metadata = f"lstm-{suffix}.json"
                frontend = f"frontend-{suffix}.json"
                tail = f"tail-{suffix}.json"
                for name in (weights, metadata, frontend, tail):
                    (output_dir / name).write_bytes(b"present")
                variants[precision] = {
                    "lstm": {
                        "weights": {"file": weights},
                        "metadata": {"file": metadata},
                    },
                    "batches": {
                        "8": {
                            "frontend_metadata": {"file": frontend},
                            "tail_metadata": {"file": tail},
                        }
                    },
                }
            old_split = {
                "version": 1,
                "frontend": {"batches": {}},
                "tail": {"batches": {}},
                "buffer_bytes_by_batch": {},
                "direct_webgpu": {"precision_variants": variants},
            }
            (output_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "models": {"segmentation": {"split": old_split}},
                    }
                )
            )
            new_split = {
                "version": 1,
                "frontend": {"batches": {}},
                "tail": {"batches": {}},
                "buffer_bytes_by_batch": {},
                "direct_webgpu": {"precision_variants": {}},
            }
            merged = export_models.merge_segmentation_split(output_dir, new_split)
            self.assertIsNotNone(merged)
            self.assertEqual(
                set(merged["direct_webgpu"]["precision_variants"]),
                {"float16", "float32"},
            )

    def test_partial_export_rejects_an_incomplete_direct_vad_precision_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            new_split = {
                "version": 1,
                "frontend": {"batches": {}},
                "tail": {"batches": {}},
                "buffer_bytes_by_batch": {},
                "direct_webgpu": {
                    "precision_variants": {"float16": {"lstm": {}, "batches": {}}}
                },
            }
            with self.assertRaisesRegex(RuntimeError, "missing: float32"):
                export_models.merge_segmentation_split(output_dir, new_split)

    def test_campplus_only_export_rejects_dangling_direct_vad_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            variants = {}
            for precision, suffix in (("float16", "f16"), ("float32", "f32")):
                weights = f"lstm-{suffix}.bin"
                metadata = f"lstm-{suffix}.json"
                frontend = f"frontend-{suffix}.json"
                tail = f"tail-{suffix}.json"
                for name in (weights, metadata, frontend, tail):
                    (output_dir / name).write_bytes(b"present")
                variants[precision] = {
                    "lstm": {
                        "weights": {"file": weights},
                        "metadata": {"file": metadata},
                    },
                    "batches": {
                        "8": {
                            "frontend_metadata": {"file": frontend},
                            "tail_metadata": {"file": tail},
                        }
                    },
                }
            (output_dir / "tail-f32.json").unlink()
            split = {
                "version": 1,
                "direct_webgpu": {"precision_variants": variants},
            }
            (output_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "models": {"segmentation": {"split": split}},
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "float32 tail metadata"):
                export_models.merge_segmentation_split(output_dir, None)

    @unittest.skipUnless(
        DIRECT_CAM_FP16.is_file() and DIRECT_CAM_FP32.is_file(),
        "direct CAM++ source artifacts are unavailable",
    )
    def test_direct_campplus_export_emits_both_precision_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            shutil.copyfile(DIRECT_CAM_FP16, output_dir / DIRECT_CAM_FP16.name)
            shutil.copyfile(DIRECT_CAM_FP32, output_dir / DIRECT_CAM_FP32.name)
            result = export_models.write_or_verify_direct_webgpu_campplus(
                output_dir,
                False,
            )
            variants = result["precision_variants"]
            self.assertEqual(set(variants), {"float16", "float32"})
            self.assertEqual(
                variants["float16"]["explicit_gpu_buffer_bytes_by_batch"]["16"],
                39_855_360,
            )
            self.assertEqual(
                variants["float32"]["explicit_gpu_buffer_bytes_by_batch"]["16"],
                78_587_392,
            )
            self.assertEqual(variants["float32"]["weights"]["bytes"], 27_394_048)
            self.assertEqual(
                export_models.write_or_verify_direct_webgpu_campplus(
                    output_dir,
                    True,
                ),
                result,
            )

    def test_partial_export_preserves_current_direct_campplus_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            variants = {}
            for precision, suffix in (("float16", "f16"), ("float32", "f32")):
                metadata = f"camp-{suffix}.json"
                weights = f"camp-{suffix}.bin"
                (output_dir / metadata).write_bytes(b"metadata")
                (output_dir / weights).write_bytes(b"weights")
                variants[precision] = {
                    "format": f"senko-campplus-direct-webgpu-{suffix}-v1",
                    "metadata": {"file": metadata},
                    "weights": {"file": weights},
                }
            direct = {"precision_variants": variants}
            (output_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "models": {"campplus": {"direct_webgpu": direct}},
                    }
                )
            )
            self.assertEqual(
                export_models.load_existing_campplus_direct(output_dir),
                direct,
            )

    def test_freeze_pyannote_sincnet_filterbank_is_exact(self) -> None:
        class FakeEncoder(nn.Module):
            as_conv1d = True
            is_pinv = False
            stride = 2
            padding = 0

            def __init__(self) -> None:
                super().__init__()
                values = torch.arange(80 * 251, dtype=torch.float32).reshape(80, 1, 251)
                self.register_buffer("filters", values / values.numel())

            def get_filters(self) -> torch.Tensor:
                return self.filters

            def forward(self, waveform: torch.Tensor) -> torch.Tensor:
                return F.conv1d(
                    waveform,
                    self.get_filters(),
                    stride=self.stride,
                    padding=self.padding,
                )

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sincnet = nn.Module()
                self.sincnet.conv1d = nn.ModuleList([FakeEncoder()])

        model = FakeModel()
        waveform = torch.linspace(-1.0, 1.0, 1024).reshape(2, 1, 512)
        expected = model.sincnet.conv1d[0](waveform)

        export_models.freeze_pyannote_sincnet_filterbank(model)

        frozen = model.sincnet.conv1d[0]
        self.assertIsInstance(frozen, nn.Conv1d)
        self.assertFalse(frozen.weight.requires_grad)
        torch.testing.assert_close(frozen(waveform), expected, rtol=0.0, atol=0.0)

    def test_static_model_contracts(self) -> None:
        segmentation = export_models.specs_for("pyannote", (1, 16))
        self.assertEqual(segmentation[0].input_shape, (1, 1, 160_000))
        self.assertEqual(segmentation[1].output_shape, (16, 589, 7))
        self.assertEqual(segmentation[0].input_name, "waveform")
        self.assertEqual(segmentation[0].output_name, "logits")

        frontend = export_models.specs_for("pyannote_frontend", (8,))[0]
        self.assertEqual(frontend.input_shape, (8, 1, 160_000))
        self.assertEqual(frontend.output_shape, (8, 589, 60))
        self.assertEqual(frontend.output_name, "features")

        tail = export_models.specs_for("pyannote_tail", (8,))[0]
        self.assertEqual(tail.input_shape, (8, 589, 256))
        self.assertEqual(tail.output_shape, (8, 589, 7))
        self.assertEqual(tail.input_name, "recurrent")

        campplus = export_models.specs_for("campplus", (32, 128))
        self.assertEqual(campplus[0].input_shape, (32, 150, 80))
        self.assertEqual(campplus[1].output_shape, (128, 192))
        self.assertEqual(campplus[0].input_name, "features")
        self.assertEqual(campplus[0].output_name, "embeddings")

        campplus_fp16 = export_models.specs_for("campplus_fp16", (32, 128))
        self.assertEqual(campplus_fp16[0].file_name, "campplus-t150-b32-fp16.onnx")
        self.assertEqual(campplus_fp16[1].file_name, "campplus-t150-b128-fp16.onnx")
        self.assertEqual(campplus_fp16[0].precision, "float16")
        self.assertEqual(campplus_fp16[0].input_shape, campplus[0].input_shape)

    def test_runtime_manifest_uses_string_batch_keys(self) -> None:
        record = {
            "family": "pyannote",
            "batch_size": 8,
            "file": "segmentation-b8.onnx",
            "bytes": 123,
            "sha256": "0" * 64,
            "precision": "float32",
            "opset": 18,
            "input": {"name": "waveform", "shape": [8, 1, 160_000]},
            "output": {"name": "logits", "shape": [8, 589, 7]},
            "graph": {},
            "verification": {},
        }
        result = export_models.manifest([record])
        self.assertEqual(result["version"], 1)
        batch = result["models"]["segmentation"]["batches"]["8"]
        self.assertEqual(batch["file"], "segmentation-b8.onnx")
        self.assertEqual(batch["bytes"], 123)
        self.assertEqual(batch["input_shape"], [8, 1, 160_000])
        self.assertNotIn("8", result["models"]["campplus"]["batches"])

    def test_manifest_adds_fp16_without_changing_fp32_batches(self) -> None:
        def record(family: str, precision: str, file_name: str) -> dict:
            return {
                "family": family,
                "batch_size": 32,
                "file": file_name,
                "bytes": 123,
                "sha256": "0" * 64,
                "precision": precision,
                "opset": 18,
                "input": {"name": "features", "shape": [32, 150, 80]},
                "output": {"name": "embeddings", "shape": [32, 192]},
                "graph": {},
                "verification": {},
            }

        result = export_models.manifest(
            [
                record("campplus", "float32", "campplus-t150-b32.onnx"),
                record(
                    "campplus_fp16",
                    "float16",
                    "campplus-t150-b32-fp16.onnx",
                ),
            ]
        )
        campplus = result["models"]["campplus"]
        self.assertEqual(
            campplus["batches"]["32"]["file"],
            "campplus-t150-b32.onnx",
        )
        fp16 = campplus["precision_variants"]["float16"]
        self.assertEqual(
            fp16["batches"]["32"]["file"],
            "campplus-t150-b32-fp16.onnx",
        )
        self.assertEqual(fp16["input_dtype"], "float32")
        self.assertEqual(fp16["internal_dtype"], "float16")
        self.assertEqual(
            fp16["ort_web"]["required_graph_optimization_level"],
            "basic",
        )

    def test_sha256_is_lowercase_hex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            path.write_bytes(b"senko")
            actual = export_models.sha256(path)
        self.assertEqual(actual, hashlib.sha256(b"senko").hexdigest())
        self.assertRegex(actual, r"^[0-9a-f]{64}$")

    def test_split_boundary_buffer_sizes(self) -> None:
        buffers = export_models.pyannote_boundary_buffers(16)
        self.assertEqual(buffers["frontend_output_bytes"], 2_261_760)
        self.assertEqual(buffers["recurrent_output_bytes"], 9_650_176)
        self.assertEqual(buffers["first_convolution_activation_bytes"], 81_792_000)
        self.assertEqual(buffers["input_affine_scratch_bytes"], 38_600_704)
        self.assertEqual(buffers["hidden_and_cell_state_bytes_per_layer"], 32_768)

    def test_batch_parser_rejects_unplanned_shapes(self) -> None:
        self.assertEqual(
            export_models.parse_batches("8,1,8", export_models.PYANNOTE_BATCHES),
            (8, 1),
        )
        with self.assertRaises(Exception):
            export_models.parse_batches("4", export_models.PYANNOTE_BATCHES)


if __name__ == "__main__":
    unittest.main()
