import { describe, expect, it, vi } from "vitest";

import {
  chooseVadBatchSize,
  loadModelManifest,
  selectCampPlusDirect,
  selectSegmentationSplit,
  selectModelVariant,
  type BrowserModel,
  type BrowserSegmentationModel,
} from "./model-manifest";

const model: BrowserModel = {
  id: "test",
  input: { name: "input", dtype: "float32", shape: ["batch", 1] },
  output: { name: "output", dtype: "float32", shape: ["batch", 1] },
  batches: Object.fromEntries(
    [1, 8, 16, 32].map((batch) => [
      String(batch),
      {
        file: `test-b${batch}.onnx`,
        bytes: batch,
        sha256: "a".repeat(64),
        opset: 18,
        precision: "fp32",
        graph: "static",
        verification: {},
      },
    ]),
  ),
};

function segmentationModel(): BrowserSegmentationModel {
  return {
    ...model,
    split: {
      version: 1,
      boundary_layout: "batch,frame,feature",
      frontend: model,
      tail: model,
      lstm: {
        format: "senko-persistent-lstm-f16-gc4h",
        boundary_layout: "batch,frame,feature",
        frames: 589,
        input_features: 60,
        output_features: 256,
        weights: { file: "lstm.bin", bytes: 64, sha256: "b".repeat(64) },
        metadata: { file: "lstm.json", bytes: 32, sha256: "c".repeat(64) },
      },
      direct_webgpu: {
        batches: {
          "8": {
            format: "senko-pyannote-direct-webgpu-f16-v1",
            frontend_metadata: {
              file: "frontend-webgpu.json",
              bytes: 100,
              sha256: "d".repeat(64),
            },
            tail_metadata: {
              file: "tail-webgpu.json",
              bytes: 80,
              sha256: "e".repeat(64),
            },
            explicit_gpu_bytes: 24_845_312,
          },
        },
      },
      buffer_bytes_by_batch: {
        "8": {
          waveform_bytes: 5_120_000,
          first_convolution_activation_bytes: 40_896_000,
          frontend_output_bytes: 1_130_880,
          recurrent_output_bytes: 4_825_088,
          tail_output_bytes: 131_936,
          two_recurrent_ping_pong_buffers_bytes: 9_650_176,
          hidden_and_cell_state_bytes_per_layer: 16_384,
        },
      },
    },
  };
}

function directCampPlusModel(): BrowserModel {
  return {
    ...model,
    direct_webgpu: {
      format: "senko-campplus-direct-webgpu-f16-v1",
      metadata: { file: "campplus.json", bytes: 178_145, sha256: "b".repeat(64) },
      weights: { file: "campplus.bin", bytes: 13_852_416, sha256: "c".repeat(64) },
      production_batch: 16,
      supported_batches: [4, 8, 16, 32],
      explicit_gpu_buffer_bytes_by_batch: {
        "4": 21_434_112,
        "8": 27_164_928,
        "16": 39_855_360,
        "32": 64_621_824,
      },
    },
  };
}

describe("model manifest selection", () => {
  it("resolves a selected asset relative to the manifest", () => {
    const selected = selectModelVariant("https://example.test/models/manifest.json", model, 16);
    expect(selected.batchSize).toBe(16);
    expect(selected.asset.url).toBe("https://example.test/models/test-b16.onnx");
  });

  it("keeps VAD at B16 on 128 MiB adapters", () => {
    expect(chooseVadBatchSize(model, 128 * 1024 * 1024)).toBe(16);
    expect(chooseVadBatchSize(model, 256 * 1024 * 1024)).toBe(32);
  });

  it("selects the pinned B16 direct CAM++ package by default", () => {
    const directModel = directCampPlusModel();
    const selected = selectCampPlusDirect(
      "https://example.test/models/manifest.json",
      directModel,
    );
    expect(selected.batchSize).toBe(16);
    expect(selected.metadata.url).toBe("https://example.test/models/campplus.json");
    expect(selected.weights.url).toBe("https://example.test/models/campplus.bin");
    expect(selected.explicitGpuBufferBytes).toBe(39_855_360);
  });

  it("keeps diagnostic B64 out of production model selection", () => {
    expect(() =>
      selectCampPlusDirect(
        "https://example.test/models/manifest.json",
        directCampPlusModel(),
        64,
      ),
    ).toThrow("does not support B64");
  });

  it("resolves every artifact in the split segmentation contract", () => {
    const segmentation = segmentationModel();
    const selected = selectSegmentationSplit(
      "https://example.test/models/manifest.json",
      segmentation,
      8,
    );
    expect(selected.frontend.asset.url).toBe(
      "https://example.test/models/test-b8.onnx",
    );
    expect(selected.tail.asset.url).toBe(
      "https://example.test/models/test-b8.onnx",
    );
    expect(selected.weights.url).toBe("https://example.test/models/lstm.bin");
    expect(selected.metadata.url).toBe("https://example.test/models/lstm.json");
    expect(selected.directWebGpu.frontendMetadata.url).toBe(
      "https://example.test/models/frontend-webgpu.json",
    );
    expect(selected.directWebGpu.tailMetadata.url).toBe(
      "https://example.test/models/tail-webgpu.json",
    );
    expect(selected.directWebGpu.explicitGpuBytes).toBe(24_845_312);
    expect(selected.declaredBufferBytes.first_convolution_activation_bytes).toBe(
      40_896_000,
    );
  });

  it("verifies the root manifest length and SHA-256 before accepting it", async () => {
    const manifest = {
      version: 1,
      generated_by: {},
      models: {
        segmentation: segmentationModel(),
        campplus: directCampPlusModel(),
      },
      sources: {},
    };
    const bytes = new TextEncoder().encode(JSON.stringify(manifest));
    const sha256 = bytesToHex(await crypto.subtle.digest("SHA-256", bytes));
    const fetchAsset = vi
      .fn<typeof fetch>()
      .mockImplementation(async () => new Response(bytes));

    await expect(
      loadModelManifest(
        "https://example.test/models/manifest.json",
        { byteLength: bytes.byteLength, sha256 },
        fetchAsset,
      ),
    ).resolves.toMatchObject({ version: 1 });

    await expect(
      loadModelManifest(
        "https://example.test/models/manifest.json",
        { byteLength: bytes.byteLength, sha256: "0".repeat(64) },
        fetchAsset,
      ),
    ).rejects.toThrow("SHA-256 mismatch");
  });

  it("rejects a root manifest with a stale declared byte length", async () => {
    const bytes = new TextEncoder().encode("{}");
    await expect(
      loadModelManifest(
        "https://example.test/models/manifest.json",
        { byteLength: bytes.byteLength + 1, sha256: "0".repeat(64) },
        vi.fn<typeof fetch>().mockResolvedValue(new Response(bytes)),
      ),
    ).rejects.toThrow(`has ${bytes.byteLength} bytes; expected ${bytes.byteLength + 1}`);
  });
});

function bytesToHex(buffer: ArrayBuffer): string {
  let result = "";
  for (const byte of new Uint8Array(buffer)) {
    result += byte.toString(16).padStart(2, "0");
  }
  return result;
}
