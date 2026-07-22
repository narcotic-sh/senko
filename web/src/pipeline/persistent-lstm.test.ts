import { describe, expect, it } from "vitest";

import {
  parsePersistentLstmMetadata,
  persistentLstmWgsl,
  type PersistentLstmWeightPrecision,
} from "./persistent-lstm";

function fixture(precision: PersistentLstmWeightPrecision): {
  metadata: Record<string, unknown>;
  weightsBytes: number;
} {
  const elementBytes = precision === "float16" ? 2 : 4;
  const dtype = precision === "float16" ? "float16-le" : "float32-le";
  let offset = 0;
  const layers = Array.from({ length: 4 }, (_, layer) => {
    const inputSize = layer === 0 ? 60 : 256;
    const columns = inputSize + 128;
    const directions = ["forward", "reverse"].map((direction) => {
      const tensor = (
        shape: readonly number[],
        packedShape: readonly number[],
        layout: "row-major" | "gate-column4-hidden-input4",
      ) => {
        const lengthBytes = packedShape.reduce((product, value) => product * value, 1) * elementBytes;
        const record = {
          offset_bytes: offset,
          length_bytes: lengthBytes,
          shape,
          packed_shape: packedShape,
          dtype,
          layout,
        };
        offset = Math.ceil((offset + lengthBytes) / 256) * 256;
        return record;
      };
      return {
        direction,
        input_size: inputSize,
        hidden_size: 128,
        gate_order: ["input", "forget", "cell", "output"],
        tensors: {
          matrix: tensor(
            [512, columns],
            [4, columns / 4, 128, 4],
            "gate-column4-hidden-input4",
          ),
          bias_ih: tensor([512], [512], "row-major"),
          bias_hh: tensor([512], [512], "row-major"),
        },
      };
    });
    return {
      layer,
      input_size: inputSize,
      output_size: 256,
      directions,
    };
  });
  return {
    weightsBytes: offset,
    metadata: {
      version: precision === "float16" ? 3 : 2,
      format: `senko-persistent-lstm-${precision === "float16" ? "f16" : "f32"}-gc4h`,
      byte_order: "little-endian",
      alignment_bytes: 256,
      boundary_layout: "batch,frame,feature",
      frames: 589,
      num_layers: 4,
      bidirectional: true,
      hidden_size: 128,
      gate_order: ["input", "forget", "cell", "output"],
      weights: {
        file: `lstm-${precision}.bin`,
        bytes: offset,
        sha256: "a".repeat(64),
      },
      layers,
      ...(precision === "float16"
        ? {
            storage_dtype: "float16",
            accumulator_dtype: "float32",
            required_webgpu_features: ["shader-f16"],
          }
        : {}),
    },
  };
}

function encodeMetadata(metadata: Record<string, unknown>): ArrayBuffer {
  return new TextEncoder().encode(JSON.stringify(metadata)).buffer as ArrayBuffer;
}

describe("persistent LSTM package contract", () => {
  it.each(["float32", "float16"] as const)(
    "accepts a strict %s package",
    (precision) => {
      const { metadata, weightsBytes } = fixture(precision);
      const parsed = parsePersistentLstmMetadata(
        encodeMetadata(metadata),
        weightsBytes,
      );
      expect(parsed.weightPrecision).toBe(precision);
      expect(parsed.weightElementBytes).toBe(precision === "float16" ? 2 : 4);
      expect(weightsBytes).toBe(precision === "float16" ? 2_760_704 : 5_521_408);
    },
  );

  it("rejects a precision/dtype mismatch", () => {
    const { metadata, weightsBytes } = fixture("float16");
    const layers = metadata.layers as Array<{
      directions: Array<{ tensors: { matrix: { dtype: string } } }>;
    }>;
    layers[0]!.directions[0]!.tensors.matrix.dtype = "float32-le";
    expect(() =>
      parsePersistentLstmMetadata(encodeMetadata(metadata), weightsBytes),
    ).toThrow(/tensor is invalid/);
  });

  it("rejects overlapping or reordered tensors", () => {
    const { metadata, weightsBytes } = fixture("float32");
    const layers = metadata.layers as Array<{
      directions: Array<{ tensors: { bias_ih: { offset_bytes: number } } }>;
    }>;
    layers[0]!.directions[0]!.tensors.bias_ih.offset_bytes = 0;
    expect(() =>
      parsePersistentLstmMetadata(encodeMetadata(metadata), weightsBytes),
    ).toThrow(/tensor is invalid/);
  });

  it("generates separate f32 and shader-f16 kernels with f32 accumulation", () => {
    const fp32 = persistentLstmWgsl("float32");
    const fp16 = persistentLstmWgsl("float16");
    expect(fp32).not.toContain("enable f16;");
    expect(fp32).toContain("array<vec4<f32>>");
    expect(fp16).toContain("enable f16;");
    expect(fp16).toContain("array<vec4<f16>>");
    expect(fp16).toContain("fn weight_vector(index: u32) -> vec4<f32>");
    expect(fp16).toContain("var first_gate");
    expect(fp16).toContain("var cell = 0.0");
  });
});
