import { describe, expect, it } from "vitest";

import { parsePyannoteFrontendMetadata } from "./metadata";
import { parsePyannoteFrontendBinaryHeader } from "./package";

describe("pyannote frontend WebGPU package", () => {
  it("parses the static B8 contract and explicit memory accounting", () => {
    const metadata = parsePyannoteFrontendMetadata(fixture());
    expect(metadata.contract.inputShape).toEqual([8, 1, 160_000]);
    expect(metadata.contract.outputShape).toEqual([8, 589, 60]);
    expect(metadata.memory).toEqual({
      slotABytes: 13_632_000,
      slotBBytes: 5_120_000,
      activationArenaBytes: 18_757_120,
      statisticsBytes: 5_120,
      minimumResidentGpuBytes: 19_009_024,
    });
    expect(metadata.sections[0]).toMatchObject({
      id: "conv:0:weight",
      byteOffset: 256,
      layout: "K_I_O4_O",
    });
  });

  it("rejects sections that are not aligned to 256 bytes", () => {
    const value = fixture();
    value.sections[0]!.byte_offset = 4;
    expect(() => parsePyannoteFrontendMetadata(value)).toThrow(/alignment/);
  });

  it("parses the little-endian fixed header", () => {
    const header = new Uint8Array(256);
    header.set(new TextEncoder().encode("SNKVADF1"));
    const view = new DataView(header.buffer);
    view.setUint32(8, 1, true);
    view.setUint32(12, 256, true);
    view.setUint32(16, 256, true);
    view.setUint32(20, 10, true);
    view.setBigUint64(24, 251_904n, true);
    header.fill(0xab, 32, 64);
    header.fill(0xcd, 64, 96);
    view.setUint32(96, 8, true);
    view.setUint32(100, 160_000, true);
    view.setUint32(104, 589, true);
    view.setUint32(108, 60, true);
    expect(parsePyannoteFrontendBinaryHeader(header)).toMatchObject({
      formatVersion: 1,
      totalBytes: 251_904,
      batch: 8,
      samples: 160_000,
      frames: 589,
      features: 60,
    });
  });
});

function fixture(): ReturnType<typeof structuredClone<Record<string, unknown>>> & {
  sections: Array<Record<string, unknown>>;
} {
  return {
    schema: "senko.pyannote-frontend.webgpu-pack",
    format_version: 1,
    source: { sha256: "ab".repeat(32) },
    binary: {
      file: "frontend.bin",
      byte_length: 251_904,
      sha256: "cd".repeat(32),
      payload_sha256: "ef".repeat(32),
      header_bytes: 256,
      section_alignment: 256,
      section_count: 1,
    },
    contract: {
      input: {
        shape: [8, 1, 160_000],
        dtype: "float32",
        layout: "BCT",
      },
      output: { shape: [8, 589, 60], dtype: "float32", layout: "BTF" },
      boundary_dtype: "float32",
      intermediate_dtype: "float32",
      reduction_dtype: "float32",
      weight_dtype: "float32",
      channel_tile: 4,
    },
    memory: {
      planned_webgpu: {
        aliased_arena: {
          slot_a_bytes: 13_632_000,
          slot_b_bytes: 5_120_000,
          activation_arena_bytes: 18_757_120,
          statistics_bytes: 5_120,
          minimum_resident_gpu_bytes: 19_009_024,
        },
      },
    },
    sections: [
      {
        id: "conv:0:weight",
        kind: "conv_weight",
        byte_offset: 256,
        byte_length: 80_320,
        element_count: 20_080,
        dtype: "float32",
        logical_shape: [80, 1, 251],
        packed_shape: [251, 1, 20, 4],
        layout: "K_I_O4_O",
        source_tensors: ["weight"],
      },
    ],
  };
}
