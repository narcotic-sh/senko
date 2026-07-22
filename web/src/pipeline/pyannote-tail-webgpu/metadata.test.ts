import { describe, expect, it } from "vitest";

import { parsePyannoteTailMetadata } from "./metadata";
import { pyannoteTailWgsl } from "./runtime";

describe("pyannote tail precision contracts", () => {
  it.each(["float16", "float32"] as const)(
    "parses a %s package",
    (precision) => {
      const metadata = parsePyannoteTailMetadata(fixture(precision));
      expect(metadata.weightPrecision).toBe(precision);
      expect([...metadata.sections.values()]).toHaveLength(6);
      expect([...metadata.sections.values()].every((section) => section.dtype === precision)).toBe(
        true,
      );
    },
  );

  it("rejects mixed section precision", () => {
    const value = fixture("float32");
    value.sections[0]!.dtype = "float16";
    expect(() => parsePyannoteTailMetadata(value)).toThrow(/section precision/);
  });

  it("generates an FP32 shader without shader-f16 syntax", () => {
    expect(pyannoteTailWgsl("float32")).not.toMatch(/\bf16\b/);
  });
});

function fixture(precision: "float16" | "float32"): {
  sections: Array<Record<string, unknown>>;
  [key: string]: unknown;
} {
  const sections = [
    ["linear:0:weight", "matrix", "I_O4_O", [256, 128]],
    ["linear:0:bias", "bias", "O4", [128]],
    ["linear:1:weight", "matrix", "I_O4_O", [128, 128]],
    ["linear:1:bias", "bias", "O4", [128]],
    ["linear:2:weight", "matrix", "I_O4_O", [128, 7]],
    ["linear:2:bias", "bias", "O4", [7]],
  ].map(([id, kind, layout, shape], index) => ({
    id,
    kind,
    layout,
    logical_shape: shape,
    packed_shape: [1, 1, 4],
    dtype: precision,
    byte_offset: 256 + index * 256,
    byte_length: 256,
    element_count: precision === "float16" ? 128 : 64,
  }));
  return {
    schema: "senko.pyannote-tail.webgpu-pack",
    format_version: 1,
    source: { sha256: "ab".repeat(32) },
    binary: {
      file: "tail.bin",
      byte_length: 2_048,
      sha256: "cd".repeat(32),
      payload_sha256: "ef".repeat(32),
      header_bytes: 256,
      section_alignment: 256,
      section_count: 6,
    },
    contract: {
      input_shape: [8, 589, 256],
      output_shape: [8, 589, 7],
      boundary_dtype: "float32",
      weight_dtype: precision,
      accumulator_dtype: "float32",
    },
    memory: {
      weight_buffer_bytes: 2_048,
      output_buffer_bytes: 131_936,
      readback_buffer_bytes: 131_936,
      uniform_bytes: 64,
      explicit_gpu_bytes: 265_984,
    },
    sections,
  };
}
