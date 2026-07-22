import { describe, expect, it } from "vitest";

import { parseCampPlusMetadata } from "./metadata";
import { makeSyntheticCampPlusFixture } from "./test-fixture";

describe("parseCampPlusMetadata", () => {
  it("validates section layouts, fused references, and exact GPU memory accounting", () => {
    const { metadata } = makeSyntheticCampPlusFixture();
    const parsed = parseCampPlusMetadata(metadata);
    expect(parsed.binary.byteLength).toBe(1024);
    expect(parsed.sections).toHaveLength(3);
    expect(parsed.fusedProgram.tdnn).toEqual({ weight: "test-weight", bias: "test-bias" });
    expect(parsed.memory).toMatchObject({
      activationArenaBytes: 256,
      weightBufferBytes: 1024,
      minimumResidentGpuBytes: 1280,
    });
  });

  it("rejects overlapping sections before any GPU allocation", () => {
    const { metadata } = makeSyntheticCampPlusFixture();
    const sections = metadata.sections as Array<Record<string, unknown>>;
    sections[1]!.byte_offset = 256;
    expect(() => parseCampPlusMetadata(metadata)).toThrow(/overlap/);
  });

  it("rejects a fused-program reference with the wrong section kind", () => {
    const { metadata } = makeSyntheticCampPlusFixture();
    const fused = metadata.fused_program as Record<string, unknown>;
    const tdnn = fused.tdnn as Record<string, unknown>;
    tdnn.convolution = { weight: "test-bias", bias: "test-bias" };
    expect(() => parseCampPlusMetadata(metadata)).toThrow(/missing or incompatible/);
  });

  it("accepts a shader-f16-free FP32 package with FP32 convolution sections", () => {
    const { metadata } = makeSyntheticCampPlusFixture();
    const contract = metadata.contract as Record<string, unknown>;
    contract.internal_dtype = "float32";
    contract.required_webgpu_features = [];
    const sections = metadata.sections as Array<Record<string, unknown>>;
    sections[0]!.dtype = "float32";
    sections[0]!.byte_length = 64;
    sections[1]!.dtype = "float32";
    sections[1]!.byte_length = 16;

    const parsed = parseCampPlusMetadata(metadata);
    expect(parsed.contract).toMatchObject({
      internalDtype: "float32",
      requiredWebGpuFeatures: [],
    });
  });
});
