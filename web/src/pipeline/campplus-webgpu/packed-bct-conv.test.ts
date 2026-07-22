import { describe, expect, it } from "vitest";

import {
  DEFAULT_PACKED_BCT_CONV_VARIANT,
  LEGACY_PACKED_BCT_CONV_VARIANT,
  PACKED_BCT_CONV_VARIANTS,
  PACKED_BCT_CONV_WGSL,
  PACKED_BCT_REQUIRED_WORKGROUP_STORAGE_BYTES,
  isPackedBctConvVariant,
  packedBctConvDispatchWorkgroups,
  packedBctConvVariantConfiguration,
  packedBctDirectWgsl,
  type PackedBctConvOutputTile,
  type PackedBctConvWorkgroupSize,
} from "./packed-bct-conv";

describe("packed BCT convolution variants", () => {
  it("pins the cached tile-1 baseline and the direct diagnostic matrix", () => {
    expect(LEGACY_PACKED_BCT_CONV_VARIANT).toBe("cached-tile1-wg128");
    expect(DEFAULT_PACKED_BCT_CONV_VARIANT).toBe("direct-tile8-wg96");
    expect(PACKED_BCT_CONV_VARIANTS).toEqual([
      "cached-tile1-wg128",
      "direct-tile2-wg96",
      "direct-tile4-wg96",
      "direct-tile8-wg96",
      "direct-tile4-wg128",
    ]);
    for (const variant of PACKED_BCT_CONV_VARIANTS) {
      expect(isPackedBctConvVariant(variant)).toBe(true);
    }
    expect(isPackedBctConvVariant("direct-tile16-wg96")).toBe(false);
  });

  it("describes cache, tile, and workgroup requirements explicitly", () => {
    expect(packedBctConvVariantConfiguration("cached-tile1-wg128")).toEqual({
      outputTile: 1,
      workgroupSize: 128,
      weightSource: "workgroup-cache",
      workgroupStorageBytes: PACKED_BCT_REQUIRED_WORKGROUP_STORAGE_BYTES,
    });
    expect(packedBctConvVariantConfiguration("direct-tile2-wg96")).toEqual({
      outputTile: 2,
      workgroupSize: 96,
      weightSource: "direct",
      workgroupStorageBytes: 0,
    });
    expect(packedBctConvVariantConfiguration("direct-tile8-wg96")).toEqual({
      outputTile: 8,
      workgroupSize: 96,
      weightSource: "direct",
      workgroupStorageBytes: 0,
    });
    expect(packedBctConvVariantConfiguration("direct-tile4-wg128")).toEqual({
      outputTile: 4,
      workgroupSize: 128,
      weightSource: "direct",
      workgroupStorageBytes: 0,
    });
  });

  it("computes output-tiled dispatch geometry without changing buffers", () => {
    expect(
      packedBctConvDispatchWorkgroups("cached-tile1-wg128", 32, 16, 75),
    ).toEqual([32, 16, 1]);
    expect(
      packedBctConvDispatchWorkgroups("direct-tile2-wg96", 32, 16, 75),
    ).toEqual([16, 16, 1]);
    expect(
      packedBctConvDispatchWorkgroups("direct-tile4-wg96", 32, 16, 75),
    ).toEqual([8, 16, 1]);
    expect(
      packedBctConvDispatchWorkgroups("direct-tile8-wg96", 32, 16, 193),
    ).toEqual([4, 16, 3]);
    expect(
      packedBctConvDispatchWorkgroups("direct-tile4-wg128", 32, 16, 193),
    ).toEqual([8, 16, 2]);
  });

  it("rejects non-divisible or non-positive dispatch dimensions", () => {
    expect(() =>
      packedBctConvDispatchWorkgroups("direct-tile8-wg96", 30, 16, 75),
    ).toThrow(/multiple of tile 8/);
    expect(() =>
      packedBctConvDispatchWorkgroups("direct-tile4-wg96", 32, 0, 75),
    ).toThrow(/batch size/);
    expect(() =>
      packedBctConvDispatchWorkgroups("direct-tile4-wg96", 32, 16, 0),
    ).toThrow(/output frames/);
  });

  it("retains the byte-for-byte cached shader as the baseline", () => {
    expect(PACKED_BCT_CONV_WGSL).toContain(
      "var<workgroup> weight_cache: array<vec4<f16>, 1600>",
    );
    expect(PACKED_BCT_CONV_WGSL).toContain("workgroupBarrier()");
    expect(PACKED_BCT_CONV_WGSL).toContain("@workgroup_size(128)");
  });

  it("generates a cache-free tile-8 shader with shared input evaluation", () => {
    const shader = packedBctDirectWgsl(8, 96);
    expect(shader).toContain("@workgroup_size(96)");
    expect(shader).toContain("let first_output_group = workgroup_id.x * 8u");
    expect(shader).toContain(
      "var accumulator_0 = vec4<f32>(biases[first_output_group + 0u])",
    );
    expect(shader).toContain(
      "var accumulator_7 = vec4<f32>(biases[first_output_group + 7u])",
    );
    expect(shader).toContain("let weight_index_7_3 =");
    expect(shader).toContain("rounded_7 = max(rounded_7");
    expect(shader).toContain("arena[output_index] = rounded_7[lane_7]");
    expect(shader).toContain(
      "input_value = max(f32(f16(input_value * scale + shift)), 0.0)",
    );
    expect(shader.match(/var input_value = f32\(arena\[input_index\]\)/g)).toHaveLength(4);
    expect(shader).not.toContain("weight_cache");
    expect(shader).not.toContain("workgroupBarrier");
  });

  it("keeps kernel, input-group, and lane accumulation order stable", () => {
    const shader = packedBctDirectWgsl(4, 128);
    const kernelLoop = shader.indexOf("for (var kernel_index");
    const inputGroupLoop = shader.indexOf("for (var input_group");
    const laneZero = shader.indexOf("if (channel_base + 0u");
    const laneOne = shader.indexOf("if (channel_base + 1u");
    const accumulatorZero = shader.indexOf("accumulator_0 = fma", laneZero);
    const accumulatorThree = shader.indexOf("accumulator_3 = fma", laneZero);
    expect(kernelLoop).toBeGreaterThan(0);
    expect(inputGroupLoop).toBeGreaterThan(kernelLoop);
    expect(laneZero).toBeGreaterThan(inputGroupLoop);
    expect(accumulatorZero).toBeGreaterThan(laneZero);
    expect(accumulatorThree).toBeGreaterThan(accumulatorZero);
    expect(laneOne).toBeGreaterThan(accumulatorThree);
  });

  it("rejects unsupported generated geometries at runtime", () => {
    expect(() =>
      packedBctDirectWgsl(3 as PackedBctConvOutputTile, 96),
    ).toThrow(/output tile 3/);
    expect(() =>
      packedBctDirectWgsl(4, 64 as PackedBctConvWorkgroupSize),
    ).toThrow(/workgroup size 64/);
  });
});
