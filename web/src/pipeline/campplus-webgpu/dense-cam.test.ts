import { describe, expect, it } from "vitest";

import {
  DEFAULT_DENSE_BOTTLENECK_VARIANT,
  DENSE_BOTTLENECK_TILE2_DIRECT_WGSL,
  DENSE_BOTTLENECK_TILE4_DIRECT_WGSL,
  DENSE_BOTTLENECK_TILE1_DIRECT_WGSL,
  DENSE_BOTTLENECK_TILE1_WG96_WGSL,
  DENSE_BOTTLENECK_TILE1_WGSL,
  DENSE_BOTTLENECK_VARIANTS,
  DENSE_BOTTLENECK_WGSL,
  denseBottleneckTile4DirectWgsl,
  denseBottleneckRequiredWorkgroupStorageBytes,
  denseBottleneckVariantConfiguration,
  isDenseBottleneckVariant,
} from "./dense-cam";

describe("dense CAM++ WGSL variants", () => {
  it("materializes every accumulation template marker", () => {
    expect(DENSE_BOTTLENECK_WGSL).not.toContain("__ACCUMULATOR_");
    expect(DENSE_BOTTLENECK_WGSL).toContain("@workgroup_size(128)");
  });

  it("builds the 96-lane reduction over all 75 active frame values", () => {
    expect(DENSE_BOTTLENECK_TILE1_WGSL).toContain(
      "var<workgroup> mean_reduction: array<vec4<f32>, 128>",
    );
    expect(DENSE_BOTTLENECK_TILE1_WG96_WGSL).toContain("@workgroup_size(96)");
    expect(DENSE_BOTTLENECK_TILE1_WG96_WGSL).toContain(
      "var<workgroup> mean_reduction: array<vec4<f32>, 96>",
    );
    expect(DENSE_BOTTLENECK_TILE1_WG96_WGSL).toContain(
      "mean_reduction[local_id.x] += mean_reduction[local_id.x + 64u]",
    );
    expect(DENSE_BOTTLENECK_TILE1_WG96_WGSL).toContain("var stride = 32u");
  });

  it("materializes direct packed-weight reads without a workgroup cache", () => {
    expect(DENSE_BOTTLENECK_TILE1_DIRECT_WGSL).not.toContain("weight_cache");
    expect(DENSE_BOTTLENECK_TILE1_DIRECT_WGSL).toContain(
      "weights[(output_group * parameters.input_groups + input_group) * 4u + 3u]",
    );
  });

  it("shares activations across two direct FP32 output groups", () => {
    expect(DENSE_BOTTLENECK_TILE2_DIRECT_WGSL).toContain("@workgroup_size(128)");
    expect(DENSE_BOTTLENECK_TILE2_DIRECT_WGSL).toContain(
      "var<workgroup> mean_reduction: array<OutputPair, 128>",
    );
    expect(DENSE_BOTTLENECK_TILE2_DIRECT_WGSL).toContain(
      "let second_output_group = first_output_group + 1u",
    );
    expect(DENSE_BOTTLENECK_TILE2_DIRECT_WGSL).toContain(
      "weights[second_weight_index + 3u]",
    );
    expect(DENSE_BOTTLENECK_TILE2_DIRECT_WGSL).not.toContain("weight_cache");
  });

  it("shares activations across four direct FP32 output groups", () => {
    expect(DENSE_BOTTLENECK_TILE4_DIRECT_WGSL).toContain("@workgroup_size(128)");
    expect(DENSE_BOTTLENECK_TILE4_DIRECT_WGSL).toContain(
      "var<workgroup> mean_reduction: array<OutputQuad, 128>",
    );
    expect(DENSE_BOTTLENECK_TILE4_DIRECT_WGSL).toContain(
      "let fourth_output_group = first_output_group + 3u",
    );
    expect(DENSE_BOTTLENECK_TILE4_DIRECT_WGSL).toContain(
      "weights[fourth_weight_index + 3u]",
    );
    expect(DENSE_BOTTLENECK_TILE4_DIRECT_WGSL).not.toContain("weight_cache");
  });

  it("retains the FP32 baseline for the production FP16 tile-4 geometry", () => {
    expect(denseBottleneckTile4DirectWgsl("float32")).toBe(
      DENSE_BOTTLENECK_TILE4_DIRECT_WGSL,
    );
    const full = denseBottleneckTile4DirectWgsl("float16");
    expect(full).toContain("var fourth_accumulators = biases[fourth_output_group]");
    expect(full).toContain(
      "fma(vec4<f16>(activated_3), weights[fourth_weight_index + 3u], fourth_accumulators)",
    );
    expect(full).toContain(
      "fourth_rounded = max(fourth_accumulators, vec4<f16>(f16(0.0)))",
    );
    expect(full).not.toContain("fourth_partial");
  });

  it("pins the measured direct tile-4 production kernel and smaller-tile oracles", () => {
    expect(DEFAULT_DENSE_BOTTLENECK_VARIANT).toBe("direct-tile4-wg128");
    expect(DENSE_BOTTLENECK_VARIANTS).toEqual([
      "direct-tile1-wg128",
      "direct-tile2-wg128",
      "direct-tile4-wg128",
    ]);
    expect(denseBottleneckVariantConfiguration("direct-tile2-wg128")).toEqual({
      accumulation: "float32",
      outputTile: 2,
      workgroupSize: 128,
      weightSource: "direct",
    });
    expect(denseBottleneckVariantConfiguration("direct-tile4-wg128")).toEqual({
      accumulation: "float16",
      outputTile: 4,
      workgroupSize: 128,
      weightSource: "direct",
    });
    expect(isDenseBottleneckVariant("direct-tile2-wg128")).toBe(true);
    expect(isDenseBottleneckVariant("direct-tile4-wg128")).toBe(true);
    expect(isDenseBottleneckVariant("direct-tile8-wg128")).toBe(false);
  });

  it("accounts cached FP32 diagnostic scratch before shader compilation", () => {
    expect(
      denseBottleneckRequiredWorkgroupStorageBytes(
        1,
        "workgroup-cache",
        "float32",
      ),
    ).toBe(17_920);
    expect(
      denseBottleneckRequiredWorkgroupStorageBytes(
        2,
        "workgroup-cache",
        "float32",
      ),
    ).toBe(35_840);
    expect(
      denseBottleneckRequiredWorkgroupStorageBytes(4, "direct", "float32"),
    ).toBe(8_192);
  });
});
