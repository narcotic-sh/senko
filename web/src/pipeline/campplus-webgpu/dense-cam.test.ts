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
      accumulation: "float32",
      outputTile: 4,
      workgroupSize: 128,
      weightSource: "direct",
    });
    expect(isDenseBottleneckVariant("direct-tile2-wg128")).toBe(true);
    expect(isDenseBottleneckVariant("direct-tile4-wg128")).toBe(true);
    expect(isDenseBottleneckVariant("direct-tile8-wg128")).toBe(false);
  });
});
