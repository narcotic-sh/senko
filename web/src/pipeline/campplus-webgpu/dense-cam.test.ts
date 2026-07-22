import { describe, expect, it } from "vitest";

import {
  DENSE_BOTTLENECK_TILE1_DIRECT_WGSL,
  DENSE_BOTTLENECK_TILE1_WG96_WGSL,
  DENSE_BOTTLENECK_TILE1_WGSL,
  DENSE_BOTTLENECK_WGSL,
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
});
