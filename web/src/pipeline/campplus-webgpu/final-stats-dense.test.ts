import { describe, expect, it } from "vitest";

import { FINAL_STATS_DENSE_WGSL } from "./final-stats-dense";

describe("final CAM++ graph fusion", () => {
  it("preserves the model's FP16 statistic and dense boundaries", () => {
    expect(FINAL_STATS_DENSE_WGSL).toContain("let mean = f16(");
    expect(FINAL_STATS_DENSE_WGSL).toContain("let centered = f16(");
    expect(FINAL_STATS_DENSE_WGSL).toContain("let variance = f16(");
    expect(FINAL_STATS_DENSE_WGSL).toContain("let dense = vec4<f16>(accumulator)");
    expect(FINAL_STATS_DENSE_WGSL).toContain("output[batch * parameters.output_channels");
  });

  it("uses one persistent 128-lane workgroup per batch item", () => {
    expect(FINAL_STATS_DENSE_WGSL).toContain("@workgroup_size(128)");
    expect(FINAL_STATS_DENSE_WGSL).toContain("var<workgroup> statistics: array<f16, 1024>");
  });
});
