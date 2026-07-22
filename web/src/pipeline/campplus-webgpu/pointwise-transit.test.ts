import { describe, expect, it } from "vitest";

import {
  DEFAULT_POINTWISE_TRANSIT_ACCUMULATION,
  DEFAULT_POINTWISE_TRANSIT_VARIANT,
  POINTWISE_TRANSIT_VARIANTS,
  isPointwiseTransitVariant,
  pointwiseTransitChunk512Wgsl,
  pointwiseTransitWgsl,
} from "./pointwise-transit";

describe("CAM++ pointwise transit variants", () => {
  it("pins the measured chunked production kernel and diagnostic names", () => {
    expect(DEFAULT_POINTWISE_TRANSIT_VARIANT).toBe("chunk512");
    expect(DEFAULT_POINTWISE_TRANSIT_ACCUMULATION).toBe("float16");
    expect(POINTWISE_TRANSIT_VARIANTS).toEqual(["full-cache", "chunk512"]);
    expect(isPointwiseTransitVariant("chunk512")).toBe(true);
    expect(isPointwiseTransitVariant("chunk256")).toBe(false);
  });

  it("keeps the tile-4 baseline's 32 KiB cache", () => {
    const code = pointwiseTransitWgsl(4);
    expect(code).toContain("var<workgroup> weight_cache: array<vec4<f16>, 4096>");
    expect(code.match(/workgroupBarrier\(\)/g)).toHaveLength(1);
  });

  it("strip-mines tile-4 weights through a 16 KiB cache", () => {
    const code = pointwiseTransitChunk512Wgsl(4);
    expect(code).toContain("var<workgroup> weight_cache: array<vec4<f16>, 2048>");
    expect(code).toContain("var chunk_start = 0u");
    expect(code).toContain("min(chunk_start + 512u, parameters.input_channels)");
    expect(code).toContain("if (chunk_end == parameters.input_channels) { break; }");
    expect(code.match(/workgroupBarrier\(\)/g)).toHaveLength(2);
    expect(code).toContain("var accumulator_0 = biases[first_output_group + 0u]");
    expect(code).toContain(
      "fma(vec4<f16>(activated_0), weight_cache[0u * 512u + cache_channel_base], accumulator_0)",
    );
  });

  it("retains an explicit FP32 diagnostic accumulator path", () => {
    const code = pointwiseTransitChunk512Wgsl(4, "float32");
    expect(code).toContain(
      "var accumulator_0 = vec4<f32>(biases[first_output_group + 0u])",
    );
    expect(code).toContain(
      "fma(vec4<f32>(f32(activated_0)), vec4<f32>(weight_cache[0u * 512u + cache_channel_base]), accumulator_0)",
    );
  });
});
