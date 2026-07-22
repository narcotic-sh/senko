import { describe, expect, it } from "vitest";

import { campPlusStorageBytes, campPlusStorageWgsl } from "./storage";

describe("CAM++ storage precision", () => {
  const source = `
enable f16;
@group(0) @binding(0) var<storage, read_write> values: array<f16>;
var<workgroup> cache: array<vec4<f16>, 8>;
fn round(value: f32) -> f16 { return f16(value); }
`;

  it("leaves the production FP16 WGSL byte-for-byte unchanged", () => {
    expect(campPlusStorageWgsl(source, "float16")).toBe(source);
    expect(campPlusStorageBytes("float16")).toBe(2);
  });

  it("widens storage without retaining shader-f16 syntax", () => {
    const widened = campPlusStorageWgsl(source, "float32");
    expect(widened).not.toMatch(/\bf16\b|enable\s+f16/);
    expect(widened).toContain("array<f32>");
    expect(widened).toContain("array<vec4<f32>, 8>");
    expect(campPlusStorageBytes("float32")).toBe(4);
  });
});
