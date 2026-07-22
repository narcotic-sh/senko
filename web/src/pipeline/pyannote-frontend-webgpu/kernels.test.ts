import { describe, expect, it } from "vitest";

import {
  PYANNOTE_CONV_POOL_F32_WORKGROUP_STORAGE_BYTES,
  PYANNOTE_CONV_POOL_WORKGROUP_STORAGE_BYTES,
  convPoolWgsl,
} from "./conv-pool";
import { bctNormWgsl } from "./instance-norm";
import { RAW_PYANNOTE_FRONTEND_PRODUCTION_KERNELS } from "./runtime";
import {
  PYANNOTE_SINC_WORKGROUP_STORAGE_BYTES,
  PYANNOTE_SINC_WORKGROUP_STORAGE_BYTES_BY_PRECISION,
  sincAbsPoolWgsl,
} from "./sinc-abs-pool";
import { pyannoteTailWgsl } from "../pyannote-tail-webgpu";
import {
  inputAffineLstmWgsl,
  recurrentLstmWgsl,
} from "../persistent-lstm";

describe("pyannote frontend kernel configuration", () => {
  it("accounts for exact workgroup scratch sizes", () => {
    expect(PYANNOTE_SINC_WORKGROUP_STORAGE_BYTES).toBe(10_652);
    expect(PYANNOTE_CONV_POOL_WORKGROUP_STORAGE_BYTES).toEqual({
      float32: 25_472,
      float16: 13_056,
    });
    expect(PYANNOTE_SINC_WORKGROUP_STORAGE_BYTES_BY_PRECISION).toEqual({
      float16: 10_652,
      float32: 12_660,
    });
    expect(PYANNOTE_CONV_POOL_F32_WORKGROUP_STORAGE_BYTES).toEqual({
      block8: 13_056,
      block16: 26_112,
    });
  });

  it("uses exact Sinc arithmetic and FP16 Conv5 scratch in production", () => {
    expect(RAW_PYANNOTE_FRONTEND_PRODUCTION_KERNELS).toEqual({
      convActivationTilePrecision: "float16",
      sincAccumulationSchedule: "interleaved",
    });
  });

  it("generates a complete shader-f16-free FP32 VAD graph", () => {
    const shaders = [
      sincAbsPoolWgsl("float32"),
      bctNormWgsl("float32"),
      convPoolWgsl("f32-bct", "float32", "float32", 8),
      convPoolWgsl("f32-btf", "float32", "float32", 8),
      inputAffineLstmWgsl("float32", 8),
      recurrentLstmWgsl("float32"),
      pyannoteTailWgsl("float32"),
    ];
    for (const shader of shaders) {
      expect(shader).not.toMatch(/\bf16\b/);
      expect(shader).not.toContain("enable f16");
    }
  });
});
