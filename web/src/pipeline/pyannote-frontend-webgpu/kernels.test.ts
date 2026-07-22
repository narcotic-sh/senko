import { describe, expect, it } from "vitest";

import { PYANNOTE_CONV_POOL_WORKGROUP_STORAGE_BYTES } from "./conv-pool";
import { RAW_PYANNOTE_FRONTEND_PRODUCTION_KERNELS } from "./runtime";
import { PYANNOTE_SINC_WORKGROUP_STORAGE_BYTES } from "./sinc-abs-pool";

describe("pyannote frontend kernel configuration", () => {
  it("accounts for exact workgroup scratch sizes", () => {
    expect(PYANNOTE_SINC_WORKGROUP_STORAGE_BYTES).toBe(10_652);
    expect(PYANNOTE_CONV_POOL_WORKGROUP_STORAGE_BYTES).toEqual({
      float32: 25_472,
      float16: 13_056,
    });
  });

  it("uses exact Sinc arithmetic and FP16 Conv5 scratch in production", () => {
    expect(RAW_PYANNOTE_FRONTEND_PRODUCTION_KERNELS).toEqual({
      convActivationTilePrecision: "float16",
      sincAccumulationSchedule: "interleaved",
    });
  });
});
