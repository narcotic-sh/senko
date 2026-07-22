import { describe, expect, it } from "vitest";

import {
  campPlusRawArenaPlan,
  campPlusRawGraphVariableGpuBytes,
  campPlusRawRequiredBufferBytes,
} from "./graph";
import {
  RAW_CAMPPLUS_PREFERRED_LIMITS,
  preferredRawCampPlusDeviceLimits,
  requireRawCampPlusAdapterLimits,
} from "./runtime";

const B64_ARENA_BYTES = 98_304_000;

function adapterWithLimits(overrides: Record<string, number> = {}): GPUAdapter {
  return {
    limits: {
      maxBufferSize: 256 * 1024 * 1024,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupStorageSize: 32 * 1024,
      ...overrides,
    },
  } as unknown as GPUAdapter;
}

describe("raw CAM++ graph batch plans", () => {
  it("derives the exact B64 arena from non-overlapping FCM and dense lifetimes", () => {
    expect(campPlusRawArenaPlan(64)).toEqual({
      activationArenaBytes: B64_ARENA_BYTES,
      minimumActivationArenaBytes: B64_ARENA_BYTES,
      fcmPeakBytes: B64_ARENA_BYTES,
      denseBackbonePeakBytes: 20_905_984,
    });
  });

  it("keeps the production B16 allocation unchanged", () => {
    expect(campPlusRawArenaPlan(16)).toMatchObject({
      activationArenaBytes: 25_190_400,
      minimumActivationArenaBytes: 24_576_000,
    });
  });

  it("doubles only internal activation storage for the FP32 fallback", () => {
    expect(campPlusRawArenaPlan(16, "float32")).toEqual({
      activationArenaBytes: 50_380_800,
      minimumActivationArenaBytes: 49_152_000,
      fcmPeakBytes: 49_152_000,
      denseBackbonePeakBytes: 10_452_992,
    });
    expect(campPlusRawGraphVariableGpuBytes(16, "float32")).toMatchObject({
      activationArena: 50_380_800,
      input: 768_000,
      output: 12_288,
      readback: 24_576,
    });
  });

  it("accounts every batch-dependent B64 GPUBuffer byte", () => {
    const bytes = campPlusRawGraphVariableGpuBytes(64);
    expect(bytes).toEqual({
      activationArena: B64_ARENA_BYTES,
      input: 3_072_000,
      output: 49_152,
      readback: 98_304,
      total: 101_523_456,
    });
    expect(campPlusRawRequiredBufferBytes(64)).toBe(B64_ARENA_BYTES);

    // Checked package weights (13,852,416) plus 119 dispatch uniforms (7,680).
    expect(bytes.total + 13_852_416 + 7_680).toBe(115_383_552);
  });

  it("requests and validates the B64 storage-buffer limit explicitly", () => {
    const adapter = adapterWithLimits();
    expect(preferredRawCampPlusDeviceLimits(adapter, B64_ARENA_BYTES)).toEqual({
      maxComputeWorkgroupStorageSize:
        RAW_CAMPPLUS_PREFERRED_LIMITS.maxComputeWorkgroupStorageSize,
      maxBufferSize: B64_ARENA_BYTES,
      maxStorageBufferBindingSize: B64_ARENA_BYTES,
    });
    expect(() =>
      requireRawCampPlusAdapterLimits(
        adapterWithLimits({ maxStorageBufferBindingSize: 90_000_000 }),
        B64_ARENA_BYTES,
      ),
    ).toThrow(/storage binding/);
  });
});
