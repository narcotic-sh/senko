import { describe, expect, it } from "vitest";
import {
  assessRuntimeCapabilities,
  type RuntimeCapabilities,
} from "./capabilities";

function capableRuntime(): RuntimeCapabilities {
  return {
    secureContext: true,
    crossOriginIsolated: true,
    dedicatedWorker: true,
    sharedArrayBuffer: true,
    wasm: true,
    wasmSimd: true,
    wasmThreads: true,
    webgpu: {
      available: true,
      features: ["shader-f16"],
    },
  };
}

describe("assessRuntimeCapabilities", () => {
  it("accepts the full performance runtime", () => {
    expect(assessRuntimeCapabilities(capableRuntime())).toEqual({
      canRun: true,
      errors: [],
      warnings: [],
    });
  });

  it("makes WebGPU a hard requirement", () => {
    const runtime = capableRuntime();
    const assessment = assessRuntimeCapabilities({
      ...runtime,
      webgpu: { available: false, features: [] },
    });

    expect(assessment.canRun).toBe(false);
    expect(assessment.errors).toContain("No WebGPU adapter is available.");
  });

  it("reports optional acceleration features as warnings", () => {
    const runtime = capableRuntime();
    const assessment = assessRuntimeCapabilities({
      ...runtime,
      crossOriginIsolated: false,
      sharedArrayBuffer: false,
      wasmSimd: false,
      wasmThreads: false,
      webgpu: { available: true, features: [] },
    });

    expect(assessment.canRun).toBe(true);
    expect(assessment.warnings).toHaveLength(3);
  });
});
