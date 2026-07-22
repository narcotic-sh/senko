import { describe, expect, it } from "vitest";

import {
  PipelineMemoryTracker,
  readChromiumUsedJsHeapBytes,
  readExposedWasmHeapBytes,
} from "./memory";

describe("runtime memory telemetry", () => {
  it("feature-probes Chromium's non-standard JS heap counter", () => {
    expect(
      readChromiumUsedJsHeapBytes({ memory: { usedJSHeapSize: 12_345 } }),
    ).toBe(12_345);
    expect(readChromiumUsedJsHeapBytes({})).toBeUndefined();
    expect(
      readChromiumUsedJsHeapBytes({ memory: { usedJSHeapSize: -1 } }),
    ).toBeUndefined();
    expect(
      readChromiumUsedJsHeapBytes({ memory: { usedJSHeapSize: 1.5 } }),
    ).toBeUndefined();
  });

  it("reads only valid structurally exposed WASM heap sizes", () => {
    expect(readExposedWasmHeapBytes({ memoryStats: { heapBytes: 524_288 } })).toBe(
      524_288,
    );
    expect(readExposedWasmHeapBytes({ memoryStats: {} })).toBeUndefined();
    expect(readExposedWasmHeapBytes(null)).toBeUndefined();
  });

  it("summarizes deterministic allocations and checkpoint heap peaks", () => {
    let usedJSHeapSize = 1_000;
    const source = {
      get memory() {
        usedJSHeapSize += 250;
        return { usedJSHeapSize };
      },
    };
    const tracker = new PipelineMemoryTracker(4_096, source);
    tracker.setWasmHeapBytes(524_288);
    tracker.setKnownGpuBufferBytes(8_000_000);
    tracker.recordAllocation("vadInputBatchBytes", 5_120_000);
    tracker.recordAllocation("vadLogitsBatchBytes", 131_936);
    tracker.recordAllocation("pcmCachePeakBytes", 96_000);
    tracker.recordAllocation("camInputBatchBytes", 1_536_000);
    tracker.recordAllocation("camOutputBatchBytes", 24_576);
    tracker.setCurrentKnownCpuBytes(1_632_000);
    tracker.checkpoint("fbank", "complete");
    tracker.recordAllocation("clusteringPeakWorkingBytes", 7_545_352);
    tracker.observeKnownCpuPeakBytes(11_900_000);
    tracker.setCurrentKnownCpuBytes(4_500_000);
    tracker.checkpoint("clustering", "complete");

    expect(tracker.summary()).toMatchObject({
      knownCpuPeakBytes: 11_900_000,
      wasmHeapBytes: 524_288,
      knownGpuBufferBytes: 8_000_000,
      jsHeapPeakBytes: 1_500,
      allocations: {
        audioBlobBytes: 4_096,
        audioBlobCopied: false,
        wavReadBufferBytes: 0,
        vadInputBatchBytes: 5_120_000,
        vadLogitsBatchBytes: 131_936,
        pcmCachePeakBytes: 96_000,
        camInputBatchBytes: 1_536_000,
        camOutputBatchBytes: 24_576,
        clusteringPeakWorkingBytes: 7_545_352,
      },
      checkpoints: [
        {
          stage: "fbank",
          phase: "complete",
          knownCpuBytes: 1_632_000,
          jsHeapBytes: 1_250,
        },
        {
          stage: "clustering",
          phase: "complete",
          knownCpuBytes: 4_500_000,
          jsHeapBytes: 1_500,
        },
      ],
    });
  });

  it("omits unsupported optional counters", () => {
    const tracker = new PipelineMemoryTracker(0, {});
    tracker.checkpoint("pipeline", "start");
    expect(tracker.summary()).toEqual({
      knownCpuPeakBytes: 0,
      allocations: {
        audioBlobBytes: 0,
        audioBlobCopied: false,
        wavReadBufferBytes: 0,
        vadInputBatchBytes: 0,
        vadLogitsBatchBytes: 0,
        pcmCachePeakBytes: 0,
        camInputBatchBytes: 0,
        camOutputBatchBytes: 0,
        retainedEmbeddingsBytes: 0,
        clusterLabelsBytes: 0,
        clusteringPeakWorkingBytes: 0,
      },
      checkpoints: [
        { stage: "pipeline", phase: "start", knownCpuBytes: 0 },
      ],
    });
  });
});
