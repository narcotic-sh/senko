import { describe, expect, it } from "vitest";
import {
  isPipelineWorkerRequest,
  isPipelineWorkerResponse,
} from "./protocol";
import { DEFAULT_PIPELINE_OPTIONS } from "./types";

describe("isPipelineWorkerResponse", () => {
  it("accepts a valid stage completion", () => {
    expect(
      isPipelineWorkerResponse({
        type: "stage-completed",
        requestId: "request-1",
        result: {
          stage: "vad",
          elapsedMs: 12.5,
          metrics: { modelRuns: 3, regionCount: 2, speechSeconds: 10 },
        },
      }),
    ).toBe(true);
  });

  it("accepts initialization progress", () => {
    expect(
      isPipelineWorkerResponse({
        type: "initialization-progress",
        requestId: "request-1",
        message: "Loading CAM++ B128",
      }),
    ).toBe(true);
  });

  it("rejects an unknown stage", () => {
    expect(
      isPipelineWorkerResponse({
        type: "stage-started",
        requestId: "request-1",
        stage: "upload",
      }),
    ).toBe(false);
  });

  it("rejects responses without a correlation id", () => {
    expect(
      isPipelineWorkerResponse({
        type: "pipeline-cancelled",
        requestId: "",
      }),
    ).toBe(false);
  });

  it("rejects superficially valid responses with malformed payloads", () => {
    expect(
      isPipelineWorkerResponse({
        type: "initialized",
        requestId: "request-1",
        runtime: {},
      }),
    ).toBe(false);
    expect(
      isPipelineWorkerResponse({
        type: "stage-completed",
        requestId: "request-1",
        result: { stage: "vad", elapsedMs: -1, metrics: {} },
      }),
    ).toBe(false);
  });

  it("validates pipeline memory summaries and checkpoints", () => {
    const response = completedPipelineResponse();
    expect(isPipelineWorkerResponse(response)).toBe(true);
    expect(
      isPipelineWorkerResponse({
        ...response,
        result: {
          ...response.result,
          memory: {
            ...response.result.memory,
            checkpoints: [
              { stage: "clustering", phase: "complete", knownCpuBytes: -1 },
            ],
          },
        },
      }),
    ).toBe(false);
    expect(
      isPipelineWorkerResponse({
        ...response,
        result: {
          ...response.result,
          memory: {
            ...response.result.memory,
            jsHeapPeakBytes: 1.5,
          },
        },
      }),
    ).toBe(false);
    expect(
      isPipelineWorkerResponse({
        ...response,
        result: {
          ...response.result,
          memory: {
            ...response.result.memory,
            knownGpuBufferBytes: -1,
          },
        },
      }),
    ).toBe(false);
  });
});

describe("isPipelineWorkerRequest", () => {
  it("accepts a valid initialization request", () => {
    expect(
      isPipelineWorkerRequest({
        type: "initialize",
        requestId: "request-1",
        manifest: {
          schemaVersion: 1,
          pipelineVersion: "test",
          assets: [],
        },
        options: DEFAULT_PIPELINE_OPTIONS,
      }),
    ).toBe(true);
  });

  it("validates message-specific fields", () => {
    expect(
      isPipelineWorkerRequest({
        type: "diarize",
        requestId: "request-1",
        audio: "not a blob",
        fileName: "test.wav",
      }),
    ).toBe(false);
    expect(
      isPipelineWorkerRequest({
        type: "cancel",
        requestId: "request-2",
        targetRequestId: "",
      }),
    ).toBe(false);
  });
});

function completedPipelineResponse() {
  return {
    type: "pipeline-completed" as const,
    requestId: "request-1",
    result: {
      durationSeconds: 1,
      speakerCount: 1,
      segments: [
        { startSeconds: 0, endSeconds: 1, speaker: "SPEAKER_01" },
      ],
      stages: [],
      totalElapsedMs: 10,
      memory: {
        knownCpuPeakBytes: 12_000_000,
        wasmHeapBytes: 524_288,
        knownGpuBufferBytes: 8_000_000,
        jsHeapPeakBytes: 20_000_000,
        allocations: {
          audioBlobBytes: 32_044,
          audioBlobCopied: false as const,
          wavReadBufferBytes: 320_000,
          vadInputBatchBytes: 5_120_000,
          vadLogitsBatchBytes: 131_936,
          pcmCachePeakBytes: 96_000,
          camInputBatchBytes: 1_536_000,
          camOutputBatchBytes: 24_576,
          retainedEmbeddingsBytes: 4_000_000,
          clusterLabelsBytes: 24_000,
          clusteringPeakWorkingBytes: 7_500_000,
        },
        checkpoints: [
          {
            stage: "pipeline" as const,
            phase: "start" as const,
            knownCpuBytes: 0,
            jsHeapBytes: 10_000_000,
          },
        ],
      },
    },
  };
}
