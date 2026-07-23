import { describe, expect, it } from "vitest";

import { SenkoFbank } from "../audio";
import type { ClusteringNumericKernels } from "../clustering";
import { makePcm16Wav } from "../audio/test-helpers";
import { DEFAULT_PIPELINE_OPTIONS, type PipelineStage } from "../runtime/types";
import {
  BrowserPipelineCancelledError,
  type BrowserPipelineModels,
  BrowserPipelineStageError,
  runBrowserPipeline,
  selectClusteringAlgorithm,
} from "./browser-pipeline";
import type { EmbeddingBatchBackend, VadBatchBackend } from "./types";
import {
  VAD_CHUNK_SAMPLES,
  VAD_FRAME_STEP_SECONDS,
  VAD_OUTPUT_CLASSES,
  VAD_OUTPUT_FRAMES,
} from "./vad";

class AllSpeechVad implements VadBatchBackend {
  readonly batchSize = 2;
  readonly chunkSamples = VAD_CHUNK_SAMPLES;
  readonly outputFrames = VAD_OUTPUT_FRAMES;
  readonly outputClasses = VAD_OUTPUT_CLASSES;
  runs = 0;

  async run(input: Float32Array): Promise<Float32Array> {
    this.runs += 1;
    const logits = new Float32Array(
      this.batchSize * this.outputFrames * this.outputClasses,
    );
    for (let batch = 0; batch < this.batchSize; batch += 1) {
      const sampleStart = batch * this.chunkSamples;
      let logicalSampleCount = 0;
      for (
        let sample = sampleStart + this.chunkSamples - 1;
        sample >= sampleStart;
        sample -= 1
      ) {
        if (input[sample] !== 0) {
          logicalSampleCount = sample - sampleStart + 1;
          break;
        }
      }
      const speechFrames = Math.min(
        this.outputFrames,
        Math.ceil(
          logicalSampleCount / 16_000 / VAD_FRAME_STEP_SECONDS,
        ),
      );
      for (let frame = 0; frame < speechFrames; frame += 1) {
        logits[
          (batch * this.outputFrames + frame) * this.outputClasses + 1
        ] = 1;
      }
    }
    return logits;
  }
}

class ConstantEmbedding implements EmbeddingBatchBackend {
  readonly batchSize = 4;
  readonly frames = 150;
  readonly featureDim = 80;
  readonly embeddingDim = 2;
  runs = 0;

  async run(features: Float32Array): Promise<Float32Array> {
    this.runs += 1;
    expect(features).toHaveLength(
      this.batchSize * this.frames * this.featureDim,
    );
    const output = new Float32Array(this.batchSize * this.embeddingDim);
    for (let row = 0; row < this.batchSize; row += 1) {
      output[row * this.embeddingDim] = 1;
    }
    return output;
  }
}

class DisposableTestFbank extends SenkoFbank {
  readonly memoryStats = {
    heapBytes: 524_288,
  };
  disposeCalls = 0;

  dispose(): void {
    this.disposeCalls += 1;
  }
}

class DeferredFirstEmbedding extends ConstantEmbedding {
  readonly maxInFlightRuns = 2;
  readonly inputBuffers: Float32Array[] = [];
  readonly secondRunStarted: Promise<void>;
  firstInputSnapshot: Float32Array | undefined;
  private resolveFirst: (() => void) | undefined;
  private markSecondRunStarted: (() => void) | undefined;

  constructor() {
    super();
    this.secondRunStarted = new Promise<void>((resolve) => {
      this.markSecondRunStarted = resolve;
    });
  }

  override run(features: Float32Array): Promise<Float32Array> {
    this.runs += 1;
    this.inputBuffers.push(features);
    const output = new Float32Array(this.batchSize * this.embeddingDim);
    for (let row = 0; row < this.batchSize; row += 1) {
      output[row * this.embeddingDim] = 1;
    }
    if (this.runs !== 1) {
      this.markSecondRunStarted?.();
      this.markSecondRunStarted = undefined;
      return Promise.resolve(output);
    }

    this.firstInputSnapshot = features.slice();
    return new Promise<Float32Array>((resolve) => {
      this.resolveFirst = () => resolve(output);
    });
  }

  releaseFirstRun(): void {
    const resolve = this.resolveFirst;
    if (resolve === undefined) throw new Error("First embedding run not pending");
    this.resolveFirst = undefined;
    resolve();
  }
}

class DeferredSecondVad extends AllSpeechVad {
  readonly secondRunStarted: Promise<void>;
  private markSecondRunStarted: (() => void) | undefined;
  private releaseSecond: (() => void) | undefined;

  constructor() {
    super();
    this.secondRunStarted = new Promise<void>((resolve) => {
      this.markSecondRunStarted = resolve;
    });
  }

  override async run(input: Float32Array): Promise<Float32Array> {
    const output = await super.run(input);
    if (this.runs !== 2) return output;
    this.markSecondRunStarted?.();
    this.markSecondRunStarted = undefined;
    await new Promise<void>((resolve) => {
      this.releaseSecond = resolve;
    });
    return output;
  }

  releaseSecondRun(): void {
    const release = this.releaseSecond;
    if (release === undefined) throw new Error("Second VAD run is not pending");
    this.releaseSecond = undefined;
    release();
  }
}

class DeferredTwoEmbeddings extends ConstantEmbedding {
  readonly maxInFlightRuns = 2;
  readonly secondRunStarted: Promise<void>;
  readonly thirdRunStarted: Promise<void>;
  private readonly resolvers = new Map<number, () => void>();
  private markSecondRunStarted: (() => void) | undefined;
  private markThirdRunStarted: (() => void) | undefined;

  constructor() {
    super();
    this.secondRunStarted = new Promise<void>((resolve) => {
      this.markSecondRunStarted = resolve;
    });
    this.thirdRunStarted = new Promise<void>((resolve) => {
      this.markThirdRunStarted = resolve;
    });
  }

  override run(features: Float32Array): Promise<Float32Array> {
    this.runs += 1;
    expect(features).toHaveLength(
      this.batchSize * this.frames * this.featureDim,
    );
    const run = this.runs;
    const output = new Float32Array(this.batchSize * this.embeddingDim);
    for (let row = 0; row < this.batchSize; row += 1) {
      output[row * this.embeddingDim] = 1;
    }
    if (run === 2) {
      this.markSecondRunStarted?.();
      this.markSecondRunStarted = undefined;
    }
    if (run >= 3) {
      if (run === 3) {
        this.markThirdRunStarted?.();
        this.markThirdRunStarted = undefined;
      }
      return Promise.resolve(output);
    }
    return new Promise<Float32Array>((resolve) => {
      this.resolvers.set(run, () => resolve(output));
    });
  }

  releaseRun(run: 1 | 2): void {
    const resolve = this.resolvers.get(run);
    if (resolve === undefined) throw new Error(`Embedding run ${run} is not pending`);
    this.resolvers.delete(run);
    resolve();
  }
}

class ObservedTestFbank extends DisposableTestFbank {
  computeCalls = 0;

  constructor(private readonly onCompute: (count: number) => void) {
    super();
  }

  override compute(samples: Float32Array) {
    this.computeCalls += 1;
    this.onCompute(this.computeCalls);
    return super.compute(samples);
  }
}

describe("runBrowserPipeline", () => {
  it("uses native Senko's strict last-subsegment 1200-second branch", () => {
    expect(selectClusteringAlgorithm([])).toBe("umap-hdbscan");
    expect(
      selectClusteringAlgorithm([{ index: 0, start: 1_198, end: 1_199.999 }]),
    ).toBe("spectral");
    expect(
      selectClusteringAlgorithm([{ index: 0, start: 1_198.5, end: 1_200 }]),
    ).toBe("umap-hdbscan");
    // Earlier boundaries do not matter; native reads subsegments[-1][1].
    expect(
      selectClusteringAlgorithm([
        { index: 0, start: 0, end: 1_300 },
        { index: 1, start: 10, end: 11.5 },
      ]),
    ).toBe("spectral");
  });

  it("streams a WAV through every stage and reports attributed metrics", async () => {
    const vad = new AllSpeechVad();
    const embedding = new ConstantEmbedding();
    const started: PipelineStage[] = [];
    const completed: PipelineStage[] = [];
    const fbank = new DisposableTestFbank();
    const clusteringKernels = {
      memoryStats: { heapBytes: 9 * 1024 * 1024 },
    } as ClusteringNumericKernels;
    const audio = wavBlob(2);
    let tick = 0;
    let usedJSHeapSize = 1_000_000;

    const result = await runBrowserPipeline(
      audio,
      { vad, embedding, knownGpuBufferBytes: 18_661_888 },
      DEFAULT_PIPELINE_OPTIONS,
      {
        createFbank: async () => fbank,
        clusteringKernels,
        now: () => ++tick,
        memoryPerformance: {
          get memory() {
            usedJSHeapSize += 100;
            return { usedJSHeapSize };
          },
        },
        onStageStarted: (stage) => {
          if (stage === "clustering") expect(fbank.disposeCalls).toBe(1);
          started.push(stage);
        },
        onStageCompleted: (stage) => completed.push(stage.stage),
      },
    );
    expect(started).toEqual([
      "decode",
      "vad",
      "fbank",
      "embedding",
      "clustering",
      "postprocess",
    ]);
    expect(completed).toEqual([
      "decode",
      "vad",
      "fbank",
      "embedding",
      "clustering",
      "postprocess",
    ]);
    expect(vad.runs).toBe(1);
    expect(embedding.runs).toBe(1);
    expect(result).toMatchObject({
      durationSeconds: 2,
      speakerCount: 1,
      segments: [
        {
          startSeconds: 0,
          endSeconds: 2.008125,
          speaker: "SPEAKER_01",
        },
      ],
    });
    expect(result.stages).toHaveLength(6);
    expect(result.totalElapsedMs).toBeGreaterThan(0);
    expect(fbank.disposeCalls).toBe(1);
    expect(result.memory).toMatchObject({
      knownCpuPeakBytes: 1_632_984,
      knownGpuBufferBytes: 18_661_888,
      wasmHeapBytes: 9.5 * 1024 * 1024,
      jsHeapPeakBytes: 1_001_400,
      allocations: {
        audioBlobBytes: audio.size,
        audioBlobCopied: false,
        wavReadBufferBytes: 320_000,
        vadInputBatchBytes: 1_280_000,
        vadLogitsBatchBytes: 32_984,
        pcmCachePeakBytes: 96_000,
        camInputBatchBytes: 192_000,
        camOutputBatchBytes: 32,
        retainedEmbeddingsBytes: 16,
        clusterLabelsBytes: 8,
        clusteringPeakWorkingBytes: 0,
      },
    });
    expect(result.memory.checkpoints).toHaveLength(14);
    expect(result.memory.checkpoints.at(-1)).toMatchObject({
      stage: "pipeline",
      phase: "complete",
    });

    expect(stage(result, "vad").metrics).toEqual({
      modelRuns: 1,
      regionCount: 1,
      speechSeconds: 2.008125,
    });
    expect(stage(result, "fbank").metrics).toEqual({
      frameCount: 295,
      melBinCount: 80,
      windowCount: 2,
    });
    expect(stage(result, "embedding").metrics).toEqual({
      batchCount: 1,
      embeddingCount: 2,
      dimensions: 2,
    });
    expect(stage(result, "clustering").metrics).toEqual({
      algorithm: "spectral",
      clusterCount: 1,
      noiseCount: 0,
    });
  });

  it("does not run CAM++ when VAD finds no speech", async () => {
    const vad = new AllSpeechVad();
    vad.run = async () =>
      new Float32Array(
        vad.batchSize * vad.outputFrames * vad.outputClasses,
      );
    const embedding = new ConstantEmbedding();
    const fbank = new DisposableTestFbank();

    const result = await runBrowserPipeline(
      wavBlob(1),
      { vad, embedding },
      DEFAULT_PIPELINE_OPTIONS,
      { createFbank: async () => fbank },
    );

    expect(embedding.runs).toBe(0);
    expect(result.segments).toEqual([]);
    expect(result.speakerCount).toBe(0);
    expect(stage(result, "fbank").metrics.windowCount).toBe(0);
    expect(stage(result, "embedding").metrics.batchCount).toBe(0);
    expect(stage(result, "clustering").metrics.algorithm).toBe("umap-hdbscan");
    expect(fbank.disposeCalls).toBe(1);
  });

  it("records the concurrently resident model buffers", async () => {
    const result = await runBrowserPipeline(
      wavBlob(2),
      { ...modelsWithSpeech(), knownGpuBufferBytes: 84_001_024 },
      DEFAULT_PIPELINE_OPTIONS,
      { createFbank: async () => new DisposableTestFbank() },
    );
    expect(result.memory.knownGpuBufferBytes).toBe(84_001_024);
  });

  it("refreshes growable clustering WASM ownership after clustering", async () => {
    let clusteringHeapBytes = 9 * 1024 * 1024;
    const clusteringKernels = {
      get memoryStats() {
        return {
          heapBytes: clusteringHeapBytes,
          arenaCapacityBytes: clusteringHeapBytes - 1024 * 1024,
          peakArenaUsedBytes: 0,
          peakReturnedJsBytes: 0,
        };
      },
    } as ClusteringNumericKernels;

    const result = await runBrowserPipeline(
      wavBlob(2),
      modelsWithSpeech(),
      DEFAULT_PIPELINE_OPTIONS,
      {
        createFbank: async () => new DisposableTestFbank(),
        clusteringKernels,
        onStageStarted(stage) {
          if (stage === "clustering") {
            clusteringHeapBytes = 64 * 1024 * 1024;
          }
        },
      },
    );

    expect(result.memory.wasmHeapBytes).toBe(64.5 * 1024 * 1024);
  });

  it("extracts the next FBank batch while CAM++ is in flight using distinct bounded buffers", async () => {
    const vad = new AllSpeechVad();
    const embedding = new DeferredFirstEmbedding();
    let markSecondBatchStarted: (() => void) | undefined;
    const secondBatchStarted = new Promise<void>((resolve) => {
      markSecondBatchStarted = resolve;
    });
    const fbank = new ObservedTestFbank((count) => {
      if (count === embedding.batchSize + 1) markSecondBatchStarted?.();
    });

    const pipeline = runBrowserPipeline(
      wavBlob(5),
      { vad, embedding },
      DEFAULT_PIPELINE_OPTIONS,
      { createFbank: async () => fbank },
    );

    await secondBatchStarted;
    expect(embedding.runs).toBe(1);
    expect(embedding.firstInputSnapshot).toBeDefined();
    expect(embedding.inputBuffers[0]).toEqual(embedding.firstInputSnapshot);
    await embedding.secondRunStarted;
    expect(embedding.runs).toBe(2);
    expect(embedding.inputBuffers[0]).toEqual(embedding.firstInputSnapshot);
    embedding.releaseFirstRun();

    const result = await pipeline;
    expect(embedding.inputBuffers[0]).not.toBe(embedding.inputBuffers[1]);
    expect(result.memory.allocations.camInputBatchBytes).toBe(384_000);
    expect(result.memory.allocations.camOutputBatchBytes).toBe(64);
    expect(stage(result, "embedding").metrics.batchCount).toBe(2);
    expect(fbank.disposeCalls).toBe(1);
  });

  it("submits two CAM++ batches while the next VAD batch is in flight", async () => {
    const vad = new DeferredSecondVad();
    const embedding = new DeferredFirstEmbedding();
    let capturedSubsegments: readonly { readonly index: number; readonly end: number }[] = [];
    const pipeline = runBrowserPipeline(
      wavBlob(25),
      { vad, embedding },
      DEFAULT_PIPELINE_OPTIONS,
      {
        createFbank: async () => new DisposableTestFbank(),
        onSubsegmentsCreated: (subsegments) => {
          capturedSubsegments = subsegments;
        },
      },
    );

    await vad.secondRunStarted;
    await embedding.secondRunStarted;
    expect(vad.runs).toBe(2);
    expect(embedding.runs).toBe(2);

    embedding.releaseFirstRun();
    vad.releaseSecondRun();
    const result = await pipeline;
    expect(stage(result, "vad").metrics.modelRuns).toBe(2);
    expect(stage(result, "embedding").metrics.batchCount).toBe(
      Math.ceil(capturedSubsegments.length / embedding.batchSize),
    );
    expect(capturedSubsegments.map((item) => item.index)).toEqual(
      capturedSubsegments.map((_, index) => index),
    );
    expect(capturedSubsegments.at(-1)!.end).toBeGreaterThan(25);
  });

  it("drains prefetched VAD and CAM++ work before completing cancellation", async () => {
    const vad = new DeferredSecondVad();
    const embedding = new DeferredFirstEmbedding();
    const controller = new AbortController();
    const fbank = new DisposableTestFbank();
    const pipeline = runBrowserPipeline(
      wavBlob(25),
      { vad, embedding },
      DEFAULT_PIPELINE_OPTIONS,
      {
        signal: controller.signal,
        createFbank: async () => fbank,
      },
    );
    let settled = false;
    void pipeline.then(
      () => {
        settled = true;
      },
      () => {
        settled = true;
      },
    );

    await vad.secondRunStarted;
    await embedding.secondRunStarted;
    controller.abort();
    vad.releaseSecondRun();
    await Promise.resolve();
    await Promise.resolve();
    expect(settled).toBe(false);

    embedding.releaseFirstRun();
    await expect(pipeline).rejects.toBeInstanceOf(
      BrowserPipelineCancelledError,
    );
    expect(fbank.disposeCalls).toBe(1);
  });

  it("caps CAM++ at two submissions and settles readbacks in FIFO order", async () => {
    const embedding = new DeferredTwoEmbeddings();
    const pipeline = runBrowserPipeline(
      wavBlob(9),
      { vad: new AllSpeechVad(), embedding },
      DEFAULT_PIPELINE_OPTIONS,
      { createFbank: async () => new DisposableTestFbank() },
    );

    await embedding.secondRunStarted;
    expect(embedding.runs).toBe(2);
    embedding.releaseRun(2);
    await Promise.resolve();
    expect(embedding.runs).toBe(2);

    embedding.releaseRun(1);
    await embedding.thirdRunStarted;
    expect(embedding.runs).toBe(3);

    const result = await pipeline;
    expect(stage(result, "embedding").metrics.batchCount).toBe(4);
    expect(result.memory.allocations.camInputBatchBytes).toBe(384_000);
    expect(result.memory.allocations.camOutputBatchBytes).toBe(64);
  });

  it("annotates an embedding inference failure", async () => {
    const base = modelsWithSpeech();
    base.embedding.run = async () => {
      throw new Error("CAM failed");
    };

    await expect(
      runBrowserPipeline(
        wavBlob(2),
        base,
        DEFAULT_PIPELINE_OPTIONS,
        { createFbank: async () => new DisposableTestFbank() },
      ),
    ).rejects.toMatchObject({
      name: "BrowserPipelineStageError",
      stage: "embedding",
      message: "CAM failed",
    });
  });

  it("honors cancellation and annotates model failures with their stage", async () => {
    const models = modelsWithSpeech();
    const controller = new AbortController();
    controller.abort();
    await expect(
      runBrowserPipeline(
        wavBlob(1),
        models,
        DEFAULT_PIPELINE_OPTIONS,
        { signal: controller.signal },
      ),
    ).rejects.toBeInstanceOf(BrowserPipelineCancelledError);

    models.embedding.run = async () => {
      throw new Error("GPU dispatch failed");
    };
    const fbank = new DisposableTestFbank();
    const failed = runBrowserPipeline(
      wavBlob(2),
      models,
      DEFAULT_PIPELINE_OPTIONS,
      { createFbank: async () => fbank },
    );
    await expect(failed).rejects.toMatchObject(
      {
        name: "BrowserPipelineStageError",
        stage: "embedding",
        message: "GPU dispatch failed",
      } satisfies Partial<BrowserPipelineStageError>,
    );
    expect(fbank.disposeCalls).toBe(1);
  });

  it("disposes FBank when cancellation happens after initialization", async () => {
    const controller = new AbortController();
    const models = modelsWithSpeech();
    const runVad = models.vad.run.bind(models.vad);
    models.vad.run = async (input) => {
      const output = await runVad(input);
      controller.abort();
      return output;
    };
    const fbank = new DisposableTestFbank();

    await expect(
      runBrowserPipeline(
        wavBlob(2),
        models,
        DEFAULT_PIPELINE_OPTIONS,
        {
          signal: controller.signal,
          createFbank: async () => fbank,
        },
      ),
    ).rejects.toBeInstanceOf(BrowserPipelineCancelledError);
    expect(fbank.disposeCalls).toBe(1);
  });
});

function modelsWithSpeech(): BrowserPipelineModels & {
  vad: AllSpeechVad;
  embedding: ConstantEmbedding;
} {
  return { vad: new AllSpeechVad(), embedding: new ConstantEmbedding() };
}

function wavBlob(durationSeconds: number): Blob {
  const samples = new Int16Array(durationSeconds * 16_000);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = Math.round(Math.sin(index / 20) * 10_000);
  }
  const bytes = makePcm16Wav(samples);
  const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
  return new Blob([buffer], { type: "audio/wav" });
}

function stage<S extends PipelineStage>(
  result: Awaited<ReturnType<typeof runBrowserPipeline>>,
  name: S,
) {
  const found = result.stages.find((candidate) => candidate.stage === name);
  if (found === undefined) throw new Error(`Missing ${name} stage`);
  return found as Extract<(typeof result.stages)[number], { stage: S }>;
}
