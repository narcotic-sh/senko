import {
  type FbankComputer,
  Pcm16WavReader,
  secondsToFbankWindow,
  SENKO_FBANK_BINS,
  StreamingFbankExtractor,
  WasmSenkoFbank,
} from "../audio";
import {
  clusterEmbeddings,
  clusterEmbeddingsSpectral,
  DEFAULT_CLUSTERING_OPTIONS,
  estimatePostUmapPeakWorkingBytes,
  type ClusteringNumericKernels,
  type NativeUmapThreadedResult,
} from "../clustering";
import type {
  AnyStageResult,
  PipelineOptions,
  PipelineResult,
  PipelineStage,
  StageResult,
} from "../runtime/types";
import { DEFAULT_PIPELINE_OPTIONS } from "../runtime/types";
import {
  PipelineMemoryTracker,
  readExposedWasmHeapBytes,
  type MemoryPerformanceSource,
} from "../runtime/memory";
import { IncrementalVadSubsegmentReducer } from "./incremental-vad-subsegments";
import { postprocessClustering } from "./postprocess";
import type {
  EmbeddingBatchBackend,
  MonoPcmSource,
  Subsegment,
  VadBatchBackend,
} from "./types";
import {
  createVadChunks,
  runVadBatches,
  type VadBatchResult,
  VAD_SAMPLE_RATE,
} from "./vad";

export interface BrowserPipelineModels {
  readonly vad: VadBatchBackend;
  readonly embedding: EmbeddingBatchBackend;
  /** Explicit GPU buffers owned by the model set; opaque ORT arenas excluded. */
  readonly knownGpuBufferBytes?: number;
}

export interface NativeUmapClusteringBackend {
  readonly memoryStats?: { readonly heapBytes: number };
  clusterNativeUmap(
    embeddings: Float32Array,
    count: number,
    dimension: number,
    signal?: AbortSignal,
  ): Promise<NativeUmapThreadedResult>;
}

export interface BrowserPipelineHooks {
  readonly signal?: AbortSignal;
  readonly onStageStarted?: (stage: PipelineStage) => void;
  readonly onStageCompleted?: (result: AnyStageResult) => void;
  /** Injectable monotonic clock for focused orchestration tests. */
  readonly now?: () => number;
  /** Tests can inject the TypeScript reference without fetching a WASM asset. */
  readonly createFbank?: () => Promise<FbankComputer>;
  /** Preloaded reusable numeric kernels; production supplies one per worker. */
  readonly clusteringKernels?: ClusteringNumericKernels;
  /** Native-parity UMAP/HDBSCAN plus its threaded layout worker pool. */
  readonly nativeUmapClustering?: NativeUmapClusteringBackend;
  /** Injectable source for Chromium's non-standard performance.memory probe. */
  readonly memoryPerformance?: MemoryPerformanceSource;
  /** Query-gated correctness harnesses can align embeddings to native windows. */
  readonly onSubsegmentsCreated?: (subsegments: readonly Subsegment[]) => void;
}

export class BrowserPipelineCancelledError extends Error {
  constructor() {
    super("Pipeline cancelled");
    this.name = "BrowserPipelineCancelledError";
  }
}

export class BrowserPipelineStageError extends Error {
  readonly stage: PipelineStage;

  constructor(stage: PipelineStage, cause: unknown) {
    super(errorMessage(cause), { cause });
    this.name = "BrowserPipelineStageError";
    this.stage = stage;
  }
}

export type BrowserClusteringAlgorithm = "spectral" | "umap-hdbscan";

/** Native Senko chooses from the last generated subsegment, not WAV duration. */
export function selectClusteringAlgorithm(
  subsegments: readonly Subsegment[],
): BrowserClusteringAlgorithm {
  return subsegments.length > 0 && subsegments[subsegments.length - 1]!.end < 1200
    ? "spectral"
    : "umap-hdbscan";
}

type EmbeddingRunOutcome =
  | {
      readonly ok: true;
      readonly output: Float32Array;
      readonly completedAtMs: number;
    }
  | { readonly ok: false; readonly error: unknown; readonly completedAtMs: number };

interface PendingEmbeddingRun {
  readonly outcome: Promise<EmbeddingRunOutcome>;
  readonly batchStart: number;
  readonly actualBatchSize: number;
  readonly featureSlot: number;
  readonly startedAtMs: number;
}

interface SettledEmbeddingRun {
  readonly attributedElapsedMs: number;
  readonly coveredUntilMs: number;
  readonly chunk?: SettledEmbeddingChunk;
}

interface SettledEmbeddingChunk {
  readonly batchStart: number;
  readonly values: Float32Array;
}

type VadBatchOutcome =
  | {
      readonly ok: true;
      readonly result: IteratorResult<VadBatchResult, void>;
      readonly completedAtMs: number;
    }
  | { readonly ok: false; readonly error: unknown; readonly completedAtMs: number };

interface PendingVadBatch {
  readonly outcome: Promise<VadBatchOutcome>;
  readonly startedAtMs: number;
}

/**
 * Run the complete browser pipeline while keeping hour-long audio and FBank
 * features streaming. At most two CAM++ input/output batches and the final
 * embeddings are retained; the WAV Blob is decoded on demand through slices.
 */
export async function runBrowserPipeline(
  audio: Blob,
  models: BrowserPipelineModels,
  options: PipelineOptions = DEFAULT_PIPELINE_OPTIONS,
  hooks: BrowserPipelineHooks = {},
): Promise<PipelineResult> {
  const now = hooks.now ?? (() => performance.now());
  const stages: AnyStageResult[] = [];
  const memory = new PipelineMemoryTracker(audio.size, hooks.memoryPerformance);
  memory.setKnownGpuBufferBytes(models.knownGpuBufferBytes);
  memory.checkpoint("pipeline", "start");
  throwIfCancelled(hooks.signal);
  const totalStart = now();
  const clusteringMemorySource =
    hooks.nativeUmapClustering ?? hooks.clusteringKernels;
  let fbank: FbankComputer | undefined;
  let extractor: StreamingFbankExtractor | undefined;
  let fbankWasmHeapBytes: number | undefined;

  try {
    const vad = models.vad;
    const embedding = models.embedding;
    validateModelOptions(vad, embedding, options);

    try {
      fbank = await (hooks.createFbank ?? (() => WasmSenkoFbank.create()))();
      fbankWasmHeapBytes = readExposedWasmHeapBytes(fbank);
      memory.setWasmHeapBytes(
        sumDefinedBytes(
          fbankWasmHeapBytes,
          readExposedWasmHeapBytes(clusteringMemorySource),
        ),
      );
      throwIfCancelled(hooks.signal);
    } catch (error) {
      rethrowForStage("fbank", error, hooks.signal);
    }

    memory.checkpoint("decode", "start");
    hooks.onStageStarted?.("decode");
    const decodeStart = now();
    let reader: Pcm16WavReader;
    try {
      reader = await Pcm16WavReader.open(audio);
      throwIfCancelled(hooks.signal);
    } catch (error) {
      rethrowForStage("decode", error, hooks.signal);
    }
    const decodeResult: StageResult<"decode"> = {
      stage: "decode",
      elapsedMs: elapsedSince(now, decodeStart),
      metrics: {
        durationSeconds: reader.info.durationSeconds,
        sampleRate: reader.info.sampleRate,
        channelCount: reader.info.channels,
        sampleCount: reader.info.sampleCount,
      },
    };
    completeStage(decodeResult, stages, hooks);
    memory.checkpoint("decode", "complete");

    const wavReadBufferBytes = reader.reusableReadBufferBytes;
    memory.recordAllocation("wavReadBufferBytes", wavReadBufferBytes);
    const vadChunkCount = createVadChunks(reader.sampleCount).length;
    const vadInputBatchBytes =
      vadChunkCount === 0
        ? 0
        : vad.batchSize *
          vad.chunkSamples *
          Float32Array.BYTES_PER_ELEMENT;
    const vadLogitsBatchBytes =
      vadChunkCount === 0
        ? 0
        : vad.batchSize *
          vad.outputFrames *
          vad.outputClasses *
          Float32Array.BYTES_PER_ELEMENT;
    memory.recordAllocation("vadInputBatchBytes", vadInputBatchBytes);
    memory.recordAllocation("vadLogitsBatchBytes", vadLogitsBatchBytes);
    memory.setCurrentKnownCpuBytes(
      wavReadBufferBytes + vadInputBatchBytes + vadLogitsBatchBytes,
    );
    memory.checkpoint("vad", "start");
    hooks.onStageStarted?.("vad");
    memory.checkpoint("fbank", "start");
    memory.checkpoint("embedding", "start");
    hooks.onStageStarted?.("fbank");
    hooks.onStageStarted?.("embedding");

    // VAD, feature extraction, and embedding inference are streamed together.
    // The first VAD batch establishes a safe subsegment prefix. Each subsequent
    // VAD dispatch then overlaps with at most two already-known CAM++ batches.
    // Elapsed stage values remain attributable intervals; totalElapsedMs is the
    // authoritative end-to-end wall-clock measurement.
    let vadElapsedMs = 0;
    let vadModelRuns = 0;
    let fbankElapsedMs = 0;
    let embeddingElapsedMs = 0;
    let totalFrameCount = 0;
    let embeddingBatchCount = 0;
    const embeddingRunLimit = embeddingRunConcurrency(embedding);
    const featureValueCount =
      embedding.batchSize * embedding.frames * embedding.featureDim;
    const featureBatchBytes =
      featureValueCount * Float32Array.BYTES_PER_ELEMENT;
    const outputBatchBytes =
      embedding.batchSize *
      embedding.embeddingDim *
      Float32Array.BYTES_PER_ELEMENT;
    const reducer = new IncrementalVadSubsegmentReducer({
      durationSeconds: options.embeddingWindowSeconds,
      shiftSeconds: options.embeddingShiftSeconds,
    });
    const streamedSubsegments: Subsegment[] = [];
    let availableSubsegments: readonly Subsegment[] = streamedSubsegments;
    const vadIterator = runVadBatches(
      wavReaderAsMonoSource(reader),
      vad,
    )[Symbol.asyncIterator]();
    let activeVad: PendingVadBatch | undefined;
    let batchFeatures: Float32Array[] = [];
    const pendingEmbeddings: PendingEmbeddingRun[] = [];
    const bufferedEmbeddingChunks: SettledEmbeddingChunk[] = [];
    let bufferedEmbeddingBytes = 0;
    let peakRetainedEmbeddingBytes = 0;
    let peakPendingEmbeddingRuns = 0;
    let coveredEmbeddingRunUntilMs: number | undefined;
    let nextEmbeddingStart = 0;
    let embeddingBatchIndex = 0;
    let embeddings: Float32Array | undefined;

    const observeStreamingCpu = (includeVadBuffers: boolean): void => {
      const pcmCacheBytes =
        (extractor?.stats.peakCachedSamples ?? 0) *
        Float32Array.BYTES_PER_ELEMENT;
      const featureBytes = batchFeatures.length * featureBatchBytes;
      const pendingOutputBytes = pendingEmbeddings.length * outputBatchBytes;
      const retainedBytes =
        bufferedEmbeddingBytes + (embeddings?.byteLength ?? 0);
      memory.observeKnownCpuPeakBytes(
        wavReadBufferBytes +
          (includeVadBuffers ? vadInputBatchBytes + vadLogitsBatchBytes : 0) +
          pcmCacheBytes +
          featureBytes +
          pendingOutputBytes +
          retainedBytes,
      );
    };

    const ensureEmbeddingBuffers = (slotCount: number): void => {
      if (batchFeatures.length > 0) return;
      if (slotCount <= 0 || slotCount > 2) {
        throw new RangeError(`Invalid CAM++ host slot count: ${slotCount}`);
      }
      extractor = new StreamingFbankExtractor(reader, fbank);
      batchFeatures = Array.from(
        { length: slotCount },
        () => new Float32Array(featureValueCount),
      );
      observeStreamingCpu(activeVad !== undefined);
    };

    const settleOldestEmbedding = async (): Promise<void> => {
      const pending = pendingEmbeddings.shift();
      if (pending === undefined) {
        throw new Error("CAM++ pending-run queue is unexpectedly empty");
      }
      const settled = await settleEmbeddingRun(
        pending,
        embedding,
        embeddings,
        hooks.signal,
        coveredEmbeddingRunUntilMs,
      );
      embeddingElapsedMs += settled.attributedElapsedMs;
      coveredEmbeddingRunUntilMs = settled.coveredUntilMs;
      embeddingBatchCount += 1;
      if (settled.chunk !== undefined) {
        bufferedEmbeddingChunks.push(settled.chunk);
        bufferedEmbeddingBytes += settled.chunk.values.byteLength;
        peakRetainedEmbeddingBytes = Math.max(
          peakRetainedEmbeddingBytes,
          bufferedEmbeddingBytes,
        );
      }
      observeStreamingCpu(activeVad !== undefined);
    };

    const pumpEmbeddingBatches = async (
      maxNewBatches: number,
      allowPartialBatch: boolean,
    ): Promise<void> => {
      let submitted = 0;
      while (
        nextEmbeddingStart < availableSubsegments.length &&
        submitted < maxNewBatches
      ) {
        const remaining = availableSubsegments.length - nextEmbeddingStart;
        if (!allowPartialBatch && remaining < embedding.batchSize) break;
        const actualBatchSize = Math.min(embedding.batchSize, remaining);
        if (batchFeatures.length === 0) {
          const remainingBatches = Math.ceil(remaining / embedding.batchSize);
          ensureEmbeddingBuffers(
            allowPartialBatch ? Math.min(2, remainingBatches) : 2,
          );
        }
        const featureSlot = embeddingBatchIndex % batchFeatures.length;
        const features = batchFeatures[featureSlot]!;
        const occupyingRun = pendingEmbeddings.findIndex(
          (pending) => pending.featureSlot === featureSlot,
        );
        if (occupyingRun > 0) {
          throw new Error("CAM++ host feature slots lost FIFO ordering");
        }
        if (occupyingRun === 0) await settleOldestEmbedding();
        features.fill(0);

        const requests = availableSubsegments
          .slice(nextEmbeddingStart, nextEmbeddingStart + actualBatchSize)
          .map((subsegment) =>
            secondsToFbankWindow(
              subsegment.start,
              subsegment.end,
              subsegment.index,
            )
          );
        const fbankIterator = extractor!
          .extract(requests)
          [Symbol.asyncIterator]();
        for (let row = 0; row < actualBatchSize; row += 1) {
          const fbankStart = now();
          try {
            throwIfCancelled(hooks.signal);
            const next = await fbankIterator.next();
            if (next.done) {
              throw new Error(
                `FBank stream ended after ${row} of ${actualBatchSize} windows`,
              );
            }
            const matrix = next.value.features;
            if (matrix.binCount !== embedding.featureDim) {
              throw new Error(
                `FBank produced ${matrix.binCount} bins; CAM++ expects ${embedding.featureDim}`,
              );
            }
            copyFeatureWindow(
              matrix.data,
              matrix.frameCount,
              matrix.binCount,
              features,
              row,
              embedding.frames,
            );
            totalFrameCount += matrix.frameCount;
            throwIfCancelled(hooks.signal);
          } catch (error) {
            rethrowForStage("fbank", error, hooks.signal);
          } finally {
            fbankElapsedMs += elapsedSince(now, fbankStart);
          }
        }

        // A serial backend still overlaps extraction with its preceding run.
        // Direct CAM++ advertises two safe submissions and readback slots.
        while (pendingEmbeddings.length >= embeddingRunLimit) {
          await settleOldestEmbedding();
        }
        pendingEmbeddings.push(
          startEmbeddingRun(
            embedding,
            features,
            nextEmbeddingStart,
            actualBatchSize,
            featureSlot,
            now,
          ),
        );
        peakPendingEmbeddingRuns = Math.max(
          peakPendingEmbeddingRuns,
          pendingEmbeddings.length,
        );
        nextEmbeddingStart += actualBatchSize;
        embeddingBatchIndex += 1;
        submitted += 1;
        observeStreamingCpu(activeVad !== undefined);
      }
    };

    const consumeActiveVad = async (): Promise<VadBatchResult> => {
      const pending = activeVad;
      if (pending === undefined) {
        throw new Error("VAD batch queue is unexpectedly empty");
      }
      const outcome = await pending.outcome;
      activeVad = undefined;
      vadElapsedMs += Math.max(0, outcome.completedAtMs - pending.startedAtMs);
      try {
        if (!outcome.ok) throw outcome.error;
        if (outcome.result.done) {
          throw new Error("VAD stream ended before all advertised chunks ran");
        }
        throwIfCancelled(hooks.signal);
        const batch = outcome.result.value;
        const emission = reducer.consumeBatch(
          batch.rawSegments,
          batch.nextUnprocessedChunkTime,
        );
        streamedSubsegments.push(...emission.emittedSubsegments);
        vadModelRuns += 1;
        return batch;
      } catch (error) {
        rethrowForStage("vad", error, hooks.signal);
      }
    };

    let subsegments: readonly Subsegment[];
    let vadSegments: readonly { readonly start: number; readonly end: number }[];
    try {
      if (vadChunkCount > 0) {
        activeVad = startVadBatch(vadIterator, now);
        let completed = await consumeActiveVad();
        while (completed.completedChunks < completed.totalChunks) {
          // Submit VAD first so its GPU work is live while CPU FBank and the
          // separate CAM++ device consume the stable prefix from prior logits.
          activeVad = startVadBatch(vadIterator, now);
          await pumpEmbeddingBatches(2, false);
          completed = await consumeActiveVad();
        }
      }
      await vadIterator.return?.();
      const finished = reducer.finish();
      vadSegments = finished.vadSegments;
      subsegments = finished.subsegments;
      availableSubsegments = subsegments;
      hooks.onSubsegmentsCreated?.(subsegments);
      throwIfCancelled(hooks.signal);

      const vadResult: StageResult<"vad"> = {
        stage: "vad",
        elapsedMs: vadElapsedMs,
        metrics: {
          modelRuns: vadModelRuns,
          regionCount: vadSegments.length,
          speechSeconds: sumDurations(vadSegments),
        },
      };
      completeStage(vadResult, stages, hooks);
      const currentPcmCacheBytes =
        (extractor?.stats.peakCachedSamples ?? 0) *
        Float32Array.BYTES_PER_ELEMENT;
      memory.setCurrentKnownCpuBytes(
        wavReadBufferBytes +
          currentPcmCacheBytes +
          batchFeatures.length * featureBatchBytes +
          pendingEmbeddings.length * outputBatchBytes +
          bufferedEmbeddingBytes,
      );
      memory.checkpoint("vad", "complete");

      embeddings = new Float32Array(
        subsegments.length * embedding.embeddingDim,
      );
      peakRetainedEmbeddingBytes = Math.max(
        peakRetainedEmbeddingBytes,
        embeddings.byteLength + bufferedEmbeddingBytes,
      );
      observeStreamingCpu(false);
      for (const chunk of bufferedEmbeddingChunks) {
        embeddings.set(
          chunk.values,
          chunk.batchStart * embedding.embeddingDim,
        );
      }
      bufferedEmbeddingChunks.length = 0;
      bufferedEmbeddingBytes = 0;

      await pumpEmbeddingBatches(Number.POSITIVE_INFINITY, true);
      while (pendingEmbeddings.length > 0) {
        await settleOldestEmbedding();
      }
      if (nextEmbeddingStart !== subsegments.length) {
        throw new Error(
          `CAM++ embedded ${nextEmbeddingStart}/${subsegments.length} windows`,
        );
      }
    } catch (error) {
      // All submitted outcomes are non-rejecting. Drain both devices before
      // propagating a cancellation/failure so neither fixed readback slot can
      // be reused while work from this job is still in flight.
      await Promise.all([
        ...(activeVad === undefined ? [] : [activeVad.outcome]),
        ...pendingEmbeddings.map((pending) => pending.outcome),
      ]);
      try {
        await vadIterator.return?.();
      } catch {
        // Preserve the original stage error.
      }
      throwIfCancelled(hooks.signal);
      throw error;
    }

    const finalEmbeddings = embeddings!;
    const retainedEmbeddingsBytes = finalEmbeddings.byteLength;
    const camInputBatchBytes = batchFeatures.length * featureBatchBytes;
    const camOutputBatchBytes = peakPendingEmbeddingRuns * outputBatchBytes;
    memory.recordAllocation(
      "retainedEmbeddingsBytes",
      peakRetainedEmbeddingBytes,
    );
    memory.recordAllocation("camInputBatchBytes", camInputBatchBytes);
    memory.recordAllocation("camOutputBatchBytes", camOutputBatchBytes);
    // Drop the two 768 KB host upload slots before CPU-only clustering. Their
    // contents have already been copied into GPU-owned input buffers.
    batchFeatures = [];

    const pcmCachePeakBytes =
      (extractor?.stats.peakCachedSamples ?? 0) *
      Float32Array.BYTES_PER_ELEMENT;
    memory.recordAllocation("pcmCachePeakBytes", pcmCachePeakBytes);
    memory.observeKnownCpuPeakBytes(
      wavReadBufferBytes +
        pcmCachePeakBytes +
        camInputBatchBytes +
        camOutputBatchBytes +
        peakRetainedEmbeddingBytes,
    );
    // The CAM input batches are no longer logically retained after the loop. The
    // PCM cache remains owned by the streaming extractor until final disposal.
    memory.setCurrentKnownCpuBytes(
      wavReadBufferBytes + pcmCachePeakBytes + retainedEmbeddingsBytes,
    );

    const fbankResult: StageResult<"fbank"> = {
      stage: "fbank",
      elapsedMs: fbankElapsedMs,
      metrics: {
        frameCount: totalFrameCount,
        melBinCount: SENKO_FBANK_BINS,
        windowCount: subsegments.length,
      },
    };
    completeStage(fbankResult, stages, hooks);
    memory.checkpoint("fbank", "complete");
    const embeddingResult: StageResult<"embedding"> = {
      stage: "embedding",
      elapsedMs: embeddingElapsedMs,
      metrics: {
        batchCount: embeddingBatchCount,
        embeddingCount: subsegments.length,
        dimensions: embedding.embeddingDim,
      },
    };
    completeStage(embeddingResult, stages, hooks);
    memory.checkpoint("embedding", "complete");

    // FBank and its decoded PCM cache are dead before clustering allocates its
    // working graph. Keep the smaller WAV reader buffer for post-processing.
    if (extractor === undefined) {
      fbank?.dispose?.();
    } else {
      extractor.dispose();
    }
    extractor = undefined;
    fbank = undefined;
    memory.setCurrentKnownCpuBytes(
      wavReadBufferBytes + retainedEmbeddingsBytes,
    );

    memory.checkpoint("clustering", "start");
    hooks.onStageStarted?.("clustering");
    const clusteringStart = now();
    const clusteringAlgorithm = selectClusteringAlgorithm(subsegments);
    const recordClusteringMemory = (peakWorkingBytes: number): void => {
      memory.recordAllocation(
        "clusteringPeakWorkingBytes",
        peakWorkingBytes,
      );
      memory.observeKnownCpuPeakBytes(
        wavReadBufferBytes + retainedEmbeddingsBytes + peakWorkingBytes,
      );
    };
    let labels: Int32Array;
    try {
      if (clusteringAlgorithm === "spectral") {
        labels = clusterEmbeddingsSpectral(
          finalEmbeddings,
          subsegments.length,
          embedding.embeddingDim,
          {
            onStats(stats) {
              recordClusteringMemory(stats.peakWorkingBytes);
            },
          },
        );
      } else if (hooks.nativeUmapClustering !== undefined) {
        const native = await hooks.nativeUmapClustering.clusterNativeUmap(
          finalEmbeddings,
          subsegments.length,
          embedding.embeddingDim,
          hooks.signal,
        );
        labels = native.labels;
        recordClusteringMemory(native.stats.peakWorkingBytes);
      } else {
        labels = clusterEmbeddings(
          finalEmbeddings,
          subsegments.length,
          embedding.embeddingDim,
          {
            onUmapStats(stats) {
              recordClusteringMemory(
                Math.max(
                  stats.peakWorkingBytes,
                  estimatePostUmapPeakWorkingBytes(
                    stats.count,
                    stats.outputDimension,
                    DEFAULT_CLUSTERING_OPTIONS.neighborCount,
                  ),
                ),
              );
            },
          },
          hooks.clusteringKernels,
        );
      }
      throwIfCancelled(hooks.signal);
    } catch (error) {
      rethrowForStage("clustering", error, hooks.signal);
    }
    // A growable clustering backend may reserve more linear memory while
    // processing this recording. Retain the disposed FBank heap's known size
    // and refresh clustering ownership after its final numeric operation.
    memory.setWasmHeapBytes(
      sumDefinedBytes(
        fbankWasmHeapBytes,
        readExposedWasmHeapBytes(clusteringMemorySource),
      ),
    );
    const labelStats = countLabels(labels);
    const clusteringResult: StageResult<"clustering"> = {
      stage: "clustering",
      elapsedMs: elapsedSince(now, clusteringStart),
      metrics: {
        algorithm: clusteringAlgorithm,
        clusterCount: labelStats.clusterCount,
        noiseCount: labelStats.noiseCount,
      },
    };
    completeStage(clusteringResult, stages, hooks);
    const clusterLabelsBytes = labels.byteLength;
    memory.recordAllocation("clusterLabelsBytes", clusterLabelsBytes);
    memory.setCurrentKnownCpuBytes(
      wavReadBufferBytes +
        retainedEmbeddingsBytes +
        clusterLabelsBytes,
    );
    memory.checkpoint("clustering", "complete");

    memory.checkpoint("postprocess", "start");
    hooks.onStageStarted?.("postprocess");
    const postprocessStart = now();
    let processed;
    try {
      processed = postprocessClustering(finalEmbeddings, labels, subsegments);
      throwIfCancelled(hooks.signal);
    } catch (error) {
      rethrowForStage("postprocess", error, hooks.signal);
    }
    const postprocessResult: StageResult<"postprocess"> = {
      stage: "postprocess",
      elapsedMs: elapsedSince(now, postprocessStart),
      metrics: {
        segmentCount: processed.mergedSegments.length,
        speakerCount: processed.speakerCount,
      },
    };
    completeStage(postprocessResult, stages, hooks);
    memory.checkpoint("postprocess", "complete");

    const totalElapsedMs = elapsedSince(now, totalStart);
    memory.checkpoint("pipeline", "complete");

    return {
      durationSeconds: reader.info.durationSeconds,
      speakerCount: processed.speakerCount,
      segments: processed.mergedSegments,
      stages,
      totalElapsedMs,
      memory: memory.summary(),
    };
  } finally {
    if (extractor === undefined) {
      fbank?.dispose?.();
    } else {
      extractor.dispose();
    }
  }
}

function startVadBatch(
  iterator: AsyncIterator<VadBatchResult, void, void>,
  now: () => number,
): PendingVadBatch {
  const startedAtMs = now();
  let next: Promise<IteratorResult<VadBatchResult, void>>;
  try {
    next = iterator.next();
  } catch (error) {
    next = Promise.reject(error);
  }
  const outcome = next.then(
    (result): VadBatchOutcome => ({
      ok: true,
      result,
      completedAtMs: now(),
    }),
    (error: unknown): VadBatchOutcome => ({
      ok: false,
      error,
      completedAtMs: now(),
    }),
  );
  return { outcome, startedAtMs };
}

function startEmbeddingRun(
  embedding: EmbeddingBatchBackend,
  features: Float32Array,
  batchStart: number,
  actualBatchSize: number,
  featureSlot: number,
  now: () => number,
): PendingEmbeddingRun {
  const startedAtMs = now();
  let run: Promise<Float32Array>;
  try {
    run = embedding.run(features);
  } catch (error) {
    run = Promise.reject(error);
  }
  // Attach both handlers immediately: FBank for the next batch can span many
  // task turns, and a fast model failure must not become an unhandled rejection
  // before the orchestrator reaches the settlement point.
  const outcome: Promise<EmbeddingRunOutcome> = run.then(
    (output): EmbeddingRunOutcome => ({
      ok: true,
      output,
      completedAtMs: now(),
    }),
    (error: unknown): EmbeddingRunOutcome => ({
      ok: false,
      error,
      completedAtMs: now(),
    }),
  );
  return {
    outcome,
    batchStart,
    actualBatchSize,
    featureSlot,
    startedAtMs,
  };
}

async function settleEmbeddingRun(
  pending: PendingEmbeddingRun,
  embedding: EmbeddingBatchBackend,
  embeddings: Float32Array | undefined,
  signal: AbortSignal | undefined,
  coveredUntilMs: number | undefined,
): Promise<SettledEmbeddingRun> {
  const outcome = await pending.outcome;
  try {
    if (!outcome.ok) throw outcome.error;
    const actualValueCount =
      pending.actualBatchSize * embedding.embeddingDim;
    if (outcome.output.length < actualValueCount) {
      throw new Error(
        `CAM++ produced ${outcome.output.length} values; expected at least ${actualValueCount}`,
      );
    }
    const values = outcome.output.subarray(0, actualValueCount);
    const chunk =
      embeddings === undefined
        ? { batchStart: pending.batchStart, values: values.slice() }
        : undefined;
    embeddings?.set(
      values,
      pending.batchStart * embedding.embeddingDim,
    );
    throwIfCancelled(signal);
    const attributedStartMs = Math.max(
      pending.startedAtMs,
      coveredUntilMs ?? pending.startedAtMs,
    );
    return {
      attributedElapsedMs: Math.max(
        0,
        outcome.completedAtMs - attributedStartMs,
      ),
      coveredUntilMs: Math.max(
        coveredUntilMs ?? pending.startedAtMs,
        outcome.completedAtMs,
      ),
      ...(chunk === undefined ? {} : { chunk }),
    };
  } catch (error) {
    rethrowForStage("embedding", error, signal);
  }
}

function embeddingRunConcurrency(embedding: EmbeddingBatchBackend): number {
  const declared = embedding.maxInFlightRuns ?? 1;
  if (!Number.isSafeInteger(declared) || declared <= 0) {
    throw new RangeError("CAM++ maxInFlightRuns must be a positive safe integer");
  }
  return Math.min(2, declared);
}

function validateModelOptions(
  vad: VadBatchBackend,
  embedding: EmbeddingBatchBackend,
  options: PipelineOptions,
): void {
  const modelVadSeconds = vad.chunkSamples / VAD_SAMPLE_RATE;
  if (options.vadChunkSeconds !== modelVadSeconds) {
    throw new RangeError(
      `The loaded VAD graph requires ${modelVadSeconds}-second chunks; received ${options.vadChunkSeconds}`,
    );
  }
  if (embedding.featureDim !== SENKO_FBANK_BINS) {
    throw new Error(
      `The loaded CAM++ graph expects ${embedding.featureDim} features; Senko FBank emits ${SENKO_FBANK_BINS}`,
    );
  }
}

function wavReaderAsMonoSource(reader: Pcm16WavReader): MonoPcmSource {
  return {
    sampleRate: reader.sampleRate,
    sampleCount: reader.sampleCount,
    async readInto(
      sampleOffset,
      sampleCount,
      destination,
      destinationOffset = 0,
    ) {
      const written = await reader.readSamplesInto(
        sampleOffset,
        destination,
        destinationOffset,
        sampleCount,
      );
      if (written !== sampleCount) {
        throw new Error(
          `Short PCM read at sample ${sampleOffset}: expected ${sampleCount}, got ${written}`,
        );
      }
    },
  };
}

function copyFeatureWindow(
  source: Float32Array,
  sourceFrames: number,
  featureDim: number,
  destination: Float32Array,
  destinationRow: number,
  targetFrames: number,
): void {
  if (source.length !== sourceFrames * featureDim) {
    throw new Error("FBank matrix shape does not match its data length");
  }
  const copiedFrames = Math.min(sourceFrames, targetFrames);
  // Native Senko center-crops the exceptional over-length case. Normal
  // 1.5-second windows produce 148 frames and are zero-padded to 150.
  const sourceStartFrame = Math.floor((sourceFrames - copiedFrames) / 2);
  const sourceStart = sourceStartFrame * featureDim;
  const destinationStart = destinationRow * targetFrames * featureDim;
  destination.set(
    source.subarray(sourceStart, sourceStart + copiedFrames * featureDim),
    destinationStart,
  );
}

function completeStage<S extends PipelineStage>(
  result: StageResult<S>,
  stages: AnyStageResult[],
  hooks: BrowserPipelineHooks,
): void {
  stages.push(result as AnyStageResult);
  hooks.onStageCompleted?.(result as AnyStageResult);
}

function countLabels(labels: Int32Array): {
  clusterCount: number;
  noiseCount: number;
} {
  // `clusterEmbeddings` returns CommonClustering's final normalized labels,
  // not raw HDBSCAN output. In particular, a sufficiently large `-1` group is
  // a speaker after Senko's post-processing and must not be counted as noise.
  const clusters = new Set<number>();
  let noiseCount = 0;
  for (const label of labels) {
    if (label < 0) noiseCount += 1;
    else clusters.add(label);
  }
  return { clusterCount: clusters.size, noiseCount };
}

function sumDurations(
  segments: readonly { start: number; end: number }[],
): number {
  let total = 0;
  for (const segment of segments) total += segment.end - segment.start;
  return total;
}

function sumDefinedBytes(
  left: number | undefined,
  right: number | undefined,
): number | undefined {
  if (left === undefined && right === undefined) return undefined;
  return (left ?? 0) + (right ?? 0);
}

function throwIfCancelled(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) throw new BrowserPipelineCancelledError();
}

function rethrowForStage(
  stage: PipelineStage,
  error: unknown,
  signal: AbortSignal | undefined,
): never {
  if (
    error instanceof BrowserPipelineCancelledError || signal?.aborted === true
  ) {
    throw new BrowserPipelineCancelledError();
  }
  if (error instanceof BrowserPipelineStageError) throw error;
  throw new BrowserPipelineStageError(stage, error);
}

function elapsedSince(now: () => number, startedAt: number): number {
  return Math.max(0, now() - startedAt);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
