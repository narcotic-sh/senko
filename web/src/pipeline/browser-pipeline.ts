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
  type ClusteringNumericKernels,
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
import { postprocessClustering } from "./postprocess";
import { createSubsegments } from "./subsegments";
import type {
  EmbeddingBatchBackend,
  MonoPcmSource,
  Subsegment,
  VadBatchBackend,
} from "./types";
import { createVadChunks, runVad, VAD_SAMPLE_RATE } from "./vad";

export interface BrowserPipelineModels {
  readonly vad: VadBatchBackend;
  readonly embedding: EmbeddingBatchBackend;
  /** Explicit GPU buffers owned by the model set; opaque ORT arenas excluded. */
  readonly knownGpuBufferBytes?: number;
  /** Load and warm VAD after a previous pipeline run released its models. */
  readonly prepareVadStage?: () => Promise<void>;
  /** Release VAD, then load and warm CAM++ only when embeddings are needed. */
  readonly prepareEmbeddingStage?: (
    hasEmbeddingWork: boolean,
  ) => Promise<void>;
  /** Release the embedding model before CPU-only clustering begins. */
  readonly finishEmbeddingStage?: () => Promise<void>;
}

export interface BrowserPipelineHooks {
  readonly signal?: AbortSignal;
  readonly onStageStarted?: (stage: PipelineStage) => void;
  readonly onStageCompleted?: (result: AnyStageResult) => void;
  /** Injectable monotonic clock for focused orchestration tests. */
  readonly now?: () => number;
  /** Tests can inject the TypeScript reference without fetching a WASM asset. */
  readonly createFbank?: () => Promise<FbankComputer>;
  /** Preloaded fixed-memory numeric kernels; production supplies one per worker. */
  readonly clusteringKernels?: ClusteringNumericKernels;
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
  let fbank: FbankComputer | undefined;
  let extractor: StreamingFbankExtractor | undefined;
  let embeddingStageNeedsFinish = false;
  const clusteringWasmHeapBytes = readExposedWasmHeapBytes(
    hooks.clusteringKernels,
  );

  try {
    try {
      await models.prepareVadStage?.();
      throwIfCancelled(hooks.signal);
    } catch (error) {
      rethrowForStage("vad", error, hooks.signal);
    }
    const vad = models.vad;
    const embedding = models.embedding;
    memory.setKnownGpuBufferBytes(models.knownGpuBufferBytes);
    validateModelOptions(vad, embedding, options);

    try {
      fbank = await (hooks.createFbank ?? (() => WasmSenkoFbank.create()))();
      memory.setWasmHeapBytes(
        sumDefinedBytes(
          readExposedWasmHeapBytes(fbank),
          clusteringWasmHeapBytes,
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
    const vadStart = now();
    const pcmSource = wavReaderAsMonoSource(reader);
    let vadSegments;
    try {
      vadSegments = await runVad(pcmSource, vad, () => {
        throwIfCancelled(hooks.signal);
      });
      throwIfCancelled(hooks.signal);
    } catch (error) {
      rethrowForStage("vad", error, hooks.signal);
    }
    const vadResult: StageResult<"vad"> = {
      stage: "vad",
      elapsedMs: elapsedSince(now, vadStart),
      metrics: {
        modelRuns: Math.ceil(vadChunkCount / vad.batchSize),
        regionCount: vadSegments.length,
        speechSeconds: sumDurations(vadSegments),
      },
    };
    completeStage(vadResult, stages, hooks);
    // runVad retains only decoded time segments after its final batch.
    memory.setCurrentKnownCpuBytes(wavReadBufferBytes);
    memory.checkpoint("vad", "complete");

    const subsegments = createSubsegments(vadSegments, {
      durationSeconds: options.embeddingWindowSeconds,
      shiftSeconds: options.embeddingShiftSeconds,
    });
    hooks.onSubsegmentsCreated?.(subsegments);

    // Feature extraction and embedding inference are deliberately interleaved.
    // Their elapsed values measure only work attributable to each stage, while
    // totalElapsedMs remains the end-to-end wall-clock measurement.
    let fbankElapsedMs = 0;
    let embeddingElapsedMs = 0;
    let totalFrameCount = 0;
    let embeddingBatchCount = 0;
    const embeddings = new Float32Array(
      subsegments.length * embedding.embeddingDim,
    );
    const retainedEmbeddingsBytes = embeddings.byteLength;
    const embeddingBatchTotal = Math.ceil(
      subsegments.length / embedding.batchSize,
    );
    const embeddingRunLimit = embeddingRunConcurrency(embedding);
    const camInputBatchBytes =
      Math.min(2, embeddingBatchTotal) *
      embedding.batchSize *
      embedding.frames *
      embedding.featureDim *
      Float32Array.BYTES_PER_ELEMENT;
    const camOutputBatchBytes =
      Math.min(embeddingRunLimit, embeddingBatchTotal) *
      embedding.batchSize *
      embedding.embeddingDim *
      Float32Array.BYTES_PER_ELEMENT;
    memory.recordAllocation("retainedEmbeddingsBytes", retainedEmbeddingsBytes);
    memory.recordAllocation("camInputBatchBytes", camInputBatchBytes);
    memory.recordAllocation("camOutputBatchBytes", camOutputBatchBytes);
    memory.setCurrentKnownCpuBytes(
      wavReadBufferBytes +
        retainedEmbeddingsBytes +
        camInputBatchBytes +
        camOutputBatchBytes,
    );
    memory.checkpoint("fbank", "start");
    memory.checkpoint("embedding", "start");
    hooks.onStageStarted?.("fbank");
    hooks.onStageStarted?.("embedding");

    if (models.prepareEmbeddingStage !== undefined) {
      const prepareStart = now();
      try {
        throwIfCancelled(hooks.signal);
        await models.prepareEmbeddingStage(subsegments.length > 0);
        embeddingStageNeedsFinish =
          subsegments.length > 0 && models.finishEmbeddingStage !== undefined;
        memory.setKnownGpuBufferBytes(models.knownGpuBufferBytes);
        throwIfCancelled(hooks.signal);
      } catch (error) {
        rethrowForStage("embedding", error, hooks.signal);
      } finally {
        embeddingElapsedMs += elapsedSince(now, prepareStart);
      }
    }

    if (subsegments.length > 0) {
      extractor = new StreamingFbankExtractor(reader, fbank);
      const featureValueCount =
        embedding.batchSize * embedding.frames * embedding.featureDim;
      const batchFeatures = Array.from(
        { length: Math.min(2, embeddingBatchTotal) },
        () => new Float32Array(featureValueCount),
      );
      const pendingEmbeddings: PendingEmbeddingRun[] = [];
      let coveredEmbeddingRunUntilMs: number | undefined;
      let batchIndex = 0;

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
      };

      try {
        for (
          let batchStart = 0;
          batchStart < subsegments.length;
          batchStart += embedding.batchSize
        ) {
          throwIfCancelled(hooks.signal);
          const actualBatchSize = Math.min(
            embedding.batchSize,
            subsegments.length - batchStart,
          );
          const featureSlot = batchIndex % batchFeatures.length;
          const features = batchFeatures[featureSlot]!;
          const occupyingRun = pendingEmbeddings.findIndex(
            (pending) => pending.featureSlot === featureSlot,
          );
          if (occupyingRun > 0) {
            throw new Error("CAM++ host feature slots lost FIFO ordering");
          }
          if (occupyingRun === 0) await settleOldestEmbedding();
          features.fill(0);

          const requests = subsegments
            .slice(batchStart, batchStart + actualBatchSize)
            .map((subsegment) =>
              secondsToFbankWindow(
                subsegment.start,
                subsegment.end,
                subsegment.index,
              )
            );
          const iterator = extractor.extract(requests)[Symbol.asyncIterator]();

          for (let row = 0; row < actualBatchSize; row += 1) {
            const fbankStart = now();
            try {
              throwIfCancelled(hooks.signal);
              const next = await iterator.next();
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
          // Direct CAM++ advertises two safe slots, so it can additionally queue
          // this batch before the preceding readback maps on the host.
          while (pendingEmbeddings.length >= embeddingRunLimit) {
            await settleOldestEmbedding();
          }
          pendingEmbeddings.push(startEmbeddingRun(
            embedding,
            features,
            batchStart,
            actualBatchSize,
            featureSlot,
            now,
          ));
          batchIndex += 1;
        }

        while (pendingEmbeddings.length > 0) {
          await settleOldestEmbedding();
        }
      } catch (error) {
        // Outcome handlers never reject. Drain all submitted work before model
        // release so a later failure cannot become unhandled or destroy an
        // in-use readback slot.
        await Promise.all(pendingEmbeddings.map((pending) => pending.outcome));
        throw error;
      }
    }

    if (embeddingStageNeedsFinish) {
      const finishStart = now();
      try {
        await models.finishEmbeddingStage?.();
        embeddingStageNeedsFinish = false;
        throwIfCancelled(hooks.signal);
      } catch (error) {
        rethrowForStage("embedding", error, hooks.signal);
      } finally {
        embeddingElapsedMs += elapsedSince(now, finishStart);
      }
    }

    const pcmCachePeakBytes =
      (extractor?.stats.peakCachedSamples ?? 0) *
      Float32Array.BYTES_PER_ELEMENT;
    memory.recordAllocation("pcmCachePeakBytes", pcmCachePeakBytes);
    memory.observeKnownCpuPeakBytes(
      wavReadBufferBytes +
        pcmCachePeakBytes +
        camInputBatchBytes +
        camOutputBatchBytes +
        retainedEmbeddingsBytes,
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
      labels =
        clusteringAlgorithm === "spectral"
          ? clusterEmbeddingsSpectral(
              embeddings,
              subsegments.length,
              embedding.embeddingDim,
              {
                onStats(stats) {
                  recordClusteringMemory(stats.peakWorkingBytes);
                },
              },
            )
          : clusterEmbeddings(
              embeddings,
              subsegments.length,
              embedding.embeddingDim,
              {
                onUmapStats(stats) {
                  recordClusteringMemory(stats.peakWorkingBytes);
                },
              },
              hooks.clusteringKernels,
            );
      throwIfCancelled(hooks.signal);
    } catch (error) {
      rethrowForStage("clustering", error, hooks.signal);
    }
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
      processed = postprocessClustering(embeddings, labels, subsegments);
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
    if (embeddingStageNeedsFinish) {
      try {
        await models.finishEmbeddingStage?.();
      } catch {
        // Preserve the original cancellation or stage failure. Model release
        // is idempotent and the worker can reconstruct the stage on retry.
      }
    }
  }
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
  embeddings: Float32Array,
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
    embeddings.set(
      outcome.output.subarray(0, actualValueCount),
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
