export const PIPELINE_STAGES = [
  "decode",
  "vad",
  "fbank",
  "embedding",
  "clustering",
  "postprocess",
] as const;

export type PipelineStage = (typeof PIPELINE_STAGES)[number];

export const PIPELINE_STAGE_LABELS: Readonly<Record<PipelineStage, string>> = {
  decode: "Decode",
  vad: "Voice activity",
  fbank: "Filter bank",
  embedding: "CAM++ embeddings",
  clustering: "Clustering",
  postprocess: "Postprocess",
};

export type ModelAssetRole =
  | "vad-graph"
  | "embedding-graph"
  | "vad-weights"
  | "embedding-weights"
  | "runtime-data";

export type ModelAssetFormat = "onnx" | "safetensors" | "binary" | "json";

/** A cacheable binary needed by a pipeline backend. URLs are resolved by the worker. */
export interface ModelAsset {
  readonly id: string;
  readonly role: ModelAssetRole;
  readonly format: ModelAssetFormat;
  readonly url: string;
  readonly byteLength: number;
  /** Lower-case, hex-encoded SHA-256 digest used to reject stale or corrupt assets. */
  readonly sha256: string;
}

export interface PipelineAssetManifest {
  readonly schemaVersion: 1;
  readonly pipelineVersion: string;
  readonly assets: readonly ModelAsset[];
}

export interface PipelineOptions {
  readonly vadChunkSeconds: number;
  readonly embeddingWindowSeconds: number;
  readonly embeddingShiftSeconds: number;
  readonly maxSpeakers: number;
  readonly preferFloat16: boolean;
}

export const DEFAULT_PIPELINE_OPTIONS: PipelineOptions = {
  vadChunkSeconds: 10,
  embeddingWindowSeconds: 1.5,
  embeddingShiftSeconds: 0.6,
  maxSpeakers: 15,
  preferFloat16: true,
};

export interface DecodeStageMetrics {
  readonly durationSeconds: number;
  readonly sampleRate: number;
  readonly channelCount: number;
  readonly sampleCount: number;
}

export interface VadStageMetrics {
  readonly modelRuns: number;
  readonly regionCount: number;
  readonly speechSeconds: number;
}

export interface FbankStageMetrics {
  readonly frameCount: number;
  readonly melBinCount: number;
  readonly windowCount: number;
}

export interface EmbeddingStageMetrics {
  readonly batchCount: number;
  readonly embeddingCount: number;
  readonly dimensions: number;
}

export interface ClusteringStageMetrics {
  readonly algorithm: "spectral" | "umap-hdbscan";
  /** Number of labels after Senko's minor-cluster and centroid-merge passes. */
  readonly clusterCount: number;
  /**
   * Negative labels after CommonClustering. A surviving HDBSCAN `-1`
   * population is an ordinary speaker label in Senko and is normalized to a
   * non-negative integer, so it is not reported here as raw HDBSCAN noise.
   */
  readonly noiseCount: number;
}

export interface PostprocessStageMetrics {
  readonly segmentCount: number;
  readonly speakerCount: number;
}

export interface StageMetricsByStage {
  readonly decode: DecodeStageMetrics;
  readonly vad: VadStageMetrics;
  readonly fbank: FbankStageMetrics;
  readonly embedding: EmbeddingStageMetrics;
  readonly clustering: ClusteringStageMetrics;
  readonly postprocess: PostprocessStageMetrics;
}

export interface StageResult<S extends PipelineStage> {
  readonly stage: S;
  readonly elapsedMs: number;
  readonly metrics: StageMetricsByStage[S];
}

export type AnyStageResult = {
  readonly [S in PipelineStage]: StageResult<S>;
}[PipelineStage];

export interface DiarizationSegment {
  readonly startSeconds: number;
  readonly endSeconds: number;
  readonly speaker: string;
}

export type PipelineMemoryCheckpointStage = PipelineStage | "pipeline";
export type PipelineMemoryCheckpointPhase = "start" | "complete";

export interface PipelineMemoryCheckpoint {
  readonly stage: PipelineMemoryCheckpointStage;
  readonly phase: PipelineMemoryCheckpointPhase;
  /** Deterministic live typed-array bytes known to the orchestrator. */
  readonly knownCpuBytes: number;
  /** Chromium's non-standard used JS heap counter, when exposed in the worker. */
  readonly jsHeapBytes?: number;
}

export interface PipelineMemoryAllocations {
  /** The input Blob stays externally backed and is read through slices. */
  readonly audioBlobBytes: number;
  readonly audioBlobCopied: false;
  /** One transferred/recycled BYOB backing store for random-access WAV reads. */
  readonly wavReadBufferBytes: number;
  readonly vadInputBatchBytes: number;
  readonly vadLogitsBatchBytes: number;
  readonly pcmCachePeakBytes: number;
  /** Total host bytes across the one or two pipelined CAM++ input batches. */
  readonly camInputBatchBytes: number;
  /** Returned CAM++ arrays retained by the bounded in-flight submission queue. */
  readonly camOutputBatchBytes: number;
  readonly retainedEmbeddingsBytes: number;
  readonly clusterLabelsBytes: number;
  /** Peak temporary/retained working state of the selected clustering branch. */
  readonly clusteringPeakWorkingBytes: number;
}

/**
 * Low-overhead memory telemetry. `knownCpuPeakBytes` is a deterministic lower
 * bound for explicitly owned buffers; it excludes engine object overhead,
 * WebGPU allocations, and opaque inference-runtime buffers.
 */
export interface PipelineMemorySummary {
  readonly knownCpuPeakBytes: number;
  readonly wasmHeapBytes?: number;
  /** Statically accounted WebGPU buffers, when a backend exposes them. */
  readonly knownGpuBufferBytes?: number;
  readonly jsHeapPeakBytes?: number;
  readonly allocations: PipelineMemoryAllocations;
  readonly checkpoints: readonly PipelineMemoryCheckpoint[];
}

export interface PipelineResult {
  readonly durationSeconds: number;
  readonly speakerCount: number;
  readonly segments: readonly DiarizationSegment[];
  readonly stages: readonly AnyStageResult[];
  readonly totalElapsedMs: number;
  readonly memory: PipelineMemorySummary;
}
