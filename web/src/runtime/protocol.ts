import {
  PIPELINE_STAGES,
  type AnyStageResult,
  type PipelineAssetManifest,
  type PipelineOptions,
  type PipelineResult,
  type PipelineStage,
} from "./types";

export interface InitializeRequest {
  readonly type: "initialize";
  readonly requestId: string;
  readonly manifest: PipelineAssetManifest;
  readonly options: PipelineOptions;
}

export interface DiarizeRequest {
  readonly type: "diarize";
  readonly requestId: string;
  /** Blob is structured-cloned without copying an hour of PCM into the main thread. */
  readonly audio: Blob;
  readonly fileName: string;
}

export interface CancelRequest {
  readonly type: "cancel";
  readonly requestId: string;
  readonly targetRequestId: string;
}

export type PipelineWorkerRequest =
  | InitializeRequest
  | DiarizeRequest
  | CancelRequest;

export interface WorkerRuntimeInfo {
  readonly crossOriginIsolated: boolean;
  readonly sharedArrayBuffer: boolean;
  readonly webgpu: boolean;
}

export interface InitializedResponse {
  readonly type: "initialized";
  readonly requestId: string;
  readonly runtime: WorkerRuntimeInfo;
}

export interface InitializationProgressResponse {
  readonly type: "initialization-progress";
  readonly requestId: string;
  readonly message: string;
}

export interface PipelineStartedResponse {
  readonly type: "pipeline-started";
  readonly requestId: string;
  readonly fileName: string;
  readonly byteLength: number;
}

export interface StageStartedResponse {
  readonly type: "stage-started";
  readonly requestId: string;
  readonly stage: PipelineStage;
}

export interface StageCompletedResponse {
  readonly type: "stage-completed";
  readonly requestId: string;
  readonly result: AnyStageResult;
}

export interface PipelineCompletedResponse {
  readonly type: "pipeline-completed";
  readonly requestId: string;
  readonly result: PipelineResult;
}

export type PipelineErrorCode =
  | "INVALID_REQUEST"
  | "NOT_INITIALIZED"
  | "NOT_IMPLEMENTED"
  | "ASSET_LOAD_FAILED"
  | "UNSUPPORTED_RUNTIME"
  | "PIPELINE_FAILED";

export interface PipelineFailedResponse {
  readonly type: "pipeline-failed";
  readonly requestId: string;
  readonly code: PipelineErrorCode;
  readonly message: string;
  readonly stage?: PipelineStage;
}

export interface PipelineCancelledResponse {
  readonly type: "pipeline-cancelled";
  readonly requestId: string;
}

export type PipelineWorkerResponse =
  | InitializationProgressResponse
  | InitializedResponse
  | PipelineStartedResponse
  | StageStartedResponse
  | StageCompletedResponse
  | PipelineCompletedResponse
  | PipelineFailedResponse
  | PipelineCancelledResponse;

export type PipelineProgressResponse =
  | PipelineStartedResponse
  | StageStartedResponse
  | StageCompletedResponse;

const RESPONSE_TYPES = new Set<PipelineWorkerResponse["type"]>([
  "initialized",
  "initialization-progress",
  "pipeline-started",
  "stage-started",
  "stage-completed",
  "pipeline-completed",
  "pipeline-failed",
  "pipeline-cancelled",
]);

const STAGES = new Set<string>(PIPELINE_STAGES);
const ASSET_ROLES = new Set([
  "vad-graph",
  "embedding-graph",
  "vad-weights",
  "embedding-weights",
  "runtime-data",
]);
const ASSET_FORMATS = new Set(["onnx", "safetensors", "binary", "json"]);
const ERROR_CODES = new Set<PipelineErrorCode>([
  "INVALID_REQUEST",
  "NOT_INITIALIZED",
  "NOT_IMPLEMENTED",
  "ASSET_LOAD_FAILED",
  "UNSUPPORTED_RUNTIME",
  "PIPELINE_FAILED",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNonNegativeNumber(value: unknown): value is number {
  return isFiniteNumber(value) && value >= 0;
}

function isByteCount(value: unknown): value is number {
  return isNonNegativeNumber(value) && Number.isSafeInteger(value);
}

function isStage(value: unknown): value is PipelineStage {
  return typeof value === "string" && STAGES.has(value);
}

function isAssetManifest(value: unknown): value is PipelineAssetManifest {
  if (!isRecord(value) || value.schemaVersion !== 1) {
    return false;
  }
  if (typeof value.pipelineVersion !== "string" || !Array.isArray(value.assets)) {
    return false;
  }
  return value.assets.every(
    (asset) =>
      isRecord(asset) &&
      typeof asset.id === "string" &&
      typeof asset.role === "string" &&
      ASSET_ROLES.has(asset.role) &&
      typeof asset.format === "string" &&
      ASSET_FORMATS.has(asset.format) &&
      typeof asset.url === "string" &&
      isNonNegativeNumber(asset.byteLength) &&
      Number.isSafeInteger(asset.byteLength) &&
      typeof asset.sha256 === "string" &&
      /^[0-9a-f]{64}$/.test(asset.sha256),
  );
}

function isPipelineOptions(value: unknown): value is PipelineOptions {
  return (
    isRecord(value) &&
    isFiniteNumber(value.vadChunkSeconds) &&
    value.vadChunkSeconds > 0 &&
    isFiniteNumber(value.embeddingWindowSeconds) &&
    value.embeddingWindowSeconds > 0 &&
    isFiniteNumber(value.embeddingShiftSeconds) &&
    value.embeddingShiftSeconds > 0 &&
    isFiniteNumber(value.maxSpeakers) &&
    Number.isSafeInteger(value.maxSpeakers) &&
    value.maxSpeakers > 0 &&
    typeof value.preferFloat16 === "boolean"
  );
}

function hasMetricNumbers(
  metrics: Record<string, unknown>,
  keys: readonly string[],
): boolean {
  return keys.every((key) => isNonNegativeNumber(metrics[key]));
}

function isStageMetrics(stage: PipelineStage, value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }
  switch (stage) {
    case "decode":
      return hasMetricNumbers(value, [
        "durationSeconds",
        "sampleRate",
        "channelCount",
        "sampleCount",
      ]);
    case "vad":
      return hasMetricNumbers(value, [
        "modelRuns",
        "regionCount",
        "speechSeconds",
      ]);
    case "fbank":
      return hasMetricNumbers(value, [
        "frameCount",
        "melBinCount",
        "windowCount",
      ]);
    case "embedding":
      return hasMetricNumbers(value, [
        "batchCount",
        "embeddingCount",
        "dimensions",
      ]);
    case "clustering":
      return (
        (value.algorithm === "spectral" || value.algorithm === "umap-hdbscan") &&
        hasMetricNumbers(value, ["clusterCount", "noiseCount"])
      );
    case "postprocess":
      return hasMetricNumbers(value, ["segmentCount", "speakerCount"]);
  }
}

function isStageResult(value: unknown): value is AnyStageResult {
  return (
    isRecord(value) &&
    isStage(value.stage) &&
    isNonNegativeNumber(value.elapsedMs) &&
    isStageMetrics(value.stage, value.metrics)
  );
}

function isPipelineMemoryCheckpoint(value: unknown): boolean {
  return (
    isRecord(value) &&
    (value.stage === "pipeline" || isStage(value.stage)) &&
    (value.phase === "start" || value.phase === "complete") &&
    isByteCount(value.knownCpuBytes) &&
    (value.jsHeapBytes === undefined || isByteCount(value.jsHeapBytes))
  );
}

function isPipelineMemorySummary(value: unknown): boolean {
  if (!isRecord(value)) return false;
  const allocations = value.allocations;
  if (
    !isByteCount(value.knownCpuPeakBytes) ||
    (value.wasmHeapBytes !== undefined && !isByteCount(value.wasmHeapBytes)) ||
    (value.knownGpuBufferBytes !== undefined &&
      !isByteCount(value.knownGpuBufferBytes)) ||
    (value.jsHeapPeakBytes !== undefined && !isByteCount(value.jsHeapPeakBytes)) ||
    !isRecord(allocations) ||
    allocations.audioBlobCopied !== false ||
    !Array.isArray(value.checkpoints)
  ) {
    return false;
  }
  return (
    [
      "audioBlobBytes",
      "wavReadBufferBytes",
      "vadInputBatchBytes",
      "vadLogitsBatchBytes",
      "pcmCachePeakBytes",
      "camInputBatchBytes",
      "camOutputBatchBytes",
      "retainedEmbeddingsBytes",
      "clusterLabelsBytes",
      "clusteringPeakWorkingBytes",
    ].every((key) => isByteCount(allocations[key])) &&
    value.checkpoints.every(isPipelineMemoryCheckpoint)
  );
}

function isPipelineResult(value: unknown): value is PipelineResult {
  return (
    isRecord(value) &&
    isNonNegativeNumber(value.durationSeconds) &&
    isNonNegativeNumber(value.speakerCount) &&
    Array.isArray(value.segments) &&
    value.segments.every(
      (segment) =>
        isRecord(segment) &&
        isNonNegativeNumber(segment.startSeconds) &&
        isNonNegativeNumber(segment.endSeconds) &&
        segment.endSeconds >= segment.startSeconds &&
        typeof segment.speaker === "string",
    ) &&
    Array.isArray(value.stages) &&
    value.stages.every(isStageResult) &&
    isNonNegativeNumber(value.totalElapsedMs) &&
    isPipelineMemorySummary(value.memory)
  );
}

export function isPipelineWorkerRequest(
  value: unknown,
): value is PipelineWorkerRequest {
  if (
    !isRecord(value) ||
    typeof value.type !== "string" ||
    typeof value.requestId !== "string" ||
    value.requestId.length === 0
  ) {
    return false;
  }

  switch (value.type) {
    case "initialize":
      return isAssetManifest(value.manifest) && isPipelineOptions(value.options);
    case "diarize":
      return value.audio instanceof Blob && typeof value.fileName === "string";
    case "cancel":
      return (
        typeof value.targetRequestId === "string" &&
        value.targetRequestId.length > 0
      );
    default:
      return false;
  }
}

/** Reject malformed cross-thread messages before they enter application state. */
export function isPipelineWorkerResponse(
  value: unknown,
): value is PipelineWorkerResponse {
  if (!isRecord(value)) {
    return false;
  }

  if (
    typeof value.type !== "string" ||
    !RESPONSE_TYPES.has(value.type as PipelineWorkerResponse["type"]) ||
    typeof value.requestId !== "string" ||
    value.requestId.length === 0
  ) {
    return false;
  }

  switch (value.type) {
    case "initialized":
      return (
        isRecord(value.runtime) &&
        typeof value.runtime.crossOriginIsolated === "boolean" &&
        typeof value.runtime.sharedArrayBuffer === "boolean" &&
        typeof value.runtime.webgpu === "boolean"
      );
    case "initialization-progress":
      return typeof value.message === "string" && value.message.length > 0;
    case "pipeline-started":
      return (
        typeof value.fileName === "string" &&
        typeof value.byteLength === "number"
      );
    case "stage-started":
      return isStage(value.stage);
    case "stage-completed":
      return isStageResult(value.result);
    case "pipeline-completed":
      return isPipelineResult(value.result);
    case "pipeline-failed":
      return (
        typeof value.code === "string" &&
        ERROR_CODES.has(value.code as PipelineErrorCode) &&
        typeof value.message === "string" &&
        (value.stage === undefined || isStage(value.stage))
      );
    case "pipeline-cancelled":
      return true;
    default:
      return false;
  }
}
