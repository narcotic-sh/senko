import type {
  PipelineMemoryAllocations,
  PipelineMemoryCheckpoint,
  PipelineMemoryCheckpointPhase,
  PipelineMemoryCheckpointStage,
  PipelineMemorySummary,
} from "./types";

interface ChromiumPerformanceMemory {
  readonly usedJSHeapSize: number;
}

/** Minimal injectable shape used by the typed Chromium feature probe. */
export interface MemoryPerformanceSource {
  readonly memory?: unknown;
}

type MutableAllocations = {
  -readonly [Key in keyof PipelineMemoryAllocations]: PipelineMemoryAllocations[Key];
};

/** Read Chromium's non-standard heap counter without assuming it exists. */
export function readChromiumUsedJsHeapBytes(
  source: MemoryPerformanceSource | undefined = defaultPerformanceSource(),
): number | undefined {
  const candidate = source?.memory;
  if (typeof candidate !== "object" || candidate === null) {
    return undefined;
  }
  const used = (candidate as Partial<ChromiumPerformanceMemory>).usedJSHeapSize;
  return isByteCount(used) ? used : undefined;
}

/** Probe a structurally exposed WASM heap, including growable implementations. */
export function readExposedWasmHeapBytes(value: unknown): number | undefined {
  if (typeof value !== "object" || value === null || !("memoryStats" in value)) {
    return undefined;
  }
  const memoryStats = value.memoryStats;
  if (
    typeof memoryStats !== "object" ||
    memoryStats === null ||
    !("heapBytes" in memoryStats)
  ) {
    return undefined;
  }
  return isByteCount(memoryStats.heapBytes) ? memoryStats.heapBytes : undefined;
}

export class PipelineMemoryTracker {
  readonly #source: MemoryPerformanceSource | undefined;
  readonly #allocations: MutableAllocations;
  readonly #checkpoints: PipelineMemoryCheckpoint[] = [];
  #currentKnownCpuBytes = 0;
  #knownCpuPeakBytes = 0;
  #wasmHeapBytes: number | undefined;
  #knownGpuBufferBytes: number | undefined;
  #jsHeapPeakBytes: number | undefined;

  constructor(
    audioBlobBytes: number,
    source: MemoryPerformanceSource | undefined = defaultPerformanceSource(),
  ) {
    requireByteCount("audioBlobBytes", audioBlobBytes);
    this.#source = source;
    this.#allocations = {
      audioBlobBytes,
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
    };
  }

  setWasmHeapBytes(bytes: number | undefined): void {
    if (bytes === undefined) return;
    requireByteCount("wasmHeapBytes", bytes);
    this.#wasmHeapBytes = Math.max(this.#wasmHeapBytes ?? 0, bytes);
  }

  setKnownGpuBufferBytes(bytes: number | undefined): void {
    if (bytes === undefined) return;
    requireByteCount("knownGpuBufferBytes", bytes);
    this.#knownGpuBufferBytes = Math.max(
      this.#knownGpuBufferBytes ?? 0,
      bytes,
    );
  }

  recordAllocation(
    name: Exclude<keyof PipelineMemoryAllocations, "audioBlobBytes" | "audioBlobCopied">,
    bytes: number,
  ): void {
    requireByteCount(name, bytes);
    this.#allocations[name] = Math.max(this.#allocations[name], bytes);
  }

  setCurrentKnownCpuBytes(bytes: number): void {
    requireByteCount("currentKnownCpuBytes", bytes);
    this.#currentKnownCpuBytes = bytes;
    this.observeKnownCpuPeakBytes(bytes);
  }

  observeKnownCpuPeakBytes(bytes: number): void {
    requireByteCount("knownCpuPeakBytes", bytes);
    this.#knownCpuPeakBytes = Math.max(this.#knownCpuPeakBytes, bytes);
  }

  checkpoint(
    stage: PipelineMemoryCheckpointStage,
    phase: PipelineMemoryCheckpointPhase,
  ): void {
    const jsHeapBytes = readChromiumUsedJsHeapBytes(this.#source);
    if (jsHeapBytes !== undefined) {
      this.#jsHeapPeakBytes = Math.max(this.#jsHeapPeakBytes ?? 0, jsHeapBytes);
    }
    this.#checkpoints.push({
      stage,
      phase,
      knownCpuBytes: this.#currentKnownCpuBytes,
      ...(jsHeapBytes === undefined ? {} : { jsHeapBytes }),
    });
  }

  summary(): PipelineMemorySummary {
    return {
      knownCpuPeakBytes: this.#knownCpuPeakBytes,
      ...(this.#wasmHeapBytes === undefined
        ? {}
        : { wasmHeapBytes: this.#wasmHeapBytes }),
      ...(this.#knownGpuBufferBytes === undefined
        ? {}
        : { knownGpuBufferBytes: this.#knownGpuBufferBytes }),
      ...(this.#jsHeapPeakBytes === undefined
        ? {}
        : { jsHeapPeakBytes: this.#jsHeapPeakBytes }),
      allocations: { ...this.#allocations },
      checkpoints: [...this.#checkpoints],
    };
  }
}

function defaultPerformanceSource(): MemoryPerformanceSource | undefined {
  if (typeof performance === "undefined") return undefined;
  return performance as Performance & MemoryPerformanceSource;
}

function isByteCount(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isSafeInteger(value) &&
    value >= 0
  );
}

function requireByteCount(name: string, value: number): void {
  if (!isByteCount(value)) {
    throw new RangeError(`${name} must be a non-negative safe integer`);
  }
}
