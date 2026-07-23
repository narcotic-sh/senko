import {
  mergeSimilarCentroids,
  normalizeLabels,
  reassignMinorClusters,
} from "./postprocess";
import {
  WasmClusteringKernels,
  type NativeUmapFuzzyGraph,
} from "./wasm-kernels";
import type {
  ThreadedUmapLayoutInput,
  ThreadedUmapLayoutResult,
} from "./threaded-umap-layout";

export const NATIVE_UMAP_NEIGHBOR_COUNT = 40;
export const NATIVE_UMAP_COMPONENT_COUNT = 60;
export const NATIVE_UMAP_A = 1.932808397545408;
export const NATIVE_UMAP_B = 0.7904949735905139;
export const NATIVE_HDBSCAN_MIN_SAMPLES = 20;
export const NATIVE_HDBSCAN_MIN_CLUSTER_SIZE = 10;
export const NATIVE_CENTROID_MERGE_THRESHOLD = 0.875;

export interface NativeUmapSerialStats {
  readonly count: number;
  readonly outputDimension: number;
  readonly epochCount: number;
  readonly retainedEdgeCount: number;
  readonly neighborMs: number;
  readonly fuzzyGraphMs: number;
  readonly spectralMs: number;
  readonly initializationMs: number;
  readonly layoutMs: number;
  readonly hdbscanMs: number;
  /** Peak caller-owned typed-array working set, excluding embeddings/WASM. */
  readonly peakWorkingBytes: number;
}

export interface NativeUmapSerialResult {
  readonly labels: Int32Array;
  readonly projection: Float32Array;
  readonly stats: NativeUmapSerialStats;
}

export interface NativeUmapThreadedStats extends NativeUmapSerialStats {
  readonly layoutWorkerCount: number;
  readonly layoutSharedMemoryBytes: number;
}

export interface NativeUmapThreadedResult {
  readonly labels: Int32Array;
  readonly projection: Float32Array;
  readonly stats: NativeUmapThreadedStats;
}

export interface NativeUmapLayoutExecutor {
  readonly workerCount: number;
  optimize(
    input: ThreadedUmapLayoutInput,
    signal?: AbortSignal,
  ): Promise<ThreadedUmapLayoutResult>;
}

interface LayoutGraph {
  readonly rowOffsets: Int32Array;
  readonly columnIndices: Int32Array;
  readonly values: Float32Array;
  readonly head: Int32Array;
  readonly epochsPerSample: Float64Array;
}

/**
 * Gated serial assembly of the native Senko UMAP/HDBSCAN path.
 *
 * Production keeps using the established fast implementation until the
 * threaded version of the 500-epoch layout clears this same end-to-end gate.
 */
export function clusterEmbeddingsNativeSerial(
  embeddings: Float32Array,
  count: number,
  dimension: number,
  randomSeed: number,
  kernels: WasmClusteringKernels,
): NativeUmapSerialResult {
  validateMatrix(embeddings, count, dimension);
  if (count < 10) {
    return {
      labels: new Int32Array(count),
      projection: new Float32Array(),
      stats: {
        count,
        outputDimension: 0,
        epochCount: 0,
        retainedEdgeCount: 0,
        neighborMs: 0,
        fuzzyGraphMs: 0,
        spectralMs: 0,
        initializationMs: 0,
        layoutMs: 0,
        hdbscanMs: 0,
        peakWorkingBytes: 0,
      },
    };
  }

  const outputDimension = Math.min(
    NATIVE_UMAP_COMPONENT_COUNT,
    count - 2,
  );
  const epochCount = count <= 10_000 ? 500 : 200;
  const neighborCount = Math.min(NATIVE_UMAP_NEIGHBOR_COUNT, count);

  let startedAt = performance.now();
  let neighbors = kernels.buildNativeUmapCosineKnn(
    embeddings,
    count,
    dimension,
    neighborCount,
    randomSeed,
  );
  const neighborMs = elapsedSince(startedAt);

  startedAt = performance.now();
  let fuzzy: NativeUmapFuzzyGraph | undefined =
    kernels.buildNativeUmapFuzzyGraph(neighbors, count);
  const fuzzyGraphMs = elapsedSince(startedAt);
  // Make the large k-NN arrays unreachable before spectral/layout allocation.
  neighbors = {
    indices: new Int32Array(),
    distances: new Float32Array(),
    neighborCount: 0,
  };
  const layoutGraph = prepareNativeLayoutGraph(fuzzy, count, epochCount);
  fuzzy = undefined;
  // The spectral solver currently uses temporary C++ vectors after the
  // reusable sbrk arena. Reserve layout's larger monotonic arena first so a
  // later grow cannot be obstructed by those allocator-owned pages.
  kernels.reserveNativeUmapLayoutSerial(
    count,
    outputDimension,
    layoutGraph.values.length,
  );

  startedAt = performance.now();
  const spectral = kernels.initializeNativeUmapSpectral(
    layoutGraph,
    count,
    outputDimension,
  );
  const spectralMs = elapsedSince(startedAt);

  startedAt = performance.now();
  const initialization = kernels.initializeNativeUmapLayout(
    spectral.values,
    count,
    outputDimension,
    randomSeed,
    count >= 4_096,
  );
  const initializationMs = elapsedSince(startedAt);

  startedAt = performance.now();
  const projection = kernels.optimizeNativeUmapLayoutSerial(
    initialization,
    count,
    outputDimension,
    layoutGraph.head,
    layoutGraph.columnIndices,
    layoutGraph.epochsPerSample,
    epochCount,
    NATIVE_UMAP_A,
    NATIVE_UMAP_B,
  );
  const layoutMs = elapsedSince(startedAt);

  startedAt = performance.now();
  const labels = kernels.clusterHdbscanF64Semantics(
    projection,
    count,
    outputDimension,
    NATIVE_HDBSCAN_MIN_SAMPLES,
    NATIVE_HDBSCAN_MIN_CLUSTER_SIZE,
  );
  const hdbscanMs = elapsedSince(startedAt);
  reassignMinorClusters(
    labels,
    embeddings,
    count,
    dimension,
    NATIVE_HDBSCAN_MIN_CLUSTER_SIZE,
  );
  mergeSimilarCentroids(
    labels,
    embeddings,
    count,
    dimension,
    NATIVE_CENTROID_MERGE_THRESHOLD,
  );
  normalizeLabels(labels);

  return {
    labels,
    projection,
    stats: {
      count,
      outputDimension,
      epochCount,
      retainedEdgeCount: layoutGraph.values.length,
      neighborMs,
      fuzzyGraphMs,
      spectralMs,
      initializationMs,
      layoutMs,
      hdbscanMs,
      peakWorkingBytes: estimateNativeUmapPeakWorkingBytes(
        count,
        outputDimension,
        layoutGraph.values.length,
        neighborCount,
      ),
    },
  };
}

/**
 * Native Senko UMAP/HDBSCAN orchestration with only UMAP's Hogwild layout
 * delegated to the persistent shared-memory worker pool.
 */
export async function clusterEmbeddingsNativeThreaded(
  embeddings: Float32Array,
  count: number,
  dimension: number,
  randomSeed: number,
  kernels: WasmClusteringKernels,
  layoutPool: NativeUmapLayoutExecutor,
  signal?: AbortSignal,
): Promise<NativeUmapThreadedResult> {
  validateMatrix(embeddings, count, dimension);
  throwIfAborted(signal);
  if (count < 10) {
    return {
      labels: new Int32Array(count),
      projection: new Float32Array(),
      stats: {
        count,
        outputDimension: 0,
        epochCount: 0,
        retainedEdgeCount: 0,
        neighborMs: 0,
        fuzzyGraphMs: 0,
        spectralMs: 0,
        initializationMs: 0,
        layoutMs: 0,
        hdbscanMs: 0,
        peakWorkingBytes: 0,
        layoutWorkerCount: layoutPool.workerCount,
        layoutSharedMemoryBytes: 0,
      },
    };
  }

  const outputDimension = Math.min(
    NATIVE_UMAP_COMPONENT_COUNT,
    count - 2,
  );
  const epochCount = count <= 10_000 ? 500 : 200;
  const neighborCount = Math.min(NATIVE_UMAP_NEIGHBOR_COUNT, count);

  let startedAt = performance.now();
  let neighbors = kernels.buildNativeUmapCosineKnn(
    embeddings,
    count,
    dimension,
    neighborCount,
    randomSeed,
  );
  const neighborMs = elapsedSince(startedAt);
  throwIfAborted(signal);

  startedAt = performance.now();
  let fuzzy: NativeUmapFuzzyGraph | undefined =
    kernels.buildNativeUmapFuzzyGraph(neighbors, count);
  const fuzzyGraphMs = elapsedSince(startedAt);
  neighbors = {
    indices: new Int32Array(),
    distances: new Float32Array(),
    neighborCount: 0,
  };
  const layoutGraph = prepareNativeLayoutGraph(fuzzy, count, epochCount);
  fuzzy = undefined;
  throwIfAborted(signal);

  startedAt = performance.now();
  const spectral = kernels.initializeNativeUmapSpectral(
    layoutGraph,
    count,
    outputDimension,
  );
  const spectralMs = elapsedSince(startedAt);
  throwIfAborted(signal);

  startedAt = performance.now();
  const initialization = kernels.initializeNativeUmapLayout(
    spectral.values,
    count,
    outputDimension,
    randomSeed,
    count >= 4_096,
  );
  const initializationMs = elapsedSince(startedAt);
  throwIfAborted(signal);

  const layout = await layoutPool.optimize(
    {
      embedding: initialization.embedding,
      rngState: initialization.rngState,
      head: layoutGraph.head,
      tail: layoutGraph.columnIndices,
      epochsPerSample: layoutGraph.epochsPerSample,
      vertexCount: count,
      dimension: outputDimension,
      epochCount,
      a: NATIVE_UMAP_A,
      b: NATIVE_UMAP_B,
    },
    signal,
  );
  throwIfAborted(signal);

  startedAt = performance.now();
  const labels = kernels.clusterHdbscanF64Semantics(
    layout.projection,
    count,
    outputDimension,
    NATIVE_HDBSCAN_MIN_SAMPLES,
    NATIVE_HDBSCAN_MIN_CLUSTER_SIZE,
  );
  const hdbscanMs = elapsedSince(startedAt);
  reassignMinorClusters(
    labels,
    embeddings,
    count,
    dimension,
    NATIVE_HDBSCAN_MIN_CLUSTER_SIZE,
  );
  mergeSimilarCentroids(
    labels,
    embeddings,
    count,
    dimension,
    NATIVE_CENTROID_MERGE_THRESHOLD,
  );
  normalizeLabels(labels);
  throwIfAborted(signal);

  return {
    labels,
    projection: layout.projection,
    stats: {
      count,
      outputDimension,
      epochCount,
      retainedEdgeCount: layoutGraph.values.length,
      neighborMs,
      fuzzyGraphMs,
      spectralMs,
      initializationMs,
      layoutMs: layout.layoutMs,
      hdbscanMs,
      layoutWorkerCount: layout.workerCount,
      layoutSharedMemoryBytes: layout.sharedMemoryBytes,
      peakWorkingBytes: estimateNativeUmapPeakWorkingBytes(
        count,
        outputDimension,
        layoutGraph.values.length,
        neighborCount,
      ),
    },
  };
}

export function prepareNativeLayoutGraph(
  graph: Pick<
    NativeUmapFuzzyGraph,
    "rowOffsets" | "columnIndices" | "values"
  >,
  count: number,
  epochCount: number,
): LayoutGraph {
  if (
    graph.rowOffsets.length !== count + 1 ||
    graph.columnIndices.length !== graph.values.length ||
    graph.rowOffsets[0] !== 0 ||
    graph.rowOffsets[count] !== graph.values.length ||
    !Number.isSafeInteger(epochCount) ||
    epochCount <= 0
  ) {
    throw new RangeError("native UMAP fuzzy CSR shape is invalid");
  }
  let maximumWeight = 0;
  for (const value of graph.values) {
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError("native UMAP fuzzy graph has an invalid weight");
    }
    maximumWeight = Math.max(maximumWeight, value);
  }
  if (!(maximumWeight > 0)) {
    throw new RangeError("native UMAP fuzzy graph has no positive edges");
  }
  const cutoff = maximumWeight / epochCount;
  let retainedEdgeCount = 0;
  for (const value of graph.values) {
    if (value !== 0 && value >= cutoff) retainedEdgeCount += 1;
  }

  const rowOffsets = new Int32Array(count + 1);
  const columnIndices = new Int32Array(retainedEdgeCount);
  const values = new Float32Array(retainedEdgeCount);
  const head = new Int32Array(retainedEdgeCount);
  const epochsPerSample = new Float64Array(retainedEdgeCount);
  let write = 0;
  for (let row = 0; row < count; row += 1) {
    const begin = graph.rowOffsets[row]!;
    const end = graph.rowOffsets[row + 1]!;
    if (begin < 0 || end < begin || end > graph.values.length) {
      throw new RangeError("native UMAP fuzzy CSR offsets are invalid");
    }
    for (let edge = begin; edge < end; edge += 1) {
      const weight = graph.values[edge]!;
      if (weight === 0 || weight < cutoff) continue;
      const column = graph.columnIndices[edge]!;
      if (column < 0 || column >= count) {
        throw new RangeError("native UMAP fuzzy CSR column is invalid");
      }
      rowOffsets[row + 1] = rowOffsets[row + 1]! + 1;
      columnIndices[write] = column;
      values[write] = weight;
      head[write] = row;
      // NumPy 2.x keeps both operations in Float32 before the final Float64
      // division in make_epochs_per_sample.
      const normalizedWeight = Math.fround(weight / maximumWeight);
      const sampleCount = Math.fround(epochCount * normalizedWeight);
      epochsPerSample[write] = epochCount / sampleCount;
      write += 1;
    }
  }
  for (let row = 0; row < count; row += 1) {
    rowOffsets[row + 1] =
      rowOffsets[row + 1]! + rowOffsets[row]!;
  }
  if (write !== retainedEdgeCount) {
    throw new Error("native UMAP retained-edge count changed while compacting");
  }
  return { rowOffsets, columnIndices, values, head, epochsPerSample };
}

/**
 * Current native-path typed-array high-water outside WebAssembly.
 *
 * The largest lifetime is either fuzzy CSR plus the compact layout graph, or
 * the compact graph plus Float64 spectral and Float32 layout initialization.
 * The caller-owned CAM++ embeddings and both WASM memories are excluded.
 */
export function estimateNativeUmapPeakWorkingBytes(
  count: number,
  outputDimension: number,
  retainedEdgeCount: number,
  neighborCount: number,
): number {
  for (const [name, value] of [
    ["count", count],
    ["outputDimension", outputDimension],
    ["retainedEdgeCount", retainedEdgeCount],
    ["neighborCount", neighborCount],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative safe integer`);
    }
  }
  const matrixValues = count * outputDimension;
  const knnEntries = count * neighborCount;
  if (
    !Number.isSafeInteger(matrixValues) ||
    !Number.isSafeInteger(knnEntries)
  ) {
    throw new RangeError("native UMAP working set exceeds safe integer range");
  }

  const fuzzyGraphBytes =
    (count + 1) * Int32Array.BYTES_PER_ELEMENT +
    retainedEdgeCount *
      (Int32Array.BYTES_PER_ELEMENT + Float32Array.BYTES_PER_ELEMENT) +
    count * 2 * Float32Array.BYTES_PER_ELEMENT;
  const layoutGraphBytes =
    (count + 1) * Int32Array.BYTES_PER_ELEMENT +
    retainedEdgeCount *
      (Int32Array.BYTES_PER_ELEMENT * 2 +
        Float32Array.BYTES_PER_ELEMENT +
        Float64Array.BYTES_PER_ELEMENT);
  const spectralAndInitializationBytes =
    matrixValues *
      (Float64Array.BYTES_PER_ELEMENT + Float32Array.BYTES_PER_ELEMENT) +
    (outputDimension + 1) * Float64Array.BYTES_PER_ELEMENT +
    3 * BigInt64Array.BYTES_PER_ELEMENT;
  const knnBytes =
    knnEntries *
    (Int32Array.BYTES_PER_ELEMENT + Float32Array.BYTES_PER_ELEMENT);
  return Math.max(
    fuzzyGraphBytes + layoutGraphBytes,
    layoutGraphBytes + spectralAndInitializationBytes,
    knnBytes + fuzzyGraphBytes,
  );
}

function elapsedSince(startedAt: number): number {
  return Math.max(0, performance.now() - startedAt);
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted !== true) return;
  throw new DOMException(
    signal.reason instanceof Error
      ? signal.reason.message
      : "Native UMAP clustering aborted",
    "AbortError",
  );
}

function validateMatrix(
  values: Float32Array,
  count: number,
  dimension: number,
): void {
  if (
    !Number.isSafeInteger(count) ||
    count < 0 ||
    !Number.isSafeInteger(dimension) ||
    dimension <= 0 ||
    values.length !== count * dimension
  ) {
    throw new RangeError("native clustering input shape is invalid");
  }
}
