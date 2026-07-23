import {
  mergeSimilarCentroids,
  normalizeLabels,
  reassignMinorClusters,
} from "./postprocess";
import {
  WasmClusteringKernels,
  type NativeUmapFuzzyGraph,
} from "./wasm-kernels";

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
}

export interface NativeUmapSerialResult {
  readonly labels: Int32Array;
  readonly projection: Float32Array;
  readonly stats: NativeUmapSerialStats;
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

function elapsedSince(startedAt: number): number {
  return Math.max(0, performance.now() - startedAt);
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
