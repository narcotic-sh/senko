import clusteringWasmUrl from "./wasm/senko-clustering.wasm?url";

import type {
  ClusteringKernelMemoryStats,
  ClusteringNumericKernels,
  NumericKnnGraph,
  NumericNeighborHeap,
} from "./numeric-kernels";
import {
  DEFAULT_CLUSTERING_OPTIONS,
  type ResolvedClusteringOptions,
} from "./types";

export interface NativeUmapKnnGraph {
  readonly indices: Int32Array;
  readonly distances: Float32Array;
  readonly neighborCount: number;
}

export interface NativeUmapFuzzyGraph {
  readonly rowOffsets: Int32Array;
  readonly columnIndices: Int32Array;
  readonly values: Float32Array;
  readonly sigmas: Float32Array;
  readonly rhos: Float32Array;
}

export interface NativeUmapSpectralEmbedding {
  readonly values: Float64Array;
  readonly eigenvalues: Float64Array;
  readonly stats: {
    readonly requestedEigenpairs: number;
    readonly basisSize: number;
    readonly restartCount: number;
    readonly convergedEigenpairs: number;
    readonly maximumResidual: number;
    readonly smallestEigenvalue: number;
    readonly largestReturnedEigenvalue: number;
    readonly peakWorkingBytes: number;
  };
}

export interface NativeUmapLayoutInitialization {
  readonly embedding: Float32Array;
  readonly rngState: BigInt64Array;
}

let compiledModule: Promise<WebAssembly.Module> | undefined;
const clusteringWasmImports = {
  env: {
    emscripten_notify_memory_growth(memoryIndex: number): void {
      // The wrapper creates fresh typed-array views after every reserve.
      void memoryIndex;
    },
  },
} satisfies WebAssembly.Imports;

interface ClusteringWasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;
  readonly _initialize: () => void;
  readonly cluster_reset: () => void;
  readonly cluster_reserve: (requiredBytes: number) => number;
  readonly cluster_alloc: (bytes: number, alignment: number) => number;
  readonly cluster_heap_base: () => number;
  readonly cluster_heap_capacity: () => number;
  readonly cluster_heap_used: () => number;
  readonly cluster_normalize_rows: (
    input: number,
    output: number,
    count: number,
    dim: number,
  ) => number;
  readonly cluster_approximate_cosine_knn: (
    values: number,
    count: number,
    dim: number,
    neighborCount: number,
    tableCount: number,
    bits: number,
    bucketSampleLimit: number,
    temporalNeighborRadius: number,
    outputIndices: number,
    outputSimilarities: number,
  ) => number;
  readonly cluster_refine_euclidean_knn: (
    embeddings: number,
    count: number,
    dim: number,
    neighborCount: number,
    seedIndices: number,
    seedNeighborCount: number,
    randomSeed: number,
    outputIndices: number,
    outputDistances: number,
    outputIsNew: number,
  ) => number;
  readonly cluster_exact_euclidean_knn: (
    values: number,
    count: number,
    dim: number,
    neighborCount: number,
    outputIndices: number,
    outputSimilarities: number,
  ) => number;
  readonly cluster_umap_cosine_knn_workspace_bytes: (
    count: number,
    dim: number,
    neighborCount: number,
    randomSeed: number,
  ) => number;
  readonly cluster_umap_cosine_knn: (
    values: number,
    count: number,
    dim: number,
    neighborCount: number,
    randomSeed: number,
    outputIndices: number,
    outputDistances: number,
  ) => number;
  readonly cluster_umap_fuzzy_workspace_bytes: (
    count: number,
    neighborCount: number,
  ) => number;
  readonly cluster_umap_fuzzy_max_entries: (
    count: number,
    neighborCount: number,
  ) => number;
  readonly cluster_umap_fuzzy_graph: (
    knnIndices: number,
    knnDistances: number,
    count: number,
    neighborCount: number,
    outputSigmas: number,
    outputRhos: number,
    outputRowOffsets: number,
    outputColumnIndices: number,
    outputValues: number,
    outputEntryCount: number,
  ) => number;
  readonly cluster_umap_spectral: (
    rowOffsets: number,
    columnIndices: number,
    values: number,
    count: number,
    edgeCount: number,
    dimension: number,
    outputVectors: number,
    outputEigenvalues: number,
    outputIntegerStats: number,
    outputNumericStats: number,
    outputPeakWorkingBytes: number,
  ) => number;
  readonly cluster_umap_initialization_workspace_bytes: (
    dimension: number,
  ) => number;
  readonly cluster_umap_initialize_layout: (
    spectralEmbedding: number,
    count: number,
    dimension: number,
    randomSeed: number,
    approximateNeighbors: number,
    outputEmbedding: number,
    outputLayoutRngState: number,
  ) => number;
  readonly cluster_umap_layout_workspace_bytes: (
    vertexCount: number,
    edgeCount: number,
  ) => number;
  readonly cluster_umap_optimize_layout_serial: (
    embedding: number,
    vertexCount: number,
    dimension: number,
    head: number,
    tail: number,
    epochsPerSample: number,
    edgeCount: number,
    rngState: number,
    epochCount: number,
    a: number,
    b: number,
    gamma: number,
    negativeSampleRate: number,
  ) => number;
  readonly cluster_hdbscan_workspace_bytes: (
    count: number,
    dim: number,
    minSamples: number,
    minClusterSize: number,
  ) => number;
  readonly cluster_hdbscan_f64_semantics: (
    projection: number,
    count: number,
    dim: number,
    minSamples: number,
    minClusterSize: number,
    outputLabels: number,
  ) => number;
  readonly cluster_hdbscan_f64_diagnostics: (
    projection: number,
    count: number,
    dim: number,
    minSamples: number,
    minClusterSize: number,
    outputLabels: number,
    outputCoreDistances: number,
    outputMstRows: number,
  ) => number;
}

/** Reusable SIMD WebAssembly backend for clustering's numeric hotspots. */
export class WasmClusteringKernels implements ClusteringNumericKernels {
  private exports: ClusteringWasmExports | undefined;
  private peakArenaUsedBytes = 0;
  private peakReturnedJsBytes = 0;
  private lastHeapBytes = 0;
  private lastArenaCapacityBytes = 0;
  private lastRefinementMode:
    | "dense-pair-bitset"
    | "row-stamps"
    | undefined;

  private constructor(exports: ClusteringWasmExports) {
    exports._initialize();
    this.exports = exports;
    this.reserveArena(exports, INITIAL_ARENA_BYTES);
    const heapBytes = exports.memory.buffer.byteLength;
    const arenaBase = exports.cluster_heap_base();
    const arenaCapacityBytes = exports.cluster_heap_capacity();
    if (
      arenaBase <= 0 ||
      arenaCapacityBytes <= 0 ||
      arenaBase + arenaCapacityBytes > heapBytes
    ) {
      throw new Error("Clustering WASM scratch arena is outside linear memory");
    }
    this.lastArenaCapacityBytes = arenaCapacityBytes;
  }

  static async create(): Promise<WasmClusteringKernels> {
    compiledModule ??= WebAssembly.compileStreaming(fetch(clusteringWasmUrl));
    const instance = await WebAssembly.instantiate(
      await compiledModule,
      clusteringWasmImports,
    );
    return WasmClusteringKernels.fromInstance(instance);
  }

  /** Instantiate supplied bytes, primarily for Node/Vitest verification. */
  static async fromBytes(bytes: BufferSource): Promise<WasmClusteringKernels> {
    const instantiated = await WebAssembly.instantiate(
      bytes,
      clusteringWasmImports,
    );
    const instance =
      instantiated instanceof WebAssembly.Instance
        ? instantiated
        : instantiated.instance;
    return WasmClusteringKernels.fromInstance(instance);
  }

  get memoryStats(): ClusteringKernelMemoryStats {
    const arenaCapacityBytes =
      this.exports?.cluster_heap_capacity() ?? this.lastArenaCapacityBytes;
    const heapBytes =
      this.exports?.memory.buffer.byteLength ?? this.lastHeapBytes;
    this.lastHeapBytes = heapBytes;
    this.lastArenaCapacityBytes = arenaCapacityBytes;
    return {
      heapBytes,
      arenaCapacityBytes,
      peakArenaUsedBytes: this.peakArenaUsedBytes,
      peakReturnedJsBytes: this.peakReturnedJsBytes,
      ...(this.lastRefinementMode === undefined
        ? {}
        : { lastRefinementMode: this.lastRefinementMode }),
    };
  }

  /** Exercise production kernels so V8 tiers them before the first recording. */
  warmup(): void {
    const count = 512;
    const dim = 192;
    const seedNeighborCount = 64;
    const values = new Float32Array(count * dim);
    let state = 0x243f6a88;
    for (let index = 0; index < values.length; index += 1) {
      state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
      values[index] = ((state >>> 8) / 0x01000000) * 2 - 1;
    }
    const seed = this.buildNormalizedApproximateCosineKnn(
      values,
      count,
      dim,
      {
        ...DEFAULT_CLUSTERING_OPTIONS,
        neighborCount: seedNeighborCount,
      },
    );
    this.refineEuclideanNeighbors(
      values,
      count,
      dim,
      20,
      seed.indices,
      seedNeighborCount,
      0x6d2b79f5,
    );

    const projection = values.subarray(0, count * 10);
    this.buildExactEuclideanKnn(projection, count, 10, 40);
  }

  normalizeRows(
    embeddings: Float32Array,
    count: number,
    dim: number,
  ): Float32Array {
    requireMatrix("embeddings", embeddings, count, dim);
    const exports = this.beginOperation(matrixArenaBytes(count, dim));
    const valuesPointer = this.copyFloat32(exports, embeddings);
    this.requireSuccess(
      "normalize rows",
      exports.cluster_normalize_rows(valuesPointer, valuesPointer, count, dim),
      exports,
    );
    this.observeReturnedJsBytes(embeddings.byteLength);
    return this.copyFloat32Result(exports, valuesPointer, embeddings.length);
  }

  buildApproximateCosineKnn(
    normalized: Float32Array,
    count: number,
    dim: number,
    options: ResolvedClusteringOptions,
  ): NumericKnnGraph {
    requireMatrix("normalized embeddings", normalized, count, dim);
    const neighborCount = Math.min(
      options.neighborCount,
      Math.max(0, count - 1),
    );
    if (neighborCount === 0) {
      return emptyKnnGraph();
    }
    const exports = this.beginOperation(
      approximateCosineArenaBytes(count, dim, neighborCount, options),
    );
    const valuesPointer = this.copyFloat32(exports, normalized);
    return this.buildApproximateCosineKnnInCurrentArena(
      exports,
      valuesPointer,
      count,
      dim,
      neighborCount,
      options,
    );
  }

  buildNormalizedApproximateCosineKnn(
    embeddings: Float32Array,
    count: number,
    dim: number,
    options: ResolvedClusteringOptions,
  ): NumericKnnGraph {
    requireMatrix("embeddings", embeddings, count, dim);
    const neighborCount = Math.min(options.neighborCount, Math.max(0, count - 1));
    if (neighborCount === 0) {
      return emptyKnnGraph();
    }
    const exports = this.beginOperation(
      approximateCosineArenaBytes(count, dim, neighborCount, options),
    );
    const valuesPointer = this.copyFloat32(exports, embeddings);
    this.requireSuccess(
      "normalize rows",
      exports.cluster_normalize_rows(valuesPointer, valuesPointer, count, dim),
      exports,
    );
    return this.buildApproximateCosineKnnInCurrentArena(
      exports,
      valuesPointer,
      count,
      dim,
      neighborCount,
      options,
    );
  }

  private buildApproximateCosineKnnInCurrentArena(
    exports: ClusteringWasmExports,
    valuesPointer: number,
    count: number,
    dim: number,
    neighborCount: number,
    options: ResolvedClusteringOptions,
  ): NumericKnnGraph {
    const outputLength = checkedElementCount(
      "approximate k-NN output",
      count,
      neighborCount,
    );
    const indicesPointer = this.allocate(
      exports,
      outputLength * Int32Array.BYTES_PER_ELEMENT,
      Int32Array.BYTES_PER_ELEMENT,
    );
    const similaritiesPointer = this.allocate(
      exports,
      outputLength * Float32Array.BYTES_PER_ELEMENT,
      Float32Array.BYTES_PER_ELEMENT,
    );
    this.requireSuccess(
      "approximate cosine k-NN",
      exports.cluster_approximate_cosine_knn(
        valuesPointer,
        count,
        dim,
        neighborCount,
        options.hashTableCount,
        options.hashBits,
        options.bucketSampleLimit,
        options.temporalNeighborRadius,
        indicesPointer,
        similaritiesPointer,
      ),
      exports,
    );
    this.observeReturnedJsBytes(
      outputLength *
        (Int32Array.BYTES_PER_ELEMENT + Float32Array.BYTES_PER_ELEMENT),
    );
    return {
      indices: this.copyInt32Result(exports, indicesPointer, outputLength),
      similarities: this.copyFloat32Result(
        exports,
        similaritiesPointer,
        outputLength,
      ),
      neighborCount,
    };
  }

  refineEuclideanNeighbors(
    embeddings: Float32Array,
    count: number,
    dim: number,
    neighborCount: number,
    seedIndices: Int32Array,
    seedNeighborCount: number,
    randomSeed: number,
  ): NumericNeighborHeap {
    requireMatrix("embeddings", embeddings, count, dim);
    const seedLength = checkedElementCount(
      "seed indices",
      count,
      seedNeighborCount,
    );
    requireLength("seed indices", seedIndices, seedLength);
    const arenaPlan = refinementArenaPlan(
      count,
      dim,
      seedNeighborCount,
      neighborCount,
    );
    this.lastRefinementMode = arenaPlan.mode;
    const exports = this.beginOperation(arenaPlan.requiredBytes);
    const embeddingsPointer = this.copyFloat32(exports, embeddings);
    const seedPointer = this.copyInt32(exports, seedIndices);
    const outputLength = checkedElementCount(
      "neighbor-refinement output",
      count,
      neighborCount,
    );
    const indicesPointer = this.allocate(
      exports,
      outputLength * Int32Array.BYTES_PER_ELEMENT,
      Int32Array.BYTES_PER_ELEMENT,
    );
    const distancesPointer = this.allocate(
      exports,
      outputLength * Float32Array.BYTES_PER_ELEMENT,
      Float32Array.BYTES_PER_ELEMENT,
    );
    const flagsPointer = this.allocate(
      exports,
      outputLength * Uint8Array.BYTES_PER_ELEMENT,
      Uint8Array.BYTES_PER_ELEMENT,
    );
    this.requireSuccess(
      "Euclidean neighbor refinement",
      exports.cluster_refine_euclidean_knn(
        embeddingsPointer,
        count,
        dim,
        neighborCount,
        seedPointer,
        seedNeighborCount,
        randomSeed,
        indicesPointer,
        distancesPointer,
        flagsPointer,
      ),
      exports,
    );
    this.observeReturnedJsBytes(
      outputLength *
        (Int32Array.BYTES_PER_ELEMENT +
          Float32Array.BYTES_PER_ELEMENT +
          Uint8Array.BYTES_PER_ELEMENT),
    );
    return {
      indices: this.copyInt32Result(exports, indicesPointer, outputLength),
      distances: this.copyFloat32Result(exports, distancesPointer, outputLength),
      isNew: this.copyUint8Result(exports, flagsPointer, outputLength),
      size: neighborCount,
    };
  }

  buildExactEuclideanKnn(
    values: Float32Array,
    count: number,
    dim: number,
    requestedNeighborCount: number,
  ): NumericKnnGraph {
    requireMatrix("values", values, count, dim);
    const neighborCount = Math.min(
      requestedNeighborCount,
      Math.max(0, count - 1),
    );
    if (neighborCount === 0) {
      return emptyKnnGraph();
    }
    const exports = this.beginOperation(
      exactEuclideanArenaBytes(count, dim, neighborCount),
    );
    const valuesPointer = this.copyFloat32(exports, values);
    const outputLength = checkedElementCount(
      "exact k-NN output",
      count,
      neighborCount,
    );
    const indicesPointer = this.allocate(
      exports,
      outputLength * Int32Array.BYTES_PER_ELEMENT,
      Int32Array.BYTES_PER_ELEMENT,
    );
    const similaritiesPointer = this.allocate(
      exports,
      outputLength * Float32Array.BYTES_PER_ELEMENT,
      Float32Array.BYTES_PER_ELEMENT,
    );
    this.requireSuccess(
      "exact Euclidean k-NN",
      exports.cluster_exact_euclidean_knn(
        valuesPointer,
        count,
        dim,
        neighborCount,
        indicesPointer,
        similaritiesPointer,
      ),
      exports,
    );
    this.observeReturnedJsBytes(
      outputLength *
        (Int32Array.BYTES_PER_ELEMENT + Float32Array.BYTES_PER_ELEMENT),
    );
    return {
      indices: this.copyInt32Result(exports, indicesPointer, outputLength),
      similarities: this.copyFloat32Result(
        exports,
        similaritiesPointer,
        outputLength,
      ),
      neighborCount,
    };
  }

  /**
   * Native UMAP 0.5.12 cosine-neighbor semantics: exact below 4,096 rows,
   * otherwise angular RP trees followed by bounded-memory NNDescent.
   */
  buildNativeUmapCosineKnn(
    values: Float32Array,
    count: number,
    dim: number,
    neighborCount: number,
    randomSeed: number,
  ): NativeUmapKnnGraph {
    requireMatrix("native UMAP input", values, count, dim);
    if (
      !Number.isSafeInteger(neighborCount) ||
      neighborCount <= 0 ||
      neighborCount > count
    ) {
      throw new RangeError(
        "native UMAP neighbor count must be in [1, count]",
      );
    }
    if (
      !Number.isSafeInteger(randomSeed) ||
      randomSeed < 0 ||
      randomSeed > UINT32_MAX
    ) {
      throw new RangeError(
        "native UMAP random seed must be an unsigned 32-bit integer",
      );
    }

    const exports = this.requireExports();
    const workspaceBytes =
      exports.cluster_umap_cosine_knn_workspace_bytes(
        count,
        dim,
        neighborCount,
        randomSeed,
      );
    if (workspaceBytes === 0) {
      throw new RangeError(
        "native UMAP neighbor dimensions or parameters are unsupported",
      );
    }
    const outputLength = checkedElementCount(
      "native UMAP k-NN output",
      count,
      neighborCount,
    );
    const operationExports = this.beginOperation(
      nativeUmapKnnArenaBytes(
        count,
        dim,
        neighborCount,
        workspaceBytes,
      ),
    );
    const valuesPointer = this.copyFloat32(operationExports, values);
    const indicesPointer = this.allocate(
      operationExports,
      outputLength * Int32Array.BYTES_PER_ELEMENT,
      Int32Array.BYTES_PER_ELEMENT,
    );
    const distancesPointer = this.allocate(
      operationExports,
      outputLength * Float32Array.BYTES_PER_ELEMENT,
      Float32Array.BYTES_PER_ELEMENT,
    );
    this.requireSuccess(
      "native UMAP cosine k-NN",
      operationExports.cluster_umap_cosine_knn(
        valuesPointer,
        count,
        dim,
        neighborCount,
        randomSeed,
        indicesPointer,
        distancesPointer,
      ),
      operationExports,
    );
    this.observeReturnedJsBytes(
      outputLength *
        (Int32Array.BYTES_PER_ELEMENT + Float32Array.BYTES_PER_ELEMENT),
    );
    return {
      indices: this.copyInt32Result(
        operationExports,
        indicesPointer,
        outputLength,
      ),
      distances: this.copyFloat32Result(
        operationExports,
        distancesPointer,
        outputLength,
      ),
      neighborCount,
    };
  }

  /** Native UMAP smooth-kNN membership strengths and fuzzy-union CSR graph. */
  buildNativeUmapFuzzyGraph(
    knn: NativeUmapKnnGraph,
    count: number,
  ): NativeUmapFuzzyGraph {
    const neighborCount = knn.neighborCount;
    const inputLength = checkedElementCount(
      "native UMAP fuzzy k-NN input",
      count,
      neighborCount,
    );
    if (
      knn.indices.length !== inputLength ||
      knn.distances.length !== inputLength
    ) {
      throw new RangeError("native UMAP fuzzy k-NN arrays have invalid lengths");
    }
    const exports = this.requireExports();
    const workspaceBytes = exports.cluster_umap_fuzzy_workspace_bytes(
      count,
      neighborCount,
    );
    const maximumEntries = exports.cluster_umap_fuzzy_max_entries(
      count,
      neighborCount,
    );
    if (workspaceBytes === 0 || maximumEntries === 0) {
      throw new RangeError(
        "native UMAP fuzzy graph dimensions are unsupported",
      );
    }
    const operationExports = this.beginOperation(
      nativeUmapFuzzyArenaBytes(
        count,
        neighborCount,
        workspaceBytes,
        maximumEntries,
      ),
    );
    const indicesPointer = this.copyInt32(operationExports, knn.indices);
    const distancesPointer = this.copyFloat32(
      operationExports,
      knn.distances,
    );
    const sigmasPointer = this.allocate(
      operationExports,
      count * Float32Array.BYTES_PER_ELEMENT,
      Float32Array.BYTES_PER_ELEMENT,
    );
    const rhosPointer = this.allocate(
      operationExports,
      count * Float32Array.BYTES_PER_ELEMENT,
      Float32Array.BYTES_PER_ELEMENT,
    );
    const rowOffsetsPointer = this.allocate(
      operationExports,
      (count + 1) * Int32Array.BYTES_PER_ELEMENT,
      Int32Array.BYTES_PER_ELEMENT,
    );
    const columnIndicesPointer = this.allocate(
      operationExports,
      maximumEntries * Int32Array.BYTES_PER_ELEMENT,
      Int32Array.BYTES_PER_ELEMENT,
    );
    const valuesPointer = this.allocate(
      operationExports,
      maximumEntries * Float32Array.BYTES_PER_ELEMENT,
      Float32Array.BYTES_PER_ELEMENT,
    );
    const entryCountPointer = this.allocate(
      operationExports,
      Uint32Array.BYTES_PER_ELEMENT,
      Uint32Array.BYTES_PER_ELEMENT,
    );
    this.requireSuccess(
      "native UMAP fuzzy graph",
      operationExports.cluster_umap_fuzzy_graph(
        indicesPointer,
        distancesPointer,
        count,
        neighborCount,
        sigmasPointer,
        rhosPointer,
        rowOffsetsPointer,
        columnIndicesPointer,
        valuesPointer,
        entryCountPointer,
      ),
      operationExports,
    );
    const memory = new Uint32Array(operationExports.memory.buffer);
    const entryCount =
      memory[entryCountPointer / Uint32Array.BYTES_PER_ELEMENT]!;
    if (entryCount > maximumEntries) {
      throw new Error("native UMAP fuzzy graph returned an invalid edge count");
    }
    this.observeReturnedJsBytes(
      (count * 2 + entryCount) * Float32Array.BYTES_PER_ELEMENT +
        (count + 1 + entryCount) * Int32Array.BYTES_PER_ELEMENT,
    );
    return {
      rowOffsets: this.copyInt32Result(
        operationExports,
        rowOffsetsPointer,
        count + 1,
      ),
      columnIndices: this.copyInt32Result(
        operationExports,
        columnIndicesPointer,
        entryCount,
      ),
      values: this.copyFloat32Result(
        operationExports,
        valuesPointer,
        entryCount,
      ),
      sigmas: this.copyFloat32Result(
        operationExports,
        sigmasPointer,
        count,
      ),
      rhos: this.copyFloat32Result(
        operationExports,
        rhosPointer,
        count,
      ),
    };
  }

  /**
   * Native UMAP connected-graph spectral initialization. The graph must
   * already have UMAP's max-weight / epoch cutoff compacted out.
   */
  initializeNativeUmapSpectral(
    graph: Pick<
      NativeUmapFuzzyGraph,
      "rowOffsets" | "columnIndices" | "values"
    >,
    count: number,
    dimension: number,
  ): NativeUmapSpectralEmbedding {
    if (
      !Number.isSafeInteger(count) ||
      count < 3 ||
      !Number.isSafeInteger(dimension) ||
      dimension < 1 ||
      dimension > count - 2 ||
      graph.rowOffsets.length !== count + 1 ||
      graph.columnIndices.length !== graph.values.length
    ) {
      throw new RangeError("native UMAP spectral graph shape is invalid");
    }
    const edgeCount = graph.values.length;
    if (
      graph.rowOffsets[0] !== 0 ||
      graph.rowOffsets[count] !== edgeCount
    ) {
      throw new RangeError("native UMAP spectral CSR offsets are invalid");
    }
    const outputLength = checkedElementCount(
      "native UMAP spectral output",
      count,
      dimension,
    );
    const operationExports = this.beginOperation(
      nativeUmapSpectralArenaBytes(count, dimension, edgeCount),
    );
    const rowOffsetsPointer = this.copyInt32(
      operationExports,
      graph.rowOffsets,
    );
    const columnIndicesPointer = this.copyInt32(
      operationExports,
      graph.columnIndices,
    );
    const valuesPointer = this.copyFloat32(operationExports, graph.values);
    const vectorsPointer = this.allocate(
      operationExports,
      outputLength * Float64Array.BYTES_PER_ELEMENT,
      Float64Array.BYTES_PER_ELEMENT,
    );
    const eigenvaluesPointer = this.allocate(
      operationExports,
      dimension * Float64Array.BYTES_PER_ELEMENT,
      Float64Array.BYTES_PER_ELEMENT,
    );
    const integerStatsPointer = this.allocate(
      operationExports,
      4 * Int32Array.BYTES_PER_ELEMENT,
      Int32Array.BYTES_PER_ELEMENT,
    );
    const numericStatsPointer = this.allocate(
      operationExports,
      3 * Float64Array.BYTES_PER_ELEMENT,
      Float64Array.BYTES_PER_ELEMENT,
    );
    const peakWorkingBytesPointer = this.allocate(
      operationExports,
      Uint32Array.BYTES_PER_ELEMENT,
      Uint32Array.BYTES_PER_ELEMENT,
    );
    this.requireSuccess(
      "native UMAP spectral initialization",
      operationExports.cluster_umap_spectral(
        rowOffsetsPointer,
        columnIndicesPointer,
        valuesPointer,
        count,
        edgeCount,
        dimension,
        vectorsPointer,
        eigenvaluesPointer,
        integerStatsPointer,
        numericStatsPointer,
        peakWorkingBytesPointer,
      ),
      operationExports,
    );
    const integerStats = checkedInt32View(
      operationExports.memory,
      integerStatsPointer,
      4,
    );
    const numericStats = checkedFloat64View(
      operationExports.memory,
      numericStatsPointer,
      3,
    );
    const peakWorkingBytes = new Uint32Array(
      operationExports.memory.buffer,
      peakWorkingBytesPointer,
      1,
    )[0]!;
    this.observeReturnedJsBytes(
      (outputLength + dimension) * Float64Array.BYTES_PER_ELEMENT,
    );
    return {
      values: this.copyFloat64Result(
        operationExports,
        vectorsPointer,
        outputLength,
      ),
      eigenvalues: this.copyFloat64Result(
        operationExports,
        eigenvaluesPointer,
        dimension,
      ),
      stats: {
        requestedEigenpairs: integerStats[0]!,
        basisSize: integerStats[1]!,
        restartCount: integerStats[2]!,
        convergedEigenpairs: integerStats[3]!,
        maximumResidual: numericStats[0]!,
        smallestEigenvalue: numericStats[1]!,
        largestReturnedEigenvalue: numericStats[2]!,
        peakWorkingBytes,
      },
    };
  }

  /** Prepare native UMAP's noisy Float32 layout coordinates and tau RNG. */
  initializeNativeUmapLayout(
    spectralEmbedding: Float64Array,
    count: number,
    dimension: number,
    randomSeed: number,
    approximateNeighbors: boolean,
  ): NativeUmapLayoutInitialization {
    const outputLength = checkedElementCount(
      "native UMAP layout initialization",
      count,
      dimension,
    );
    if (spectralEmbedding.length !== outputLength) {
      throw new RangeError(
        "native UMAP spectral embedding has an invalid length",
      );
    }
    if (
      !Number.isSafeInteger(randomSeed) ||
      randomSeed < 0 ||
      randomSeed > UINT32_MAX
    ) {
      throw new RangeError(
        "native UMAP random seed must be an unsigned 32-bit integer",
      );
    }
    const exports = this.requireExports();
    const workspaceBytes =
      exports.cluster_umap_initialization_workspace_bytes(dimension);
    if (workspaceBytes === 0) {
      throw new RangeError(
        "native UMAP initialization dimensions are unsupported",
      );
    }
    const operationExports = this.beginOperation(
      nativeUmapInitializationArenaBytes(
        count,
        dimension,
        workspaceBytes,
      ),
    );
    const spectralPointer = this.copyFloat64(
      operationExports,
      spectralEmbedding,
    );
    const embeddingPointer = this.allocate(
      operationExports,
      outputLength * Float32Array.BYTES_PER_ELEMENT,
      Float32Array.BYTES_PER_ELEMENT,
    );
    const rngStatePointer = this.allocate(
      operationExports,
      3 * BigInt64Array.BYTES_PER_ELEMENT,
      BigInt64Array.BYTES_PER_ELEMENT,
    );
    this.requireSuccess(
      "native UMAP layout initialization",
      operationExports.cluster_umap_initialize_layout(
        spectralPointer,
        count,
        dimension,
        randomSeed,
        approximateNeighbors ? 1 : 0,
        embeddingPointer,
        rngStatePointer,
      ),
      operationExports,
    );
    this.observeReturnedJsBytes(
      outputLength * Float32Array.BYTES_PER_ELEMENT +
        3 * BigInt64Array.BYTES_PER_ELEMENT,
    );
    return {
      embedding: this.copyFloat32Result(
        operationExports,
        embeddingPointer,
        outputLength,
      ),
      rngState: this.copyBigInt64Result(
        operationExports,
        rngStatePointer,
        3,
      ),
    };
  }

  /** Deterministic scalar-order diagnostic and no-thread compatibility path. */
  reserveNativeUmapLayoutSerial(
    count: number,
    dimension: number,
    edgeCount: number,
  ): void {
    const exports = this.requireExports();
    const workspaceBytes = exports.cluster_umap_layout_workspace_bytes(
      count,
      edgeCount,
    );
    if (workspaceBytes === 0) {
      throw new RangeError("native UMAP layout dimensions are unsupported");
    }
    this.reserveArena(
      exports,
      nativeUmapLayoutArenaBytes(
        count,
        dimension,
        edgeCount,
        workspaceBytes,
      ),
    );
  }

  /** Deterministic scalar-order diagnostic and no-thread compatibility path. */
  optimizeNativeUmapLayoutSerial(
    initialization: NativeUmapLayoutInitialization,
    count: number,
    dimension: number,
    head: Int32Array,
    tail: Int32Array,
    epochsPerSample: Float64Array,
    epochCount: number,
    a: number,
    b: number,
  ): Float32Array {
    const embeddingLength = checkedElementCount(
      "native UMAP layout embedding",
      count,
      dimension,
    );
    if (initialization.embedding.length !== embeddingLength) {
      throw new RangeError("native UMAP layout embedding length is invalid");
    }
    if (initialization.rngState.length !== 3) {
      throw new RangeError("native UMAP layout RNG state must have 3 values");
    }
    const edgeCount = head.length;
    if (
      tail.length !== edgeCount ||
      epochsPerSample.length !== edgeCount ||
      !Number.isSafeInteger(epochCount) ||
      epochCount <= 0
    ) {
      throw new RangeError("native UMAP layout edge arrays are invalid");
    }
    const exports = this.requireExports();
    const workspaceBytes = exports.cluster_umap_layout_workspace_bytes(
      count,
      edgeCount,
    );
    if (workspaceBytes === 0) {
      throw new RangeError("native UMAP layout dimensions are unsupported");
    }
    const operationExports = this.beginOperation(
      nativeUmapLayoutArenaBytes(
        count,
        dimension,
        edgeCount,
        workspaceBytes,
      ),
    );
    const embeddingPointer = this.copyFloat32(
      operationExports,
      initialization.embedding,
    );
    const headPointer = this.copyInt32(operationExports, head);
    const tailPointer = this.copyInt32(operationExports, tail);
    const epochsPerSamplePointer = this.copyFloat64(
      operationExports,
      epochsPerSample,
    );
    const rngStatePointer = this.copyBigInt64(
      operationExports,
      initialization.rngState,
    );
    this.requireSuccess(
      "native UMAP serial layout",
      operationExports.cluster_umap_optimize_layout_serial(
        embeddingPointer,
        count,
        dimension,
        headPointer,
        tailPointer,
        epochsPerSamplePointer,
        edgeCount,
        rngStatePointer,
        epochCount,
        a,
        b,
        1,
        5,
      ),
      operationExports,
    );
    this.observeReturnedJsBytes(
      embeddingLength * Float32Array.BYTES_PER_ELEMENT,
    );
    return this.copyFloat32Result(
      operationExports,
      embeddingPointer,
      embeddingLength,
    );
  }

  /**
   * Native-compatible HDBSCAN 0.8.44 Float64 hierarchy semantics. The exact
   * core-distance and native approximate-Boruvka provider remains gated from
   * production until the native UMAP path is ready.
   */
  clusterHdbscanF64Semantics(
    projection: Float32Array,
    count: number,
    dim: number,
    minSamples: number,
    minClusterSize: number,
  ): Int32Array {
    requireMatrix("HDBSCAN projection", projection, count, dim);
    const exports = this.requireExports();
    const workspaceBytes = exports.cluster_hdbscan_workspace_bytes(
      count,
      dim,
      minSamples,
      minClusterSize,
    );
    if (workspaceBytes === 0) {
      throw new RangeError("HDBSCAN dimensions or parameters are unsupported");
    }
    const arenaBytes = hdbscanArenaBytes(
      count,
      dim,
      workspaceBytes,
    );
    const operationExports = this.beginOperation(arenaBytes);
    const projectionPointer = this.copyFloat32(
      operationExports,
      projection,
    );
    const labelsPointer = this.allocate(
      operationExports,
      count * Int32Array.BYTES_PER_ELEMENT,
      Int32Array.BYTES_PER_ELEMENT,
    );
    this.requireSuccess(
      "HDBSCAN Float64 semantics",
      operationExports.cluster_hdbscan_f64_semantics(
        projectionPointer,
        count,
        dim,
        minSamples,
        minClusterSize,
        labelsPointer,
      ),
      operationExports,
    );
    this.observeReturnedJsBytes(count * Int32Array.BYTES_PER_ELEMENT);
    return this.copyInt32Result(operationExports, labelsPointer, count);
  }

  /** Return raw numeric stages for gated differential tests. */
  diagnoseHdbscanF64Semantics(
    projection: Float32Array,
    count: number,
    dim: number,
    minSamples: number,
    minClusterSize: number,
  ): {
    readonly labels: Int32Array;
    readonly coreDistances: Float64Array;
    readonly mstRows: Float64Array;
  } {
    requireMatrix("HDBSCAN projection", projection, count, dim);
    const exports = this.requireExports();
    const workspaceBytes = exports.cluster_hdbscan_workspace_bytes(
      count,
      dim,
      minSamples,
      minClusterSize,
    );
    if (workspaceBytes === 0) {
      throw new RangeError("HDBSCAN dimensions or parameters are unsupported");
    }
    const mstElementCount = checkedElementCount(
      "HDBSCAN diagnostic MST",
      count - 1,
      3,
    );
    const operationExports = this.beginOperation(
      hdbscanDiagnosticArenaBytes(count, dim, workspaceBytes),
    );
    const projectionPointer = this.copyFloat32(
      operationExports,
      projection,
    );
    const labelsPointer = this.allocate(
      operationExports,
      count * Int32Array.BYTES_PER_ELEMENT,
      Int32Array.BYTES_PER_ELEMENT,
    );
    const coreDistancesPointer = this.allocate(
      operationExports,
      count * Float64Array.BYTES_PER_ELEMENT,
      Float64Array.BYTES_PER_ELEMENT,
    );
    const mstPointer = this.allocate(
      operationExports,
      mstElementCount * Float64Array.BYTES_PER_ELEMENT,
      Float64Array.BYTES_PER_ELEMENT,
    );
    this.requireSuccess(
      "HDBSCAN Float64 diagnostics",
      operationExports.cluster_hdbscan_f64_diagnostics(
        projectionPointer,
        count,
        dim,
        minSamples,
        minClusterSize,
        labelsPointer,
        coreDistancesPointer,
        mstPointer,
      ),
      operationExports,
    );
    const returnedBytes =
      count * Int32Array.BYTES_PER_ELEMENT +
      (count + mstElementCount) * Float64Array.BYTES_PER_ELEMENT;
    this.observeReturnedJsBytes(returnedBytes);
    return {
      labels: this.copyInt32Result(operationExports, labelsPointer, count),
      coreDistances: this.copyFloat64Result(
        operationExports,
        coreDistancesPointer,
        count,
      ),
      mstRows: this.copyFloat64Result(
        operationExports,
        mstPointer,
        mstElementCount,
      ),
    };
  }

  dispose(): void {
    if (this.exports !== undefined) {
      this.lastHeapBytes = this.exports.memory.buffer.byteLength;
      this.lastArenaCapacityBytes = this.exports.cluster_heap_capacity();
    }
    this.exports = undefined;
  }

  private static fromInstance(
    instance: WebAssembly.Instance,
  ): WasmClusteringKernels {
    return new WasmClusteringKernels(
      instance.exports as ClusteringWasmExports,
    );
  }

  private beginOperation(requiredArenaBytes: number): ClusteringWasmExports {
    const exports = this.requireExports();
    this.reserveArena(exports, requiredArenaBytes);
    exports.cluster_reset();
    return exports;
  }

  private reserveArena(
    exports: ClusteringWasmExports,
    requiredArenaBytes: number,
  ): void {
    if (
      !Number.isSafeInteger(requiredArenaBytes) ||
      requiredArenaBytes < 0 ||
      requiredArenaBytes > UINT32_MAX
    ) {
      throw new RangeError(
        `Clustering WASM scratch requirement ${requiredArenaBytes} is outside the wasm32 address range`,
      );
    }
    if (requiredArenaBytes > exports.cluster_heap_capacity()) {
      const status = exports.cluster_reserve(requiredArenaBytes);
      if (status !== 1) {
        throw new RangeError(
          `Clustering WASM could not reserve ${requiredArenaBytes} scratch bytes`,
        );
      }
    }
    const arenaBase = exports.cluster_heap_base();
    const arenaCapacityBytes = exports.cluster_heap_capacity();
    if (
      arenaBase <= 0 ||
      arenaCapacityBytes < requiredArenaBytes ||
      arenaBase + arenaCapacityBytes > exports.memory.buffer.byteLength
    ) {
      throw new Error("Clustering WASM scratch arena is outside linear memory");
    }
    this.lastHeapBytes = exports.memory.buffer.byteLength;
    this.lastArenaCapacityBytes = arenaCapacityBytes;
  }

  private requireExports(): ClusteringWasmExports {
    if (this.exports === undefined) {
      throw new Error("WasmClusteringKernels has been disposed");
    }
    return this.exports;
  }

  private allocate(
    exports: ClusteringWasmExports,
    bytes: number,
    alignment: number,
  ): number {
    const pointer = exports.cluster_alloc(bytes, alignment);
    if (pointer === 0) {
      throw new RangeError(
        `Clustering WASM scratch arena exhausted while allocating ${bytes} bytes`,
      );
    }
    return pointer;
  }

  private copyFloat32(
    exports: ClusteringWasmExports,
    input: Float32Array,
  ): number {
    const pointer = this.allocate(
      exports,
      input.byteLength,
      Float32Array.BYTES_PER_ELEMENT,
    );
    checkedFloat32View(exports.memory, pointer, input.length).set(input);
    return pointer;
  }

  private copyInt32(
    exports: ClusteringWasmExports,
    input: Int32Array,
  ): number {
    const pointer = this.allocate(
      exports,
      input.byteLength,
      Int32Array.BYTES_PER_ELEMENT,
    );
    checkedInt32View(exports.memory, pointer, input.length).set(input);
    return pointer;
  }

  private copyFloat64(
    exports: ClusteringWasmExports,
    input: Float64Array,
  ): number {
    const pointer = this.allocate(
      exports,
      input.byteLength,
      Float64Array.BYTES_PER_ELEMENT,
    );
    checkedFloat64View(exports.memory, pointer, input.length).set(input);
    return pointer;
  }

  private copyBigInt64(
    exports: ClusteringWasmExports,
    input: BigInt64Array,
  ): number {
    const pointer = this.allocate(
      exports,
      input.byteLength,
      BigInt64Array.BYTES_PER_ELEMENT,
    );
    checkedBigInt64View(exports.memory, pointer, input.length).set(input);
    return pointer;
  }

  private copyFloat32Result(
    exports: ClusteringWasmExports,
    pointer: number,
    length: number,
  ): Float32Array {
    return checkedFloat32View(exports.memory, pointer, length).slice();
  }

  private copyInt32Result(
    exports: ClusteringWasmExports,
    pointer: number,
    length: number,
  ): Int32Array {
    return checkedInt32View(exports.memory, pointer, length).slice();
  }

  private copyFloat64Result(
    exports: ClusteringWasmExports,
    pointer: number,
    length: number,
  ): Float64Array {
    return checkedFloat64View(exports.memory, pointer, length).slice();
  }

  private copyBigInt64Result(
    exports: ClusteringWasmExports,
    pointer: number,
    length: number,
  ): BigInt64Array {
    return checkedBigInt64View(exports.memory, pointer, length).slice();
  }

  private copyUint8Result(
    exports: ClusteringWasmExports,
    pointer: number,
    length: number,
  ): Uint8Array {
    return checkedUint8View(exports.memory, pointer, length).slice();
  }

  private requireSuccess(
    operation: string,
    status: number,
    exports: ClusteringWasmExports,
  ): void {
    this.peakArenaUsedBytes = Math.max(
      this.peakArenaUsedBytes,
      exports.cluster_heap_used(),
    );
    if (status !== 1) {
      throw new Error(`Clustering WASM ${operation} failed (${status})`);
    }
  }

  private observeReturnedJsBytes(bytes: number): void {
    this.peakReturnedJsBytes = Math.max(this.peakReturnedJsBytes, bytes);
  }
}

function checkedFloat32View(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
): Float32Array {
  checkedRange(memory, pointer, length, Float32Array.BYTES_PER_ELEMENT);
  return new Float32Array(memory.buffer, pointer, length);
}

function checkedInt32View(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
): Int32Array {
  checkedRange(memory, pointer, length, Int32Array.BYTES_PER_ELEMENT);
  return new Int32Array(memory.buffer, pointer, length);
}

function checkedFloat64View(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
): Float64Array {
  checkedRange(memory, pointer, length, Float64Array.BYTES_PER_ELEMENT);
  return new Float64Array(memory.buffer, pointer, length);
}

function checkedBigInt64View(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
): BigInt64Array {
  checkedRange(memory, pointer, length, BigInt64Array.BYTES_PER_ELEMENT);
  return new BigInt64Array(memory.buffer, pointer, length);
}

function checkedUint8View(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
): Uint8Array {
  checkedRange(memory, pointer, length, Uint8Array.BYTES_PER_ELEMENT);
  return new Uint8Array(memory.buffer, pointer, length);
}

function checkedRange(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
  elementBytes: number,
): void {
  const byteEnd = pointer + length * elementBytes;
  if (
    pointer < 0 ||
    length < 0 ||
    pointer % elementBytes !== 0 ||
    byteEnd > memory.buffer.byteLength
  ) {
    throw new Error("Clustering WASM typed-array view is outside linear memory");
  }
}

function requireMatrix(
  name: string,
  values: Float32Array,
  count: number,
  dim: number,
): void {
  requireLength(name, values, checkedElementCount(name, count, dim));
}

function requireLength(
  name: string,
  values: { readonly length: number },
  expected: number,
): void {
  if (values.length !== expected) {
    throw new RangeError(
      `${name} length ${values.length} does not match expected ${expected}`,
    );
  }
}

function checkedElementCount(
  name: string,
  ...dimensions: readonly number[]
): number {
  let result = 1n;
  for (const dimension of dimensions) {
    if (
      !Number.isSafeInteger(dimension) ||
      dimension < 0 ||
      dimension > INT32_MAX
    ) {
      throw new RangeError(
        `${name} dimension ${dimension} is outside the signed wasm32 range`,
      );
    }
    result *= BigInt(dimension);
    if (result > BigInt(INT32_MAX)) {
      throw new RangeError(
        `${name} element count exceeds the signed wasm32 index range`,
      );
    }
  }
  return Number(result);
}

class ArenaSizer {
  private cursor: bigint;

  constructor(cursor = 0n) {
    this.cursor = cursor;
  }

  copy(): ArenaSizer {
    return new ArenaSizer(this.cursor);
  }

  add(
    elementCount: bigint,
    elementBytes: bigint,
    alignment = 16n,
  ): void {
    const bytes = elementCount * elementBytes;
    if (bytes < 0n || bytes > UINT32_MAX_BIGINT) {
      throw new RangeError(
        `Clustering WASM allocation of ${bytes} bytes is outside the wasm32 address range`,
      );
    }
    this.cursor = alignBigInt(this.cursor, alignment) + bytes;
    if (this.cursor > UINT32_MAX_BIGINT) {
      throw new RangeError(
        `Clustering WASM scratch requirement ${this.cursor} is outside the wasm32 address range`,
      );
    }
  }

  bytes(): number {
    return Number(this.cursor);
  }
}

function alignBigInt(value: bigint, alignment: bigint): bigint {
  return (value + alignment - 1n) & ~(alignment - 1n);
}

function matrixArenaBytes(count: number, dim: number): number {
  const sizer = new ArenaSizer();
  sizer.add(BigInt(count) * BigInt(dim), 4n, 4n);
  return sizer.bytes();
}

function hdbscanArenaBytes(
  count: number,
  dim: number,
  workspaceBytes: number,
): number {
  checkedElementCount("HDBSCAN projection", count, dim);
  checkedElementCount("HDBSCAN labels", count);
  if (
    !Number.isSafeInteger(workspaceBytes) ||
    workspaceBytes <= 0 ||
    workspaceBytes > UINT32_MAX
  ) {
    throw new RangeError(
      `HDBSCAN workspace ${workspaceBytes} is outside the wasm32 address range`,
    );
  }
  const sizer = new ArenaSizer();
  sizer.add(BigInt(count) * BigInt(dim), 4n, 4n);
  sizer.add(BigInt(count), 4n, 4n);
  sizer.add(BigInt(workspaceBytes), 1n, 16n);
  return sizer.bytes();
}

function nativeUmapKnnArenaBytes(
  count: number,
  dim: number,
  neighborCount: number,
  workspaceBytes: number,
): number {
  checkedElementCount("native UMAP input", count, dim);
  const outputLength = checkedElementCount(
    "native UMAP k-NN output",
    count,
    neighborCount,
  );
  if (
    !Number.isSafeInteger(workspaceBytes) ||
    workspaceBytes <= 0 ||
    workspaceBytes > UINT32_MAX
  ) {
    throw new RangeError(
      `native UMAP workspace ${workspaceBytes} is outside the wasm32 address range`,
    );
  }
  const sizer = new ArenaSizer();
  sizer.add(BigInt(count) * BigInt(dim), 4n, 4n);
  sizer.add(BigInt(outputLength), 4n, 4n);
  sizer.add(BigInt(outputLength), 4n, 4n);
  sizer.add(BigInt(workspaceBytes), 1n, 16n);
  return sizer.bytes();
}

function nativeUmapFuzzyArenaBytes(
  count: number,
  neighborCount: number,
  workspaceBytes: number,
  maximumEntries: number,
): number {
  const inputLength = checkedElementCount(
    "native UMAP fuzzy k-NN input",
    count,
    neighborCount,
  );
  checkedElementCount("native UMAP fuzzy maximum entries", maximumEntries);
  if (
    !Number.isSafeInteger(workspaceBytes) ||
    workspaceBytes <= 0 ||
    workspaceBytes > UINT32_MAX
  ) {
    throw new RangeError(
      `native UMAP fuzzy workspace ${workspaceBytes} is outside the wasm32 address range`,
    );
  }
  const sizer = new ArenaSizer();
  sizer.add(BigInt(inputLength), 4n, 4n);
  sizer.add(BigInt(inputLength), 4n, 4n);
  sizer.add(BigInt(count), 4n, 4n);
  sizer.add(BigInt(count), 4n, 4n);
  sizer.add(BigInt(count + 1), 4n, 4n);
  sizer.add(BigInt(maximumEntries), 4n, 4n);
  sizer.add(BigInt(maximumEntries), 4n, 4n);
  sizer.add(1n, 4n, 4n);
  sizer.add(BigInt(workspaceBytes), 1n, 16n);
  return sizer.bytes();
}

function nativeUmapSpectralArenaBytes(
  count: number,
  dimension: number,
  edgeCount: number,
): number {
  const outputLength = checkedElementCount(
    "native UMAP spectral output",
    count,
    dimension,
  );
  checkedElementCount("native UMAP spectral edges", edgeCount);
  const sizer = new ArenaSizer();
  sizer.add(BigInt(count + 1), 4n, 4n);
  sizer.add(BigInt(edgeCount), 4n, 4n);
  sizer.add(BigInt(edgeCount), 4n, 4n);
  sizer.add(BigInt(outputLength), 8n, 8n);
  sizer.add(BigInt(dimension), 8n, 8n);
  sizer.add(4n, 4n, 4n);
  sizer.add(3n, 8n, 8n);
  sizer.add(1n, 4n, 4n);
  return sizer.bytes();
}

function nativeUmapInitializationArenaBytes(
  count: number,
  dimension: number,
  workspaceBytes: number,
): number {
  const outputLength = checkedElementCount(
    "native UMAP layout initialization",
    count,
    dimension,
  );
  if (
    !Number.isSafeInteger(workspaceBytes) ||
    workspaceBytes <= 0 ||
    workspaceBytes > UINT32_MAX
  ) {
    throw new RangeError(
      `native UMAP initialization workspace ${workspaceBytes} is outside the wasm32 address range`,
    );
  }
  const sizer = new ArenaSizer();
  sizer.add(BigInt(outputLength), 8n, 8n);
  sizer.add(BigInt(outputLength), 4n, 4n);
  sizer.add(3n, 8n, 8n);
  sizer.add(BigInt(workspaceBytes), 1n, 16n);
  return sizer.bytes();
}

function nativeUmapLayoutArenaBytes(
  count: number,
  dimension: number,
  edgeCount: number,
  workspaceBytes: number,
): number {
  const embeddingLength = checkedElementCount(
    "native UMAP layout embedding",
    count,
    dimension,
  );
  checkedElementCount("native UMAP layout edges", edgeCount);
  if (
    !Number.isSafeInteger(workspaceBytes) ||
    workspaceBytes <= 0 ||
    workspaceBytes > UINT32_MAX
  ) {
    throw new RangeError(
      `native UMAP layout workspace ${workspaceBytes} is outside the wasm32 address range`,
    );
  }
  const sizer = new ArenaSizer();
  sizer.add(BigInt(embeddingLength), 4n, 4n);
  sizer.add(BigInt(edgeCount), 4n, 4n);
  sizer.add(BigInt(edgeCount), 4n, 4n);
  sizer.add(BigInt(edgeCount), 8n, 8n);
  sizer.add(3n, 8n, 8n);
  sizer.add(BigInt(workspaceBytes), 1n, 16n);
  return sizer.bytes();
}

function hdbscanDiagnosticArenaBytes(
  count: number,
  dim: number,
  workspaceBytes: number,
): number {
  checkedElementCount("HDBSCAN projection", count, dim);
  checkedElementCount("HDBSCAN labels", count);
  const mstElementCount = checkedElementCount(
    "HDBSCAN diagnostic MST",
    count - 1,
    3,
  );
  if (
    !Number.isSafeInteger(workspaceBytes) ||
    workspaceBytes <= 0 ||
    workspaceBytes > UINT32_MAX
  ) {
    throw new RangeError(
      `HDBSCAN workspace ${workspaceBytes} is outside the wasm32 address range`,
    );
  }
  const sizer = new ArenaSizer();
  sizer.add(BigInt(count) * BigInt(dim), 4n, 4n);
  sizer.add(BigInt(count), 4n, 4n);
  sizer.add(BigInt(count), 8n, 8n);
  sizer.add(BigInt(mstElementCount), 8n, 8n);
  sizer.add(BigInt(workspaceBytes), 1n, 16n);
  return sizer.bytes();
}

function approximateCosineArenaBytes(
  count: number,
  dim: number,
  neighborCount: number,
  options: ResolvedClusteringOptions,
): number {
  checkedElementCount("approximate k-NN output", count, neighborCount);
  checkedElementCount(
    "approximate hash planes",
    options.hashTableCount,
    options.hashBits,
    dim,
  );
  checkedElementCount(
    "approximate signatures",
    count,
    options.hashTableCount,
  );
  const numericBucketCount = 2 ** options.hashBits;
  checkedElementCount(
    "approximate hash buckets",
    options.hashTableCount,
    numericBucketCount,
  );
  checkedElementCount(
    "approximate hash bucket offsets",
    options.hashTableCount,
    numericBucketCount + 1,
  );
  const n = BigInt(count);
  const d = BigInt(dim);
  const neighbors = BigInt(neighborCount);
  const tables = BigInt(options.hashTableCount);
  const bits = BigInt(options.hashBits);
  const bucketSampleLimit = BigInt(options.bucketSampleLimit);
  const temporalRadius = BigInt(options.temporalNeighborRadius);
  const bucketCount = 1n << bits;
  const planeCount = tables * bits;
  const candidateCapacity =
    tables * bucketSampleLimit * 2n + temporalRadius * 2n < n
      ? tables * bucketSampleLimit * 2n + temporalRadius * 2n
      : n;

  const sizer = new ArenaSizer();
  // Wrapper allocations use their natural typed-array alignment.
  sizer.add(n * d, 4n, 4n);
  sizer.add(n * neighbors, 4n, 4n);
  sizer.add(n * neighbors, 4n, 4n);
  // Kernel scratch allocations are deliberately 16-byte aligned.
  sizer.add(planeCount * d, 1n);
  sizer.add(n * tables, 2n);
  sizer.add(tables * bucketCount, 4n);
  sizer.add(tables * (bucketCount + 1n), 4n);
  sizer.add(tables * n, 4n);
  sizer.add(n, 4n);
  sizer.add(candidateCapacity, 4n);
  return sizer.bytes();
}

interface RefinementArenaPlan {
  readonly mode: "dense-pair-bitset" | "row-stamps";
  readonly requiredBytes: number;
}

/** Exact wrapper-plus-kernel high-water mark for Euclidean refinement. */
function refinementArenaPlan(
  count: number,
  dim: number,
  seedNeighborCount: number,
  neighborCount: number,
): RefinementArenaPlan {
  checkedElementCount(
    "neighbor-refinement output",
    count,
    neighborCount,
  );
  const n = BigInt(count);
  const d = BigInt(dim);
  const seed = BigInt(seedNeighborCount);
  const neighbors = BigInt(neighborCount);

  const wrapper = new ArenaSizer();
  wrapper.add(n * d, 4n, 4n);
  wrapper.add(n * seed, 4n, 4n);
  wrapper.add(n * neighbors, 4n, 4n);
  wrapper.add(n * neighbors, 4n, 4n);
  wrapper.add(n * neighbors, 1n, 1n);

  const pairCount = (n * (n - 1n)) / 2n;
  const dense = wrapper.copy();
  dense.add((pairCount + 31n) / 32n, 4n);
  dense.add(n * neighbors, 4n);
  dense.add(n * neighbors, 1n);
  if (dense.bytes() <= INITIAL_ARENA_BYTES) {
    return { mode: "dense-pair-bitset", requiredBytes: dense.bytes() };
  }

  const scalable = wrapper.copy();
  scalable.add(n * neighbors, 4n);
  scalable.add(n * neighbors, 1n);
  scalable.add(n, 4n);
  return { mode: "row-stamps", requiredBytes: scalable.bytes() };
}

function exactEuclideanArenaBytes(
  count: number,
  dim: number,
  neighborCount: number,
): number {
  checkedElementCount("exact k-NN output", count, neighborCount);
  const n = BigInt(count);
  const d = BigInt(dim);
  const neighbors = BigInt(neighborCount);
  const sizer = new ArenaSizer();
  sizer.add(n * d, 4n, 4n);
  sizer.add(n * neighbors, 4n, 4n);
  sizer.add(n * neighbors, 4n, 4n);
  return sizer.bytes();
}

const INT32_MAX = 0x7fff_ffff;
const UINT32_MAX = 0xffff_ffff;
const UINT32_MAX_BIGINT = 0xffff_ffffn;
const INITIAL_ARENA_BYTES = 10 * 1024 * 1024;

function emptyKnnGraph(): NumericKnnGraph {
  return {
    indices: new Int32Array(),
    similarities: new Float32Array(),
    neighborCount: 0,
  };
}
