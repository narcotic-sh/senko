import clusteringWasmUrl from "./wasm/senko-clustering.wasm?url";

import type {
  ClusteringKernelMemoryStats,
  ClusteringNumericKernels,
  NumericKnnGraph,
  NumericNeighborHeap,
} from "./numeric-kernels";
import type { ResolvedClusteringOptions } from "./types";

let compiledModule: Promise<WebAssembly.Module> | undefined;

interface ClusteringWasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;
  readonly _initialize: () => void;
  readonly cluster_reset: () => void;
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
}

/** Fixed-heap SIMD WebAssembly backend for clustering's numeric hotspots. */
export class WasmClusteringKernels implements ClusteringNumericKernels {
  private exports: ClusteringWasmExports | undefined;
  private peakArenaUsedBytes = 0;
  private peakReturnedJsBytes = 0;
  private readonly heapBytes: number;
  private readonly arenaCapacityBytes: number;

  private constructor(exports: ClusteringWasmExports) {
    exports._initialize();
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
    this.exports = exports;
    this.heapBytes = heapBytes;
    this.arenaCapacityBytes = arenaCapacityBytes;
  }

  static async create(): Promise<WasmClusteringKernels> {
    compiledModule ??= WebAssembly.compileStreaming(fetch(clusteringWasmUrl));
    const instance = await WebAssembly.instantiate(await compiledModule, {});
    return WasmClusteringKernels.fromInstance(instance);
  }

  /** Instantiate supplied bytes, primarily for Node/Vitest verification. */
  static async fromBytes(bytes: BufferSource): Promise<WasmClusteringKernels> {
    const instantiated = await WebAssembly.instantiate(bytes, {});
    const instance =
      instantiated instanceof WebAssembly.Instance
        ? instantiated
        : instantiated.instance;
    return WasmClusteringKernels.fromInstance(instance);
  }

  get memoryStats(): ClusteringKernelMemoryStats {
    return {
      heapBytes: this.heapBytes,
      arenaCapacityBytes: this.arenaCapacityBytes,
      peakArenaUsedBytes: this.peakArenaUsedBytes,
      peakReturnedJsBytes: this.peakReturnedJsBytes,
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
    const seedIndices = new Int32Array(count * seedNeighborCount);
    for (let row = 0; row < count; row += 1) {
      for (let rank = 0; rank < seedNeighborCount; rank += 1) {
        seedIndices[row * seedNeighborCount + rank] =
          (row + rank + 1) % count;
      }
    }
    this.refineEuclideanNeighbors(
      values,
      count,
      dim,
      20,
      seedIndices,
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
    const exports = this.beginOperation();
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
    const neighborCount = Math.min(options.neighborCount, Math.max(0, count - 1));
    if (neighborCount === 0) {
      return emptyKnnGraph();
    }
    const exports = this.beginOperation();
    const valuesPointer = this.copyFloat32(exports, normalized);
    const outputLength = count * neighborCount;
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
    requireLength("seed indices", seedIndices, count * seedNeighborCount);
    const exports = this.beginOperation();
    const embeddingsPointer = this.copyFloat32(exports, embeddings);
    const seedPointer = this.copyInt32(exports, seedIndices);
    const outputLength = count * neighborCount;
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
    const exports = this.beginOperation();
    const valuesPointer = this.copyFloat32(exports, values);
    const outputLength = count * neighborCount;
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

  dispose(): void {
    this.exports = undefined;
  }

  private static fromInstance(
    instance: WebAssembly.Instance,
  ): WasmClusteringKernels {
    return new WasmClusteringKernels(
      instance.exports as ClusteringWasmExports,
    );
  }

  private beginOperation(): ClusteringWasmExports {
    const exports = this.requireExports();
    exports.cluster_reset();
    return exports;
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
        `Clustering WASM fixed scratch arena exhausted while allocating ${bytes} bytes`,
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
  requireLength(name, values, count * dim);
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

function emptyKnnGraph(): NumericKnnGraph {
  return {
    indices: new Int32Array(),
    similarities: new Float32Array(),
    neighborCount: 0,
  };
}
