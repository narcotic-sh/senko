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
