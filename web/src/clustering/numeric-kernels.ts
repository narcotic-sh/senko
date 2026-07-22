import type { ResolvedClusteringOptions } from "./types";

export interface NumericKnnGraph {
  readonly indices: Int32Array;
  readonly similarities: Float32Array;
  readonly neighborCount: number;
}

export interface NumericNeighborHeap {
  readonly indices: Int32Array;
  readonly distances: Float32Array;
  readonly isNew: Uint8Array;
  readonly size: number;
}

export interface ClusteringKernelMemoryStats {
  /** Fixed WebAssembly linear-memory reservation. */
  readonly heapBytes: number;
  /** Reusable scratch arena within the linear-memory reservation. */
  readonly arenaCapacityBytes: number;
  /** Highest arena cursor observed across completed operations. */
  readonly peakArenaUsedBytes: number;
  /** Largest output set copied once into JS by a completed operation. */
  readonly peakReturnedJsBytes: number;
}

/**
 * Numeric clustering stages that can run in a reusable fixed-memory backend.
 * Graph construction and density hierarchy orchestration remain in TypeScript.
 */
export interface ClusteringNumericKernels {
  readonly memoryStats: ClusteringKernelMemoryStats;

  normalizeRows(
    embeddings: Float32Array,
    count: number,
    dim: number,
  ): Float32Array;

  buildApproximateCosineKnn(
    normalized: Float32Array,
    count: number,
    dim: number,
    options: ResolvedClusteringOptions,
  ): NumericKnnGraph;

  /** Normalize in place in the numeric backend and build the UMAP seed graph. */
  buildNormalizedApproximateCosineKnn(
    embeddings: Float32Array,
    count: number,
    dim: number,
    options: ResolvedClusteringOptions,
  ): NumericKnnGraph;

  refineEuclideanNeighbors(
    embeddings: Float32Array,
    count: number,
    dim: number,
    neighborCount: number,
    seedIndices: Int32Array,
    seedNeighborCount: number,
    randomSeed: number,
  ): NumericNeighborHeap;

  buildExactEuclideanKnn(
    values: Float32Array,
    count: number,
    dim: number,
    requestedNeighborCount: number,
  ): NumericKnnGraph;

  dispose(): void;
}
