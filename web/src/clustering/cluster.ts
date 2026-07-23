import { clusterSparseGraph } from "./hierarchy";
import {
  buildApproximateCosineKnn,
  buildExactEuclideanKnn,
  normalizeRows,
} from "./knn";
import {
  mergeSimilarCentroids,
  normalizeLabels,
  reassignMinorClusters,
} from "./postprocess";
import {
  resolveClusteringOptions,
  type ClusteringOptions,
} from "./types";
import { projectWithUmap } from "./umap";
import type { ClusteringNumericKernels } from "./numeric-kernels";

/**
 * Cluster row-major speaker embeddings with TypeScript orchestration and an
 * optional reusable numeric backend.
 *
 * The default parameters use a deterministic 10-dimensional UMAP layout with
 * 20 neighbors and 50 epochs, followed by Euclidean k-NN with 40 neighbors,
 * mutual-reachability density using min_samples=20 and min_cluster_size=10,
 * Senko-compatible minor-label reassignment, then centroid merging at cosine
 * similarity 0.875. Labels are consecutive integers starting at zero.
 */
export function clusterEmbeddings(
  embeddings: Float32Array,
  count: number,
  dim: number,
  options: ClusteringOptions = {},
  kernels?: ClusteringNumericKernels,
): Int32Array {
  validateInput(embeddings, count, dim);
  if (count === 0) {
    return new Int32Array();
  }
  const resolved = resolveClusteringOptions(options);
  if (count < 10) {
    return new Int32Array(count);
  }

  const clusteringInput = resolved.useUmap
    ? projectWithUmap(embeddings, count, dim, resolved, kernels)
    : { values: embeddings, dimension: dim };
  const graph = resolved.useUmap
    ? buildExactEuclideanKnn(
        clusteringInput.values,
        count,
        clusteringInput.dimension,
        resolved.neighborCount,
        kernels,
      )
    : buildApproximateCosineKnn(
        normalizeRows(
          clusteringInput.values,
          count,
          clusteringInput.dimension,
          kernels,
        ),
        count,
        clusteringInput.dimension,
        resolved,
        kernels,
      );
  const labels = clusterSparseGraph(
    graph,
    count,
    resolved.minSamples,
    resolved.minClusterSize,
  );
  reassignMinorClusters(
    labels,
    embeddings,
    count,
    dim,
    resolved.minClusterSize,
  );
  if (resolved.mergeThreshold !== null) {
    mergeSimilarCentroids(
      labels,
      embeddings,
      count,
      dim,
      resolved.mergeThreshold,
    );
  }
  normalizeLabels(labels);
  return labels;
}

/**
 * Typed-array high-water mark after UMAP has returned its projection.
 *
 * The caller-owned CAM++ embeddings and the WASM heap are deliberately
 * excluded. The estimate includes the retained projection and exact k-NN
 * graph plus the larger of hierarchy radix sorting and dendrogram assembly.
 * Engine object overhead remains outside the pipeline's lower-bound ledger.
 */
export function estimatePostUmapPeakWorkingBytes(
  count: number,
  projectionDimension: number,
  requestedNeighborCount: number,
): number {
  requireNonNegativeSafeInteger("count", count);
  requireNonNegativeSafeInteger("projectionDimension", projectionDimension);
  requireNonNegativeSafeInteger(
    "requestedNeighborCount",
    requestedNeighborCount,
  );
  if (count === 0) return 0;

  const neighborCount = Math.min(
    requestedNeighborCount,
    Math.max(0, count - 1),
  );
  const edgeCapacity = checkedProduct(
    "sparse hierarchy edge capacity",
    count,
    neighborCount,
  );
  const projectionBytes = checkedProduct(
    "UMAP projection bytes",
    count,
    projectionDimension,
    Float32Array.BYTES_PER_ELEMENT,
  );
  const graphBytes = checkedProduct(
    "exact k-NN graph bytes",
    edgeCapacity,
    Int32Array.BYTES_PER_ELEMENT + Float32Array.BYTES_PER_ELEMENT,
  );
  const coreDistanceBytes = checkedProduct(
    "core-distance bytes",
    count,
    Float64Array.BYTES_PER_ELEMENT,
  );

  // Sorting holds Float64 weights, two Uint32 edge-order arrays, and the
  // fixed 65,536-entry radix count table. Dendrogram construction retains one
  // order array plus weights and allocates 52 bytes of typed state per row.
  const radixSortBytes = checkedSum(
    "hierarchy radix bytes",
    checkedProduct("hierarchy radix edge bytes", edgeCapacity, 16),
    (1 << 16) * Uint32Array.BYTES_PER_ELEMENT,
  );
  const dendrogramBuildBytes = checkedSum(
    "dendrogram build bytes",
    checkedProduct("sorted hierarchy edge bytes", edgeCapacity, 12),
    checkedProduct("dendrogram row bytes", count, 52),
  );
  return checkedSum(
    "post-UMAP peak working bytes",
    projectionBytes,
    graphBytes,
    coreDistanceBytes,
    Math.max(radixSortBytes, dendrogramBuildBytes),
  );
}

function validateInput(embeddings: Float32Array, count: number, dim: number): void {
  if (!Number.isInteger(count) || count < 0) {
    throw new RangeError("count must be a non-negative integer");
  }
  if (!Number.isInteger(dim) || dim <= 0) {
    throw new RangeError("dim must be a positive integer");
  }
  if (embeddings.length !== count * dim) {
    throw new RangeError(
      `embeddings length ${embeddings.length} does not match count * dim (${count * dim})`,
    );
  }
  for (let i = 0; i < embeddings.length; i += 1) {
    if (!Number.isFinite(embeddings[i]!)) {
      throw new RangeError(`embeddings contains a non-finite value at index ${i}`);
    }
  }
}

function requireNonNegativeSafeInteger(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer`);
  }
}

function checkedProduct(name: string, ...factors: readonly number[]): number {
  let result = 1;
  for (const factor of factors) {
    requireNonNegativeSafeInteger(name, factor);
    result *= factor;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`${name} exceeds safe integer accounting`);
    }
  }
  return result;
}

function checkedSum(name: string, ...terms: readonly number[]): number {
  let result = 0;
  for (const term of terms) {
    requireNonNegativeSafeInteger(name, term);
    result += term;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`${name} exceeds safe integer accounting`);
    }
  }
  return result;
}
