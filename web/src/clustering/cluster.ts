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
 * optional fixed-memory numeric backend.
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
