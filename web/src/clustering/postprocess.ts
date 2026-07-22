/**
 * Reassign labels whose population is smaller than `minClusterSize` to their
 * closest major centroid.
 *
 * This intentionally treats HDBSCAN's `-1` label exactly like every other
 * label. That is the behavior of Senko's `CommonClustering.filter_minor_cluster`:
 * a `-1` population is only reassigned when it is smaller than the configured
 * threshold, and otherwise remains eligible for the later centroid merge.
 */
export function reassignMinorClusters(
  labels: Int32Array,
  embeddings: Float32Array,
  count: number,
  dim: number,
  minClusterSize: number,
): void {
  const sizes = countLabels(labels);
  const majorLabels: number[] = [];
  const minorLabels = new Set<number>();
  for (const [label, size] of sizes) {
    if (size >= minClusterSize) {
      majorLabels.push(label);
    } else {
      minorLabels.add(label);
    }
  }
  if (minorLabels.size === 0) {
    return;
  }
  if (majorLabels.length === 0) {
    labels.fill(0);
    return;
  }
  majorLabels.sort((left, right) => left - right);
  const centroids = calculateCentroids(
    labels,
    embeddings,
    count,
    dim,
    majorLabels,
  );
  normalizeCentroidRows(centroids, majorLabels.length, dim);

  for (let row = 0; row < count; row += 1) {
    if (!minorLabels.has(labels[row]!)) {
      continue;
    }
    const embeddingOffset = row * dim;
    const embeddingNorm = rowNorm(embeddings, embeddingOffset, dim);
    let bestMajor = 0;
    let bestSimilarity = Number.NEGATIVE_INFINITY;
    for (let major = 0; major < majorLabels.length; major += 1) {
      const similarity = dotRows(
        embeddings,
        embeddingOffset,
        centroids,
        major * dim,
        dim,
      );
      const cosineSimilarity = embeddingNorm > 0 ? similarity / embeddingNorm : 0;
      if (cosineSimilarity > bestSimilarity) {
        bestSimilarity = cosineSimilarity;
        bestMajor = major;
      }
    }
    labels[row] = majorLabels[bestMajor]!;
  }
}

/** Repeatedly merge the most similar pair of cluster centroids. */
export function mergeSimilarCentroids(
  labels: Int32Array,
  embeddings: Float32Array,
  count: number,
  dim: number,
  threshold: number,
): void {
  for (;;) {
    const unique = [...countLabels(labels).keys()].sort((left, right) => left - right);
    if (unique.length <= 1) {
      return;
    }
    const centroids = calculateCentroids(
      labels,
      embeddings,
      count,
      dim,
      unique,
    );
    normalizeCentroidRows(centroids, unique.length, dim);
    let bestLeft = -1;
    let bestRight = -1;
    let bestSimilarity = Number.NEGATIVE_INFINITY;
    for (let left = 0; left < unique.length; left += 1) {
      for (let right = left + 1; right < unique.length; right += 1) {
        const similarity = dotRows(
          centroids,
          left * dim,
          centroids,
          right * dim,
          dim,
        );
        if (similarity > bestSimilarity) {
          bestSimilarity = similarity;
          bestLeft = left;
          bestRight = right;
        }
      }
    }
    if (bestSimilarity < threshold) {
      return;
    }
    const keep = unique[bestLeft]!;
    const remove = unique[bestRight]!;
    for (let row = 0; row < count; row += 1) {
      if (labels[row] === remove) {
        labels[row] = keep;
      }
    }
  }
}

/** Normalize arbitrary labels to consecutive integers in sorted-label order. */
export function normalizeLabels(labels: Int32Array): void {
  const unique = [...new Set(labels)].sort((left, right) => left - right);
  const replacements = new Map<number, number>(
    unique.map((label, index) => [label, index]),
  );
  for (let row = 0; row < labels.length; row += 1) {
    labels[row] = replacements.get(labels[row]!)!;
  }
}

function countLabels(labels: Int32Array): Map<number, number> {
  const sizes = new Map<number, number>();
  for (const label of labels) {
    sizes.set(label, (sizes.get(label) ?? 0) + 1);
  }
  return sizes;
}

function calculateCentroids(
  labels: Int32Array,
  embeddings: Float32Array,
  count: number,
  dim: number,
  clusterLabels: readonly number[],
): Float32Array {
  const labelToRow = new Map<number, number>();
  for (let i = 0; i < clusterLabels.length; i += 1) {
    labelToRow.set(clusterLabels[i]!, i);
  }
  const centroids = new Float32Array(clusterLabels.length * dim);
  const sizes = new Int32Array(clusterLabels.length);
  for (let row = 0; row < count; row += 1) {
    const centroidRow = labelToRow.get(labels[row]!);
    if (centroidRow === undefined) {
      continue;
    }
    sizes[centroidRow] = sizes[centroidRow]! + 1;
    const sourceOffset = row * dim;
    const targetOffset = centroidRow * dim;
    for (let column = 0; column < dim; column += 1) {
      centroids[targetOffset + column] =
        centroids[targetOffset + column]! + embeddings[sourceOffset + column]!;
    }
  }
  for (let row = 0; row < clusterLabels.length; row += 1) {
    const scale = 1 / sizes[row]!;
    const offset = row * dim;
    for (let column = 0; column < dim; column += 1) {
      centroids[offset + column] = centroids[offset + column]! * scale;
    }
  }
  return centroids;
}

function normalizeCentroidRows(values: Float32Array, count: number, dim: number): void {
  for (let row = 0; row < count; row += 1) {
    const offset = row * dim;
    let squaredNorm = 0;
    for (let column = 0; column < dim; column += 1) {
      const value = values[offset + column]!;
      squaredNorm += value * value;
    }
    const scale = squaredNorm > 0 ? 1 / Math.sqrt(squaredNorm) : 0;
    for (let column = 0; column < dim; column += 1) {
      values[offset + column] = values[offset + column]! * scale;
    }
  }
}

function dotRows(
  left: Float32Array,
  leftOffset: number,
  right: Float32Array,
  rightOffset: number,
  dim: number,
): number {
  let result = 0;
  for (let column = 0; column < dim; column += 1) {
    result += left[leftOffset + column]! * right[rightOffset + column]!;
  }
  return result;
}

function rowNorm(values: Float32Array, offset: number, dim: number): number {
  let squaredNorm = 0;
  for (let column = 0; column < dim; column += 1) {
    const value = values[offset + column]!;
    squaredNorm += value * value;
  }
  return Math.sqrt(squaredNorm);
}
