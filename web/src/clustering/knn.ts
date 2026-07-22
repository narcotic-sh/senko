import type { ResolvedClusteringOptions } from "./types";
import type { ClusteringNumericKernels } from "./numeric-kernels";

export interface KnnGraph {
  /** Row-major, descending-similarity neighbor IDs; unused entries are -1. */
  readonly indices: Int32Array;
  /** Row-major affinity values corresponding to `indices`. */
  readonly similarities: Float32Array;
  readonly neighborCount: number;
}

/**
 * Build an exact Euclidean k-NN graph without an N-by-N distance matrix.
 * Similarities are encoded as `1 - distance` so the shared hierarchy can
 * recover Euclidean distance with its existing `1 - similarity` convention.
 */
export function buildExactEuclideanKnn(
  values: Float32Array,
  count: number,
  dim: number,
  requestedNeighborCount: number,
  kernels?: ClusteringNumericKernels,
): KnnGraph {
  if (kernels !== undefined) {
    return kernels.buildExactEuclideanKnn(
      values,
      count,
      dim,
      requestedNeighborCount,
    );
  }
  const neighborCount = Math.min(requestedNeighborCount, Math.max(0, count - 1));
  const indices = new Int32Array(count * neighborCount);
  indices.fill(-1);
  const similarities = new Float32Array(count * neighborCount);
  similarities.fill(Number.NEGATIVE_INFINITY);
  if (neighborCount === 0) {
    return { indices, similarities, neighborCount };
  }
  for (let left = 0; left < count; left += 1) {
    for (let right = left + 1; right < count; right += 1) {
      const distance = euclideanDistance(values, left * dim, right * dim, dim);
      const similarity = 1 - distance;
      insertNeighbor(
        indices,
        similarities,
        left * neighborCount,
        neighborCount,
        right,
        similarity,
      );
      insertNeighbor(
        indices,
        similarities,
        right * neighborCount,
        neighborCount,
        left,
        similarity,
      );
    }
  }
  return { indices, similarities, neighborCount };
}

/**
 * Build a bounded-memory approximate cosine k-NN graph.
 *
 * Candidate generation uses deterministic random-hyperplane LSH plus temporal
 * neighbors. Candidate distances are always re-ranked with exact cosine
 * similarity, so hashing affects recall but never the reported edge weights.
 */
export function buildApproximateCosineKnn(
  normalized: Float32Array,
  count: number,
  dim: number,
  options: ResolvedClusteringOptions,
  kernels?: ClusteringNumericKernels,
): KnnGraph {
  if (kernels !== undefined) {
    return kernels.buildApproximateCosineKnn(normalized, count, dim, options);
  }
  const k = Math.min(options.neighborCount, Math.max(0, count - 1));
  const indices = new Int32Array(count * k);
  indices.fill(-1);
  const similarities = new Float32Array(count * k);
  similarities.fill(Number.NEGATIVE_INFINITY);
  if (k === 0) {
    return { indices, similarities, neighborCount: k };
  }

  const signatures = computeSignatures(
    normalized,
    count,
    dim,
    options.hashTableCount,
    options.hashBits,
  );
  const bucketCount = 1 << options.hashBits;
  const tables = buildBuckets(
    signatures,
    count,
    options.hashTableCount,
    bucketCount,
  );
  const seen = new Int32Array(count);
  const candidates = new Int32Array(
    Math.min(
      count,
      options.hashTableCount * options.bucketSampleLimit * 2 +
        options.temporalNeighborRadius * 2,
    ),
  );

  for (let row = 0; row < count; row += 1) {
    const stamp = row + 1;
    seen[row] = stamp;
    let candidateCount = 0;

    const start = Math.max(0, row - options.temporalNeighborRadius);
    const end = Math.min(count, row + options.temporalNeighborRadius + 1);
    for (let candidate = start; candidate < end; candidate += 1) {
      candidateCount = appendCandidate(
        candidate,
        stamp,
        seen,
        candidates,
        candidateCount,
      );
    }

    for (let table = 0; table < options.hashTableCount; table += 1) {
      const signature = signatures[row * options.hashTableCount + table]!;
      const buckets = tables[table]!;
      candidateCount = appendBucket(
        buckets[signature]!,
        row,
        table,
        options.bucketSampleLimit,
        stamp,
        seen,
        candidates,
        candidateCount,
      );
    }

    // Random data has small base buckets. Probe Hamming-distance-one buckets
    // until there are enough points to estimate a stable 40-neighbor density.
    const desiredCandidates = Math.min(count - 1, Math.max(k * 3, 64));
    if (candidateCount < desiredCandidates) {
      outer: for (let table = 0; table < options.hashTableCount; table += 1) {
        const signature = signatures[row * options.hashTableCount + table]!;
        const buckets = tables[table]!;
        for (let bit = 0; bit < options.hashBits; bit += 1) {
          candidateCount = appendBucket(
            buckets[signature ^ (1 << bit)]!,
            row,
            table + bit + 1,
            options.bucketSampleLimit,
            stamp,
            seen,
            candidates,
            candidateCount,
          );
          if (candidateCount >= desiredCandidates || candidateCount === candidates.length) {
            break outer;
          }
        }
      }
    }

    for (let cursor = 0; cursor < candidateCount; cursor += 1) {
      const candidate = candidates[cursor]!;
      const similarity = dot(normalized, row * dim, candidate * dim, dim);
      insertNeighbor(indices, similarities, row * k, k, candidate, similarity);
    }
  }

  return { indices, similarities, neighborCount: k };
}

export function normalizeRows(
  embeddings: Float32Array,
  count: number,
  dim: number,
  kernels?: ClusteringNumericKernels,
): Float32Array {
  if (kernels !== undefined) {
    return kernels.normalizeRows(embeddings, count, dim);
  }
  const normalized = new Float32Array(embeddings.length);
  for (let row = 0; row < count; row += 1) {
    const offset = row * dim;
    let squaredNorm = 0;
    for (let column = 0; column < dim; column += 1) {
      const value = embeddings[offset + column]!;
      squaredNorm += value * value;
    }
    const scale = squaredNorm > 0 ? 1 / Math.sqrt(squaredNorm) : 0;
    for (let column = 0; column < dim; column += 1) {
      normalized[offset + column] = embeddings[offset + column]! * scale;
    }
  }
  return normalized;
}

function computeSignatures(
  values: Float32Array,
  count: number,
  dim: number,
  tableCount: number,
  bits: number,
): Uint16Array {
  const planeCount = tableCount * bits;
  const planes = new Int8Array(planeCount * dim);
  let randomState = 0x9e3779b9;
  for (let i = 0; i < planes.length; i += 1) {
    randomState = xorshift32(randomState);
    planes[i] = (randomState & 1) === 0 ? -1 : 1;
  }

  const signatures = new Uint16Array(count * tableCount);
  for (let row = 0; row < count; row += 1) {
    const rowOffset = row * dim;
    for (let table = 0; table < tableCount; table += 1) {
      let signature = 0;
      for (let bit = 0; bit < bits; bit += 1) {
        const planeOffset = (table * bits + bit) * dim;
        let projection = 0;
        for (let column = 0; column < dim; column += 1) {
          projection += values[rowOffset + column]! * planes[planeOffset + column]!;
        }
        if (projection >= 0) {
          signature |= 1 << bit;
        }
      }
      signatures[row * tableCount + table] = signature;
    }
  }
  return signatures;
}

function buildBuckets(
  signatures: Uint16Array,
  count: number,
  tableCount: number,
  bucketCount: number,
): readonly (readonly Int32Array[])[] {
  const result: Int32Array[][] = [];
  for (let table = 0; table < tableCount; table += 1) {
    const sizes = new Int32Array(bucketCount);
    for (let row = 0; row < count; row += 1) {
      const key = signatures[row * tableCount + table]!;
      sizes[key] = sizes[key]! + 1;
    }
    const buckets: Int32Array[] = new Array<Int32Array>(bucketCount);
    for (let key = 0; key < bucketCount; key += 1) {
      buckets[key] = new Int32Array(sizes[key]!);
    }
    sizes.fill(0);
    for (let row = 0; row < count; row += 1) {
      const key = signatures[row * tableCount + table]!;
      buckets[key]![sizes[key]!] = row;
      sizes[key] = sizes[key]! + 1;
    }
    result.push(buckets);
  }
  return result;
}

function appendBucket(
  bucket: Int32Array,
  row: number,
  salt: number,
  limit: number,
  stamp: number,
  seen: Int32Array,
  candidates: Int32Array,
  candidateCount: number,
): number {
  if (bucket.length <= limit) {
    for (let i = 0; i < bucket.length && candidateCount < candidates.length; i += 1) {
      candidateCount = appendCandidate(
        bucket[i]!,
        stamp,
        seen,
        candidates,
        candidateCount,
      );
    }
    return candidateCount;
  }

  // Evenly sample a large bucket, with a deterministic per-row phase. This
  // avoids quadratic work for long single-speaker recordings while retaining
  // candidates spread over the entire timeline.
  const phase = unsignedHash(row, salt) % bucket.length;
  for (let sample = 0; sample < limit && candidateCount < candidates.length; sample += 1) {
    const index = (phase + Math.floor((sample * bucket.length) / limit)) % bucket.length;
    candidateCount = appendCandidate(
      bucket[index]!,
      stamp,
      seen,
      candidates,
      candidateCount,
    );
  }
  return candidateCount;
}

function appendCandidate(
  candidate: number,
  stamp: number,
  seen: Int32Array,
  candidates: Int32Array,
  candidateCount: number,
): number {
  if (seen[candidate] === stamp || candidateCount === candidates.length) {
    return candidateCount;
  }
  seen[candidate] = stamp;
  candidates[candidateCount] = candidate;
  return candidateCount + 1;
}

function insertNeighbor(
  indices: Int32Array,
  similarities: Float32Array,
  offset: number,
  count: number,
  candidate: number,
  similarity: number,
): void {
  let position = count - 1;
  const lastSimilarity = similarities[offset + position]!;
  const lastIndex = indices[offset + position]!;
  if (
    similarity < lastSimilarity ||
    (similarity === lastSimilarity && lastIndex >= 0 && candidate > lastIndex)
  ) {
    return;
  }
  while (position > 0) {
    const previousSimilarity = similarities[offset + position - 1]!;
    const previousIndex = indices[offset + position - 1]!;
    if (
      similarity < previousSimilarity ||
      (similarity === previousSimilarity && candidate > previousIndex)
    ) {
      break;
    }
    similarities[offset + position] = previousSimilarity;
    indices[offset + position] = previousIndex;
    position -= 1;
  }
  similarities[offset + position] = similarity;
  indices[offset + position] = candidate;
}

function dot(
  values: Float32Array,
  leftOffset: number,
  rightOffset: number,
  dim: number,
): number {
  let result = 0;
  let column = 0;
  const unrolledEnd = dim - (dim % 4);
  for (; column < unrolledEnd; column += 4) {
    result +=
      values[leftOffset + column]! * values[rightOffset + column]! +
      values[leftOffset + column + 1]! * values[rightOffset + column + 1]! +
      values[leftOffset + column + 2]! * values[rightOffset + column + 2]! +
      values[leftOffset + column + 3]! * values[rightOffset + column + 3]!;
  }
  for (; column < dim; column += 1) {
    result += values[leftOffset + column]! * values[rightOffset + column]!;
  }
  return Math.max(-1, Math.min(1, result));
}

function euclideanDistance(
  values: Float32Array,
  leftOffset: number,
  rightOffset: number,
  dim: number,
): number {
  let squaredDistance = 0;
  let column = 0;
  const unrolledEnd = dim - (dim % 4);
  for (; column < unrolledEnd; column += 4) {
    const difference0 = values[leftOffset + column]! - values[rightOffset + column]!;
    const difference1 =
      values[leftOffset + column + 1]! - values[rightOffset + column + 1]!;
    const difference2 =
      values[leftOffset + column + 2]! - values[rightOffset + column + 2]!;
    const difference3 =
      values[leftOffset + column + 3]! - values[rightOffset + column + 3]!;
    squaredDistance +=
      difference0 * difference0 +
      difference1 * difference1 +
      difference2 * difference2 +
      difference3 * difference3;
  }
  for (; column < dim; column += 1) {
    const difference = values[leftOffset + column]! - values[rightOffset + column]!;
    squaredDistance += difference * difference;
  }
  return Math.sqrt(squaredDistance);
}

function xorshift32(value: number): number {
  let result = value | 0;
  result ^= result << 13;
  result ^= result >>> 17;
  result ^= result << 5;
  return result >>> 0;
}

function unsignedHash(left: number, right: number): number {
  let value = (left + 1) * 0x85ebca6b ^ (right + 1) * 0xc2b2ae35;
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb352d);
  value ^= value >>> 15;
  return value >>> 0;
}
