import { buildApproximateCosineKnn, normalizeRows } from "./knn";
import type { ClusteringNumericKernels } from "./numeric-kernels";
import type {
  ResolvedClusteringOptions,
  UmapProjectionStats,
} from "./types";

const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;
const INT_BYTES = Int32Array.BYTES_PER_ELEMENT;
const UMAP_A_MIN_DIST_ZERO = 1.9330828875904762;
const UMAP_B_MIN_DIST_ZERO = 0.7922918021756439;

export interface UmapProjection {
  readonly values: Float32Array;
  readonly dimension: number;
  readonly stats: UmapProjectionStats;
}

interface NeighborHeap {
  readonly indices: Int32Array;
  readonly distances: Float32Array;
  readonly isNew: Uint8Array;
  readonly size: number;
}

interface FuzzyGraph {
  readonly head: Int32Array;
  readonly tail: Int32Array;
  readonly weights: Float32Array;
  readonly edgeCount: number;
}

/**
 * Deterministic UMAP specialized for Senko's dense, row-major embeddings.
 *
 * All substantial state is held in flat typed arrays. Approximate cosine LSH
 * provides a strong seed which neighbor-of-neighbor descent refines using the
 * Euclidean metric used by the validated browser baseline. The fuzzy graph is
 * symmetrized with a radix sort and optimized in place for 50 epochs.
 */
export function projectWithUmap(
  embeddings: Float32Array,
  count: number,
  dim: number,
  options: ResolvedClusteringOptions,
  kernels?: ClusteringNumericKernels,
): UmapProjection {
  const started = performance.now();
  const outputDimension = Math.min(options.umapComponents, count - 2);
  const neighborCount = Math.min(options.umapNeighborCount, count);
  if (outputDimension < 2 || neighborCount < 2) {
    const stats: UmapProjectionStats = {
      count,
      inputDimension: dim,
      outputDimension: dim,
      neighborCount,
      epochs: 0,
      seedKnnMs: 0,
      refineKnnMs: 0,
      fuzzyGraphMs: 0,
      optimizeMs: 0,
      totalMs: performance.now() - started,
      graphEdgeCount: 0,
      outputBytes: 0,
      peakWorkingBytes: 0,
      peakTemporaryBytes: 0,
    };
    options.onUmapStats?.(stats);
    return { values: embeddings, dimension: dim, stats };
  }

  const allocations = new AllocationTracker();
  const random = createDeterministicRandom(options.umapRandomSeed);

  const seedStarted = performance.now();
  const normalized = normalizeRows(embeddings, count, dim);
  allocations.retain(normalized.byteLength);
  const seedNeighborCount = Math.min(64, count - 1);
  allocations.observeAdditional(
    estimateApproximateKnnPeakBytes(count, dim, seedNeighborCount, options),
  );
  const seed = buildApproximateCosineKnn(normalized, count, dim, {
    ...options,
    neighborCount: seedNeighborCount,
  });
  allocations.release(normalized.byteLength);
  allocations.retain(seed.indices.byteLength + seed.similarities.byteLength);
  const seedKnnMs = performance.now() - seedStarted;

  const refineStarted = performance.now();
  let neighbors: NeighborHeap;
  if (kernels === undefined) {
    neighbors = initializeEuclideanHeap(
      embeddings,
      count,
      dim,
      neighborCount,
      seed.indices,
      seed.neighborCount,
      random,
      allocations,
    );
  } else {
    neighbors = kernels.refineEuclideanNeighbors(
      embeddings,
      count,
      dim,
      neighborCount,
      seed.indices,
      seed.neighborCount,
      options.umapRandomSeed,
    );
    allocations.retain(
      neighbors.indices.byteLength +
        neighbors.distances.byteLength +
        neighbors.isNew.byteLength,
    );
  }
  allocations.release(seed.indices.byteLength + seed.similarities.byteLength);
  if (kernels === undefined) {
    refineNeighborGraph(embeddings, count, dim, neighbors, allocations);
    sortNeighborRows(neighbors, count);
  }
  const refineKnnMs = performance.now() - refineStarted;

  const graphStarted = performance.now();
  const graph = buildFuzzyGraph(
    neighbors,
    count,
    options.umapEpochs,
    allocations,
  );
  allocations.release(
    neighbors.indices.byteLength +
      neighbors.distances.byteLength +
      neighbors.isNew.byteLength,
  );
  const fuzzyGraphMs = performance.now() - graphStarted;

  const optimizeStarted = performance.now();
  // The WASM refinement owns an equivalent local PRNG. Advance the TS stream
  // past those four per-row samples so layout initialization remains identical.
  if (kernels !== undefined) {
    for (let skipped = 0; skipped < count * 4; skipped += 1) random();
  }
  const values = optimizeLayout(
    graph,
    count,
    outputDimension,
    options.umapEpochs,
    options.umapMinDistance,
    random,
    allocations,
  );
  allocations.release(
    graph.head.byteLength + graph.tail.byteLength + graph.weights.byteLength,
  );
  const optimizeMs = performance.now() - optimizeStarted;
  const outputBytes = values.byteLength;
  const peakWorkingBytes = allocations.peakBytes;
  const stats: UmapProjectionStats = {
    count,
    inputDimension: dim,
    outputDimension,
    neighborCount,
    epochs: options.umapEpochs,
    seedKnnMs,
    refineKnnMs,
    fuzzyGraphMs,
    optimizeMs,
    totalMs: performance.now() - started,
    graphEdgeCount: graph.edgeCount,
    outputBytes,
    peakWorkingBytes,
    peakTemporaryBytes: Math.max(0, peakWorkingBytes - outputBytes),
  };
  options.onUmapStats?.(stats);
  return { values, dimension: outputDimension, stats };
}

function initializeEuclideanHeap(
  embeddings: Float32Array,
  count: number,
  dim: number,
  neighborCount: number,
  seedIndices: Int32Array,
  seedNeighborCount: number,
  random: () => number,
  allocations: AllocationTracker,
): NeighborHeap {
  const indices = new Int32Array(count * neighborCount);
  indices.fill(-1);
  const distances = new Float32Array(count * neighborCount);
  distances.fill(Number.POSITIVE_INFINITY);
  const isNew = new Uint8Array(count * neighborCount);
  allocations.retain(indices.byteLength + distances.byteLength + isNew.byteLength);
  const heap: NeighborHeap = { indices, distances, isNew, size: neighborCount };

  for (let row = 0; row < count; row += 1) {
    heapPush(heap, row, 0, row, 1);
    const seedOffset = row * seedNeighborCount;
    for (let rank = 0; rank < seedNeighborCount; rank += 1) {
      const candidate = seedIndices[seedOffset + rank]!;
      if (candidate < 0) {
        continue;
      }
      const distance = euclideanDistance(embeddings, row, candidate, dim);
      heapPush(heap, row, distance, candidate, 1);
      heapPush(heap, candidate, distance, row, 1);
    }
    // A few global edges make refinement robust when an LSH bucket misses an
    // isolated manifold branch. They are deterministic and quickly evicted if
    // they are not competitive.
    for (let sample = 0; sample < 4; sample += 1) {
      const candidate = Math.floor(random() * count);
      const distance = euclideanDistance(embeddings, row, candidate, dim);
      heapPush(heap, row, distance, candidate, 1);
      heapPush(heap, candidate, distance, row, 1);
    }
  }
  return heap;
}

function refineNeighborGraph(
  embeddings: Float32Array,
  count: number,
  dim: number,
  heap: NeighborHeap,
  allocations: AllocationTracker,
): void {
  const snapshotIndices = new Int32Array(heap.indices.length);
  const snapshotFlags = new Uint8Array(heap.isNew.length);
  allocations.retain(snapshotIndices.byteLength + snapshotFlags.byteLength);
  const convergenceLimit = Math.max(1, Math.floor(0.001 * heap.size * count));

  for (let iteration = 0; iteration < 6; iteration += 1) {
    snapshotIndices.set(heap.indices);
    snapshotFlags.set(heap.isNew);
    heap.isNew.fill(0);
    let changes = 0;
    for (let row = 0; row < count; row += 1) {
      const rowOffset = row * heap.size;
      for (let rank = 0; rank < heap.size; rank += 1) {
        const pivotOffset = rowOffset + rank;
        const pivot = snapshotIndices[pivotOffset]!;
        if (pivot < 0 || pivot === row) {
          continue;
        }
        const pivotIsNew = snapshotFlags[pivotOffset] !== 0;
        const neighborOffset = pivot * heap.size;
        for (let candidateRank = 0; candidateRank < heap.size; candidateRank += 1) {
          const candidateOffset = neighborOffset + candidateRank;
          const candidate = snapshotIndices[candidateOffset]!;
          if (
            candidate < 0 ||
            candidate === row ||
            (!pivotIsNew && snapshotFlags[candidateOffset] === 0)
          ) {
            continue;
          }
          const distance = euclideanDistance(embeddings, row, candidate, dim);
          changes += heapPush(heap, row, distance, candidate, 1);
          changes += heapPush(heap, candidate, distance, row, 1);
        }
      }
    }
    if (changes <= convergenceLimit) {
      break;
    }
  }
  allocations.release(snapshotIndices.byteLength + snapshotFlags.byteLength);
}

function heapPush(
  heap: NeighborHeap,
  row: number,
  distance: number,
  index: number,
  flag: number,
): number {
  const offset = row * heap.size;
  if (distance >= heap.distances[offset]!) {
    return 0;
  }
  for (let position = 0; position < heap.size; position += 1) {
    if (heap.indices[offset + position] === index) {
      return 0;
    }
  }

  let position = 0;
  while (true) {
    const left = position * 2 + 1;
    if (left >= heap.size) {
      break;
    }
    const right = left + 1;
    let swap = left;
    if (
      right < heap.size &&
      heap.distances[offset + right]! > heap.distances[offset + left]!
    ) {
      swap = right;
    }
    if (distance >= heap.distances[offset + swap]!) {
      break;
    }
    heap.distances[offset + position] = heap.distances[offset + swap]!;
    heap.indices[offset + position] = heap.indices[offset + swap]!;
    heap.isNew[offset + position] = heap.isNew[offset + swap]!;
    position = swap;
  }
  heap.distances[offset + position] = distance;
  heap.indices[offset + position] = index;
  heap.isNew[offset + position] = flag;
  return 1;
}

function sortNeighborRows(heap: NeighborHeap, count: number): void {
  for (let row = 0; row < count; row += 1) {
    const offset = row * heap.size;
    for (let current = 1; current < heap.size; current += 1) {
      const distance = heap.distances[offset + current]!;
      const index = heap.indices[offset + current]!;
      const flag = heap.isNew[offset + current]!;
      let position = current;
      while (
        position > 0 &&
        (distance < heap.distances[offset + position - 1]! ||
          (distance === heap.distances[offset + position - 1]! &&
            index < heap.indices[offset + position - 1]!))
      ) {
        heap.distances[offset + position] = heap.distances[offset + position - 1]!;
        heap.indices[offset + position] = heap.indices[offset + position - 1]!;
        heap.isNew[offset + position] = heap.isNew[offset + position - 1]!;
        position -= 1;
      }
      heap.distances[offset + position] = distance;
      heap.indices[offset + position] = index;
      heap.isNew[offset + position] = flag;
    }
  }
}

function buildFuzzyGraph(
  neighbors: NeighborHeap,
  count: number,
  epochs: number,
  allocations: AllocationTracker,
): FuzzyGraph {
  const sigmas = new Float32Array(count);
  const rhos = new Float32Array(count);
  allocations.retain(sigmas.byteLength + rhos.byteLength);
  smoothKnnDistances(neighbors, count, sigmas, rhos);

  const directedCount = count * Math.max(0, neighbors.size - 1);
  const capacity = directedCount * 2;
  let keys: Uint32Array = new Uint32Array(capacity);
  let weights: Float32Array = new Float32Array(capacity);
  allocations.retain(keys.byteLength + weights.byteLength);
  let cursor = 0;
  for (let row = 0; row < count; row += 1) {
    const offset = row * neighbors.size;
    for (let rank = 1; rank < neighbors.size; rank += 1) {
      const column = neighbors.indices[offset + rank]!;
      if (column < 0 || column === row) {
        continue;
      }
      const distance = neighbors.distances[offset + rank]!;
      const membership =
        distance - rhos[row]! <= 0
          ? 1
          : Math.exp(-((distance - rhos[row]!) / sigmas[row]!));
      keys[cursor] = row * count + column;
      weights[cursor] = membership;
      cursor += 1;
      keys[cursor] = column * count + row;
      weights[cursor] = membership;
      cursor += 1;
    }
  }
  allocations.release(sigmas.byteLength + rhos.byteLength);

  const sorted = radixSortKeyValues(keys, weights, cursor, allocations);
  keys = sorted.keys;
  weights = sorted.values;
  let uniqueCount = 0;
  for (let read = 0; read < cursor; ) {
    const key = keys[read]!;
    let complement = 1;
    do {
      complement *= 1 - weights[read]!;
      read += 1;
    } while (read < cursor && keys[read] === key);
    const unionWeight = 1 - complement;
    if (unionWeight >= 1 / epochs) {
      keys[uniqueCount] = key;
      weights[uniqueCount] = unionWeight;
      uniqueCount += 1;
    }
  }

  const head = new Int32Array(uniqueCount);
  const tail = new Int32Array(uniqueCount);
  const compactWeights = new Float32Array(uniqueCount);
  allocations.retain(head.byteLength + tail.byteLength + compactWeights.byteLength);
  for (let edge = 0; edge < uniqueCount; edge += 1) {
    const key = keys[edge]!;
    const row = Math.floor(key / count);
    tail[edge] = row;
    head[edge] = key - row * count;
    compactWeights[edge] = weights[edge]!;
  }
  allocations.release(keys.byteLength + weights.byteLength);
  return { head, tail, weights: compactWeights, edgeCount: uniqueCount };
}

function smoothKnnDistances(
  neighbors: NeighborHeap,
  count: number,
  sigmas: Float32Array,
  rhos: Float32Array,
): void {
  const target = Math.log2(neighbors.size);
  let globalSum = 0;
  let globalCount = 0;
  for (let i = 0; i < neighbors.distances.length; i += 1) {
    const value = neighbors.distances[i]!;
    if (Number.isFinite(value)) {
      globalSum += value;
      globalCount += 1;
    }
  }
  const globalMean = globalCount > 0 ? globalSum / globalCount : 1;
  for (let row = 0; row < count; row += 1) {
    const offset = row * neighbors.size;
    let rho = 0;
    let rowSum = 0;
    let rowCount = 0;
    for (let rank = 0; rank < neighbors.size; rank += 1) {
      const distance = neighbors.distances[offset + rank]!;
      if (Number.isFinite(distance)) {
        rowSum += distance;
        rowCount += 1;
        if (rho === 0 && distance > 0) {
          rho = distance;
        }
      }
    }
    rhos[row] = rho;
    let low = 0;
    let high = Number.POSITIVE_INFINITY;
    let sigma = 1;
    for (let iteration = 0; iteration < 64; iteration += 1) {
      let sum = 0;
      for (let rank = 1; rank < neighbors.size; rank += 1) {
        const distance = neighbors.distances[offset + rank]! - rho;
        sum += distance > 0 ? Math.exp(-(distance / sigma)) : 1;
      }
      if (Math.abs(sum - target) < 1e-5) {
        break;
      }
      if (sum > target) {
        high = sigma;
        sigma = (low + high) / 2;
      } else {
        low = sigma;
        sigma = high === Number.POSITIVE_INFINITY ? sigma * 2 : (low + high) / 2;
      }
    }
    const mean = rowCount > 0 ? rowSum / rowCount : globalMean;
    sigmas[row] = Math.max(sigma, 1e-3 * (rho > 0 ? mean : globalMean));
  }
}

function radixSortKeyValues(
  inputKeys: Uint32Array,
  inputValues: Float32Array,
  length: number,
  allocations: AllocationTracker,
): { keys: Uint32Array; values: Float32Array } {
  let keys = inputKeys;
  let values = inputValues;
  let temporaryKeys: Uint32Array = new Uint32Array(inputKeys.length);
  let temporaryValues: Float32Array = new Float32Array(inputValues.length);
  const counts = new Uint32Array(1 << 16);
  allocations.retain(
    temporaryKeys.byteLength + temporaryValues.byteLength + counts.byteLength,
  );
  for (const shift of [0, 16]) {
    counts.fill(0);
    for (let i = 0; i < length; i += 1) {
      const bucket = (keys[i]! >>> shift) & 0xffff;
      counts[bucket] = counts[bucket]! + 1;
    }
    let position = 0;
    for (let bucket = 0; bucket < counts.length; bucket += 1) {
      const size = counts[bucket]!;
      counts[bucket] = position;
      position += size;
    }
    for (let i = 0; i < length; i += 1) {
      const bucket = (keys[i]! >>> shift) & 0xffff;
      const destination = counts[bucket]!;
      temporaryKeys[destination] = keys[i]!;
      temporaryValues[destination] = values[i]!;
      counts[bucket] = destination + 1;
    }
    const oldKeys = keys;
    const oldValues = values;
    keys = temporaryKeys;
    values = temporaryValues;
    temporaryKeys = oldKeys;
    temporaryValues = oldValues;
  }
  // Two passes return data to the input arrays. The temporary pair can be
  // released while the sorted input remains live.
  allocations.release(
    temporaryKeys.byteLength + temporaryValues.byteLength + counts.byteLength,
  );
  return { keys, values };
}

function optimizeLayout(
  graph: FuzzyGraph,
  count: number,
  dim: number,
  epochs: number,
  minDistance: number,
  random: () => number,
  allocations: AllocationTracker,
): Float32Array {
  const embedding = new Float32Array(count * dim);
  for (let i = 0; i < embedding.length; i += 1) {
    embedding[i] = random() * 20 - 10;
  }
  const epochsPerSample = new Float32Array(graph.edgeCount);
  const epochOfNextSample = new Float32Array(graph.edgeCount);
  const epochsPerNegativeSample = new Float32Array(graph.edgeCount);
  const epochOfNextNegativeSample = new Float32Array(graph.edgeCount);
  allocations.retain(
    embedding.byteLength +
      epochsPerSample.byteLength +
      epochOfNextSample.byteLength +
      epochsPerNegativeSample.byteLength +
      epochOfNextNegativeSample.byteLength,
  );
  let maximumWeight = 0;
  for (let edge = 0; edge < graph.edgeCount; edge += 1) {
    maximumWeight = Math.max(maximumWeight, graph.weights[edge]!);
  }
  for (let edge = 0; edge < graph.edgeCount; edge += 1) {
    const period = maximumWeight / graph.weights[edge]!;
    epochsPerSample[edge] = period;
    epochOfNextSample[edge] = period;
    epochsPerNegativeSample[edge] = period / 5;
    epochOfNextNegativeSample[edge] = period / 5;
  }

  const { a, b } = curveParameters(minDistance);
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const alpha = 1 - epoch / epochs;
    for (let edge = 0; edge < graph.edgeCount; edge += 1) {
      if (epochOfNextSample[edge]! > epoch) {
        continue;
      }
      const currentRow = graph.head[edge]!;
      const otherRow = graph.tail[edge]!;
      const currentOffset = currentRow * dim;
      const otherOffset = otherRow * dim;
      let squaredDistance = 0;
      for (let component = 0; component < dim; component += 1) {
        const difference =
          embedding[currentOffset + component]! - embedding[otherOffset + component]!;
        squaredDistance += difference * difference;
      }
      let coefficient = 0;
      if (squaredDistance > 0) {
        coefficient =
          (-2 * a * b * squaredDistance ** (b - 1)) /
          (a * squaredDistance ** b + 1);
      }
      for (let component = 0; component < dim; component += 1) {
        const difference =
          embedding[currentOffset + component]! - embedding[otherOffset + component]!;
        const gradient = clip(coefficient * difference, 4) * alpha;
        embedding[currentOffset + component] =
          embedding[currentOffset + component]! + gradient;
        embedding[otherOffset + component] =
          embedding[otherOffset + component]! - gradient;
      }
      epochOfNextSample[edge] =
        epochOfNextSample[edge]! + epochsPerSample[edge]!;

      const negativePeriod = epochsPerNegativeSample[edge]!;
      const negativeSamples = Math.floor(
        (epoch - epochOfNextNegativeSample[edge]!) / negativePeriod,
      );
      for (let sample = 0; sample < negativeSamples; sample += 1) {
        const negativeRow = Math.floor(random() * count);
        if (negativeRow === currentRow) {
          continue;
        }
        const negativeOffset = negativeRow * dim;
        squaredDistance = 0;
        for (let component = 0; component < dim; component += 1) {
          const difference =
            embedding[currentOffset + component]! -
            embedding[negativeOffset + component]!;
          squaredDistance += difference * difference;
        }
        let repulsion = 0;
        if (squaredDistance > 0) {
          repulsion =
            (2 * b) /
            ((0.001 + squaredDistance) * (a * squaredDistance ** b + 1));
        }
        for (let component = 0; component < dim; component += 1) {
          const difference =
            embedding[currentOffset + component]! -
            embedding[negativeOffset + component]!;
          const gradient =
            (repulsion > 0 ? clip(repulsion * difference, 4) : 4) * alpha;
          embedding[currentOffset + component] =
            embedding[currentOffset + component]! + gradient;
        }
      }
      epochOfNextNegativeSample[edge] =
        epochOfNextNegativeSample[edge]! + negativeSamples * negativePeriod;
    }
  }
  allocations.release(
    epochsPerSample.byteLength +
      epochOfNextSample.byteLength +
      epochsPerNegativeSample.byteLength +
      epochOfNextNegativeSample.byteLength,
  );
  return embedding;
}

function euclideanDistance(
  embeddings: Float32Array,
  leftRow: number,
  rightRow: number,
  dim: number,
): number {
  const leftOffset = leftRow * dim;
  const rightOffset = rightRow * dim;
  let squaredDistance = 0;
  let column = 0;
  const unrolledEnd = dim - (dim % 4);
  for (; column < unrolledEnd; column += 4) {
    const difference0 = embeddings[leftOffset + column]! - embeddings[rightOffset + column]!;
    const difference1 =
      embeddings[leftOffset + column + 1]! - embeddings[rightOffset + column + 1]!;
    const difference2 =
      embeddings[leftOffset + column + 2]! - embeddings[rightOffset + column + 2]!;
    const difference3 =
      embeddings[leftOffset + column + 3]! - embeddings[rightOffset + column + 3]!;
    squaredDistance +=
      difference0 * difference0 +
      difference1 * difference1 +
      difference2 * difference2 +
      difference3 * difference3;
  }
  for (; column < dim; column += 1) {
    const difference = embeddings[leftOffset + column]! - embeddings[rightOffset + column]!;
    squaredDistance += difference * difference;
  }
  return Math.sqrt(squaredDistance);
}

function curveParameters(minDistance: number): { a: number; b: number } {
  if (minDistance === 0) {
    return { a: UMAP_A_MIN_DIST_ZERO, b: UMAP_B_MIN_DIST_ZERO };
  }
  // Accurate at the library's conventional minDist=0.1 and smoothly
  // interpolated for diagnostic overrides. Production uses minDist=0.
  const ratio = Math.min(1, minDistance / 0.1);
  return {
    a: UMAP_A_MIN_DIST_ZERO + (1.5694704762346365 - UMAP_A_MIN_DIST_ZERO) * ratio,
    b: UMAP_B_MIN_DIST_ZERO + (0.8941996053733949 - UMAP_B_MIN_DIST_ZERO) * ratio,
  };
}

function clip(value: number, limit: number): number {
  return Math.max(-limit, Math.min(limit, value));
}

function createDeterministicRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function estimateApproximateKnnPeakBytes(
  count: number,
  dim: number,
  neighborCount: number,
  options: ResolvedClusteringOptions,
): number {
  const output = count * neighborCount * (INT_BYTES + FLOAT_BYTES);
  const signatures = count * options.hashTableCount * Uint16Array.BYTES_PER_ELEMENT;
  const buckets = count * options.hashTableCount * INT_BYTES;
  const queryState = count * INT_BYTES;
  const candidateCapacity = Math.min(
    count,
    options.hashTableCount * options.bucketSampleLimit * 2 +
      options.temporalNeighborRadius * 2,
  );
  const candidates = candidateCapacity * INT_BYTES;
  const bucketSizes = (1 << options.hashBits) * INT_BYTES;
  const planes = options.hashTableCount * options.hashBits * dim;
  return Math.max(
    output + signatures + planes,
    output + signatures + buckets + queryState + candidates + bucketSizes,
  );
}

class AllocationTracker {
  currentBytes = 0;
  peakBytes = 0;

  retain(bytes: number): void {
    this.currentBytes += bytes;
    this.peakBytes = Math.max(this.peakBytes, this.currentBytes);
  }

  release(bytes: number): void {
    this.currentBytes = Math.max(0, this.currentBytes - bytes);
  }

  observeAdditional(bytes: number): void {
    this.peakBytes = Math.max(this.peakBytes, this.currentBytes + bytes);
  }
}
