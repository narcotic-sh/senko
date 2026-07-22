import type { KnnGraph } from "./knn";

interface DensityCluster {
  readonly treeNode: number;
  readonly birthLambda: number;
  stability: number;
  readonly children: DensityCluster[];
  selected: boolean;
}

interface Dendrogram {
  readonly root: number;
  readonly left: Int32Array;
  readonly right: Int32Array;
  readonly sizes: Int32Array;
  readonly distances: Float64Array;
}

interface SortedMutualReachabilityEdges {
  readonly order: Uint32Array;
  readonly weights: Float64Array;
  readonly edgeCount: number;
}

const RADIX_BITS = 16;
const RADIX_SIZE = 1 << RADIX_BITS;
const RADIX_MASK = RADIX_SIZE - 1;
const FLOAT64_WORD_ORDER = new Uint32Array(new Float64Array([1]).buffer);
const FLOAT64_LOW_WORD_OFFSET = FLOAT64_WORD_ORDER[0] === 0 ? 0 : 1;
const FLOAT64_HIGH_WORD_OFFSET = 1 - FLOAT64_LOW_WORD_OFFSET;

/**
 * Extract HDBSCAN-style clusters from a sparse k-NN graph.
 *
 * This follows the reference algorithm's mutual-reachability MST, condensed
 * hierarchy and excess-of-mass selection. The approximation is confined to
 * neighbor discovery; hierarchy construction itself is deterministic.
 */
export function clusterSparseGraph(
  graph: KnnGraph,
  count: number,
  minSamples: number,
  minClusterSize: number,
): Int32Array {
  if (count === 0) {
    return new Int32Array();
  }
  if (count < minClusterSize || graph.neighborCount === 0) {
    return new Int32Array(count);
  }

  const coreDistances = calculateCoreDistances(graph, count, minSamples);
  const dendrogram = buildDendrogram(graph, count, coreDistances);
  const rootCluster: DensityCluster = {
    treeNode: dendrogram.root,
    birthLambda: 0,
    stability: 0,
    children: [],
    selected: false,
  };
  condense(
    dendrogram.root,
    rootCluster,
    dendrogram,
    minClusterSize,
  );

  // HDBSCAN excludes the all-points root unless allow_single_cluster is set.
  // Senko's post-processing then naturally turns an all-noise result into one
  // cluster, matching the Python pipeline's behavior.
  for (const child of rootCluster.children) {
    selectExcessOfMass(child);
  }

  const labels = new Int32Array(count);
  labels.fill(-1);
  let nextLabel = 0;
  const stack: DensityCluster[] = [...rootCluster.children].reverse();
  while (stack.length > 0) {
    const cluster = stack.pop()!;
    if (cluster.selected) {
      labelSubtree(cluster.treeNode, nextLabel, labels, dendrogram, count);
      nextLabel += 1;
      continue;
    }
    for (let i = cluster.children.length - 1; i >= 0; i -= 1) {
      stack.push(cluster.children[i]!);
    }
  }
  return labels;
}

function calculateCoreDistances(
  graph: KnnGraph,
  count: number,
  minSamples: number,
): Float64Array {
  const result = new Float64Array(count);
  const rank = Math.min(minSamples, graph.neighborCount) - 1;
  for (let row = 0; row < count; row += 1) {
    const similarity = graph.similarities[row * graph.neighborCount + rank]!;
    result[row] = Number.isFinite(similarity) ? Math.max(0, 1 - similarity) : 2;
  }
  return result;
}

function buildDendrogram(
  graph: KnnGraph,
  count: number,
  coreDistances: Float64Array,
): Dendrogram {
  const sortedEdges = buildSortedMutualReachabilityEdges(
    graph,
    count,
    coreDistances,
  );

  const maximumNodes = count * 2;
  const left = new Int32Array(maximumNodes);
  const right = new Int32Array(maximumNodes);
  left.fill(-1);
  right.fill(-1);
  const sizes = new Int32Array(maximumNodes);
  const distances = new Float64Array(maximumNodes);
  const parent = new Int32Array(count);
  const componentSize = new Int32Array(count);
  const componentNode = new Int32Array(count);
  for (let i = 0; i < count; i += 1) {
    parent[i] = i;
    componentSize[i] = 1;
    componentNode[i] = i;
    sizes[i] = 1;
  }

  let nextNode = count;
  let maximumWeight = 0;
  for (let cursor = 0; cursor < sortedEdges.edgeCount; cursor += 1) {
    const edge = sortedEdges.order[cursor]!;
    const from = Math.floor(edge / graph.neighborCount);
    const to = graph.indices[edge]!;
    let fromRoot = findRoot(parent, from);
    let toRoot = findRoot(parent, to);
    if (fromRoot === toRoot) {
      continue;
    }
    if (componentSize[fromRoot]! < componentSize[toRoot]!) {
      const swap = fromRoot;
      fromRoot = toRoot;
      toRoot = swap;
    }
    const node = nextNode;
    nextNode += 1;
    left[node] = componentNode[fromRoot]!;
    right[node] = componentNode[toRoot]!;
    sizes[node] = componentSize[fromRoot]! + componentSize[toRoot]!;
    const weight = sortedEdges.weights[edge]!;
    distances[node] = weight;
    maximumWeight = Math.max(maximumWeight, weight);

    parent[toRoot] = fromRoot;
    componentSize[fromRoot] = sizes[node]!;
    componentNode[fromRoot] = node;
    if (sizes[node] === count) {
      break;
    }
  }

  // Approximate-neighbor graphs can be disconnected. Join their trees at a
  // distance outside the observed graph, which preserves every component as a
  // separate high-level density branch.
  const roots: number[] = [];
  for (let i = 0; i < count; i += 1) {
    if (findRoot(parent, i) === i) {
      roots.push(i);
    }
  }
  let rootNode = componentNode[roots[0]!]!;
  const bridgeDistance = Math.max(2, maximumWeight + 1e-6);
  for (let i = 1; i < roots.length; i += 1) {
    const otherNode = componentNode[roots[i]!]!;
    const node = nextNode;
    nextNode += 1;
    left[node] = rootNode;
    right[node] = otherNode;
    sizes[node] = sizes[rootNode]! + sizes[otherNode]!;
    distances[node] = bridgeDistance;
    rootNode = node;
  }

  return { root: rootNode, left, right, sizes, distances };
}

/**
 * Build the deterministic Kruskal edge order without allocating boxed indices.
 *
 * Mutual-reachability weights are non-negative Float64 values, whose unsigned
 * IEEE-754 bit patterns have the same order as their numeric values. Stable LSD
 * radix passes over `(weight, from, to)` therefore reproduce the former
 * comparator sort exactly, including insertion order for duplicate edges.
 * Exported from this module so the ordering invariant can be tested directly.
 */
export function buildSortedMutualReachabilityEdges(
  graph: KnnGraph,
  count: number,
  coreDistances: Float64Array,
): SortedMutualReachabilityEdges {
  const directedEdgeCount = count * graph.neighborCount;
  const weights = new Float64Array(directedEdgeCount);
  let order = new Uint32Array(directedEdgeCount);
  let edgeCount = 0;

  for (let from = 0; from < count; from += 1) {
    const offset = from * graph.neighborCount;
    for (let rank = 0; rank < graph.neighborCount; rank += 1) {
      const edge = offset + rank;
      const to = graph.indices[edge]!;
      if (to < 0 || to === from) {
        continue;
      }
      const distance = Math.max(0, 1 - graph.similarities[edge]!);
      weights[edge] = Math.max(coreDistances[from]!, coreDistances[to]!, distance);
      order[edgeCount] = edge;
      edgeCount += 1;
    }
  }
  if (edgeCount < 2) {
    return { order, weights, edgeCount };
  }

  let temporary = new Uint32Array(directedEdgeCount);
  const counts = new Uint32Array(RADIX_SIZE);
  const weightWords = new Uint32Array(weights.buffer);
  // Production is far below 65,536 rows, but retain exact endpoint ordering for
  // larger valid Int32 graphs by adding a second 16-bit pass per endpoint.
  const endpointPasses = count <= RADIX_SIZE ? 1 : 2;
  const weightPassStart = endpointPasses * 2;
  const totalPasses = weightPassStart + 4;

  for (let pass = 0; pass < totalPasses; pass += 1) {
    counts.fill(0);
    for (let cursor = 0; cursor < edgeCount; cursor += 1) {
      const edge = order[cursor]!;
      let key: number;
      if (pass < endpointPasses) {
        key = (graph.indices[edge]! >>> (pass * RADIX_BITS)) & RADIX_MASK;
      } else if (pass < weightPassStart) {
        const from = Math.floor(edge / graph.neighborCount);
        key =
          (from >>> ((pass - endpointPasses) * RADIX_BITS)) & RADIX_MASK;
      } else {
        const weightPass = pass - weightPassStart;
        const wordOffset =
          weightPass < 2 ? FLOAT64_LOW_WORD_OFFSET : FLOAT64_HIGH_WORD_OFFSET;
        const word = weightWords[edge * 2 + wordOffset]!;
        key =
          weightPass % 2 === 0 ? word & RADIX_MASK : word >>> RADIX_BITS;
      }
      counts[key] = counts[key]! + 1;
    }

    let position = 0;
    for (let key = 0; key < RADIX_SIZE; key += 1) {
      const size = counts[key]!;
      counts[key] = position;
      position += size;
    }

    for (let cursor = 0; cursor < edgeCount; cursor += 1) {
      const edge = order[cursor]!;
      let key: number;
      if (pass < endpointPasses) {
        key = (graph.indices[edge]! >>> (pass * RADIX_BITS)) & RADIX_MASK;
      } else if (pass < weightPassStart) {
        const from = Math.floor(edge / graph.neighborCount);
        key =
          (from >>> ((pass - endpointPasses) * RADIX_BITS)) & RADIX_MASK;
      } else {
        const weightPass = pass - weightPassStart;
        const wordOffset =
          weightPass < 2 ? FLOAT64_LOW_WORD_OFFSET : FLOAT64_HIGH_WORD_OFFSET;
        const word = weightWords[edge * 2 + wordOffset]!;
        key =
          weightPass % 2 === 0 ? word & RADIX_MASK : word >>> RADIX_BITS;
      }
      const destination = counts[key]!;
      temporary[destination] = edge;
      counts[key] = destination + 1;
    }

    const previous = order;
    order = temporary;
    temporary = previous;
  }

  return { order, weights, edgeCount };
}

function condense(
  node: number,
  cluster: DensityCluster,
  dendrogram: Dendrogram,
  minClusterSize: number,
): void {
  if (dendrogram.left[node]! < 0) {
    return;
  }
  const left = dendrogram.left[node]!;
  const right = dendrogram.right[node]!;
  const lambda = distanceToLambda(dendrogram.distances[node]!);
  const leftLarge = dendrogram.sizes[left]! >= minClusterSize;
  const rightLarge = dendrogram.sizes[right]! >= minClusterSize;

  if (leftLarge && rightLarge) {
    cluster.stability +=
      dendrogram.sizes[node]! * Math.max(0, lambda - cluster.birthLambda);
    const leftCluster: DensityCluster = {
      treeNode: left,
      birthLambda: lambda,
      stability: 0,
      children: [],
      selected: false,
    };
    const rightCluster: DensityCluster = {
      treeNode: right,
      birthLambda: lambda,
      stability: 0,
      children: [],
      selected: false,
    };
    cluster.children.push(leftCluster, rightCluster);
    condense(left, leftCluster, dendrogram, minClusterSize);
    condense(right, rightCluster, dendrogram, minClusterSize);
    return;
  }

  if (leftLarge) {
    cluster.stability +=
      dendrogram.sizes[right]! * Math.max(0, lambda - cluster.birthLambda);
    condense(left, cluster, dendrogram, minClusterSize);
    return;
  }
  if (rightLarge) {
    cluster.stability +=
      dendrogram.sizes[left]! * Math.max(0, lambda - cluster.birthLambda);
    condense(right, cluster, dendrogram, minClusterSize);
    return;
  }

  cluster.stability +=
    dendrogram.sizes[node]! * Math.max(0, lambda - cluster.birthLambda);
}

function selectExcessOfMass(cluster: DensityCluster): number {
  if (cluster.children.length === 0) {
    cluster.selected = true;
    return cluster.stability;
  }
  let descendantsStability = 0;
  for (const child of cluster.children) {
    descendantsStability += selectExcessOfMass(child);
  }
  if (cluster.stability >= descendantsStability) {
    clearSelection(cluster.children);
    cluster.selected = true;
    return cluster.stability;
  }
  return descendantsStability;
}

function clearSelection(clusters: readonly DensityCluster[]): void {
  for (const cluster of clusters) {
    cluster.selected = false;
    clearSelection(cluster.children);
  }
}

function labelSubtree(
  treeNode: number,
  label: number,
  labels: Int32Array,
  dendrogram: Dendrogram,
  leafCount: number,
): void {
  const stack = [treeNode];
  while (stack.length > 0) {
    const node = stack.pop()!;
    if (node < leafCount) {
      // A selected ancestor always wins; selected clusters are disjoint.
      if (labels[node]! < 0) {
        labels[node] = label;
      }
      continue;
    }
    stack.push(dendrogram.right[node]!, dendrogram.left[node]!);
  }
}

function findRoot(parent: Int32Array, value: number): number {
  let root = value;
  while (parent[root] !== root) {
    root = parent[root]!;
  }
  let cursor = value;
  while (parent[cursor] !== cursor) {
    const next = parent[cursor]!;
    parent[cursor] = root;
    cursor = next;
  }
  return root;
}

function distanceToLambda(distance: number): number {
  return 1 / Math.max(distance, 1e-7);
}
