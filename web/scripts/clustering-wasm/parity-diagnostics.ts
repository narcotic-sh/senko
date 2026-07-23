export interface LabelHistogramEntry {
  readonly label: number;
  readonly size: number;
}

export interface LabelPartitionSummary {
  readonly sampleCount: number;
  /** Number of non-noise labels. */
  readonly clusterCount: number;
  readonly noiseCount: number;
  /** Label-invariant population sizes, excluding noise. */
  readonly clusterSizesDescending: readonly number[];
  /** Label-preserving histogram, useful when inspecting raw HDBSCAN output. */
  readonly histogram: readonly LabelHistogramEntry[];
}

export interface LabelParityDiagnostics {
  readonly adjustedRandIndex: number;
  /**
   * True when the two partitions are identical after renumbering ordinary
   * clusters by first occurrence. HDBSCAN's -1 noise label remains special.
   */
  readonly exactPartition: boolean;
  readonly exactNoiseMask: boolean;
  readonly noiseMismatchCount: number;
  readonly firstNoiseMismatchIndices: readonly number[];
  readonly reference: LabelPartitionSummary;
  readonly candidate: LabelPartitionSummary;
}

export interface NumericMismatch {
  readonly index: number;
  readonly reference: number;
  readonly candidate: number;
  readonly absoluteError: number;
  readonly relativeError: number;
}

export interface NumericParityDiagnostics {
  readonly length: number;
  readonly mismatchCount: number;
  readonly nonFiniteMismatchCount: number;
  readonly maxAbsoluteError: number;
  readonly maxRelativeError: number;
  readonly firstMismatches: readonly NumericMismatch[];
}

export interface NumericTolerance {
  readonly absolute: number;
  readonly relative: number;
  readonly mismatchSampleLimit?: number;
}

export interface UndirectedWeightedEdge {
  readonly from: number;
  readonly to: number;
  readonly weight: number;
}

export interface MstParityDiagnostics {
  readonly exactEndpoints: boolean;
  readonly missingEndpointPairs: readonly string[];
  readonly unexpectedEndpointPairs: readonly string[];
  readonly weights: NumericParityDiagnostics;
}

const NOISE_LABEL = -1;
const DEFAULT_MISMATCH_SAMPLE_LIMIT = 16;

/**
 * Compare raw HDBSCAN labels without assuming that cluster IDs have the same
 * numbers in two implementations.
 *
 * ARI alone is deliberately insufficient: it treats a retained -1 population
 * like any other cluster. Senko's post-processing gives -1 semantic meaning,
 * so parity also requires an exact noise mask.
 */
export function compareLabelPartitions(
  reference: ArrayLike<number>,
  candidate: ArrayLike<number>,
  mismatchSampleLimit = DEFAULT_MISMATCH_SAMPLE_LIMIT,
): LabelParityDiagnostics {
  requireEqualLength("label", reference, candidate);
  requireNonNegativeInteger("mismatchSampleLimit", mismatchSampleLimit);
  validateLabels("reference", reference);
  validateLabels("candidate", candidate);

  const canonicalReference = canonicalizeLabels(reference);
  const canonicalCandidate = canonicalizeLabels(candidate);
  let exactPartition = true;
  let noiseMismatchCount = 0;
  const firstNoiseMismatchIndices: number[] = [];
  for (let index = 0; index < reference.length; index += 1) {
    if (canonicalReference[index] !== canonicalCandidate[index]) {
      exactPartition = false;
    }
    if (
      (reference[index] === NOISE_LABEL) !==
      (candidate[index] === NOISE_LABEL)
    ) {
      noiseMismatchCount += 1;
      if (firstNoiseMismatchIndices.length < mismatchSampleLimit) {
        firstNoiseMismatchIndices.push(index);
      }
    }
  }

  return {
    adjustedRandIndex: adjustedRandIndex(reference, candidate),
    exactPartition,
    exactNoiseMask: noiseMismatchCount === 0,
    noiseMismatchCount,
    firstNoiseMismatchIndices,
    reference: summarizeLabels(reference),
    candidate: summarizeLabels(candidate),
  };
}

export function summarizeLabels(
  labels: ArrayLike<number>,
): LabelPartitionSummary {
  validateLabels("labels", labels);
  const counts = new Map<number, number>();
  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index]!;
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  const histogram = [...counts.entries()]
    .map(([label, size]) => ({ label, size }))
    .sort((left, right) => left.label - right.label);
  const noiseCount = counts.get(NOISE_LABEL) ?? 0;
  const clusterSizesDescending = histogram
    .filter(({ label }) => label !== NOISE_LABEL)
    .map(({ size }) => size)
    .sort((left, right) => right - left);

  return {
    sampleCount: labels.length,
    clusterCount: clusterSizesDescending.length,
    noiseCount,
    clusterSizesDescending,
    histogram,
  };
}

/**
 * sklearn-compatible adjusted Rand index. The -1 population is intentionally
 * included as a category; compareLabelPartitions separately checks its mask.
 */
export function adjustedRandIndex(
  reference: ArrayLike<number>,
  candidate: ArrayLike<number>,
): number {
  requireEqualLength("label", reference, candidate);
  validateLabels("reference", reference);
  validateLabels("candidate", candidate);

  const referenceCounts = new Map<number, number>();
  const candidateCounts = new Map<number, number>();
  const intersections = new Map<number, Map<number, number>>();
  for (let index = 0; index < reference.length; index += 1) {
    const referenceLabel = reference[index]!;
    const candidateLabel = candidate[index]!;
    referenceCounts.set(
      referenceLabel,
      (referenceCounts.get(referenceLabel) ?? 0) + 1,
    );
    candidateCounts.set(
      candidateLabel,
      (candidateCounts.get(candidateLabel) ?? 0) + 1,
    );
    let row = intersections.get(referenceLabel);
    if (row === undefined) {
      row = new Map<number, number>();
      intersections.set(referenceLabel, row);
    }
    row.set(candidateLabel, (row.get(candidateLabel) ?? 0) + 1);
  }

  let intersectionPairs = 0;
  for (const row of intersections.values()) {
    for (const count of row.values()) {
      intersectionPairs += pairCount(count);
    }
  }
  let referencePairs = 0;
  for (const count of referenceCounts.values()) {
    referencePairs += pairCount(count);
  }
  let candidatePairs = 0;
  for (const count of candidateCounts.values()) {
    candidatePairs += pairCount(count);
  }

  const totalPairs = pairCount(reference.length);
  if (totalPairs === 0) {
    return 1;
  }
  const expected = (referencePairs * candidatePairs) / totalPairs;
  const maximum = (referencePairs + candidatePairs) / 2;
  return maximum === expected
    ? 1
    : (intersectionPairs - expected) / (maximum - expected);
}

/**
 * Compare intermediate Float64 products such as core distances or flattened
 * single-linkage rows. Integer-valued linkage columns can be compared with a
 * zero tolerance while the distance column uses a numerical tolerance.
 */
export function compareNumericArrays(
  reference: ArrayLike<number>,
  candidate: ArrayLike<number>,
  tolerance: NumericTolerance,
): NumericParityDiagnostics {
  requireEqualLength("numeric", reference, candidate);
  requireFiniteNonNegative("absolute tolerance", tolerance.absolute);
  requireFiniteNonNegative("relative tolerance", tolerance.relative);
  const mismatchSampleLimit =
    tolerance.mismatchSampleLimit ?? DEFAULT_MISMATCH_SAMPLE_LIMIT;
  requireNonNegativeInteger("mismatchSampleLimit", mismatchSampleLimit);

  let mismatchCount = 0;
  let nonFiniteMismatchCount = 0;
  let maxAbsoluteError = 0;
  let maxRelativeError = 0;
  const firstMismatches: NumericMismatch[] = [];

  for (let index = 0; index < reference.length; index += 1) {
    const expected = reference[index]!;
    const actual = candidate[index]!;
    if (Object.is(expected, actual)) {
      continue;
    }
    if (!Number.isFinite(expected) || !Number.isFinite(actual)) {
      mismatchCount += 1;
      nonFiniteMismatchCount += 1;
      if (firstMismatches.length < mismatchSampleLimit) {
        firstMismatches.push({
          index,
          reference: expected,
          candidate: actual,
          absoluteError: Number.POSITIVE_INFINITY,
          relativeError: Number.POSITIVE_INFINITY,
        });
      }
      maxAbsoluteError = Number.POSITIVE_INFINITY;
      maxRelativeError = Number.POSITIVE_INFINITY;
      continue;
    }

    const absoluteError = Math.abs(actual - expected);
    const scale = Math.max(Math.abs(expected), Math.abs(actual));
    const relativeError = scale === 0 ? 0 : absoluteError / scale;
    maxAbsoluteError = Math.max(maxAbsoluteError, absoluteError);
    maxRelativeError = Math.max(maxRelativeError, relativeError);
    if (
      absoluteError >
      tolerance.absolute + tolerance.relative * Math.abs(expected)
    ) {
      mismatchCount += 1;
      if (firstMismatches.length < mismatchSampleLimit) {
        firstMismatches.push({
          index,
          reference: expected,
          candidate: actual,
          absoluteError,
          relativeError,
        });
      }
    }
  }

  return {
    length: reference.length,
    mismatchCount,
    nonFiniteMismatchCount,
    maxAbsoluteError,
    maxRelativeError,
    firstMismatches,
  };
}

/**
 * Compare an MST by undirected endpoints, independent of edge ordering and
 * endpoint orientation. This intentionally reports endpoint-set differences
 * separately from numerical weight differences.
 */
export function compareMstEdges(
  reference: readonly UndirectedWeightedEdge[],
  candidate: readonly UndirectedWeightedEdge[],
  tolerance: NumericTolerance,
): MstParityDiagnostics {
  const referenceByEndpoints = indexEdges(reference, "reference");
  const candidateByEndpoints = indexEdges(candidate, "candidate");
  const missingEndpointPairs = [...referenceByEndpoints.keys()]
    .filter((key) => !candidateByEndpoints.has(key))
    .sort();
  const unexpectedEndpointPairs = [...candidateByEndpoints.keys()]
    .filter((key) => !referenceByEndpoints.has(key))
    .sort();
  const sharedEndpointPairs = [...referenceByEndpoints.keys()]
    .filter((key) => candidateByEndpoints.has(key))
    .sort();
  const referenceWeights = new Float64Array(sharedEndpointPairs.length);
  const candidateWeights = new Float64Array(sharedEndpointPairs.length);
  for (let index = 0; index < sharedEndpointPairs.length; index += 1) {
    const key = sharedEndpointPairs[index]!;
    referenceWeights[index] = referenceByEndpoints.get(key)!;
    candidateWeights[index] = candidateByEndpoints.get(key)!;
  }

  return {
    exactEndpoints:
      missingEndpointPairs.length === 0 &&
      unexpectedEndpointPairs.length === 0,
    missingEndpointPairs,
    unexpectedEndpointPairs,
    weights: compareNumericArrays(
      referenceWeights,
      candidateWeights,
      tolerance,
    ),
  };
}

function canonicalizeLabels(labels: ArrayLike<number>): Int32Array {
  const canonical = new Int32Array(labels.length);
  const mapping = new Map<number, number>();
  let nextLabel = 0;
  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index]!;
    if (label === NOISE_LABEL) {
      canonical[index] = NOISE_LABEL;
      continue;
    }
    let replacement = mapping.get(label);
    if (replacement === undefined) {
      replacement = nextLabel;
      nextLabel += 1;
      mapping.set(label, replacement);
    }
    canonical[index] = replacement;
  }
  return canonical;
}

function pairCount(count: number): number {
  return (count * (count - 1)) / 2;
}

function indexEdges(
  edges: readonly UndirectedWeightedEdge[],
  name: string,
): Map<string, number> {
  const result = new Map<string, number>();
  for (const edge of edges) {
    if (
      !Number.isSafeInteger(edge.from) ||
      edge.from < 0 ||
      !Number.isSafeInteger(edge.to) ||
      edge.to < 0 ||
      edge.from === edge.to
    ) {
      throw new RangeError(`${name} MST contains an invalid endpoint`);
    }
    if (!Number.isFinite(edge.weight) || edge.weight < 0) {
      throw new RangeError(`${name} MST contains an invalid weight`);
    }
    const lower = Math.min(edge.from, edge.to);
    const upper = Math.max(edge.from, edge.to);
    const key = `${lower}:${upper}`;
    if (result.has(key)) {
      throw new RangeError(`${name} MST contains duplicate edge ${key}`);
    }
    result.set(key, edge.weight);
  }
  return result;
}

function validateLabels(name: string, labels: ArrayLike<number>): void {
  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index]!;
    if (!Number.isSafeInteger(label) || label < NOISE_LABEL) {
      throw new RangeError(
        `${name} label ${index} must be an integer greater than or equal to -1`,
      );
    }
  }
}

function requireEqualLength(
  kind: string,
  left: ArrayLike<number>,
  right: ArrayLike<number>,
): void {
  if (left.length !== right.length) {
    throw new RangeError(
      `${kind} arrays must have equal lengths (${left.length} !== ${right.length})`,
    );
  }
}

function requireFiniteNonNegative(name: string, value: number): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${name} must be finite and non-negative`);
  }
}

function requireNonNegativeInteger(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative integer`);
  }
}
