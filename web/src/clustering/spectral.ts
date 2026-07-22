import {
  mergeSimilarCentroids,
  normalizeLabels,
  reassignMinorClusters,
} from "./postprocess";

const DEFAULT_P_VALUE = 0.012;
const DEFAULT_MIN_PRUNED_NEIGHBORS = 6;
const DEFAULT_MIN_SPEAKERS = 1;
const DEFAULT_MAX_SPEAKERS = 15;
const DEFAULT_CLUSTER_LINE = 10;
const DEFAULT_MIN_CLUSTER_SIZE = 4;
const DEFAULT_MERGE_THRESHOLD = 0.875;
const DEFAULT_KRYLOV_BASIS_SIZE = 160;

export interface SpectralClusteringOptions {
  readonly pValue?: number;
  readonly minPrunedNeighbors?: number;
  readonly minSpeakers?: number;
  readonly maxSpeakers?: number;
  readonly oracleSpeakerCount?: number | null;
  readonly clusterLine?: number;
  readonly minClusterSize?: number;
  readonly mergeThreshold?: number | null;
  /** Receives bounded-allocation and convergence diagnostics. */
  readonly onStats?: SpectralClusteringStatsListener;
  /** Test/diagnostic override. Production should use the default. */
  readonly krylovBasisSize?: number;
}

export interface SpectralClusteringStats {
  readonly count: number;
  readonly dimension: number;
  readonly retainedPerRow: number;
  readonly undirectedEdgeCount: number;
  readonly speakerCountBeforePostprocess: number;
  readonly basisSize: number;
  readonly affinityMs: number;
  readonly eigensolverMs: number;
  readonly kmeansMs: number;
  readonly postprocessMs: number;
  readonly totalMs: number;
  readonly maximumEigenResidual: number;
  /** Smallest eigenvalues used by Senko's eigengap decision. */
  readonly eigenvalues: readonly number[];
  /** Dense float32 affinity + Laplacian bytes avoided by the sparse graph. */
  readonly avoidedDenseMatrixBytes: number;
  /** Peak live typed-array bytes, excluding caller-owned embeddings and labels. */
  readonly peakWorkingBytes: number;
}

export type SpectralClusteringStatsListener = (
  stats: SpectralClusteringStats,
) => void;

interface ResolvedSpectralOptions {
  readonly pValue: number;
  readonly minPrunedNeighbors: number;
  readonly minSpeakers: number;
  readonly maxSpeakers: number;
  readonly oracleSpeakerCount: number | null;
  readonly clusterLine: number;
  readonly minClusterSize: number;
  readonly mergeThreshold: number | null;
  readonly onStats: SpectralClusteringStatsListener | null;
  readonly krylovBasisSize: number;
}

interface SparseLaplacian {
  readonly rowOffsets: Int32Array;
  readonly columns: Int32Array;
  readonly weights: Float32Array;
  readonly degrees: Float32Array;
  readonly retainedPerRow: number;
  readonly undirectedEdgeCount: number;
  readonly buildPeakBytes: number;
}

interface SpectralEmbeddingResult {
  readonly values: Float64Array;
  readonly dimension: number;
  readonly eigenvalues: Float64Array;
  readonly maximumResidual: number;
  readonly basisSize: number;
  readonly graph: SparseLaplacian;
  readonly eigensolverPeakBytes: number;
}

interface SymmetricEigenResult {
  readonly values: Float64Array;
  /** Row-major matrix whose columns are eigenvectors. */
  readonly vectors: Float64Array;
}

/**
 * Senko's short-recording clustering branch.
 *
 * This follows `senko/cluster/cluster_cpu.py`: cosine affinity, p-pruning,
 * symmetrization, an unnormalized Laplacian, the 1..15 eigengap heuristic,
 * scikit-learn-compatible KMeans(random_state=0), minor-cluster reassignment,
 * and repeated cosine-centroid merging. The affinity and Laplacian are never
 * retained densely.
 */
export function clusterEmbeddingsSpectral(
  embeddings: Float32Array,
  count: number,
  dim: number,
  options: SpectralClusteringOptions = {},
): Int32Array {
  validateInput(embeddings, count, dim);
  const resolved = resolveSpectralOptions(options);
  const totalStart = monotonicNow();
  if (count === 0) {
    return new Int32Array();
  }
  if (count < resolved.clusterLine) {
    const labels = new Int32Array(count);
    resolved.onStats?.({
      count,
      dimension: dim,
      retainedPerRow: 0,
      undirectedEdgeCount: 0,
      speakerCountBeforePostprocess: 1,
      basisSize: 0,
      affinityMs: 0,
      eigensolverMs: 0,
      kmeansMs: 0,
      postprocessMs: 0,
      totalMs: monotonicNow() - totalStart,
      maximumEigenResidual: 0,
      eigenvalues: [],
      avoidedDenseMatrixBytes: 0,
      peakWorkingBytes: 0,
    });
    return labels;
  }

  const affinityStart = monotonicNow();
  const graph = buildPrunedCosineLaplacian(
    embeddings,
    count,
    dim,
    resolved.pValue,
    resolved.minPrunedNeighbors,
  );
  const affinityMs = monotonicNow() - affinityStart;

  const eigensolverStart = monotonicNow();
  const spectral = buildSpectralEmbedding(
    graph,
    count,
    resolved.minSpeakers,
    resolved.maxSpeakers,
    resolved.oracleSpeakerCount,
    resolved.krylovBasisSize,
  );
  const eigensolverMs = monotonicNow() - eigensolverStart;

  const kmeansStart = monotonicNow();
  const labels = sklearnKMeans(spectral.values, count, spectral.dimension);
  const kmeansMs = monotonicNow() - kmeansStart;

  const postprocessStart = monotonicNow();
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
  const postprocessMs = monotonicNow() - postprocessStart;

  const graphBytes = sparseGraphBytes(graph);
  const localTrials = 2 + Math.trunc(Math.log(spectral.dimension));
  const kmeansTemporaryBytes =
    spectral.values.byteLength +
    count * (1 + localTrials) * Float64Array.BYTES_PER_ELEMENT +
    spectral.dimension * spectral.dimension * 2 * Float64Array.BYTES_PER_ELEMENT +
    count * 2 * Int32Array.BYTES_PER_ELEMENT +
    spectral.dimension * 3 * Float64Array.BYTES_PER_ELEMENT +
    localTrials * Int32Array.BYTES_PER_ELEMENT;
  resolved.onStats?.({
    count,
    dimension: dim,
    retainedPerRow: graph.retainedPerRow,
    undirectedEdgeCount: graph.undirectedEdgeCount,
    speakerCountBeforePostprocess: spectral.dimension,
    basisSize: spectral.basisSize,
    affinityMs,
    eigensolverMs,
    kmeansMs,
    postprocessMs,
    totalMs: monotonicNow() - totalStart,
    maximumEigenResidual: spectral.maximumResidual,
    eigenvalues: Array.from(spectral.eigenvalues),
    avoidedDenseMatrixBytes:
      2 * count * count * Float32Array.BYTES_PER_ELEMENT,
    peakWorkingBytes: Math.max(
      graph.buildPeakBytes,
      graphBytes + spectral.eigensolverPeakBytes,
      spectral.values.byteLength + kmeansTemporaryBytes,
    ),
  });
  return labels;
}

/** Build Senko's p-pruned symmetric unnormalized Laplacian in CSR form. */
export function buildPrunedCosineLaplacian(
  embeddings: Float32Array,
  count: number,
  dim: number,
  pValue = DEFAULT_P_VALUE,
  minPrunedNeighbors = DEFAULT_MIN_PRUNED_NEIGHBORS,
): SparseLaplacian {
  validateInput(embeddings, count, dim);
  if (!(pValue >= 0 && pValue <= 1)) {
    throw new RangeError("pValue must be in [0, 1]");
  }
  if (count === 0) {
    return {
      rowOffsets: new Int32Array(1),
      columns: new Int32Array(),
      weights: new Float32Array(),
      degrees: new Float32Array(),
      retainedPerRow: 0,
      undirectedEdgeCount: 0,
      buildPeakBytes: 0,
    };
  }
  requireInteger("minPrunedNeighbors", minPrunedNeighbors, 1, count);

  // NumPy computes `n_elems = min(int((1-p)N), N-min_pnum)` and zeros
  // exactly that many smallest entries. Expressing the complement preserves
  // its truncation behavior at floating-point boundaries.
  const prunedPerRow = Math.min(
    Math.trunc((1 - pValue) * count),
    count - minPrunedNeighbors,
  );
  const retainedPerRow = count - prunedPerRow;
  const normalized = normalizeRowsFloat32(embeddings, count, dim);
  const heapIndices = new Int32Array(count * retainedPerRow);
  heapIndices.fill(-1);
  const heapValues = new Float32Array(count * retainedPerRow);
  heapValues.fill(Number.NEGATIVE_INFINITY);

  for (let left = 0; left < count; left += 1) {
    const leftOffset = left * dim;
    for (let right = left; right < count; right += 1) {
      const rightOffset = right * dim;
      let similarity = 0;
      for (let column = 0; column < dim; column += 1) {
        similarity +=
          normalized[leftOffset + column]! * normalized[rightOffset + column]!;
      }
      const roundedSimilarity = Math.fround(similarity);
      heapInsert(
        heapIndices,
        heapValues,
        left * retainedPerRow,
        retainedPerRow,
        right,
        roundedSimilarity,
      );
      if (right !== left) {
        heapInsert(
          heapIndices,
          heapValues,
          right * retainedPerRow,
          retainedPerRow,
          left,
          roundedSimilarity,
        );
      }
    }
  }

  const rowCounts = new Int32Array(count);
  let undirectedEdgeCount = 0;
  for (let row = 0; row < count; row += 1) {
    const base = row * retainedPerRow;
    for (let slot = 0; slot < retainedPerRow; slot += 1) {
      const column = heapIndices[base + slot]!;
      const weight = heapValues[base + slot]!;
      if (
        column < 0 ||
        column === row ||
        weight === 0 ||
        (column < row &&
          directedHeapSlot(
            heapIndices,
            column * retainedPerRow,
            retainedPerRow,
            row,
          ) >= 0)
      ) {
        continue;
      }
      rowCounts[row] = rowCounts[row]! + 1;
      rowCounts[column] = rowCounts[column]! + 1;
      undirectedEdgeCount += 1;
    }
  }

  const rowOffsets = new Int32Array(count + 1);
  for (let row = 0; row < count; row += 1) {
    rowOffsets[row + 1] = rowOffsets[row]! + rowCounts[row]!;
  }
  const columns = new Int32Array(undirectedEdgeCount * 2);
  const weights = new Float32Array(undirectedEdgeCount * 2);
  const degrees = new Float32Array(count);
  const cursors = rowOffsets.slice(0, count);

  for (let row = 0; row < count; row += 1) {
    const base = row * retainedPerRow;
    for (let slot = 0; slot < retainedPerRow; slot += 1) {
      const column = heapIndices[base + slot]!;
      const directedWeight = heapValues[base + slot]!;
      if (column < 0 || column === row || directedWeight === 0) {
        continue;
      }
      const reverseSlot = directedHeapSlot(
        heapIndices,
        column * retainedPerRow,
        retainedPerRow,
        row,
      );
      if (column < row && reverseSlot >= 0) {
        continue;
      }
      const weight = Math.fround(
        0.5 *
          (directedWeight +
            (reverseSlot >= 0 ? heapValues[reverseSlot]! : 0)),
      );
      const forward = cursors[row]!;
      const reverse = cursors[column]!;
      columns[forward] = column;
      weights[forward] = weight;
      columns[reverse] = row;
      weights[reverse] = weight;
      cursors[row] = forward + 1;
      cursors[column] = reverse + 1;
      degrees[row] = Math.fround(degrees[row]! + Math.abs(weight));
      degrees[column] = Math.fround(degrees[column]! + Math.abs(weight));
    }
  }

  const graphBytes =
    rowOffsets.byteLength +
    columns.byteLength +
    weights.byteLength +
    degrees.byteLength;
  return {
    rowOffsets,
    columns,
    weights,
    degrees,
    retainedPerRow,
    undirectedEdgeCount,
    buildPeakBytes:
      normalized.byteLength +
      heapIndices.byteLength +
      heapValues.byteLength +
      rowCounts.byteLength +
      cursors.byteLength +
      graphBytes,
  };
}

function buildSpectralEmbedding(
  graph: SparseLaplacian,
  count: number,
  minSpeakers: number,
  maxSpeakers: number,
  oracleSpeakerCount: number | null,
  requestedBasisSize: number,
): SpectralEmbeddingResult {
  const maximumInferredSpeakers = Math.min(maxSpeakers, count - 1);
  const requiredEigenpairs = Math.min(
    count,
    (oracleSpeakerCount ?? maximumInferredSpeakers) + 1,
  );
  const eigensystem = smallestSparseEigenpairs(
    graph,
    count,
    requiredEigenpairs,
    requestedBasisSize,
  );
  let speakerCount = oracleSpeakerCount;
  if (speakerCount === null) {
    let largestGap = Number.NEGATIVE_INFINITY;
    speakerCount = minSpeakers;
    for (
      let speaker = minSpeakers;
      speaker <= maximumInferredSpeakers;
      speaker += 1
    ) {
      const gap =
        eigensystem.eigenvalues[speaker]! -
        eigensystem.eigenvalues[speaker - 1]!;
      if (gap > largestGap) {
        largestGap = gap;
        speakerCount = speaker;
      }
    }
  }
  const values = new Float64Array(count * speakerCount);
  for (let row = 0; row < count; row += 1) {
    for (let column = 0; column < speakerCount; column += 1) {
      values[row * speakerCount + column] =
        eigensystem.eigenvectors[row * requiredEigenpairs + column]!;
    }
  }
  return {
    values,
    dimension: speakerCount,
    eigenvalues: eigensystem.eigenvalues,
    maximumResidual: eigensystem.maximumResidual,
    basisSize: eigensystem.basisSize,
    graph,
    eigensolverPeakBytes: eigensystem.peakWorkingBytes + values.byteLength,
  };
}

interface SparseEigenResult {
  readonly eigenvalues: Float64Array;
  /** Row-major, with one eigenvector per column. */
  readonly eigenvectors: Float64Array;
  readonly maximumResidual: number;
  readonly basisSize: number;
  readonly peakWorkingBytes: number;
}

/**
 * Block Krylov/Rayleigh-Ritz eigensolver for the smallest Laplacian modes.
 * A 16-wide starting block preserves repeated zero-eigenvalue subspaces that
 * scalar Lanczos cannot represent. Full reorthogonalization keeps the compact
 * projected problem stable.
 */
function smallestSparseEigenpairs(
  graph: SparseLaplacian,
  count: number,
  eigenpairCount: number,
  requestedBasisSize: number,
): SparseEigenResult {
  const basisSize = Math.min(
    count,
    Math.max(eigenpairCount + 1, requestedBasisSize),
  );
  const blockWidth = Math.min(eigenpairCount, basisSize);
  const basis = new Float64Array(count * basisSize);
  const candidate = new Float64Array(count);
  const product = new Float64Array(count);
  let randomState = 0x9e3779b9;

  let completed = 0;
  for (let column = 0; column < blockWidth; column += 1) {
    if (column === 0) {
      candidate.fill(1);
    } else {
      for (let row = 0; row < count; row += 1) {
        randomState = xorshift32(randomState);
        candidate[row] = randomState / 0x1_0000_0000 - 0.5;
      }
    }
    if (!appendOrthonormalColumn(basis, count, completed, candidate)) {
      throw new Error("Unable to initialize spectral Krylov basis");
    }
    completed += 1;
  }

  let sourceStart = 0;
  while (completed < basisSize) {
    const sourceEnd = completed;
    let added = 0;
    for (
      let source = sourceStart;
      source < sourceEnd && completed < basisSize;
      source += 1
    ) {
      applySparseLaplacianToBasisColumn(
        graph,
        basis,
        count,
        source,
        candidate,
      );
      if (!appendOrthonormalColumn(basis, count, completed, candidate)) {
        for (let row = 0; row < count; row += 1) {
          randomState = xorshift32(randomState);
          candidate[row] = randomState / 0x1_0000_0000 - 0.5;
        }
        if (!appendOrthonormalColumn(basis, count, completed, candidate)) {
          continue;
        }
      }
      completed += 1;
      added += 1;
    }
    if (added === 0) {
      break;
    }
    sourceStart = sourceEnd;
  }

  const actualBasisSize = completed;
  const projected = new Float64Array(actualBasisSize * actualBasisSize);
  for (let right = 0; right < actualBasisSize; right += 1) {
    applySparseLaplacianToBasisColumn(
      graph,
      basis,
      count,
      right,
      product,
    );
    for (let left = 0; left <= right; left += 1) {
      let dot = 0;
      const leftOffset = left * count;
      for (let row = 0; row < count; row += 1) {
        dot += basis[leftOffset + row]! * product[row]!;
      }
      projected[left * actualBasisSize + right] = dot;
      projected[right * actualBasisSize + left] = dot;
    }
  }

  const smallEigen = jacobiSymmetricEigen(projected, actualBasisSize);
  const order = Array.from(
    { length: actualBasisSize },
    (_, index) => index,
  ).sort((left, right) => {
    const difference = smallEigen.values[left]! - smallEigen.values[right]!;
    return difference === 0 ? left - right : difference;
  });
  const eigenvalues = new Float64Array(eigenpairCount);
  const eigenvectors = new Float64Array(count * eigenpairCount);
  for (let outputColumn = 0; outputColumn < eigenpairCount; outputColumn += 1) {
    const projectedColumn = order[outputColumn]!;
    eigenvalues[outputColumn] = smallEigen.values[projectedColumn]!;
    for (let basisColumn = 0; basisColumn < actualBasisSize; basisColumn += 1) {
      const scale =
        smallEigen.vectors[basisColumn * actualBasisSize + projectedColumn]!;
      if (scale === 0) {
        continue;
      }
      const basisOffset = basisColumn * count;
      for (let row = 0; row < count; row += 1) {
        const target = row * eigenpairCount + outputColumn;
        eigenvectors[target] =
          eigenvectors[target]! + basis[basisOffset + row]! * scale;
      }
    }
  }

  let maximumResidual = 0;
  for (let column = 0; column < eigenpairCount; column += 1) {
    applySparseLaplacianToRowMajorColumn(
      graph,
      eigenvectors,
      count,
      eigenpairCount,
      column,
      product,
    );
    let squaredResidual = 0;
    for (let row = 0; row < count; row += 1) {
      const residual =
        product[row]! -
        eigenvalues[column]! *
          eigenvectors[row * eigenpairCount + column]!;
      squaredResidual += residual * residual;
    }
    maximumResidual = Math.max(maximumResidual, Math.sqrt(squaredResidual));
  }

  return {
    eigenvalues,
    eigenvectors,
    maximumResidual,
    basisSize: actualBasisSize,
    peakWorkingBytes:
      basis.byteLength +
      candidate.byteLength +
      product.byteLength +
      projected.byteLength +
      smallEigen.values.byteLength +
      smallEigen.vectors.byteLength +
      eigenvalues.byteLength +
      eigenvectors.byteLength,
  };
}

function appendOrthonormalColumn(
  basis: Float64Array,
  count: number,
  completed: number,
  candidate: Float64Array,
): boolean {
  for (let pass = 0; pass < 2; pass += 1) {
    for (let column = 0; column < completed; column += 1) {
      const offset = column * count;
      let dot = 0;
      for (let row = 0; row < count; row += 1) {
        dot += basis[offset + row]! * candidate[row]!;
      }
      for (let row = 0; row < count; row += 1) {
        candidate[row] = candidate[row]! - dot * basis[offset + row]!;
      }
    }
  }
  let squaredNorm = 0;
  for (let row = 0; row < count; row += 1) {
    squaredNorm += candidate[row]! * candidate[row]!;
  }
  if (!(squaredNorm > 1e-24)) {
    return false;
  }
  const scale = 1 / Math.sqrt(squaredNorm);
  const targetOffset = completed * count;
  for (let row = 0; row < count; row += 1) {
    basis[targetOffset + row] = candidate[row]! * scale;
  }
  return true;
}

function applySparseLaplacianToBasisColumn(
  graph: SparseLaplacian,
  basis: Float64Array,
  count: number,
  column: number,
  output: Float64Array,
): void {
  const offset = column * count;
  for (let row = 0; row < count; row += 1) {
    let value = graph.degrees[row]! * basis[offset + row]!;
    for (
      let edge = graph.rowOffsets[row]!;
      edge < graph.rowOffsets[row + 1]!;
      edge += 1
    ) {
      value -=
        graph.weights[edge]! * basis[offset + graph.columns[edge]!]!;
    }
    output[row] = value;
  }
}

function applySparseLaplacianToRowMajorColumn(
  graph: SparseLaplacian,
  values: Float64Array,
  count: number,
  dim: number,
  column: number,
  output: Float64Array,
): void {
  for (let row = 0; row < count; row += 1) {
    let value = graph.degrees[row]! * values[row * dim + column]!;
    for (
      let edge = graph.rowOffsets[row]!;
      edge < graph.rowOffsets[row + 1]!;
      edge += 1
    ) {
      value -=
        graph.weights[edge]! *
        values[graph.columns[edge]! * dim + column]!;
    }
    output[row] = value;
  }
}

function jacobiSymmetricEigen(
  input: Float64Array,
  size: number,
): SymmetricEigenResult {
  const matrix = input.slice();
  const vectors = new Float64Array(size * size);
  for (let index = 0; index < size; index += 1) {
    vectors[index * size + index] = 1;
  }
  const maximumSweeps = 30;
  for (let sweep = 0; sweep < maximumSweeps; sweep += 1) {
    let largest = 0;
    let diagonalScale = 0;
    for (let index = 0; index < size; index += 1) {
      diagonalScale = Math.max(
        diagonalScale,
        Math.abs(matrix[index * size + index]!),
      );
    }
    for (let left = 0; left < size - 1; left += 1) {
      for (let right = left + 1; right < size; right += 1) {
        const lr = left * size + right;
        const offDiagonal = matrix[lr]!;
        largest = Math.max(largest, Math.abs(offDiagonal));
        if (
          Math.abs(offDiagonal) <=
          Number.EPSILON *
            8 *
            (Math.abs(matrix[left * size + left]!) +
              Math.abs(matrix[right * size + right]!) +
              1)
        ) {
          continue;
        }
        const leftDiagonal = matrix[left * size + left]!;
        const rightDiagonal = matrix[right * size + right]!;
        const tau = (rightDiagonal - leftDiagonal) / (2 * offDiagonal);
        const tangent =
          (tau >= 0 ? 1 : -1) /
          (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const cosine = 1 / Math.sqrt(1 + tangent * tangent);
        const sine = tangent * cosine;

        matrix[left * size + left] = leftDiagonal - tangent * offDiagonal;
        matrix[right * size + right] = rightDiagonal + tangent * offDiagonal;
        matrix[lr] = 0;
        matrix[right * size + left] = 0;
        for (let index = 0; index < size; index += 1) {
          if (index === left || index === right) {
            continue;
          }
          const indexLeft = matrix[index * size + left]!;
          const indexRight = matrix[index * size + right]!;
          const rotatedLeft = cosine * indexLeft - sine * indexRight;
          const rotatedRight = sine * indexLeft + cosine * indexRight;
          matrix[index * size + left] = rotatedLeft;
          matrix[left * size + index] = rotatedLeft;
          matrix[index * size + right] = rotatedRight;
          matrix[right * size + index] = rotatedRight;
        }
        for (let row = 0; row < size; row += 1) {
          const vectorLeft = vectors[row * size + left]!;
          const vectorRight = vectors[row * size + right]!;
          vectors[row * size + left] =
            cosine * vectorLeft - sine * vectorRight;
          vectors[row * size + right] =
            sine * vectorLeft + cosine * vectorRight;
        }
      }
    }
    if (largest <= Math.max(1, diagonalScale) * 1e-12) {
      break;
    }
  }
  const values = new Float64Array(size);
  for (let index = 0; index < size; index += 1) {
    values[index] = matrix[index * size + index]!;
  }
  return { values, vectors };
}

/** One scikit-learn 1.9 KMeans run: k-means++, MT19937 seed 0, Lloyd. */
function sklearnKMeans(
  input: Float64Array,
  count: number,
  dim: number,
): Int32Array {
  if (dim === 1) {
    return new Int32Array(count);
  }
  const values = input.slice();
  const means = new Float64Array(dim);
  for (let row = 0; row < count; row += 1) {
    for (let column = 0; column < dim; column += 1) {
      means[column] = means[column]! + values[row * dim + column]!;
    }
  }
  for (let column = 0; column < dim; column += 1) {
    means[column] = means[column]! / count;
  }
  for (let row = 0; row < count; row += 1) {
    for (let column = 0; column < dim; column += 1) {
      const index = row * dim + column;
      values[index] = values[index]! - means[column]!;
    }
  }

  let varianceSum = 0;
  for (let column = 0; column < dim; column += 1) {
    let variance = 0;
    for (let row = 0; row < count; row += 1) {
      const value = values[row * dim + column]!;
      variance += value * value;
    }
    varianceSum += variance / count;
  }
  const tolerance = (varianceSum / dim) * 0.0001;
  let centers: Float64Array = initializeKMeansPlusPlus(values, count, dim);
  let nextCenters: Float64Array = new Float64Array(dim * dim);
  const labels = new Int32Array(count);
  labels.fill(-1);
  const oldLabels = new Int32Array(count);
  oldLabels.fill(-1);
  const sizes = new Float64Array(dim);
  const centerShift = new Float64Array(dim);
  let strictConvergence = false;

  for (let iteration = 0; iteration < 300; iteration += 1) {
    nextCenters.fill(0);
    sizes.fill(0);
    assignKMeansLabels(values, centers, labels, count, dim);
    for (let row = 0; row < count; row += 1) {
      const label = labels[row]!;
      sizes[label] = sizes[label]! + 1;
      for (let column = 0; column < dim; column += 1) {
        const index = label * dim + column;
        nextCenters[index] =
          nextCenters[index]! + values[row * dim + column]!;
      }
    }
    relocateEmptyClusters(
      values,
      centers,
      nextCenters,
      labels,
      sizes,
      count,
      dim,
    );
    averageCenters(nextCenters, sizes, dim);
    let shiftTotal = 0;
    for (let center = 0; center < dim; center += 1) {
      let squaredShift = 0;
      for (let column = 0; column < dim; column += 1) {
        const difference =
          centers[center * dim + column]! -
          nextCenters[center * dim + column]!;
        squaredShift += difference * difference;
      }
      centerShift[center] = Math.sqrt(squaredShift);
      shiftTotal += centerShift[center]! * centerShift[center]!;
    }
    const swapped = centers;
    centers = nextCenters;
    nextCenters = swapped;
    if (intArraysEqual(labels, oldLabels)) {
      strictConvergence = true;
      break;
    }
    if (shiftTotal <= tolerance) {
      break;
    }
    oldLabels.set(labels);
  }
  if (!strictConvergence) {
    assignKMeansLabels(values, centers, labels, count, dim);
  }
  return labels;
}

function initializeKMeansPlusPlus(
  values: Float64Array,
  count: number,
  dim: number,
): Float64Array {
  const random = new Mt19937(0);
  const centers = new Float64Array(dim * dim);
  const closestSquared = new Float64Array(count);
  const first = Math.min(count - 1, Math.floor(random.double() * count));
  copyRow(values, first, centers, 0, dim);
  let currentPotential = 0;
  for (let row = 0; row < count; row += 1) {
    const distance = squaredDistance(values, row, centers, 0, dim);
    closestSquared[row] = distance;
    currentPotential += distance;
  }
  const localTrials = 2 + Math.trunc(Math.log(dim));
  const candidates = new Int32Array(localTrials);
  const candidateDistances = new Float64Array(count * localTrials);
  for (let center = 1; center < dim; center += 1) {
    for (let trial = 0; trial < localTrials; trial += 1) {
      const target = random.double() * currentPotential;
      let cumulative = 0;
      let candidate = count - 1;
      for (let row = 0; row < count; row += 1) {
        cumulative += closestSquared[row]!;
        if (cumulative >= target) {
          candidate = row;
          break;
        }
      }
      candidates[trial] = candidate;
    }
    let bestTrial = 0;
    let bestPotential = Number.POSITIVE_INFINITY;
    for (let trial = 0; trial < localTrials; trial += 1) {
      const candidate = candidates[trial]!;
      let potential = 0;
      for (let row = 0; row < count; row += 1) {
        const distance = squaredDistance(values, row, values, candidate, dim);
        const closest = Math.min(closestSquared[row]!, distance);
        candidateDistances[trial * count + row] = closest;
        potential += closest;
      }
      if (potential < bestPotential) {
        bestPotential = potential;
        bestTrial = trial;
      }
    }
    closestSquared.set(
      candidateDistances.subarray(bestTrial * count, (bestTrial + 1) * count),
    );
    currentPotential = bestPotential;
    copyRow(values, candidates[bestTrial]!, centers, center, dim);
  }
  return centers;
}

function assignKMeansLabels(
  values: Float64Array,
  centers: Float64Array,
  labels: Int32Array,
  count: number,
  dim: number,
): void {
  for (let row = 0; row < count; row += 1) {
    let best = 0;
    let bestDistance = squaredDistance(values, row, centers, 0, dim);
    for (let center = 1; center < dim; center += 1) {
      const distance = squaredDistance(values, row, centers, center, dim);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = center;
      }
    }
    labels[row] = best;
  }
}

function relocateEmptyClusters(
  values: Float64Array,
  centers: Float64Array,
  nextCenters: Float64Array,
  labels: Int32Array,
  sizes: Float64Array,
  count: number,
  dim: number,
): void {
  const empty: number[] = [];
  for (let center = 0; center < dim; center += 1) {
    if (sizes[center] === 0) {
      empty.push(center);
    }
  }
  if (empty.length === 0) {
    return;
  }
  const rows = Array.from({ length: count }, (_, row) => row);
  rows.sort((left, right) => {
    const leftDistance = squaredDistance(
      values,
      left,
      centers,
      labels[left]!,
      dim,
    );
    const rightDistance = squaredDistance(
      values,
      right,
      centers,
      labels[right]!,
      dim,
    );
    return rightDistance - leftDistance || right - left;
  });
  if (
    squaredDistance(values, rows[0]!, centers, labels[rows[0]!]!, dim) === 0
  ) {
    return;
  }
  for (let index = 0; index < empty.length; index += 1) {
    const row = rows[index]!;
    const oldCenter = labels[row]!;
    const newCenter = empty[index]!;
    for (let column = 0; column < dim; column += 1) {
      const value = values[row * dim + column]!;
      const oldIndex = oldCenter * dim + column;
      nextCenters[oldIndex] = nextCenters[oldIndex]! - value;
      nextCenters[newCenter * dim + column] = value;
    }
    sizes[newCenter] = 1;
    sizes[oldCenter] = sizes[oldCenter]! - 1;
  }
}

function averageCenters(
  centers: Float64Array,
  sizes: Float64Array,
  dim: number,
): void {
  let largest = 0;
  for (let center = 1; center < dim; center += 1) {
    if (sizes[center]! > sizes[largest]!) {
      largest = center;
    }
  }
  for (let center = 0; center < dim; center += 1) {
    if (sizes[center]! > 0) {
      const scale = 1 / sizes[center]!;
      for (let column = 0; column < dim; column += 1) {
        const index = center * dim + column;
        centers[index] = centers[index]! * scale;
      }
    } else {
      copyRow(centers, largest, centers, center, dim);
    }
  }
}

class Mt19937 {
  private readonly state = new Uint32Array(624);
  private index = 624;

  constructor(seed: number) {
    this.state[0] = seed >>> 0;
    for (let index = 1; index < 624; index += 1) {
      const previous = this.state[index - 1]!;
      this.state[index] =
        (Math.imul(1_812_433_253, previous ^ (previous >>> 30)) + index) >>> 0;
    }
  }

  double(): number {
    const high = this.uint32() >>> 5;
    const low = this.uint32() >>> 6;
    return (high * 67_108_864 + low) / 9_007_199_254_740_992;
  }

  private uint32(): number {
    if (this.index >= 624) {
      this.twist();
    }
    let value = this.state[this.index++]!;
    value ^= value >>> 11;
    value ^= (value << 7) & 0x9d2c5680;
    value ^= (value << 15) & 0xefc60000;
    value ^= value >>> 18;
    return value >>> 0;
  }

  private twist(): void {
    for (let index = 0; index < 624; index += 1) {
      const combined =
        (this.state[index]! & 0x80000000) |
        (this.state[(index + 1) % 624]! & 0x7fffffff);
      this.state[index] =
        this.state[(index + 397) % 624]! ^
        (combined >>> 1) ^
        (combined & 1 ? 0x9908b0df : 0);
    }
    this.index = 0;
  }
}

function heapInsert(
  indices: Int32Array,
  values: Float32Array,
  offset: number,
  size: number,
  candidateIndex: number,
  candidateValue: number,
): void {
  if (!isBetter(candidateValue, candidateIndex, values[offset]!, indices[offset]!)) {
    return;
  }
  indices[offset] = candidateIndex;
  values[offset] = candidateValue;
  let parent = 0;
  for (;;) {
    const left = parent * 2 + 1;
    if (left >= size) {
      return;
    }
    const right = left + 1;
    let worse = left;
    if (
      right < size &&
      isWorse(
        values[offset + right]!,
        indices[offset + right]!,
        values[offset + left]!,
        indices[offset + left]!,
      )
    ) {
      worse = right;
    }
    if (
      !isWorse(
        values[offset + worse]!,
        indices[offset + worse]!,
        values[offset + parent]!,
        indices[offset + parent]!,
      )
    ) {
      return;
    }
    swapHeapEntries(indices, values, offset + parent, offset + worse);
    parent = worse;
  }
}

function isBetter(
  candidateValue: number,
  candidateIndex: number,
  currentValue: number,
  currentIndex: number,
): boolean {
  return (
    candidateValue > currentValue ||
    (candidateValue === currentValue && candidateIndex < currentIndex)
  );
}

function isWorse(
  candidateValue: number,
  candidateIndex: number,
  currentValue: number,
  currentIndex: number,
): boolean {
  return (
    candidateValue < currentValue ||
    (candidateValue === currentValue && candidateIndex > currentIndex)
  );
}

function swapHeapEntries(
  indices: Int32Array,
  values: Float32Array,
  left: number,
  right: number,
): void {
  const index = indices[left]!;
  indices[left] = indices[right]!;
  indices[right] = index;
  const value = values[left]!;
  values[left] = values[right]!;
  values[right] = value;
}

function directedHeapSlot(
  indices: Int32Array,
  offset: number,
  size: number,
  target: number,
): number {
  for (let slot = 0; slot < size; slot += 1) {
    if (indices[offset + slot] === target) {
      return offset + slot;
    }
  }
  return -1;
}

function normalizeRowsFloat32(
  values: Float32Array,
  count: number,
  dim: number,
): Float32Array {
  const normalized = new Float32Array(values.length);
  for (let row = 0; row < count; row += 1) {
    const offset = row * dim;
    let squaredNorm = 0;
    for (let column = 0; column < dim; column += 1) {
      const value = values[offset + column]!;
      squaredNorm += value * value;
    }
    const scale = squaredNorm > 0 ? 1 / Math.sqrt(squaredNorm) : 1;
    for (let column = 0; column < dim; column += 1) {
      normalized[offset + column] = Math.fround(
        values[offset + column]! * scale,
      );
    }
  }
  return normalized;
}

function squaredDistance(
  left: Float64Array,
  leftRow: number,
  right: Float64Array,
  rightRow: number,
  dim: number,
): number {
  let distance = 0;
  for (let column = 0; column < dim; column += 1) {
    const difference =
      left[leftRow * dim + column]! - right[rightRow * dim + column]!;
    distance += difference * difference;
  }
  return distance;
}

function copyRow(
  source: Float64Array,
  sourceRow: number,
  target: Float64Array,
  targetRow: number,
  dim: number,
): void {
  target.set(
    source.subarray(sourceRow * dim, (sourceRow + 1) * dim),
    targetRow * dim,
  );
}

function intArraysEqual(left: Int32Array, right: Int32Array): boolean {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      return false;
    }
  }
  return true;
}

function resolveSpectralOptions(
  options: SpectralClusteringOptions,
): ResolvedSpectralOptions {
  const resolved: ResolvedSpectralOptions = {
    pValue: options.pValue ?? DEFAULT_P_VALUE,
    minPrunedNeighbors:
      options.minPrunedNeighbors ?? DEFAULT_MIN_PRUNED_NEIGHBORS,
    minSpeakers: options.minSpeakers ?? DEFAULT_MIN_SPEAKERS,
    maxSpeakers: options.maxSpeakers ?? DEFAULT_MAX_SPEAKERS,
    oracleSpeakerCount:
      options.oracleSpeakerCount === undefined
        ? null
        : options.oracleSpeakerCount,
    clusterLine: options.clusterLine ?? DEFAULT_CLUSTER_LINE,
    minClusterSize: options.minClusterSize ?? DEFAULT_MIN_CLUSTER_SIZE,
    mergeThreshold:
      options.mergeThreshold === undefined
        ? DEFAULT_MERGE_THRESHOLD
        : options.mergeThreshold,
    onStats: options.onStats ?? null,
    krylovBasisSize:
      options.krylovBasisSize ?? DEFAULT_KRYLOV_BASIS_SIZE,
  };
  if (!(resolved.pValue >= 0 && resolved.pValue <= 1)) {
    throw new RangeError("pValue must be in [0, 1]");
  }
  requireInteger("minPrunedNeighbors", resolved.minPrunedNeighbors, 1, 4096);
  requireInteger("minSpeakers", resolved.minSpeakers, 1, 64);
  requireInteger("maxSpeakers", resolved.maxSpeakers, 1, 64);
  if (resolved.maxSpeakers < resolved.minSpeakers) {
    throw new RangeError("maxSpeakers must be at least minSpeakers");
  }
  if (resolved.oracleSpeakerCount !== null) {
    requireInteger(
      "oracleSpeakerCount",
      resolved.oracleSpeakerCount,
      resolved.minSpeakers,
      resolved.maxSpeakers,
    );
  }
  requireInteger("clusterLine", resolved.clusterLine, 1, 1_000_000);
  requireInteger("minClusterSize", resolved.minClusterSize, 1, 1_000_000);
  requireInteger(
    "krylovBasisSize",
    resolved.krylovBasisSize,
    resolved.maxSpeakers + 1,
    512,
  );
  if (
    resolved.mergeThreshold !== null &&
    (!Number.isFinite(resolved.mergeThreshold) ||
      resolved.mergeThreshold <= 0 ||
      resolved.mergeThreshold > 1)
  ) {
    throw new RangeError("mergeThreshold must be null or in (0, 1]");
  }
  return resolved;
}

function validateInput(
  embeddings: Float32Array,
  count: number,
  dim: number,
): void {
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
  for (let index = 0; index < embeddings.length; index += 1) {
    if (!Number.isFinite(embeddings[index]!)) {
      throw new RangeError(
        `embeddings contains a non-finite value at index ${index}`,
      );
    }
  }
}

function requireInteger(
  name: string,
  value: number,
  minimum: number,
  maximum: number,
): void {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be an integer in [${minimum}, ${maximum}]`);
  }
}

function sparseGraphBytes(graph: SparseLaplacian): number {
  return (
    graph.rowOffsets.byteLength +
    graph.columns.byteLength +
    graph.weights.byteLength +
    graph.degrees.byteLength
  );
}

function xorshift32(value: number): number {
  let result = value | 0;
  result ^= result << 13;
  result ^= result >>> 17;
  result ^= result << 5;
  return result >>> 0;
}

function monotonicNow(): number {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}
