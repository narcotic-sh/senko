export interface ClusteringOptions {
  /** Run UMAP before density clustering. Disable only for diagnostics. */
  readonly useUmap?: boolean;
  /** UMAP neighborhood size. */
  readonly umapNeighborCount?: number;
  /** UMAP output dimensions. */
  readonly umapComponents?: number;
  /** UMAP stochastic-gradient epochs. */
  readonly umapEpochs?: number;
  /** UMAP minimum embedding distance. */
  readonly umapMinDistance?: number;
  /** Seed for UMAP's deterministic pseudo-random generator. */
  readonly umapRandomSeed?: number;
  /** Receives deterministic timing and allocation statistics for UMAP. */
  readonly onUmapStats?: UmapStatsListener;
  /** Number of approximate cosine neighbors. Mirrors Senko's UMAP setting. */
  readonly neighborCount?: number;
  /** Neighbor rank used for mutual-reachability core distances. */
  readonly minSamples?: number;
  /** Smallest cluster retained by the density hierarchy. */
  readonly minClusterSize?: number;
  /** Final cosine-centroid merge threshold. Set to null to disable merging. */
  readonly mergeThreshold?: number | null;
  /** Number of independent SimHash tables used by approximate k-NN. */
  readonly hashTableCount?: number;
  /** Bits in each SimHash table key. */
  readonly hashBits?: number;
  /** Maximum candidates sampled from one hash bucket. */
  readonly bucketSampleLimit?: number;
  /** Always consider this many preceding/following embeddings. */
  readonly temporalNeighborRadius?: number;
}

export interface ResolvedClusteringOptions {
  readonly useUmap: boolean;
  readonly umapNeighborCount: number;
  readonly umapComponents: number;
  readonly umapEpochs: number;
  readonly umapMinDistance: number;
  readonly umapRandomSeed: number;
  readonly onUmapStats: UmapStatsListener | null;
  readonly neighborCount: number;
  readonly minSamples: number;
  readonly minClusterSize: number;
  readonly mergeThreshold: number | null;
  readonly hashTableCount: number;
  readonly hashBits: number;
  readonly bucketSampleLimit: number;
  readonly temporalNeighborRadius: number;
}

export interface UmapProjectionStats {
  readonly count: number;
  readonly inputDimension: number;
  readonly outputDimension: number;
  readonly neighborCount: number;
  readonly epochs: number;
  readonly seedKnnMs: number;
  readonly refineKnnMs: number;
  readonly fuzzyGraphMs: number;
  readonly optimizeMs: number;
  readonly totalMs: number;
  readonly graphEdgeCount: number;
  /** Bytes retained by the returned Float32 projection. */
  readonly outputBytes: number;
  /** Deterministic peak bytes of live typed-array working state plus output. */
  readonly peakWorkingBytes: number;
  /** `peakWorkingBytes` minus the retained output allocation. */
  readonly peakTemporaryBytes: number;
}

export type UmapStatsListener = (stats: UmapProjectionStats) => void;

export const DEFAULT_CLUSTERING_OPTIONS: ResolvedClusteringOptions = {
  useUmap: true,
  umapNeighborCount: 20,
  umapComponents: 10,
  umapEpochs: 50,
  umapMinDistance: 0,
  umapRandomSeed: 0x6d2b79f5,
  onUmapStats: null,
  neighborCount: 40,
  minSamples: 20,
  minClusterSize: 10,
  mergeThreshold: 0.875,
  hashTableCount: 6,
  hashBits: 8,
  bucketSampleLimit: 64,
  temporalNeighborRadius: 12,
};

export function resolveClusteringOptions(
  options: ClusteringOptions = {},
): ResolvedClusteringOptions {
  const resolved: ResolvedClusteringOptions = {
    useUmap: options.useUmap ?? DEFAULT_CLUSTERING_OPTIONS.useUmap,
    umapNeighborCount:
      options.umapNeighborCount ?? DEFAULT_CLUSTERING_OPTIONS.umapNeighborCount,
    umapComponents:
      options.umapComponents ?? DEFAULT_CLUSTERING_OPTIONS.umapComponents,
    umapEpochs: options.umapEpochs ?? DEFAULT_CLUSTERING_OPTIONS.umapEpochs,
    umapMinDistance:
      options.umapMinDistance ?? DEFAULT_CLUSTERING_OPTIONS.umapMinDistance,
    umapRandomSeed:
      options.umapRandomSeed ?? DEFAULT_CLUSTERING_OPTIONS.umapRandomSeed,
    onUmapStats: options.onUmapStats ?? DEFAULT_CLUSTERING_OPTIONS.onUmapStats,
    neighborCount: options.neighborCount ?? DEFAULT_CLUSTERING_OPTIONS.neighborCount,
    minSamples: options.minSamples ?? DEFAULT_CLUSTERING_OPTIONS.minSamples,
    minClusterSize:
      options.minClusterSize ?? DEFAULT_CLUSTERING_OPTIONS.minClusterSize,
    mergeThreshold:
      options.mergeThreshold === undefined
        ? DEFAULT_CLUSTERING_OPTIONS.mergeThreshold
        : options.mergeThreshold,
    hashTableCount:
      options.hashTableCount ?? DEFAULT_CLUSTERING_OPTIONS.hashTableCount,
    hashBits: options.hashBits ?? DEFAULT_CLUSTERING_OPTIONS.hashBits,
    bucketSampleLimit:
      options.bucketSampleLimit ?? DEFAULT_CLUSTERING_OPTIONS.bucketSampleLimit,
    temporalNeighborRadius:
      options.temporalNeighborRadius ??
      DEFAULT_CLUSTERING_OPTIONS.temporalNeighborRadius,
  };

  requireInteger("umapNeighborCount", resolved.umapNeighborCount, 2, 256);
  requireInteger("umapComponents", resolved.umapComponents, 2, 100);
  requireInteger("umapEpochs", resolved.umapEpochs, 1, 10_000);
  requireInteger("umapRandomSeed", resolved.umapRandomSeed, 0, 0xffff_ffff);
  if (
    !Number.isFinite(resolved.umapMinDistance) ||
    resolved.umapMinDistance < 0 ||
    resolved.umapMinDistance > 1
  ) {
    throw new RangeError("umapMinDistance must be in [0, 1]");
  }
  requireInteger("neighborCount", resolved.neighborCount, 1, 256);
  requireInteger("minSamples", resolved.minSamples, 1, 256);
  requireInteger("minClusterSize", resolved.minClusterSize, 2, 1_000_000);
  requireInteger("hashTableCount", resolved.hashTableCount, 1, 16);
  requireInteger("hashBits", resolved.hashBits, 4, 16);
  requireInteger("bucketSampleLimit", resolved.bucketSampleLimit, 1, 4096);
  requireInteger(
    "temporalNeighborRadius",
    resolved.temporalNeighborRadius,
    0,
    4096,
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
