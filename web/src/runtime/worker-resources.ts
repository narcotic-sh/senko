export interface WorkerModelResources {
  release(): Promise<void>;
}

export interface WorkerClusteringResources {
  warmup(): void | Promise<void>;
  dispose(): void;
}

export interface LoadedWorkerResources<
  Models extends WorkerModelResources,
  Clustering extends WorkerClusteringResources,
> {
  readonly models: Models;
  readonly clustering: Clustering;
}

/** Load both worker residency sets atomically and clean up every failure path. */
export async function loadWorkerResources<
  Models extends WorkerModelResources,
  Clustering extends WorkerClusteringResources,
>(
  loadModels: () => Promise<Models>,
  loadClustering: () => Promise<Clustering>,
): Promise<LoadedWorkerResources<Models, Clustering>> {
  let loadedClustering: Clustering | undefined;
  const loadAndWarmClustering = Promise.resolve()
    .then(loadClustering)
    .then(async (clustering) => {
      loadedClustering = clustering;
      await clustering.warmup();
      return clustering;
    });
  const [modelsResult, clusteringResult] = await Promise.allSettled([
    Promise.resolve().then(loadModels),
    loadAndWarmClustering,
  ] as const);

  if (modelsResult.status === "rejected" || clusteringResult.status === "rejected") {
    await releaseQuietly(
      modelsResult.status === "fulfilled" ? modelsResult.value : undefined,
    );
    disposeQuietly(
      clusteringResult.status === "fulfilled"
        ? clusteringResult.value
        : loadedClustering,
    );
    throw modelsResult.status === "rejected"
      ? modelsResult.reason
      : clusteringResult.status === "rejected"
        ? clusteringResult.reason
        : new Error("Worker resource loading failed");
  }

  return {
    models: modelsResult.value,
    clustering: clusteringResult.value,
  };
}

async function releaseQuietly(
  models: WorkerModelResources | undefined,
): Promise<void> {
  if (models === undefined) return;
  await Promise.allSettled([models.release()]);
}

function disposeQuietly(
  clustering: WorkerClusteringResources | undefined,
): void {
  try {
    clustering?.dispose();
  } catch {
    // Preserve the original load/warm-up error.
  }
}
