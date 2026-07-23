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
  const [modelsResult, clusteringResult] = await Promise.allSettled([
    Promise.resolve().then(loadModels),
    Promise.resolve().then(loadClustering),
  ] as const);

  if (modelsResult.status === "rejected" || clusteringResult.status === "rejected") {
    await releaseQuietly(
      modelsResult.status === "fulfilled" ? modelsResult.value : undefined,
    );
    disposeQuietly(
      clusteringResult.status === "fulfilled"
        ? clusteringResult.value
        : undefined,
    );
    throw modelsResult.status === "rejected"
      ? modelsResult.reason
      : clusteringResult.status === "rejected"
        ? clusteringResult.reason
        : new Error("Worker resource loading failed");
  }

  try {
    await clusteringResult.value.warmup();
    return {
      models: modelsResult.value,
      clustering: clusteringResult.value,
    };
  } catch (error) {
    await releaseQuietly(modelsResult.value);
    disposeQuietly(clusteringResult.value);
    throw error;
  }
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
