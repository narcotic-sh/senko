import { describe, expect, it, vi } from "vitest";

import { loadWorkerResources } from "./worker-resources";

function modelResources() {
  return { release: vi.fn(async () => undefined) };
}

function clusteringResources(warmupError?: Error) {
  return {
    warmup: vi.fn(() => {
      if (warmupError !== undefined) throw warmupError;
    }),
    dispose: vi.fn(),
  };
}

function asyncClusteringResources(warmupError?: Error) {
  return {
    warmup: vi.fn(async () => {
      if (warmupError !== undefined) throw warmupError;
    }),
    dispose: vi.fn(),
  };
}

describe("loadWorkerResources", () => {
  it("warms and returns both resources only after both loads succeed", async () => {
    const models = modelResources();
    const clustering = clusteringResources();

    await expect(
      loadWorkerResources(async () => models, async () => clustering),
    ).resolves.toEqual({ models, clustering });
    expect(clustering.warmup).toHaveBeenCalledOnce();
    expect(models.release).not.toHaveBeenCalled();
    expect(clustering.dispose).not.toHaveBeenCalled();
  });

  it("releases a loaded model set when clustering fails to load", async () => {
    const models = modelResources();
    const failure = new Error("clustering failed");

    await expect(
      loadWorkerResources(
        async () => models,
        async () => {
          throw failure;
        },
      ),
    ).rejects.toBe(failure);
    expect(models.release).toHaveBeenCalledOnce();
  });

  it("warms and disposes loaded clustering kernels when model loading fails", async () => {
    const clustering = clusteringResources();
    const failure = new Error("models failed");

    await expect(
      loadWorkerResources(
        async () => {
          throw failure;
        },
        async () => clustering,
      ),
    ).rejects.toBe(failure);
    expect(clustering.warmup).toHaveBeenCalledOnce();
    expect(clustering.dispose).toHaveBeenCalledOnce();
  });

  it("cleans up both resources when warm-up fails", async () => {
    const models = modelResources();
    const failure = new Error("warm-up failed");
    const clustering = clusteringResources(failure);

    await expect(
      loadWorkerResources(async () => models, async () => clustering),
    ).rejects.toBe(failure);
    expect(models.release).toHaveBeenCalledOnce();
    expect(clustering.dispose).toHaveBeenCalledOnce();
  });

  it("awaits asynchronous warm-up before returning resources", async () => {
    const models = modelResources();
    const clustering = asyncClusteringResources();
    let releaseWarmup!: () => void;
    let reportWarmupStarted!: () => void;
    const warmupStarted = new Promise<void>((resolve) => {
      reportWarmupStarted = resolve;
    });
    const warmupGate = new Promise<void>((resolve) => {
      releaseWarmup = resolve;
    });
    clustering.warmup.mockImplementation(() => {
      reportWarmupStarted();
      return warmupGate;
    });

    let settled = false;
    const loading = loadWorkerResources(
      async () => models,
      async () => clustering,
    ).finally(() => {
      settled = true;
    });
    await warmupStarted;
    expect(settled).toBe(false);

    releaseWarmup();
    await expect(loading).resolves.toEqual({ models, clustering });
  });

  it("warms clustering while model loading is still in progress", async () => {
    const models = modelResources();
    const clustering = asyncClusteringResources();
    let releaseModels!: () => void;
    let reportWarmupStarted!: () => void;
    const modelsGate = new Promise<void>((resolve) => {
      releaseModels = resolve;
    });
    const warmupStarted = new Promise<void>((resolve) => {
      reportWarmupStarted = resolve;
    });
    clustering.warmup.mockImplementation(async () => {
      reportWarmupStarted();
    });

    const loading = loadWorkerResources(
      async () => {
        await modelsGate;
        return models;
      },
      async () => clustering,
    );
    await warmupStarted;
    expect(clustering.warmup).toHaveBeenCalledOnce();
    expect(models.release).not.toHaveBeenCalled();

    releaseModels();
    await expect(loading).resolves.toEqual({ models, clustering });
  });

  it("cleans up both resources when asynchronous warm-up rejects", async () => {
    const models = modelResources();
    const failure = new Error("async warm-up failed");
    const clustering = asyncClusteringResources(failure);

    await expect(
      loadWorkerResources(async () => models, async () => clustering),
    ).rejects.toBe(failure);
    expect(models.release).toHaveBeenCalledOnce();
    expect(clustering.dispose).toHaveBeenCalledOnce();
  });
});
