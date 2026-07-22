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

  it("disposes loaded clustering kernels when model loading fails", async () => {
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
    expect(clustering.warmup).not.toHaveBeenCalled();
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
});
