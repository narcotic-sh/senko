import { beforeEach, describe, expect, it, vi } from "vitest";

const fakes = vi.hoisted(() => {
  const events: string[] = [];
  const device = {
    queue: {
      async onSubmittedWorkDone() {
        events.push("queue-idle");
      },
    },
    destroy() {
      events.push("destroy-device");
    },
  };
  const adapter = {
    features: new Set(["shader-f16"]),
    limits: {
      maxBufferSize: 1_000_000_000,
      maxStorageBufferBindingSize: 1_000_000_000,
      maxComputeWorkgroupStorageSize: 32_768,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupsPerDimension: 65_535,
    },
    async requestDevice() {
      return device;
    },
  };
  const manifest = {
    models: {
      segmentation: { split: { frontend: {} } },
      campplus: { id: "campplus", batches: { "32": {} }, direct_webgpu: {} },
    },
  };
  return { adapter, device, events, manifest };
});

vi.mock("./model-manifest", () => ({
  chooseVadBatchSize: () => 8,
  loadModelManifest: async () => fakes.manifest,
  selectSegmentationSplit: () => ({
    batchSize: 8,
    directWebGpu: {
      frontendMetadata: { url: "https://example.test/frontend.json" },
      tailMetadata: { url: "https://example.test/tail.json" },
      explicitGpuBytes: 20_000_000,
    },
  }),
  selectCampPlusDirect: () => ({
    batchSize: 16,
    metadata: { url: "https://example.test/campplus.json" },
    weights: { url: "https://example.test/campplus.bin" },
    explicitGpuBufferBytes: 2_000_000,
  }),
}));

vi.mock("./raw-campplus-backend", () => {
  class FakeEmbeddingBackend {
    readonly batchSize = 16;
    readonly frames = 150;
    readonly featureDim = 80;
    readonly embeddingDim = 192;
    readonly gpuBufferBytes = { total: 2_000_000 };

    static async create(): Promise<FakeEmbeddingBackend> {
      fakes.events.push("create-embedding");
      return new FakeEmbeddingBackend();
    }

    async run(_input: Float32Array): Promise<Float32Array> {
      fakes.events.push("run-embedding");
      return new Float32Array(this.batchSize * this.embeddingDim);
    }

    async release(): Promise<void> {
      fakes.events.push("release-embedding");
    }
  }

  return {
    RawCampPlusEmbeddingBackend: FakeEmbeddingBackend,
  };
});

vi.mock("./raw-vad-backend", () => {
  class FakeVadBackend {
    readonly batchSize = 8;
    readonly chunkSamples = 160_000;
    readonly outputFrames = 589;
    readonly outputClasses = 7;
    readonly gpuBufferBytes = { totalOwned: 20_000_000 };

    static async create(): Promise<FakeVadBackend> {
      fakes.events.push("create-vad");
      return new FakeVadBackend();
    }

    async run(_input: Float32Array): Promise<Float32Array> {
      fakes.events.push("run-vad");
      return new Float32Array(
        this.batchSize * this.outputFrames * this.outputClasses,
      );
    }

    async release(): Promise<void> {
      fakes.events.push("release-vad");
    }
  }

  return { RawWebGpuVadBackend: FakeVadBackend };
});

import { BrowserModelSet } from "./browser-models";

describe("BrowserModelSet stage residency", () => {
  beforeEach(() => {
    fakes.events.length = 0;
  });

  it("never retains VAD and CAM++ at the same stage boundary", async () => {
    const models = await BrowserModelSet.load(
      "https://example.test/manifest.json",
      fakes.adapter as unknown as GPUAdapter,
      {
        vadBatchSize: 8,
        embeddingBatchSize: 16,
        warmupRuns: 1,
      },
    );

    expect(fakes.events).toEqual(["create-vad", "run-vad"]);
    expect(models.knownGpuBufferBytes).toBe(20_000_000);
    await expect(
      models.embedding.run(new Float32Array(16 * 150 * 80)),
    ).rejects.toThrow("has not been prepared");

    fakes.events.length = 0;
    await models.prepareEmbeddingStage();
    expect(fakes.events).toEqual([
      "release-vad",
      "queue-idle",
      "create-embedding",
      "run-embedding",
    ]);
    await models.embedding.run(new Float32Array(16 * 150 * 80));
    expect(fakes.events.at(-1)).toBe("run-embedding");
    // Residency is mutually exclusive, so this is max(VAD, CAM), not their sum.
    expect(models.knownGpuBufferBytes).toBe(20_000_000);

    fakes.events.length = 0;
    await models.finishEmbeddingStage();
    expect(fakes.events).toEqual(["release-embedding", "queue-idle"]);

    fakes.events.length = 0;
    await models.prepareVadStage();
    expect(fakes.events).toEqual(["create-vad", "run-vad"]);
    await models.release();
    expect(fakes.events.slice(-2)).toEqual(["release-vad", "destroy-device"]);
  });

  it("makes final release idempotent", async () => {
    const models = await BrowserModelSet.load(
      "https://example.test/manifest.json",
      fakes.adapter as unknown as GPUAdapter,
      {
        vadBatchSize: 8,
        embeddingBatchSize: 16,
        warmupRuns: 0,
      },
    );
    fakes.events.length = 0;

    await models.release();
    await models.release();

    expect(fakes.events).toEqual(["release-vad", "destroy-device"]);
    await expect(models.prepareVadStage()).rejects.toThrow("has been released");
  });

  it("releases VAD without loading CAM++ when there is no speech", async () => {
    const models = await BrowserModelSet.load(
      "https://example.test/manifest.json",
      fakes.adapter as unknown as GPUAdapter,
      {
        vadBatchSize: 8,
        embeddingBatchSize: 16,
        warmupRuns: 0,
      },
    );
    fakes.events.length = 0;

    await models.prepareEmbeddingStage(false);

    expect(fakes.events).toEqual(["release-vad", "queue-idle"]);
    await expect(
      models.embedding.run(new Float32Array(16 * 150 * 80)),
    ).rejects.toThrow("has not been prepared");
    await models.release();
  });
});
