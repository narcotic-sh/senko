import { beforeEach, describe, expect, it, vi } from "vitest";

const fakes = vi.hoisted(() => ({
  events: [] as string[],
  selections: [] as string[],
  deviceDescriptors: [] as GPUDeviceDescriptor[],
  failEmbeddingCreate: false,
  manifest: {
    models: {
      segmentation: { split: { frontend: {} } },
      campplus: { id: "campplus", batches: { "32": {} }, direct_webgpu: {} },
    },
  },
}));

vi.mock("./model-manifest", () => ({
  loadModelManifest: async () => fakes.manifest,
  selectSegmentationSplit: (
    _url: string,
    _model: unknown,
    _batch: number,
    precision: "float16" | "float32",
  ) => ({
    precision,
    batchSize: 8,
    directWebGpu: {
      frontendMetadata: { url: "https://example.test/frontend.json" },
      tailMetadata: { url: "https://example.test/tail.json" },
      explicitGpuBytes: 44_145_664,
    },
  }),
  selectCampPlusDirect: (
    _url: string,
    _model: unknown,
    _batch: number,
    precision: "float16" | "float32",
  ) => ({
    precision,
    batchSize: 16,
    metadata: { url: "https://example.test/campplus.json" },
    weights: { url: "https://example.test/campplus.bin" },
    explicitGpuBufferBytes: 39_855_360,
  }),
}));

vi.mock("./raw-campplus-backend", () => {
  class FakeEmbeddingBackend {
    readonly batchSize = 16;
    readonly frames = 150;
    readonly featureDim = 80;
    readonly embeddingDim = 192;
    readonly gpuBufferBytes = { total: 39_855_360 };
    private released = false;

    static async create(device: GPUDevice): Promise<FakeEmbeddingBackend> {
      fakes.events.push(`create-embedding:${fakeDeviceRole(device)}`);
      if (fakes.failEmbeddingCreate) throw new Error("embedding load failed");
      return new FakeEmbeddingBackend();
    }

    async run(_input: Float32Array): Promise<Float32Array> {
      if (this.released) throw new Error("Direct WebGPU CAM++ has been released");
      fakes.events.push("run-embedding");
      return new Float32Array(this.batchSize * this.embeddingDim);
    }

    async release(): Promise<void> {
      this.released = true;
      fakes.events.push("release-embedding");
    }
  }

  return { RawCampPlusEmbeddingBackend: FakeEmbeddingBackend };
});

vi.mock("./raw-vad-backend", () => {
  class FakeVadBackend {
    readonly batchSize = 8;
    readonly chunkSamples = 160_000;
    readonly outputFrames = 589;
    readonly outputClasses = 7;
    readonly gpuBufferBytes = { totalOwned: 44_145_664 };

    static async create(device: GPUDevice): Promise<FakeVadBackend> {
      fakes.events.push(`create-vad:${fakeDeviceRole(device)}`);
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

interface FakeDevice {
  readonly role: "vad" | "embedding";
  readonly queue: { onSubmittedWorkDone(): Promise<void> };
  destroy(): void;
}

function fakeDeviceRole(device: GPUDevice): string {
  return (device as unknown as FakeDevice).role;
}

function createGpu(shaderF16 = true): {
  readonly gpu: GPU;
  readonly devices: readonly [FakeDevice, FakeDevice];
} {
  const devices = (["vad", "embedding"] as const).map((role): FakeDevice => ({
    role,
    queue: {
      async onSubmittedWorkDone() {
        fakes.events.push(`queue-idle:${role}`);
      },
    },
    destroy() {
      fakes.events.push(`destroy-device:${role}`);
    },
  })) as unknown as readonly [FakeDevice, FakeDevice];
  let adapterIndex = 0;
  const gpu = {
    async requestAdapter() {
      const index = adapterIndex;
      adapterIndex += 1;
      const device = devices[index];
      if (device === undefined) throw new Error("unexpected third adapter request");
      fakes.events.push(`request-adapter:${device.role}`);
      return {
        features: new Set(shaderF16 ? ["shader-f16"] : []),
        limits: {
          maxBufferSize: 1_000_000_000,
          maxStorageBufferBindingSize: 1_000_000_000,
          maxComputeWorkgroupStorageSize: 32_768,
          maxComputeInvocationsPerWorkgroup: 256,
          maxComputeWorkgroupSizeX: 256,
          maxComputeWorkgroupsPerDimension: 65_535,
        },
        async requestDevice(descriptor: GPUDeviceDescriptor) {
          fakes.events.push(`request-device:${device.role}`);
          fakes.deviceDescriptors.push(descriptor);
          return device;
        },
      };
    },
  };
  return { gpu: gpu as unknown as GPU, devices };
}

describe("BrowserModelSet dual residency", () => {
  beforeEach(() => {
    fakes.events.length = 0;
    fakes.selections.length = 0;
    fakes.deviceDescriptors.length = 0;
    fakes.failEmbeddingCreate = false;
  });

  it("loads, warms, and retains VAD and CAM++ on separate devices", async () => {
    const { gpu, devices } = createGpu();
    const models = await BrowserModelSet.load(
      "https://example.test/manifest.json",
      gpu,
      { vadBatchSize: 8, embeddingBatchSize: 16, warmupRuns: 1 },
    );

    expect(models.vadDevice).toBe(devices[0]);
    expect(models.embeddingDevice).toBe(devices[1]);
    expect(models.precision).toBe("float16");
    expect(fakes.events).toEqual([
      "request-adapter:vad",
      "request-adapter:embedding",
      "request-device:vad",
      "request-device:embedding",
      "create-vad:vad",
      "create-embedding:embedding",
      "run-vad",
      "run-embedding",
    ]);
    expect(models.knownGpuBufferBytes).toBe(84_001_024);
    expect(fakes.deviceDescriptors).toHaveLength(2);
    expect(fakes.deviceDescriptors).toEqual([
      expect.objectContaining({ requiredFeatures: ["shader-f16"] }),
      expect.objectContaining({ requiredFeatures: ["shader-f16"] }),
    ]);

    fakes.events.length = 0;
    await models.vad.run(new Float32Array(8 * 160_000));
    await models.embedding.run(new Float32Array(16 * 150 * 80));
    expect(fakes.events).toEqual(["run-vad", "run-embedding"]);
  });

  it("forces fully FP32 models without requesting shader-f16", async () => {
    const { gpu } = createGpu(true);
    const models = await BrowserModelSet.load(
      "https://example.test/manifest.json",
      gpu,
      { preferFloat16: false, warmupRuns: 0 },
    );

    expect(models.precision).toBe("float32");
    expect(models.vadVariant.precision).toBe("float32");
    expect(models.embeddingVariant.precision).toBe("float32");
    expect(fakes.deviceDescriptors).toHaveLength(2);
    expect(
      fakes.deviceDescriptors.every(
        (descriptor) => descriptor.requiredFeatures === undefined,
      ),
    ).toBe(true);
  });

  it("automatically chooses FP32 when shader-f16 is unavailable", async () => {
    const { gpu } = createGpu(false);
    const models = await BrowserModelSet.load(
      "https://example.test/manifest.json",
      gpu,
      { warmupRuns: 0 },
    );

    expect(models.precision).toBe("float32");
    expect(fakes.deviceDescriptors).toHaveLength(2);
    expect(
      fakes.deviceDescriptors.every(
        (descriptor) => descriptor.requiredFeatures === undefined,
      ),
    ).toBe(true);
  });

  it("reports both concurrent load and warmup paths", async () => {
    const progress: Array<{ readonly stage: string; readonly message: string }> = [];
    const { gpu } = createGpu();

    await BrowserModelSet.load("https://example.test/manifest.json", gpu, {
      warmupRuns: 1,
      onProgress: (event) => progress.push(event),
    });

    expect(progress.map((event) => event.stage)).toEqual([
      "manifest",
      "vad",
      "embedding",
      "warmup",
      "warmup",
    ]);
    expect(progress.map((event) => event.message)).toEqual([
      "Loading model manifest",
      "Loading pyannote segmentation B8",
      "Loading CAM++ embedding B16",
      "Compiling pyannote WebGPU kernels",
      "Compiling CAM++ WebGPU kernels",
    ]);
  });

  it("drains, releases, and destroys both devices exactly once", async () => {
    const { gpu } = createGpu();
    const models = await BrowserModelSet.load(
      "https://example.test/manifest.json",
      gpu,
      { warmupRuns: 0 },
    );
    fakes.events.length = 0;

    await models.release();
    await models.release();

    expect(fakes.events).toEqual([
      "release-vad",
      "release-embedding",
      "queue-idle:vad",
      "queue-idle:embedding",
      "destroy-device:vad",
      "destroy-device:embedding",
    ]);
    expect(() => models.vad).toThrow("has been released");
    await expect(
      models.embedding.run(new Float32Array(16 * 150 * 80)),
    ).rejects.toThrow("has been released");
  });

  it("cleans up both devices after a partial concurrent load failure", async () => {
    const { gpu } = createGpu();
    fakes.failEmbeddingCreate = true;

    await expect(
      BrowserModelSet.load("https://example.test/manifest.json", gpu, {
        warmupRuns: 0,
      }),
    ).rejects.toThrow("embedding load failed");

    expect(fakes.events).toContain("release-vad");
    expect(fakes.events.slice(-4)).toEqual([
      "queue-idle:vad",
      "queue-idle:embedding",
      "destroy-device:vad",
      "destroy-device:embedding",
    ]);
  });
});
