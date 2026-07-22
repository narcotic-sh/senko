import {
  loadModelManifest,
  selectCampPlusDirect,
  selectSegmentationSplit,
  type BrowserModelManifest,
  type ModelManifestIntegrity,
  type SelectedCampPlusDirect,
  type SelectedSegmentationSplit,
} from "./model-manifest";
import { CAMPPLUS_RAW_MAX_IN_FLIGHT_RUNS } from "./campplus-webgpu";
import { RawCampPlusEmbeddingBackend } from "./raw-campplus-backend";
import {
  RawWebGpuVadBackend,
  type RawVadModelAssets,
} from "./raw-vad-backend";
import type { EmbeddingBatchBackend } from "./types";

export interface BrowserModelLoadProgress {
  readonly stage: "manifest" | "vad" | "embedding" | "warmup";
  readonly message: string;
}

export interface BrowserModelSetOptions {
  readonly manifestIntegrity?: ModelManifestIntegrity;
  readonly vadBatchSize?: number;
  readonly embeddingBatchSize?: number;
  readonly warmupRuns?: number;
  readonly onProgress?: (progress: BrowserModelLoadProgress) => void;
}

export class BrowserModelSet {
  private released = false;
  private vadBackend: RawWebGpuVadBackend | undefined;
  private embeddingBackend: RawCampPlusEmbeddingBackend | undefined;
  private peakKnownGpuBufferBytes = 0;

  readonly embedding: EmbeddingBatchBackend;

  /**
   * Peak explicitly owned WebGPU buffers across the mutually exclusive model
   * stages. Both direct-WebGPU backends expose every owned GPUBuffer.
   */
  get knownGpuBufferBytes(): number {
    return this.peakKnownGpuBufferBytes;
  }

  get vad(): RawWebGpuVadBackend {
    const backend = this.vadBackend;
    if (backend === undefined) {
      throw new Error("The pyannote VAD stage has not been prepared");
    }
    return backend;
  }

  private constructor(
    readonly device: GPUDevice,
    readonly manifest: BrowserModelManifest,
    readonly vadVariant: SelectedSegmentationSplit,
    readonly embeddingVariant: SelectedCampPlusDirect,
    readonly embeddingPrecision: "float16",
    vad: RawWebGpuVadBackend,
    private readonly rawVadModelAssets: RawVadModelAssets,
    private readonly warmupRuns: number,
    readonly loadElapsedMs: number,
  ) {
    this.vadBackend = vad;
    this.observeKnownGpuBufferBytes(vad.gpuBufferBytes.totalOwned);
    // Shape metadata is available before the CAM++ session exists, while run
    // dispatches only to the backend active during the embedding stage.
    this.embedding = {
      batchSize: embeddingVariant.batchSize,
      frames: 150,
      featureDim: 80,
      embeddingDim: 192,
      maxInFlightRuns: CAMPPLUS_RAW_MAX_IN_FLIGHT_RUNS,
      run: (features) => this.runEmbedding(features),
    };
  }

  static async load(
    manifestUrl: string,
    adapter: GPUAdapter,
    options: BrowserModelSetOptions = {},
  ): Promise<BrowserModelSet> {
    const start = performance.now();
    options.onProgress?.({ stage: "manifest", message: "Loading model manifest" });
    const manifest = await loadModelManifest(
      manifestUrl,
      options.manifestIntegrity,
    );
    // Direct-WebGPU pyannote is currently packaged and tuned at B8.
    const vadBatchSize = options.vadBatchSize ?? 8;
    const embeddingPrecision = "float16" as const;
    const vadVariant = selectSegmentationSplit(
      manifestUrl,
      manifest.models.segmentation,
      vadBatchSize,
    );
    const embeddingVariant = selectCampPlusDirect(
      manifestUrl,
      manifest.models.campplus,
      options.embeddingBatchSize,
    );
    const device = await requestMaximumPerformanceDevice(adapter);

    options.onProgress?.({
      stage: "vad",
      message: `Loading pyannote segmentation B${vadBatchSize}`,
    });
    const warmupRuns = options.warmupRuns ?? 1;
    let vad: RawWebGpuVadBackend | undefined;
    try {
      const directVadAssets = rawVadAssets(vadVariant);
      vad = await RawWebGpuVadBackend.create(
        device,
        vadVariant,
        directVadAssets,
        (message) =>
          options.onProgress?.({ stage: "vad", message: `Pyannote: ${message}` }),
      );
      if (warmupRuns > 0) {
        options.onProgress?.({
          stage: "warmup",
          message: "Compiling pyannote WebGPU kernels",
        });
        const vadInput = new Float32Array(vad.batchSize * vad.chunkSamples);
        for (let index = 0; index < warmupRuns; index += 1) {
          await vad.run(vadInput);
        }
      }

      return new BrowserModelSet(
        device,
        manifest,
        vadVariant,
        embeddingVariant,
        embeddingPrecision,
        vad,
        directVadAssets,
        warmupRuns,
        performance.now() - start,
      );
    } catch (error) {
      await vad?.release();
      device.destroy();
      throw error;
    }
  }

  /** Ensure a VAD-only model residency set, loading it after a previous run. */
  async prepareVadStage(): Promise<void> {
    this.ensureNotReleased();
    if (this.vadBackend !== undefined) return;

    await this.releaseEmbeddingBackend();
    this.ensureNotReleased();

    let vad: RawWebGpuVadBackend | undefined;
    try {
      vad = await RawWebGpuVadBackend.create(
        this.device,
        this.vadVariant,
        this.rawVadModelAssets,
      );
      await warmVad(vad, this.warmupRuns);
      this.ensureNotReleased();
      this.vadBackend = vad;
      this.observeKnownGpuBufferBytes(vad.gpuBufferBytes.totalOwned);
    } catch (error) {
      await vad?.release();
      throw error;
    }
  }

  /**
   * Drop all VAD sessions and buffers before constructing and warming CAM++.
   * The two direct-WebGPU residency sets are deliberately mutually exclusive.
   */
  async prepareEmbeddingStage(hasEmbeddingWork = true): Promise<void> {
    this.ensureNotReleased();
    if (hasEmbeddingWork && this.embeddingBackend !== undefined) return;

    const vad = this.vadBackend;
    this.vadBackend = undefined;
    if (vad !== undefined) await vad.release();
    await this.device.queue.onSubmittedWorkDone();
    this.ensureNotReleased();
    if (!hasEmbeddingWork) {
      await this.releaseEmbeddingBackend();
      return;
    }

    let embedding: RawCampPlusEmbeddingBackend | undefined;
    try {
      embedding = await RawCampPlusEmbeddingBackend.create(
        this.device,
        this.embeddingVariant,
        (message) => console.info(`[senko] ${message}`),
      );
      await warmEmbedding(embedding, this.warmupRuns);
      this.ensureNotReleased();
      this.embeddingBackend = embedding;
      this.observeKnownGpuBufferBytes(embedding.gpuBufferBytes.total);
    } catch (error) {
      await embedding?.release();
      throw error;
    }
  }

  /** Release CAM++ before the CPU-only clustering and post-processing stages. */
  async finishEmbeddingStage(): Promise<void> {
    this.ensureNotReleased();
    await this.releaseEmbeddingBackend();
  }

  async release(): Promise<void> {
    if (this.released) return;
    this.released = true;
    const vad = this.vadBackend;
    const embedding = this.embeddingBackend;
    this.vadBackend = undefined;
    this.embeddingBackend = undefined;
    await Promise.allSettled([vad?.release(), embedding?.release()]);
    this.device.destroy();
  }

  private async runEmbedding(features: Float32Array): Promise<Float32Array> {
    this.ensureNotReleased();
    const backend = this.embeddingBackend;
    if (backend === undefined) {
      throw new Error("The CAM++ embedding stage has not been prepared");
    }
    return await backend.run(features);
  }

  private async releaseEmbeddingBackend(): Promise<void> {
    const embedding = this.embeddingBackend;
    this.embeddingBackend = undefined;
    if (embedding === undefined) return;
    try {
      await embedding.release();
    } finally {
      await this.device.queue.onSubmittedWorkDone();
    }
  }

  private observeKnownGpuBufferBytes(bytes: number): void {
    this.peakKnownGpuBufferBytes = Math.max(
      this.peakKnownGpuBufferBytes,
      bytes,
    );
  }

  private ensureNotReleased(): void {
    if (this.released) throw new Error("Browser model set has been released");
  }
}

async function warmVad(vad: RawWebGpuVadBackend, runs: number): Promise<void> {
  if (runs <= 0) return;
  const input = new Float32Array(vad.batchSize * vad.chunkSamples);
  for (let index = 0; index < runs; index += 1) await vad.run(input);
}

function rawVadAssets(selected: SelectedSegmentationSplit): RawVadModelAssets {
  return {
    frontendMetadata: selected.directWebGpu.frontendMetadata,
    tailMetadata: selected.directWebGpu.tailMetadata,
  };
}

async function warmEmbedding(
  embedding: RawCampPlusEmbeddingBackend,
  runs: number,
): Promise<void> {
  if (runs <= 0) return;
  const input = new Float32Array(
    embedding.batchSize * embedding.frames * embedding.featureDim,
  );
  for (let index = 0; index < runs; index += 1) await embedding.run(input);
}

export async function requestMaximumPerformanceAdapter(
  gpu: GPU,
): Promise<GPUAdapter> {
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("No high-performance WebGPU adapter is available");
  return adapter;
}

export async function requestMaximumPerformanceDevice(
  adapter: GPUAdapter,
): Promise<GPUDevice> {
  if (!adapter.features.has("shader-f16")) {
    throw new Error("Direct WebGPU inference requires shader-f16 support");
  }
  return await adapter.requestDevice({
    requiredFeatures: ["shader-f16"],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize:
        adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup:
        adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupsPerDimension:
        adapter.limits.maxComputeWorkgroupsPerDimension,
    },
  });
}
