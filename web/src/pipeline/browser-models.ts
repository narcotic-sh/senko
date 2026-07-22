import {
  loadModelManifest,
  selectCampPlusDirect,
  selectSegmentationSplit,
  type BrowserModelManifest,
  type ModelManifestIntegrity,
  type SelectedCampPlusDirect,
  type SelectedSegmentationSplit,
  type WebGpuModelPrecision,
} from "./model-manifest";
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
  /** Set false to exercise the complete FP32 fallback on shader-f16 hardware. */
  readonly preferFloat16?: boolean;
  readonly warmupRuns?: number;
  readonly onProgress?: (progress: BrowserModelLoadProgress) => void;
}

export class BrowserModelSet {
  private released = false;
  private vadBackend: RawWebGpuVadBackend | undefined;
  private embeddingBackend: RawCampPlusEmbeddingBackend | undefined;

  readonly embedding: EmbeddingBatchBackend;

  /** Exact sum of the two concurrently resident direct-WebGPU model sets. */
  get knownGpuBufferBytes(): number {
    return (
      (this.vadBackend?.gpuBufferBytes.totalOwned ?? 0) +
      (this.embeddingBackend?.gpuBufferBytes.total ?? 0)
    );
  }

  get vad(): RawWebGpuVadBackend {
    this.ensureNotReleased();
    const backend = this.vadBackend;
    if (backend === undefined) {
      throw new Error("The pyannote VAD stage has not been prepared");
    }
    return backend;
  }

  private constructor(
    readonly vadDevice: GPUDevice,
    readonly embeddingDevice: GPUDevice,
    readonly manifest: BrowserModelManifest,
    readonly vadVariant: SelectedSegmentationSplit,
    readonly embeddingVariant: SelectedCampPlusDirect,
    readonly precision: WebGpuModelPrecision,
    vad: RawWebGpuVadBackend,
    embedding: RawCampPlusEmbeddingBackend,
    readonly loadElapsedMs: number,
  ) {
    this.vadBackend = vad;
    this.embeddingBackend = embedding;
    this.embedding = embedding;
  }

  static async load(
    manifestUrl: string,
    gpu: GPU,
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
    const warmupRuns = options.warmupRuns ?? 1;
    let vadDevice: GPUDevice | undefined;
    let embeddingDevice: GPUDevice | undefined;
    let vad: RawWebGpuVadBackend | undefined;
    let embedding: RawCampPlusEmbeddingBackend | undefined;
    try {
      // Chrome/Dawn consumes a GPUAdapter handle after requestDevice(). The two
      // production residency sets therefore require independently requested
      // handles to the same high-performance physical adapter.
      const vadAdapter = await requestMaximumPerformanceAdapter(gpu);
      const embeddingAdapter = await requestMaximumPerformanceAdapter(gpu);
      const precision = selectWebGpuModelPrecision(
        [vadAdapter, embeddingAdapter],
        options.preferFloat16 ?? true,
      );
      const vadVariant = selectSegmentationSplit(
        manifestUrl,
        manifest.models.segmentation,
        vadBatchSize,
        precision,
      );
      const embeddingVariant = selectCampPlusDirect(
        manifestUrl,
        manifest.models.campplus,
        options.embeddingBatchSize,
        precision,
      );
      vadDevice = await requestMaximumPerformanceDevice(vadAdapter, precision);
      embeddingDevice = await requestMaximumPerformanceDevice(
        embeddingAdapter,
        precision,
      );

      const directVadAssets = rawVadAssets(vadVariant);
      const loadVad = async (): Promise<RawWebGpuVadBackend> => {
        options.onProgress?.({
          stage: "vad",
          message: `Loading pyannote segmentation B${vadBatchSize}`,
        });
        const backend = await RawWebGpuVadBackend.create(
          vadDevice!,
          vadVariant,
          directVadAssets,
          (message) =>
            options.onProgress?.({
              stage: "vad",
              message: `Pyannote: ${message}`,
            }),
        );
        vad = backend;
        if (warmupRuns > 0) {
          options.onProgress?.({
            stage: "warmup",
            message: "Compiling pyannote WebGPU kernels",
          });
          await warmVad(backend, warmupRuns);
        }
        return backend;
      };
      const loadEmbedding = async (): Promise<RawCampPlusEmbeddingBackend> => {
        options.onProgress?.({
          stage: "embedding",
          message: `Loading CAM++ embedding B${embeddingVariant.batchSize}`,
        });
        const backend = await RawCampPlusEmbeddingBackend.create(
          embeddingDevice!,
          embeddingVariant,
          (message) =>
            options.onProgress?.({
              stage: "embedding",
              message: `CAM++: ${message}`,
            }),
        );
        embedding = backend;
        if (warmupRuns > 0) {
          options.onProgress?.({
            stage: "warmup",
            message: "Compiling CAM++ WebGPU kernels",
          });
          await warmEmbedding(backend, warmupRuns);
        }
        return backend;
      };

      const [vadResult, embeddingResult] = await Promise.allSettled([
        loadVad(),
        loadEmbedding(),
      ]);
      if (vadResult.status === "rejected") throw vadResult.reason;
      if (embeddingResult.status === "rejected") throw embeddingResult.reason;

      return new BrowserModelSet(
        vadDevice,
        embeddingDevice,
        manifest,
        vadVariant,
        embeddingVariant,
        precision,
        vadResult.value,
        embeddingResult.value,
        performance.now() - start,
      );
    } catch (error) {
      await Promise.allSettled([vad?.release(), embedding?.release()]);
      await drainDevices(vadDevice, embeddingDevice);
      vadDevice?.destroy();
      embeddingDevice?.destroy();
      throw error;
    }
  }

  async release(): Promise<void> {
    if (this.released) return;
    this.released = true;
    const vad = this.vadBackend;
    const embedding = this.embeddingBackend;
    this.vadBackend = undefined;
    this.embeddingBackend = undefined;
    await Promise.allSettled([vad?.release(), embedding?.release()]);
    await drainDevices(this.vadDevice, this.embeddingDevice);
    this.vadDevice.destroy();
    this.embeddingDevice.destroy();
  }

  private ensureNotReleased(): void {
    if (this.released) throw new Error("Browser model set has been released");
  }
}

async function drainDevices(
  ...devices: readonly (GPUDevice | undefined)[]
): Promise<void> {
  await Promise.allSettled(
    devices.flatMap((device) =>
      device === undefined ? [] : [device.queue.onSubmittedWorkDone()],
    ),
  );
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
  precision: WebGpuModelPrecision = "float16",
): Promise<GPUDevice> {
  if (precision === "float16" && !adapter.features.has("shader-f16")) {
    throw new Error("Direct WebGPU inference requires shader-f16 support");
  }
  return await adapter.requestDevice({
    ...(precision === "float16"
      ? { requiredFeatures: ["shader-f16"] as const }
      : {}),
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

export function selectWebGpuModelPrecision(
  adapters: readonly GPUAdapter[],
  preferFloat16 = true,
): WebGpuModelPrecision {
  return preferFloat16 &&
    adapters.every((adapter) => adapter.features.has("shader-f16"))
    ? "float16"
    : "float32";
}
