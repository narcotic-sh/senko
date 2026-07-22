import {
  loadModelManifest,
  selectSegmentationSplit,
  selectModelVariant,
  chooseVadBatchSize,
} from "./model-manifest";
import {
  configureOrt,
  OrtEmbeddingBackend,
  OrtVadBackend,
} from "./ort-backends";

export interface ModelBenchmarkOptions {
  manifestUrl: string;
  adapter: GPUAdapter;
  vadChunks?: number;
  embeddingWindows?: number;
  vadBatchSize?: number;
  embeddingBatchSize?: number;
  warmupRuns?: number;
  onProgress?: (stage: string, completed: number, total: number) => void;
}

export interface ModelBenchmarkStage {
  batchSize: number;
  items: number;
  batches: number;
  loadMs: number;
  firstRunMs: number;
  warmRunMs: number;
  projectedMs: number;
  itemsPerSecond: number;
}

export interface ModelBenchmarkResult {
  vad: ModelBenchmarkStage;
  embeddings: ModelBenchmarkStage;
  projectedModelMs: number;
}

/**
 * Runs the exact one-hour model workload with zero-valued inputs. This isolates
 * model scheduling/throughput from audio and clustering and is intentionally
 * usable before the rest of the pipeline is wired up.
 */
export async function benchmarkOrtModels(
  options: ModelBenchmarkOptions,
): Promise<ModelBenchmarkResult> {
  const manifest = await loadModelManifest(options.manifestUrl);
  const limits = options.adapter.limits;
  const vadBatch =
    options.vadBatchSize ??
    chooseVadBatchSize(
      manifest.models.segmentation.split.frontend,
      limits.maxStorageBufferBindingSize,
    );
  const embeddingBatch =
    options.embeddingBatchSize ??
    Math.max(...Object.keys(manifest.models.campplus.batches).map(Number));
  const runtime = configureOrt({
    adapter: options.adapter,
    graphCapture: false,
    strictWebGpu: true,
  });

  const vadSelected = selectSegmentationSplit(
    options.manifestUrl,
    manifest.models.segmentation,
    vadBatch,
  );
  const vadLoadStart = performance.now();
  const vadBackend = await OrtVadBackend.create(runtime, vadSelected);
  const vadLoadMs = performance.now() - vadLoadStart;
  let vad: ModelBenchmarkStage;
  try {
    vad = await benchmarkBackend(
      vadBackend,
      new Float32Array(vadBackend.batchSize * vadBackend.chunkSamples),
      options.vadChunks ?? 370,
      vadLoadMs,
      options.warmupRuns ?? 2,
      "vad",
      options.onProgress,
    );
  } finally {
    await vadBackend.release();
  }

  const embeddingSelected = selectModelVariant(
    options.manifestUrl,
    manifest.models.campplus,
    embeddingBatch,
  );
  const embeddingLoadStart = performance.now();
  const embeddingBackend = await OrtEmbeddingBackend.create(
    runtime,
    embeddingSelected.asset,
    embeddingSelected.batchSize,
  );
  const embeddingLoadMs = performance.now() - embeddingLoadStart;
  let embeddings: ModelBenchmarkStage;
  try {
    embeddings = await benchmarkBackend(
      embeddingBackend,
      new Float32Array(
        embeddingBackend.batchSize * embeddingBackend.frames * embeddingBackend.featureDim,
      ),
      options.embeddingWindows ?? 5_713,
      embeddingLoadMs,
      options.warmupRuns ?? 2,
      "embeddings",
      options.onProgress,
    );
  } finally {
    await embeddingBackend.release();
  }

  return {
    vad,
    embeddings,
    projectedModelMs: vad.projectedMs + embeddings.projectedMs,
  };
}

async function benchmarkBackend(
  backend: { batchSize: number; run(input: Float32Array): Promise<Float32Array> },
  input: Float32Array,
  itemCount: number,
  loadMs: number,
  warmupRuns: number,
  stage: string,
  onProgress?: (stage: string, completed: number, total: number) => void,
): Promise<ModelBenchmarkStage> {
  const firstStart = performance.now();
  await backend.run(input);
  const firstRunMs = performance.now() - firstStart;

  for (let run = 1; run < warmupRuns; run += 1) await backend.run(input);

  const batches = Math.ceil(itemCount / backend.batchSize);
  const start = performance.now();
  for (let batch = 0; batch < batches; batch += 1) {
    await backend.run(input);
    onProgress?.(stage, Math.min((batch + 1) * backend.batchSize, itemCount), itemCount);
  }
  const warmRunMs = performance.now() - start;
  return {
    batchSize: backend.batchSize,
    items: itemCount,
    batches,
    loadMs,
    firstRunMs,
    warmRunMs,
    projectedMs: warmRunMs,
    itemsPerSecond: itemCount / (warmRunMs / 1_000),
  };
}
