import type { OrtModelAsset } from "./ort-backends";

export interface TensorDescriptor {
  name: string;
  dtype: "float32";
  shape: readonly (number | "batch")[];
}

export interface ModelVariant {
  file: string;
  bytes: number;
  sha256: string;
  opset: number;
  precision: string;
  graph: unknown;
  verification: unknown;
}

export interface BrowserModel {
  id: string;
  input: TensorDescriptor;
  output: TensorDescriptor;
  batches: Record<string, ModelVariant>;
  precision_variants?: Record<string, PrecisionVariant>;
  direct_webgpu?: DirectWebGpuCampPlusManifest;
}

export type WebGpuModelPrecision = "float16" | "float32";

const WEBGPU_MODEL_PRECISIONS = ["float16", "float32"] as const;

export interface DirectWebGpuCampPlusPrecisionVariant {
  readonly format:
    | "senko-campplus-direct-webgpu-f16-v1"
    | "senko-campplus-direct-webgpu-f32-v1";
  readonly metadata: FileAssetRecord;
  readonly weights: FileAssetRecord;
  readonly production_batch: 16;
  readonly supported_batches: readonly [4, 8, 16, 32];
  readonly explicit_gpu_buffer_bytes_by_batch: Readonly<Record<string, number>>;
}

export interface DirectWebGpuCampPlusManifest {
  readonly precision_variants: Readonly<
    Record<WebGpuModelPrecision, DirectWebGpuCampPlusPrecisionVariant>
  >;
}

export interface PrecisionVariant {
  internal_dtype: string;
  input_dtype: string;
  output_dtype: string;
  batches: Record<string, ModelVariant>;
  ort_web?: {
    required_graph_optimization_level?: "disabled" | "basic" | "extended" | "all";
    reason?: string;
  };
}

export interface FileAssetRecord {
  file: string;
  bytes: number;
  sha256: string;
}

export interface LstmSplitArtifact {
  format:
    | "senko-persistent-lstm-f16-gc4h"
    | "senko-persistent-lstm-f32-gc4h";
  boundary_layout: "batch,frame,feature";
  frames: 589;
  input_features: 60;
  output_features: 256;
  weights: FileAssetRecord;
  metadata: FileAssetRecord;
}

export interface VadBufferBytes {
  waveform_bytes: number;
  first_convolution_activation_bytes: number;
  frontend_output_bytes: number;
  recurrent_output_bytes: number;
  tail_output_bytes: number;
  two_recurrent_ping_pong_buffers_bytes: number;
  input_affine_scratch_bytes: number;
  hidden_and_cell_state_bytes_per_layer: number;
}

export interface DirectWebGpuVadVariant {
  readonly format:
    | "senko-pyannote-direct-webgpu-f16-v1"
    | "senko-pyannote-direct-webgpu-f32-v1";
  readonly frontend_metadata: FileAssetRecord;
  readonly tail_metadata: FileAssetRecord;
  readonly explicit_gpu_bytes: number;
}

export interface DirectWebGpuVadPrecisionVariant {
  readonly lstm: LstmSplitArtifact;
  readonly batches: Record<string, DirectWebGpuVadVariant>;
}

export interface DirectWebGpuVadManifest {
  readonly precision_variants: Readonly<
    Record<WebGpuModelPrecision, DirectWebGpuVadPrecisionVariant>
  >;
}

export interface SegmentationSplitManifest {
  version: 1;
  boundary_layout: "batch,frame,feature";
  frontend: BrowserModel;
  tail: BrowserModel;
  direct_webgpu: DirectWebGpuVadManifest;
  buffer_bytes_by_batch: Record<string, VadBufferBytes>;
}

export interface BrowserSegmentationModel extends BrowserModel {
  split: SegmentationSplitManifest;
}

export interface BrowserModelManifest {
  version: 1;
  generated_by: Record<string, unknown>;
  models: {
    segmentation: BrowserSegmentationModel;
    campplus: BrowserModel;
  };
  sources: Record<string, unknown>;
}

export interface SelectedModelVariant {
  batchSize: number;
  asset: OrtModelAsset;
  model: BrowserModel;
  variant: ModelVariant;
}

export interface SelectedSegmentationSplit {
  readonly precision: WebGpuModelPrecision;
  batchSize: number;
  frontend: SelectedModelVariant;
  tail: SelectedModelVariant;
  weights: OrtModelAsset;
  metadata: OrtModelAsset;
  directWebGpu: {
    readonly frontendMetadata: OrtModelAsset;
    readonly tailMetadata: OrtModelAsset;
    readonly explicitGpuBytes: number;
  };
  declaredBufferBytes: VadBufferBytes;
  artifact: LstmSplitArtifact;
}

export interface SelectedCampPlusDirect {
  readonly precision: WebGpuModelPrecision;
  readonly batchSize: 4 | 8 | 16 | 32;
  readonly metadata: OrtModelAsset;
  readonly weights: OrtModelAsset;
  readonly explicitGpuBufferBytes: number;
}

export interface ModelManifestIntegrity {
  readonly byteLength: number;
  readonly sha256: string;
}

export async function loadModelManifest(
  url: string,
  integrity?: ModelManifestIntegrity,
  fetchAsset: typeof fetch = fetch,
): Promise<BrowserModelManifest> {
  const response = await fetchAsset(url);
  if (!response.ok) {
    throw new Error(`Failed to load model manifest ${url}: HTTP ${response.status}`);
  }
  const bytes = await response.arrayBuffer();
  if (integrity !== undefined) {
    assertManifestIntegrity(integrity);
    if (bytes.byteLength !== integrity.byteLength) {
      throw new Error(
        `Model manifest ${url} has ${bytes.byteLength} bytes; expected ${integrity.byteLength}`,
      );
    }
    const actualSha256 = bytesToHex(
      await crypto.subtle.digest("SHA-256", bytes),
    );
    if (actualSha256 !== integrity.sha256.toLowerCase()) {
      throw new Error(
        `Model manifest SHA-256 mismatch for ${url}: ${actualSha256}`,
      );
    }
  }
  const value: unknown = JSON.parse(new TextDecoder().decode(bytes));
  assertManifest(value);
  return value;
}

export function selectModelVariant(
  manifestUrl: string,
  model: BrowserModel,
  preferredBatchSize?: number,
): SelectedModelVariant {
  const sizes = Object.keys(model.batches)
    .map(Number)
    .filter((value) => Number.isSafeInteger(value) && value > 0)
    .sort((left, right) => left - right);
  if (sizes.length === 0) throw new Error(`Model '${model.id}' has no batch variants`);

  const batchSize =
    preferredBatchSize === undefined
      ? sizes[sizes.length - 1]!
      : sizes.includes(preferredBatchSize)
        ? preferredBatchSize
        : (() => {
            throw new Error(
              `Model '${model.id}' has no B${preferredBatchSize} variant; available: ${sizes.join(", ")}`,
            );
          })();
  const variant = model.batches[String(batchSize)];
  if (variant === undefined) {
    throw new Error(`Model '${model.id}' has no B${batchSize} variant`);
  }
  return {
    batchSize,
    model,
    variant,
    asset: {
      url: new URL(variant.file, manifestUrl).toString(),
      inputName: model.input.name,
      outputName: model.output.name,
      sha256: variant.sha256,
      byteLength: variant.bytes,
    },
  };
}

export function selectSegmentationSplit(
  manifestUrl: string,
  model: BrowserSegmentationModel,
  batchSize: number,
  precision: WebGpuModelPrecision = "float16",
): SelectedSegmentationSplit {
  const split = model.split;
  const frontend = selectModelVariant(manifestUrl, split.frontend, batchSize);
  const tail = selectModelVariant(manifestUrl, split.tail, batchSize);
  const declaredBufferBytes = split.buffer_bytes_by_batch[String(batchSize)];
  if (declaredBufferBytes === undefined) {
    throw new Error(`Split segmentation has no B${batchSize} buffer accounting`);
  }
  const precisionVariant = split.direct_webgpu.precision_variants[precision];
  const direct = precisionVariant.batches[String(batchSize)];
  if (direct === undefined) {
    throw new Error(
      `Split segmentation has no direct WebGPU ${precision} B${batchSize} package`,
    );
  }
  return {
    precision,
    batchSize,
    frontend,
    tail,
    weights: fileAsset(manifestUrl, precisionVariant.lstm.weights),
    metadata: fileAsset(manifestUrl, precisionVariant.lstm.metadata),
    directWebGpu: {
      frontendMetadata: fileAsset(manifestUrl, direct.frontend_metadata),
      tailMetadata: fileAsset(manifestUrl, direct.tail_metadata),
      explicitGpuBytes: direct.explicit_gpu_bytes,
    },
    declaredBufferBytes,
    artifact: precisionVariant.lstm,
  };
}

export function selectCampPlusDirect(
  manifestUrl: string,
  model: BrowserModel,
  preferredBatchSize?: number,
  precision: WebGpuModelPrecision = "float16",
): SelectedCampPlusDirect {
  const directManifest = model.direct_webgpu;
  if (directManifest === undefined) {
    throw new Error("CAM++ direct WebGPU package is missing");
  }
  const direct = directManifest.precision_variants[precision];
  const batchSize = preferredBatchSize ?? direct.production_batch;
  if (batchSize !== 4 && batchSize !== 8 && batchSize !== 16 && batchSize !== 32) {
    throw new Error(`CAM++ direct WebGPU does not support B${batchSize}`);
  }
  if (!direct.supported_batches.includes(batchSize)) {
    throw new Error(`CAM++ direct WebGPU package does not declare B${batchSize}`);
  }
  const explicitGpuBufferBytes =
    direct.explicit_gpu_buffer_bytes_by_batch[String(batchSize)];
  if (
    typeof explicitGpuBufferBytes !== "number" ||
    !Number.isSafeInteger(explicitGpuBufferBytes) ||
    explicitGpuBufferBytes <= 0
  ) {
    throw new Error(`CAM++ direct WebGPU B${batchSize} memory declaration is missing`);
  }
  return {
    precision,
    batchSize,
    metadata: fileAsset(manifestUrl, direct.metadata),
    weights: fileAsset(manifestUrl, direct.weights),
    explicitGpuBufferBytes,
  };
}

function fileAsset(manifestUrl: string, record: FileAssetRecord): OrtModelAsset {
  return {
    url: new URL(record.file, manifestUrl).toString(),
    byteLength: record.bytes,
    sha256: record.sha256,
  };
}

/** Avoids the 156 MiB stock B32 Sinc activation on 128 MiB adapters. */
export function chooseVadBatchSize(
  model: BrowserModel,
  maxStorageBufferBindingSize: number,
): number {
  const available = Object.keys(model.batches).map(Number).sort((a, b) => a - b);
  const portableMaximum = maxStorageBufferBindingSize < 164_000_000 ? 16 : 32;
  const candidates = available.filter((size) => size <= portableMaximum);
  if (available.length === 0) throw new Error(`Model '${model.id}' has no batch variants`);
  if (candidates.length === 0) return available[0]!;
  return candidates[candidates.length - 1]!;
}

function assertManifest(value: unknown): asserts value is BrowserModelManifest {
  if (typeof value !== "object" || value === null) {
    throw new Error("Model manifest must be an object");
  }
  const candidate = value as Partial<BrowserModelManifest>;
  if (candidate.version !== 1 || candidate.models === undefined) {
    throw new Error("Unsupported or malformed model manifest");
  }
  assertModel(candidate.models.segmentation, "segmentation");
  assertModel(candidate.models.campplus, "campplus");
  assertSegmentationSplit(candidate.models.segmentation.split);
  assertCampPlusDirect(candidate.models.campplus.direct_webgpu);
}

function assertModel(value: unknown, key: string): asserts value is BrowserModel {
  if (typeof value !== "object" || value === null) {
    throw new Error(`Manifest model '${key}' is missing`);
  }
  const model = value as Partial<BrowserModel>;
  if (
    typeof model.id !== "string" ||
    typeof model.input?.name !== "string" ||
    typeof model.output?.name !== "string" ||
    typeof model.batches !== "object" ||
    model.batches === null
  ) {
    throw new Error(`Manifest model '${key}' is malformed`);
  }
  for (const [batch, variant] of Object.entries(model.batches)) {
    assertModelVariant(variant, key, batch);
  }
  if (model.precision_variants !== undefined) {
    if (typeof model.precision_variants !== "object" || model.precision_variants === null) {
      throw new Error(`Manifest model '${key}' precision variants are malformed`);
    }
    for (const [precision, variant] of Object.entries(model.precision_variants)) {
      if (
        typeof variant !== "object" ||
        variant === null ||
        typeof variant.batches !== "object" ||
        variant.batches === null
      ) {
        throw new Error(`Manifest model '${key}' ${precision} variant is malformed`);
      }
      const optimizationLevel =
        variant.ort_web?.required_graph_optimization_level;
      if (
        optimizationLevel !== undefined &&
        !["disabled", "basic", "extended", "all"].includes(optimizationLevel)
      ) {
        throw new Error(
          `Manifest model '${key}' ${precision} has an invalid ORT optimization level`,
        );
      }
      for (const [batch, batchVariant] of Object.entries(variant.batches)) {
        assertModelVariant(batchVariant, `${key}.${precision}`, batch);
      }
    }
  }
}

function assertModelVariant(variant: ModelVariant, key: string, batch: string): void {
    if (
      !/^\d+$/.test(batch) ||
      typeof variant.file !== "string" ||
      !Number.isSafeInteger(variant.bytes) ||
      !/^[0-9a-f]{64}$/i.test(variant.sha256)
    ) {
      throw new Error(`Manifest model '${key}' has a malformed B${batch} variant`);
    }
}

function assertSegmentationSplit(value: unknown): asserts value is SegmentationSplitManifest {
  if (typeof value !== "object" || value === null) {
    throw new Error("Segmentation split manifest is missing");
  }
  const split = value as Partial<SegmentationSplitManifest>;
  if (
    split.version !== 1 ||
    split.boundary_layout !== "batch,frame,feature"
  ) {
    throw new Error("Segmentation split manifest has an unsupported contract");
  }
  assertModel(split.frontend, "segmentation.split.frontend");
  assertModel(split.tail, "segmentation.split.tail");
  if (
    typeof split.direct_webgpu !== "object" ||
    split.direct_webgpu === null ||
    typeof split.direct_webgpu.precision_variants !== "object" ||
    split.direct_webgpu.precision_variants === null
  ) {
    throw new Error("Segmentation direct WebGPU packages are missing");
  }
  for (const precision of WEBGPU_MODEL_PRECISIONS) {
    const precisionVariant = split.direct_webgpu.precision_variants[precision];
    if (
      typeof precisionVariant !== "object" ||
      precisionVariant === null ||
      typeof precisionVariant.batches !== "object" ||
      precisionVariant.batches === null
    ) {
      throw new Error(`Segmentation direct WebGPU ${precision} package is missing`);
    }
    assertLstmSplitArtifact(precisionVariant.lstm, precision);
    for (const [batch, value] of Object.entries(precisionVariant.batches)) {
      if (!/^\d+$/.test(batch) || typeof value !== "object" || value === null) {
        throw new Error(
          `Segmentation direct WebGPU ${precision} B${batch} is malformed`,
        );
      }
      const variant = value as Partial<DirectWebGpuVadVariant>;
      const expectedFormat = `senko-pyannote-direct-webgpu-${precision === "float16" ? "f16" : "f32"}-v1`;
      if (
        variant.format !== expectedFormat ||
        !Number.isSafeInteger(variant.explicit_gpu_bytes) ||
        (variant.explicit_gpu_bytes ?? 0) <= 0
      ) {
        throw new Error(
          `Segmentation direct WebGPU ${precision} B${batch} is malformed`,
        );
      }
      assertFileAsset(
        variant.frontend_metadata,
        `segmentation.split.direct_webgpu.${precision}.${batch}.frontend_metadata`,
      );
      assertFileAsset(
        variant.tail_metadata,
        `segmentation.split.direct_webgpu.${precision}.${batch}.tail_metadata`,
      );
    }
  }
  if (
    typeof split.buffer_bytes_by_batch !== "object" ||
    split.buffer_bytes_by_batch === null
  ) {
    throw new Error("Segmentation split buffer accounting is missing");
  }
}

function assertLstmSplitArtifact(
  value: unknown,
  precision: WebGpuModelPrecision,
): asserts value is LstmSplitArtifact {
  if (typeof value !== "object" || value === null) {
    throw new Error(`Segmentation ${precision} LSTM artifact is missing`);
  }
  const artifact = value as Partial<LstmSplitArtifact>;
  const expectedFormat =
    precision === "float16"
      ? "senko-persistent-lstm-f16-gc4h"
      : "senko-persistent-lstm-f32-gc4h";
  if (
    artifact.format !== expectedFormat ||
    artifact.boundary_layout !== "batch,frame,feature" ||
    artifact.frames !== 589 ||
    artifact.input_features !== 60 ||
    artifact.output_features !== 256
  ) {
    throw new Error(`Segmentation ${precision} LSTM artifact is malformed`);
  }
  assertFileAsset(artifact.weights, `segmentation.${precision}.lstm.weights`);
  assertFileAsset(artifact.metadata, `segmentation.${precision}.lstm.metadata`);
}

function assertCampPlusDirect(value: unknown): asserts value is DirectWebGpuCampPlusManifest {
  if (typeof value !== "object" || value === null) {
    throw new Error("CAM++ direct WebGPU package is missing");
  }
  const direct = value as Partial<DirectWebGpuCampPlusManifest>;
  if (
    typeof direct.precision_variants !== "object" ||
    direct.precision_variants === null
  ) {
    throw new Error("CAM++ direct WebGPU package has an unsupported contract");
  }
  for (const precision of WEBGPU_MODEL_PRECISIONS) {
    const variant = direct.precision_variants[precision];
    const expectedFormat = `senko-campplus-direct-webgpu-${precision === "float16" ? "f16" : "f32"}-v1`;
    if (
      typeof variant !== "object" ||
      variant === null ||
      variant.format !== expectedFormat ||
      variant.production_batch !== 16 ||
      !Array.isArray(variant.supported_batches) ||
      variant.supported_batches.join(",") !== "4,8,16,32" ||
      typeof variant.explicit_gpu_buffer_bytes_by_batch !== "object" ||
      variant.explicit_gpu_buffer_bytes_by_batch === null
    ) {
      throw new Error(`CAM++ direct WebGPU ${precision} package is malformed`);
    }
    assertFileAsset(
      variant.metadata,
      `campplus.direct_webgpu.${precision}.metadata`,
    );
    assertFileAsset(
      variant.weights,
      `campplus.direct_webgpu.${precision}.weights`,
    );
    for (const batch of variant.supported_batches) {
      const bytes = variant.explicit_gpu_buffer_bytes_by_batch[String(batch)];
      if (typeof bytes !== "number" || !Number.isSafeInteger(bytes) || bytes <= 0) {
        throw new Error(
          `CAM++ direct WebGPU ${precision} B${batch} memory declaration is malformed`,
        );
      }
    }
  }
}

function assertFileAsset(value: unknown, name: string): asserts value is FileAssetRecord {
  if (typeof value !== "object" || value === null) {
    throw new Error(`Manifest asset '${name}' is missing`);
  }
  const asset = value as Partial<FileAssetRecord>;
  if (
    typeof asset.file !== "string" ||
    !Number.isSafeInteger(asset.bytes) ||
    !/^[0-9a-f]{64}$/i.test(asset.sha256 ?? "")
  ) {
    throw new Error(`Manifest asset '${name}' is malformed`);
  }
}

function assertManifestIntegrity(
  integrity: ModelManifestIntegrity,
): void {
  if (
    !Number.isSafeInteger(integrity.byteLength) ||
    integrity.byteLength < 0 ||
    !/^[0-9a-f]{64}$/i.test(integrity.sha256)
  ) {
    throw new Error("Model manifest integrity declaration is malformed");
  }
}

function bytesToHex(buffer: ArrayBuffer): string {
  let result = "";
  for (const byte of new Uint8Array(buffer)) {
    result += byte.toString(16).padStart(2, "0");
  }
  return result;
}
