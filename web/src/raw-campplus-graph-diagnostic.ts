/// <reference types="@webgpu/types" />

import {
  CAMPPLUS_RAW_NUMERIC_VARIANTS,
  CampPlusRawGraph,
  campPlusRawRequiredBufferBytes,
  isCampPlusRawNumericVariant,
  type CampPlusRawBatchSize,
  type CampPlusRawNumericVariant,
} from "./pipeline/campplus-webgpu/graph";
import {
  DEFAULT_DENSE_BOTTLENECK_VARIANT,
  DENSE_BOTTLENECK_VARIANTS,
  isDenseBottleneckVariant,
  type DenseBottleneckVariant,
} from "./pipeline/campplus-webgpu/dense-cam";
import {
  DEFAULT_FCM_VARIANT,
  FCM_VARIANTS,
  isFcmVariant,
  type FcmVariant,
} from "./pipeline/campplus-webgpu/fcm";
import {
  DEFAULT_PACKED_BCT_CONV_VARIANT,
  PACKED_BCT_CONV_VARIANTS,
  isPackedBctConvVariant,
  type PackedBctConvVariant,
} from "./pipeline/campplus-webgpu/packed-bct-conv";
import {
  DEFAULT_POINTWISE_TRANSIT_VARIANT,
  POINTWISE_TRANSIT_VARIANTS,
  isPointwiseTransitVariant,
  type PointwiseTransitVariant,
} from "./pipeline/campplus-webgpu/pointwise-transit";
import {
  preferredRawCampPlusDeviceLimits,
  requireRawCampPlusAdapterLimits,
} from "./pipeline/campplus-webgpu/runtime";

const METADATA_URL = "/models/campplus-t150-webgpu-fp16.json";
const REFERENCE_URL = "/models/campplus-t150-b32-reference.f32";
const FRAMES = 150;
const FEATURES = 80;
const EMBEDDINGS = 192;
const REFERENCE_BATCH_SIZE = 32;

interface FloatComparison {
  readonly maxAbsoluteError: number;
  readonly meanAbsoluteError: number;
  readonly cosineSimilarity: number;
}

export async function runRawCampPlusGraphDiagnostic(root: HTMLElement): Promise<void> {
  root.innerHTML = `<section><h1>Raw CAM++ full-graph diagnostic</h1><pre id="raw-campplus-graph-result">Requesting WebGPU…</pre></section>`;
  const output = root.querySelector<HTMLElement>("#raw-campplus-graph-result");
  if (output === null) throw new Error("Missing raw CAM++ graph diagnostic output");
  try {
    const parameters = new URLSearchParams(location.search);
    const batchSize = parseBatchSize(parameters.get("batch"));
    const fcmVariant = parseFcmVariant(parameters.get("fcm-variant"));
    const denseBottleneckVariant = parseDenseBottleneckVariant(
      parameters.get("dense-bottleneck-variant"),
    );
    const tdnnVariant = parseTdnnVariant(parameters.get("tdnn-variant"));
    const pointwiseTransitVariant = parsePointwiseTransitVariant(
      parameters.get("pointwise-transit-variant"),
    );
    const numericVariant = parseNumericVariant(parameters.get("numeric-variant"));
    const result = await execute(
      batchSize,
      fcmVariant,
      denseBottleneckVariant,
      tdnnVariant,
      pointwiseTransitVariant,
      numericVariant,
      (message) => {
        output.textContent = message;
      },
    );
    output.textContent = JSON.stringify(result, null, 2);
    output.dataset.status = result.ok ? "passed" : "failed";
    globalThis.dispatchEvent(
      new CustomEvent("senko-raw-campplus-graph-diagnostic", { detail: result }),
    );
  } catch (error) {
    output.textContent = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    output.dataset.status = "error";
    throw error;
  }
}

async function execute(
  batchSize: CampPlusRawBatchSize,
  fcmVariant: FcmVariant,
  denseBottleneckVariant: DenseBottleneckVariant,
  tdnnVariant: PackedBctConvVariant,
  pointwiseTransitVariant: PointwiseTransitVariant,
  numericVariant: CampPlusRawNumericVariant,
  report: (message: string) => void,
): Promise<Record<string, unknown>> {
  if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null || !adapter.features.has("shader-f16")) {
    throw new Error("A shader-f16 WebGPU adapter is required");
  }
  const requiredBufferBytes = campPlusRawRequiredBufferBytes(batchSize);
  requireRawCampPlusAdapterLimits(adapter, requiredBufferBytes);
  const timestampQuery = adapter.features.has("timestamp-query");
  const requiredFeatures: GPUFeatureName[] = ["shader-f16"];
  if (timestampQuery) requiredFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: preferredRawCampPlusDeviceLimits(
      adapter,
      requiredBufferBytes,
    ),
  });
  let graph: CampPlusRawGraph | undefined;
  try {
    report(
      `Streaming weights and compiling the raw B${batchSize} graph (${numericVariant}, ${fcmVariant}, ${tdnnVariant}, ${denseBottleneckVariant}, ${pointwiseTransitVariant})…`,
    );
    const compileStart = performance.now();
    graph = await CampPlusRawGraph.create(device, METADATA_URL, {
      batchSize,
      fcmVariant,
      denseBottleneckVariant,
      tdnnVariant,
      pointwiseTransitVariant,
      numericVariant,
    });
    const loadAndCompileMs = performance.now() - compileStart;
    const features = deterministicFeatures(batchSize);
    const expected = await fetchReference(batchSize);

    report(`Warming one B${batchSize} 119-dispatch submission…`);
    const warm = await graph.run(features, { timestamps: timestampQuery });
    report(`Timing three B${batchSize} full-graph submissions…`);
    const runs = [];
    let actual = warm.embeddings;
    for (let index = 0; index < 3; index += 1) {
      const run = await graph.run(features, { timestamps: timestampQuery });
      runs.push({ wallMs: run.wallMs, ...(run.gpuMs === undefined ? {} : { gpuMs: run.gpuMs }) });
      actual = run.embeddings;
    }
    const parity = compare(actual, expected);
    report(`Profiling B${batchSize} graph stages…`);
    const profile = timestampQuery ? await graph.profile(features) : undefined;
    const wallValues = runs.map((run) => run.wallMs);
    const gpuValues = runs.flatMap((run) => (run.gpuMs === undefined ? [] : [run.gpuMs]));
    const ok =
      parity.maxAbsoluteError <= 0.08 &&
      parity.meanAbsoluteError <= 0.005 &&
      parity.cosineSimilarity >= 0.999;
    return {
      ok,
      batchSize,
      fcmVariant: graph.fcmVariant,
      denseBottleneckVariant: graph.denseBottleneckVariant,
      tdnnVariant: graph.tdnnVariant,
      pointwiseTransitVariant: graph.pointwiseTransitVariant,
      numericVariant: graph.numericVariant,
      fcmAccumulation: graph.fcmAccumulation,
      denseBottleneckAccumulation: graph.denseBottleneckAccumulation,
      pointwiseTransitAccumulation: graph.pointwiseTransitAccumulation,
      dispatches: graph.dispatchCount,
      commandEncodersPerRun: 1,
      submissionsPerRun: 1,
      loadAndCompileMs,
      timestampQuery,
      warm: { wallMs: warm.wallMs, ...(warm.gpuMs === undefined ? {} : { gpuMs: warm.gpuMs }) },
      runs,
      settled: {
        wallMeanMs: mean(wallValues),
        wallMinMs: Math.min(...wallValues),
        ...(gpuValues.length === 0
          ? {}
          : { gpuMeanMs: mean(gpuValues), gpuMinMs: Math.min(...gpuValues) }),
      },
      ...(profile === undefined ? {} : { profile }),
      parity,
      fingerprint: fingerprint(actual),
      reference: {
        sourceBatchSize: REFERENCE_BATCH_SIZE,
        repeatedRows: batchSize > REFERENCE_BATCH_SIZE,
      },
      explicitGpuBytes: graph.gpuBytes,
      retainedCpuBytes: {
        inputFeatures: features.byteLength,
        expectedEmbeddings: expected.byteLength,
        warmEmbeddings: warm.embeddings.byteLength,
        returnedEmbeddings: actual.byteLength,
        diagnosticRetainedTypedArrays:
          features.byteLength +
          expected.byteLength +
          warm.embeddings.byteLength +
          actual.byteLength,
        diagnosticTransientPeakTypedArrays:
          features.byteLength +
          expected.byteLength +
          warm.embeddings.byteLength +
          actual.byteLength * 2,
        twoInFlightCallerStaging:
          (features.byteLength + actual.byteLength) * 2,
        metadataApproximate: JSON.stringify(graph.foundation.gpuPackage.metadata).length,
        productionBinaryAfterUpload: 0,
      },
    };
  } finally {
    graph?.destroy();
    device.destroy();
  }
}

export function parseBatchSize(value: string | null): CampPlusRawBatchSize {
  if (value === null) return 32;
  const parsed = Number(value);
  if (
    parsed === 4 ||
    parsed === 8 ||
    parsed === 16 ||
    parsed === 32 ||
    parsed === 64
  ) {
    return parsed;
  }
  throw new RangeError("Raw CAM++ batch must be 4, 8, 16, 32, or 64");
}

export function parseNumericVariant(
  value: string | null,
): CampPlusRawNumericVariant {
  if (value === null) return "production";
  if (isCampPlusRawNumericVariant(value)) return value;
  throw new RangeError(
    `Raw CAM++ numeric variant must be one of: ${CAMPPLUS_RAW_NUMERIC_VARIANTS.join(", ")}`,
  );
}

export function parseFcmVariant(value: string | null): FcmVariant {
  if (value === null) return DEFAULT_FCM_VARIANT;
  if (isFcmVariant(value)) return value;
  throw new RangeError(
    `Raw CAM++ FCM variant must be one of: ${FCM_VARIANTS.join(", ")}`,
  );
}

export function parseDenseBottleneckVariant(
  value: string | null,
): DenseBottleneckVariant {
  if (value === null) return DEFAULT_DENSE_BOTTLENECK_VARIANT;
  if (isDenseBottleneckVariant(value)) return value;
  throw new RangeError(
    `Raw CAM++ dense bottleneck variant must be one of: ${DENSE_BOTTLENECK_VARIANTS.join(", ")}`,
  );
}

export function parseTdnnVariant(value: string | null): PackedBctConvVariant {
  if (value === null) return DEFAULT_PACKED_BCT_CONV_VARIANT;
  if (isPackedBctConvVariant(value)) return value;
  throw new RangeError(
    `Raw CAM++ TDNN variant must be one of: ${PACKED_BCT_CONV_VARIANTS.join(", ")}`,
  );
}

export function parsePointwiseTransitVariant(
  value: string | null,
): PointwiseTransitVariant {
  if (value === null) return DEFAULT_POINTWISE_TRANSIT_VARIANT;
  if (isPointwiseTransitVariant(value)) return value;
  throw new RangeError(
    `Raw CAM++ pointwise transit variant must be one of: ${POINTWISE_TRANSIT_VARIANTS.join(", ")}`,
  );
}

export function deterministicFeatures(
  batchSize: CampPlusRawBatchSize,
): Float32Array<ArrayBuffer> {
  const values = new Float32Array(batchSize * FRAMES * FEATURES);
  for (let batch = 0; batch < batchSize; batch += 1) {
    // The checked oracle is B32. B64 repeats those independent rows, which
    // validates all 64 graph lanes without introducing a second model export.
    const referenceBatch = batch % REFERENCE_BATCH_SIZE;
    for (let frame = 0; frame < FRAMES; frame += 1) {
      for (let feature = 0; feature < FEATURES; feature += 1) {
        values[(batch * FRAMES + frame) * FEATURES + feature] =
          Math.sin(referenceBatch * 0.17 + frame * 0.041 + feature * 0.013) * 0.65 +
          Math.cos(referenceBatch * 0.07 - frame * 0.023 + feature * 0.009) * 0.3;
      }
    }
  }
  return values;
}

async function fetchReference(
  batchSize: CampPlusRawBatchSize,
): Promise<Float32Array<ArrayBuffer>> {
  const response = await fetch(REFERENCE_URL);
  if (!response.ok) throw new Error("Unable to load the compact CAM++ graph oracle");
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength !== REFERENCE_BATCH_SIZE * EMBEDDINGS * 4) {
    throw new Error("CAM++ graph oracle has the wrong byte length");
  }
  return repeatReferenceRows(new Float32Array(bytes), batchSize);
}

export function repeatReferenceRows(
  source: Float32Array<ArrayBuffer>,
  batchSize: CampPlusRawBatchSize,
): Float32Array<ArrayBuffer> {
  if (source.length !== REFERENCE_BATCH_SIZE * EMBEDDINGS) {
    throw new RangeError("CAM++ source oracle must contain exactly 32 embedding rows");
  }
  const expanded = new Float32Array(batchSize * EMBEDDINGS);
  for (let batch = 0; batch < batchSize; batch += 1) {
    const sourceStart = (batch % REFERENCE_BATCH_SIZE) * EMBEDDINGS;
    expanded.set(
      source.subarray(sourceStart, sourceStart + EMBEDDINGS),
      batch * EMBEDDINGS,
    );
  }
  return expanded;
}

function compare(actual: Float32Array, expected: Float32Array): FloatComparison {
  if (actual.length !== expected.length) throw new Error("CAM++ output lengths differ");
  let maxAbsoluteError = 0;
  let absoluteError = 0;
  let dot = 0;
  let actualNorm = 0;
  let expectedNorm = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const left = actual[index]!;
    const right = expected[index]!;
    const error = Math.abs(left - right);
    maxAbsoluteError = Math.max(maxAbsoluteError, error);
    absoluteError += error;
    dot += left * right;
    actualNorm += left * left;
    expectedNorm += right * right;
  }
  return {
    maxAbsoluteError,
    meanAbsoluteError: absoluteError / actual.length,
    cosineSimilarity: dot / Math.sqrt(actualNorm * expectedNorm),
  };
}

function fingerprint(values: Float32Array): Record<string, number> {
  let sum = 0;
  let squared = 0;
  let max = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    sum += value;
    squared += value * value;
    max = Math.max(max, value);
  }
  return { sum, l2: Math.sqrt(squared), max };
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
