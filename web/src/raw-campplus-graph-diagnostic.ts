/// <reference types="@webgpu/types" />

import {
  CampPlusRawGraph,
  type CampPlusRawBatchSize,
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
    const pointwiseTransitVariant = parsePointwiseTransitVariant(
      parameters.get("pointwise-transit-variant"),
    );
    const result = await execute(
      batchSize,
      fcmVariant,
      denseBottleneckVariant,
      pointwiseTransitVariant,
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
  pointwiseTransitVariant: PointwiseTransitVariant,
  report: (message: string) => void,
): Promise<Record<string, unknown>> {
  if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null || !adapter.features.has("shader-f16")) {
    throw new Error("A shader-f16 WebGPU adapter is required");
  }
  requireRawCampPlusAdapterLimits(adapter);
  const timestampQuery = adapter.features.has("timestamp-query");
  const requiredFeatures: GPUFeatureName[] = ["shader-f16"];
  if (timestampQuery) requiredFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: preferredRawCampPlusDeviceLimits(adapter),
  });
  let graph: CampPlusRawGraph | undefined;
  try {
    report(
      `Streaming weights and compiling the raw B${batchSize} graph (${fcmVariant}, ${denseBottleneckVariant}, ${pointwiseTransitVariant})…`,
    );
    const compileStart = performance.now();
    graph = await CampPlusRawGraph.create(device, METADATA_URL, {
      batchSize,
      fcmVariant,
      denseBottleneckVariant,
      pointwiseTransitVariant,
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
      pointwiseTransitVariant: graph.pointwiseTransitVariant,
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
      explicitGpuBytes: graph.gpuBytes,
      retainedCpuBytes: {
        inputFeatures: features.byteLength,
        expectedEmbeddings: expected.byteLength,
        returnedEmbeddings: actual.byteLength,
        metadataApproximate: JSON.stringify(graph.foundation.gpuPackage.metadata).length,
        productionBinaryAfterUpload: 0,
      },
    };
  } finally {
    graph?.destroy();
    device.destroy();
  }
}

function parseBatchSize(value: string | null): CampPlusRawBatchSize {
  if (value === null) return 32;
  const parsed = Number(value);
  if (parsed === 4 || parsed === 8 || parsed === 16 || parsed === 32) return parsed;
  throw new RangeError("Raw CAM++ batch must be 4, 8, 16, or 32");
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

export function parsePointwiseTransitVariant(
  value: string | null,
): PointwiseTransitVariant {
  if (value === null) return DEFAULT_POINTWISE_TRANSIT_VARIANT;
  if (isPointwiseTransitVariant(value)) return value;
  throw new RangeError(
    `Raw CAM++ pointwise transit variant must be one of: ${POINTWISE_TRANSIT_VARIANTS.join(", ")}`,
  );
}

function deterministicFeatures(batchSize: number): Float32Array<ArrayBuffer> {
  const values = new Float32Array(batchSize * FRAMES * FEATURES);
  for (let batch = 0; batch < batchSize; batch += 1) {
    for (let frame = 0; frame < FRAMES; frame += 1) {
      for (let feature = 0; feature < FEATURES; feature += 1) {
        values[(batch * FRAMES + frame) * FEATURES + feature] =
          Math.sin(batch * 0.17 + frame * 0.041 + feature * 0.013) * 0.65 +
          Math.cos(batch * 0.07 - frame * 0.023 + feature * 0.009) * 0.3;
      }
    }
  }
  return values;
}

async function fetchReference(batchSize: number): Promise<Float32Array<ArrayBuffer>> {
  const response = await fetch(REFERENCE_URL);
  if (!response.ok) throw new Error("Unable to load the compact CAM++ graph oracle");
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength !== 32 * EMBEDDINGS * 4) {
    throw new Error("CAM++ graph oracle has the wrong byte length");
  }
  return new Float32Array(bytes, 0, batchSize * EMBEDDINGS).slice();
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
