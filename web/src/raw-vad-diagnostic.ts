/// <reference types="@webgpu/types" />

import { requestMaximumPerformanceAdapter } from "./pipeline/browser-models";
import {
  loadModelManifest,
  selectModelVariant,
  type BrowserModelManifest,
  type SelectedSegmentationSplit,
} from "./pipeline/model-manifest";
import { configureOrt, OrtVadBackend } from "./pipeline/ort-backends";
import {
  DEFAULT_PERSISTENT_LSTM_VARIANT,
  PersistentWebGpuLstm,
  type PersistentLstmVariant,
} from "./pipeline/persistent-lstm";
import { RawPyannoteFrontendFoundation } from "./pipeline/pyannote-frontend-webgpu";
import { RawPyannoteTail } from "./pipeline/pyannote-tail-webgpu";

const parameters = new URLSearchParams(location.search);
const BATCH = diagnosticBatch(parameters.get("batch"));
const FRONTEND_MODE = diagnosticFrontendMode(parameters.get("frontend"));
const LSTM_VARIANT = diagnosticLstmVariant(parameters.get("lstm-variant"));
const SAMPLES = 160_000;
const FRAMES = 589;
const CLASSES = 7;
const LONG_FILE_CHUNKS = 370;

function diagnosticBatch(source: string | null): 8 | 16 | 32 {
  const batch = source === null ? 8 : Number(source);
  if (batch !== 8 && batch !== 16 && batch !== 32) {
    throw new Error(`Raw VAD diagnostic batch must be 8, 16, or 32; received ${source}`);
  }
  return batch;
}

function diagnosticFrontendMode(
  source: string | null,
): "baseline" | "production" {
  const mode = source ?? "production";
  if (mode !== "baseline" && mode !== "production") {
    throw new Error(
      `Raw VAD frontend mode must be baseline or production; received ${source}`,
    );
  }
  return mode;
}

function diagnosticLstmVariant(source: string | null): PersistentLstmVariant {
  const variant = source ?? DEFAULT_PERSISTENT_LSTM_VARIANT;
  if (
    variant !== "persistent" &&
    variant !== "input-affine-tile4" &&
    variant !== "input-affine-tile8"
  ) {
    throw new Error(
      `Raw VAD LSTM variant must be input-affine-tile8, input-affine-tile4, or persistent; received ${source}`,
    );
  }
  return variant;
}

const element = document.querySelector<HTMLPreElement>("#output");
if (element === null) throw new Error("Missing diagnostic output");
const output = element;
const started = performance.now();

function report(message: string): void {
  output.textContent += `\n${(performance.now() - started).toFixed(1)} ms  ${message}`;
}

function fillDeterministic(values: Float32Array): void {
  let state = 0x5e4b0;
  for (let index = 0; index < values.length; index += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    values[index] = (state / 0xffff_ffff - 0.5) * 0.4;
  }
}

async function floatSha256(values: Float32Array): Promise<string> {
  const bytes =
    values.buffer instanceof ArrayBuffer
      ? values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength)
      : new Uint8Array(
          new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
        ).buffer;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  let result = "";
  for (const byte of digest) result += byte.toString(16).padStart(2, "0");
  return result;
}

function parity(reference: Float32Array, actual: Float32Array): {
  readonly maxAbsolute: number;
  readonly rms: number;
  readonly nonFinite: number;
  readonly matchingArgmax: number;
  readonly totalFrames: number;
} {
  if (reference.length !== actual.length) throw new Error("VAD parity length mismatch");
  let maxAbsolute = 0;
  let squared = 0;
  let nonFinite = 0;
  let matchingArgmax = 0;
  const totalFrames = BATCH * FRAMES;
  for (let frame = 0; frame < totalFrames; frame += 1) {
    const base = frame * CLASSES;
    let expectedClass = 0;
    let actualClass = 0;
    for (let candidate = 0; candidate < CLASSES; candidate += 1) {
      const expected = reference[base + candidate]!;
      const observed = actual[base + candidate]!;
      const difference = Math.abs(expected - observed);
      if (!Number.isFinite(difference)) nonFinite += 1;
      else {
        maxAbsolute = Math.max(maxAbsolute, difference);
        squared += difference * difference;
      }
      if (expected > reference[base + expectedClass]!) expectedClass = candidate;
      if (observed > actual[base + actualClass]!) actualClass = candidate;
    }
    if (expectedClass === actualClass) matchingArgmax += 1;
  }
  return {
    maxAbsolute,
    rms: Math.sqrt(squared / reference.length),
    nonFinite,
    matchingArgmax,
    totalFrames,
  };
}

function selectDiagnosticSplit(
  manifestUrl: string,
  manifest: BrowserModelManifest,
  batchSize: number,
): SelectedSegmentationSplit {
  const split = manifest.models.segmentation.split;
  const declaredBufferBytes = split.buffer_bytes_by_batch[String(batchSize)];
  if (declaredBufferBytes === undefined) {
    throw new Error(`Split segmentation has no B${batchSize} buffer accounting`);
  }
  const asset = (record: { file: string; bytes: number; sha256: string }) => ({
    url: new URL(record.file, manifestUrl).toString(),
    byteLength: record.bytes,
    sha256: record.sha256,
  });
  return {
    batchSize,
    frontend: selectModelVariant(manifestUrl, split.frontend, batchSize),
    tail: selectModelVariant(manifestUrl, split.tail, batchSize),
    weights: asset(split.lstm.weights),
    metadata: asset(split.lstm.metadata),
    // The isolated diagnostic loads its direct packages from a batch-specific
    // directory, so no production-manifest direct entry is required.
    directWebGpu: {
      frontendMetadata: { url: "diagnostic-only" },
      tailMetadata: { url: "diagnostic-only" },
      explicitGpuBytes: 1,
    },
    declaredBufferBytes,
    artifact: split.lstm,
  };
}

interface TimestampResources {
  readonly querySet: GPUQuerySet;
  readonly resolve: GPUBuffer;
  readonly readback: GPUBuffer;
}

interface RawRunMeasurement {
  readonly output: Float32Array;
  readonly wallMs: number;
  readonly gpuMs?: number;
}

function createTimestampResources(device: GPUDevice): TimestampResources | undefined {
  if (!device.features.has("timestamp-query")) return undefined;
  return {
    querySet: device.createQuerySet({ type: "timestamp", count: 2 }),
    resolve: device.createBuffer({
      label: "senko-raw-vad-timestamp-resolve",
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    }),
    readback: device.createBuffer({
      label: "senko-raw-vad-timestamp-readback",
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
  };
}

async function runRaw(
  device: GPUDevice,
  frontend: RawPyannoteFrontendFoundation,
  lstm: PersistentWebGpuLstm,
  tail: RawPyannoteTail,
  timestamps?: TimestampResources,
): Promise<RawRunMeasurement> {
  const encoder = device.createCommandEncoder({ label: "senko-raw-vad" });
  frontend.encode(
    encoder,
    timestamps === undefined
      ? undefined
      : { querySet: timestamps.querySet, beginningOfPassWriteIndex: 0 },
  );
  lstm.encode(encoder);
  tail.encode(
    encoder,
    true,
    timestamps === undefined
      ? undefined
      : { querySet: timestamps.querySet, endOfPassWriteIndex: 1 },
  );
  if (timestamps !== undefined) {
    encoder.resolveQuerySet(timestamps.querySet, 0, 2, timestamps.resolve, 0);
    encoder.copyBufferToBuffer(timestamps.resolve, 0, timestamps.readback, 0, 16);
  }
  const wallStarted = performance.now();
  device.queue.submit([encoder.finish()]);
  const [result] = await Promise.all([
    tail.readback(),
    timestamps?.readback.mapAsync(GPUMapMode.READ),
  ]);
  const wallMs = performance.now() - wallStarted;
  if (timestamps === undefined) return { output: result, wallMs };
  try {
    const values = new BigUint64Array(timestamps.readback.getMappedRange());
    return {
      output: result,
      wallMs,
      gpuMs: Number(values[1]! - values[0]!) / 1_000_000,
    };
  } finally {
    timestamps.readback.unmap();
  }
}

async function main(): Promise<void> {
  output.textContent = `0.0 ms  Starting raw VAD B${BATCH} (${LSTM_VARIANT})`;
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await requestMaximumPerformanceAdapter(navigator.gpu);
  if (!adapter.features.has("shader-f16")) {
    throw new Error("Raw VAD requires shader-f16");
  }
  const timestampQuery = adapter.features.has("timestamp-query");
  const requiredFeatures: GPUFeatureName[] = ["shader-f16"];
  if (timestampQuery) requiredFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures,
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
  const referenceAdapter = await requestMaximumPerformanceAdapter(navigator.gpu);
  const runtime = configureOrt({
    adapter: referenceAdapter,
    graphCapture: false,
    graphOptimizationLevel: "all",
    strictWebGpu: true,
  });
  const manifestUrl = new URL("/models/manifest.json", location.href).toString();
  const manifest = await loadModelManifest(manifestUrl);
  const selected = selectDiagnosticSplit(manifestUrl, manifest, BATCH);
  const waveform = new Float32Array(BATCH * SAMPLES);
  fillDeterministic(waveform);

  const directDirectory = BATCH === 8 ? "/models" : `/models/diagnostic-vad-b${BATCH}`;
  const rawLoadStarted = performance.now();
  const frontend = await RawPyannoteFrontendFoundation.create(
    device,
    `${directDirectory}/pyannote-segmentation-3.0-frontend-webgpu-f16.json`,
    {
      sincAccumulationSchedule:
        FRONTEND_MODE === "baseline" ? "serial" : "interleaved",
      convActivationTilePrecision:
        FRONTEND_MODE === "baseline" ? "float32" : "float16",
    },
  );
  const lstm = await PersistentWebGpuLstm.create(
    device,
    BATCH,
    frontend.frontendOutputBuffer,
    selected.weights,
    selected.metadata,
    undefined,
    LSTM_VARIANT,
  );
  const tail = await RawPyannoteTail.create(
    device,
    lstm.outputBuffer,
    `${directDirectory}/pyannote-segmentation-3.0-tail-webgpu-f16.json`,
  );
  const rawLoadMs = performance.now() - rawLoadStarted;
  report(
    `raw resources loaded in ${rawLoadMs.toFixed(3)} ms; frontend=${frontend.gpuBytes.total}, lstm=${lstm.bufferBytes.total}, tail=${tail.gpuBytes.total}`,
  );

  const timestamps = createTimestampResources(device);
  frontend.uploadWaveform(waveform);
  const warm = await runRaw(device, frontend, lstm, tail, timestamps);
  const actual = warm.output;
  report(
    `raw warmup wall=${warm.wallMs.toFixed(3)} ms${warm.gpuMs === undefined ? "" : `, gpu=${warm.gpuMs.toFixed(3)} ms`}`,
  );

  const timings: number[] = [];
  const gpuTimings: number[] = [];
  for (let run = 0; run < 10; run += 1) {
    frontend.uploadWaveform(waveform);
    const measured = await runRaw(device, frontend, lstm, tail, timestamps);
    timings.push(measured.wallMs);
    if (measured.gpuMs !== undefined) gpuTimings.push(measured.gpuMs);
    report(
      `raw VAD run ${run + 1}: wall=${measured.wallMs.toFixed(3)} ms${measured.gpuMs === undefined ? "" : `, gpu=${measured.gpuMs.toFixed(3)} ms`}`,
    );
  }
  const settled = timings.slice(3);
  const settledMeanMs = settled.reduce((sum, value) => sum + value, 0) / settled.length;
  const sorted = [...settled].sort((left, right) => left - right);
  const settledMedianMs = sorted[Math.floor(sorted.length / 2)]!;
  const settledGpu = gpuTimings.slice(3);
  const settledGpuMeanMs =
    settledGpu.length === 0
      ? undefined
      : settledGpu.reduce((sum, value) => sum + value, 0) / settledGpu.length;
  const settledGpuMedianMs =
    settledGpu.length === 0
      ? undefined
      : [...settledGpu].sort((left, right) => left - right)[
          Math.floor(settledGpu.length / 2)
        ];

  const stageProfiles: Array<{
    frontendMs: number;
    lstmMs: number;
    tailReadbackMs: number;
  }> = [];
  for (let profile = 0; profile < 5; profile += 1) {
    frontend.uploadWaveform(waveform);
    const frontendStarted = performance.now();
    const frontendEncoder = device.createCommandEncoder();
    frontend.encode(frontendEncoder);
    device.queue.submit([frontendEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const frontendMs = performance.now() - frontendStarted;

    const lstmStarted = performance.now();
    const lstmEncoder = device.createCommandEncoder();
    lstm.encode(lstmEncoder);
    device.queue.submit([lstmEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const lstmMs = performance.now() - lstmStarted;

    const tailStarted = performance.now();
    const tailEncoder = device.createCommandEncoder();
    tail.encode(tailEncoder, true);
    device.queue.submit([tailEncoder.finish()]);
    await tail.readback();
    const tailReadbackMs = performance.now() - tailStarted;
    stageProfiles.push({ frontendMs, lstmMs, tailReadbackMs });
  }

  const lstmLayerProfiles: Array<{
    layerMs: readonly [number, number, number, number];
    inputAffineMs?: readonly [number, number, number, number];
    recurrentMs?: readonly [number, number, number, number];
    totalMs: number;
  }> = [];
  for (let profile = 0; profile < 5; profile += 1) {
    frontend.uploadWaveform(waveform);
    const frontendEncoder = device.createCommandEncoder();
    frontend.encode(frontendEncoder);
    device.queue.submit([frontendEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    if (LSTM_VARIANT === "persistent") {
      const measured: number[] = [];
      for (let layer = 0; layer < 4; layer += 1) {
        const layerStarted = performance.now();
        const layerEncoder = device.createCommandEncoder();
        lstm.encodeLayer(layerEncoder, layer);
        device.queue.submit([layerEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        measured.push(performance.now() - layerStarted);
      }
      const layerMs = [
        measured[0]!,
        measured[1]!,
        measured[2]!,
        measured[3]!,
      ] as const;
      lstmLayerProfiles.push({
        layerMs,
        totalMs: layerMs.reduce((sum, value) => sum + value, 0),
      });
    } else {
      const inputAffineMeasured: number[] = [];
      const recurrentMeasured: number[] = [];
      for (let layer = 0; layer < 4; layer += 1) {
        const inputAffineStarted = performance.now();
        const inputAffineEncoder = device.createCommandEncoder();
        lstm.encodeInputAffineLayer(inputAffineEncoder, layer);
        device.queue.submit([inputAffineEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        inputAffineMeasured.push(performance.now() - inputAffineStarted);

        const recurrentStarted = performance.now();
        const recurrentEncoder = device.createCommandEncoder();
        lstm.encodeRecurrentLayer(recurrentEncoder, layer);
        device.queue.submit([recurrentEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        recurrentMeasured.push(performance.now() - recurrentStarted);
      }
      const inputAffineMs = [
        inputAffineMeasured[0]!,
        inputAffineMeasured[1]!,
        inputAffineMeasured[2]!,
        inputAffineMeasured[3]!,
      ] as const;
      const recurrentMs = [
        recurrentMeasured[0]!,
        recurrentMeasured[1]!,
        recurrentMeasured[2]!,
        recurrentMeasured[3]!,
      ] as const;
      const layerMs = [
        inputAffineMs[0] + recurrentMs[0],
        inputAffineMs[1] + recurrentMs[1],
        inputAffineMs[2] + recurrentMs[2],
        inputAffineMs[3] + recurrentMs[3],
      ] as const;
      lstmLayerProfiles.push({
        layerMs,
        inputAffineMs,
        recurrentMs,
        totalMs: layerMs.reduce((sum, value) => sum + value, 0),
      });
    }
  }

  const gpuBytes = {
    frontend: frontend.gpuBytes.total,
    lstm: lstm.bufferBytes.total,
    tail: tail.gpuBytes.total,
    total: frontend.gpuBytes.total + lstm.bufferBytes.total + tail.gpuBytes.total,
  };
  const frontendMetadataSerializedBytes = new TextEncoder().encode(
    JSON.stringify(frontend.gpuPackage.metadata),
  ).byteLength;
  const tailMetadataSerializedBytes = new TextEncoder().encode(
    JSON.stringify({
      ...tail.metadata,
      sections: [...tail.metadata.sections.values()],
    }),
  ).byteLength;
  if (timestamps !== undefined) {
    timestamps.readback.destroy();
    timestamps.resolve.destroy();
    timestamps.querySet.destroy();
  }
  tail.destroy();
  lstm.release();
  frontend.destroy();
  await device.queue.onSubmittedWorkDone();
  report("raw resources released");

  const referenceLoadStarted = performance.now();
  const referenceBackend = await OrtVadBackend.create(runtime, selected);
  const referenceLoadMs = performance.now() - referenceLoadStarted;
  const referenceStarted = performance.now();
  const reference = await referenceBackend.run(waveform);
  const referenceRunMs = performance.now() - referenceStarted;
  report(
    `split ORT reference load=${referenceLoadMs.toFixed(2)} ms, run=${referenceRunMs.toFixed(2)} ms`,
  );
  await referenceBackend.release();
  const referenceDevice = await runtime.device;
  await referenceDevice.queue.onSubmittedWorkDone();

  const comparison = parity(reference, actual);
  const outputSha256 = await floatSha256(actual);
  report(
    `parity max_abs=${comparison.maxAbsolute.toExponential(6)}, rms=${comparison.rms.toExponential(6)}, nonfinite=${comparison.nonFinite}, argmax=${comparison.matchingArgmax}/${comparison.totalFrames}`,
  );
  const longBatchRuns = Math.ceil(LONG_FILE_CHUNKS / BATCH);
  const longPaddedChunks = longBatchRuns * BATCH - LONG_FILE_CHUNKS;
  const projectedLongSteadyWallMs = settledMeanMs * longBatchRuns;
  const projectedLongFirstUseWallMs =
    rawLoadMs + warm.wallMs + projectedLongSteadyWallMs;
  const ok =
    comparison.nonFinite === 0 &&
    comparison.matchingArgmax === comparison.totalFrames;
  const result = {
    ok,
    batchSize: BATCH,
    frontendMode: FRONTEND_MODE,
    lstmVariant: LSTM_VARIANT,
    timestampQuery,
    outputSha256,
    comparison,
    rawLoadMs,
    warmup: warm.gpuMs === undefined
      ? { wallMs: warm.wallMs }
      : { wallMs: warm.wallMs, gpuMs: warm.gpuMs },
    reference: { loadMs: referenceLoadMs, runMs: referenceRunMs },
    timings,
    gpuTimings,
    settledMeanMs,
    settledMedianMs,
    settledGpuMeanMs,
    settledGpuMedianMs,
    projectedLongFile: {
      chunks: LONG_FILE_CHUNKS,
      batchRuns: longBatchRuns,
      processedStaticChunks: longBatchRuns * BATCH,
      paddedChunks: longPaddedChunks,
      steadyWallMs: projectedLongSteadyWallMs,
      firstUseWallMs: projectedLongFirstUseWallMs,
    },
    stageProfiles,
    lstmLayerProfiles,
    gpuBytes,
    retainedCpuBytes: {
      frontendMetadataSerialized: frontendMetadataSerializedBytes,
      tailMetadataSerialized: tailMetadataSerializedBytes,
      packedModelBinariesRetained: 0,
      callerOwnedWaveform: waveform.byteLength,
      diagnosticOnlyReferenceAndActual: reference.byteLength + actual.byteLength,
      productionWavReadBuffer: 320_000,
      productionVadLogits: BATCH * FRAMES * CLASSES * 4,
      productionLogicalLive:
        320_000 + waveform.byteLength + BATCH * FRAMES * CLASSES * 4,
      productionFixedWasmHeaps: 12_058_624,
    },
  };
  referenceDevice.destroy();
  device.destroy();
  Object.assign(globalThis, { __senkoRawVadDiagnostic: result });
  output.textContent = JSON.stringify(result, null, 2);
  output.dataset.status = ok ? "passed" : "failed";
  globalThis.dispatchEvent(
    new CustomEvent("senko-raw-vad-diagnostic", { detail: result }),
  );
}

void main().catch((error: unknown) => {
  const failure = {
    ok: false,
    error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
  };
  Object.assign(globalThis, { __senkoRawVadDiagnostic: failure });
  output.textContent = JSON.stringify(failure, null, 2);
  output.dataset.status = "error";
  globalThis.dispatchEvent(
    new CustomEvent("senko-raw-vad-diagnostic", { detail: failure }),
  );
  console.error(failure.error);
});
