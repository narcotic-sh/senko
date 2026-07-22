/// <reference types="@webgpu/types" />

import * as ort from "onnxruntime-web";

import { requestMaximumPerformanceAdapter } from "./pipeline/browser-models";
import { configureOrt } from "./pipeline/ort-backends";
import { RawPyannoteFrontendFoundation } from "./pipeline/pyannote-frontend-webgpu";

const BATCH = 8;
const SAMPLES = 160_000;
const CHANNELS = 60;
const FRAMES = 589;
const OUTPUT_LENGTH = BATCH * CHANNELS * FRAMES;
const OUTPUT_BYTES = OUTPUT_LENGTH * 4;

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

async function runRawWithReadback(
  device: GPUDevice,
  frontend: RawPyannoteFrontendFoundation,
): Promise<Float32Array> {
  const readback = device.createBuffer({
    label: "senko-pyannote-frontend-diagnostic-readback",
    size: OUTPUT_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    frontend.encode(encoder);
    encoder.copyBufferToBuffer(frontend.frontendOutputBuffer, 0, readback, 0, OUTPUT_BYTES);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Float32Array(readback.getMappedRange()).slice();
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

async function runRawGpuOnly(
  device: GPUDevice,
  frontend: RawPyannoteFrontendFoundation,
): Promise<number> {
  const runStarted = performance.now();
  const encoder = device.createCommandEncoder();
  frontend.encode(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - runStarted;
}

function compare(reference: Float32Array, actual: Float32Array): {
  readonly maxAbsolute: number;
  readonly rms: number;
  readonly nonFinite: number;
  readonly referenceSum: number;
  readonly actualSum: number;
} {
  if (reference.length !== actual.length) throw new Error("Parity length mismatch");
  let maxAbsolute = 0;
  let squared = 0;
  let nonFinite = 0;
  let referenceSum = 0;
  let actualSum = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const expected = reference[index]!;
    const observed = actual[index]!;
    const difference = Math.abs(expected - observed);
    referenceSum += expected;
    actualSum += observed;
    if (!Number.isFinite(difference)) nonFinite += 1;
    else {
      maxAbsolute = Math.max(maxAbsolute, difference);
      squared += difference * difference;
    }
  }
  return {
    maxAbsolute,
    rms: Math.sqrt(squared / reference.length),
    nonFinite,
    referenceSum,
    actualSum,
  };
}

async function main(): Promise<void> {
  output.textContent = "0.0 ms  Starting";
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await requestMaximumPerformanceAdapter(navigator.gpu);
  const runtime = configureOrt({
    adapter,
    graphCapture: false,
    graphOptimizationLevel: "all",
    strictWebGpu: true,
  });
  const referenceBytes = await fetch(
    "/models/pyannote-segmentation-3.0-frontend-b8.onnx",
  ).then(async (response) => {
    if (!response.ok) throw new Error(`Reference ONNX request failed: ${response.status}`);
    return response.arrayBuffer();
  });
  // ORT creates the adapter's JSEP device lazily during the first session.
  // Build the short-lived parity session before handing that exact device to
  // the raw runtime; this mirrors production's single-device requirement.
  const referenceSession = await ort.InferenceSession.create(
    referenceBytes,
    runtime.sessionOptions,
  );
  const device = await runtime.device;
  report(
    `device buffer=${device.limits.maxBufferSize}, storage=${device.limits.maxStorageBufferBindingSize}`,
  );
  const frontend = await RawPyannoteFrontendFoundation.create(
    device,
    "/models/pyannote-segmentation-3.0-frontend-webgpu-f16.json",
  );
  const waveform = new Float32Array(BATCH * SAMPLES);
  fillDeterministic(waveform);
  frontend.uploadWaveform(waveform);
  report(`raw resources loaded; gpu=${JSON.stringify(frontend.gpuBytes)}`);

  let reference: Float32Array | undefined;
  try {
    const input = new ort.Tensor("float32", waveform, [BATCH, 1, SAMPLES]);
    try {
      const outputs = await referenceSession.run({ waveform: input });
      const tensor = outputs.features;
      if (tensor === undefined || !(tensor.data instanceof Float32Array)) {
        throw new Error("Reference frontend output is missing or not float32");
      }
      reference = tensor.data.slice();
      tensor.dispose();
    } finally {
      input.dispose();
    }
  } finally {
    await referenceSession.release();
  }
  await device.queue.onSubmittedWorkDone();
  report("ORT reference released");

  const actual = await runRawWithReadback(device, frontend);
  if (reference === undefined) throw new Error("Reference output was not produced");
  const parity = compare(reference, actual);
  report(
    `parity max_abs=${parity.maxAbsolute.toExponential(6)}, rms=${parity.rms.toExponential(6)}, nonfinite=${parity.nonFinite}, sums=${parity.referenceSum.toPrecision(10)}/${parity.actualSum.toPrecision(10)}`,
  );
  reference = undefined;

  const timings: number[] = [];
  for (let run = 0; run < 10; run += 1) {
    const elapsed = await runRawGpuOnly(device, frontend);
    timings.push(elapsed);
    report(`raw run ${run + 1}: ${elapsed.toFixed(3)} ms`);
  }
  const settled = timings.slice(3);
  const mean = settled.reduce((sum, value) => sum + value, 0) / settled.length;
  const sorted = [...settled].sort((left, right) => left - right);
  const median = sorted[Math.floor(sorted.length / 2)]!;
  const stageProfiles: Array<{ sincMs: number; conv1Ms: number; conv2FinalMs: number }> = [];
  for (let profile = 0; profile < 5; profile += 1) {
    frontend.uploadWaveform(waveform);
    const stageTimes: number[] = [];
    for (const encode of [
      (encoder: GPUCommandEncoder) => frontend.encodeSincStage(encoder),
      (encoder: GPUCommandEncoder) => frontend.encodeConv1Stage(encoder),
      (encoder: GPUCommandEncoder) => frontend.encodeConv2AndFinalStage(encoder),
    ]) {
      const stageStarted = performance.now();
      const encoder = device.createCommandEncoder();
      encode(encoder);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      stageTimes.push(performance.now() - stageStarted);
    }
    stageProfiles.push({
      sincMs: stageTimes[0]!,
      conv1Ms: stageTimes[1]!,
      conv2FinalMs: stageTimes[2]!,
    });
  }
  const metadataSerializedBytes = new TextEncoder().encode(
    JSON.stringify(frontend.gpuPackage.metadata),
  ).byteLength;
  const result = {
    parity,
    timings,
    settledMeanMs: mean,
    settledMedianMs: median,
    projected47BatchMs: mean * 47,
    stageProfiles,
    gpuBytes: frontend.gpuBytes,
    retainedCpuBytes: {
      runtimeMetadataSerialized: metadataSerializedBytes,
      modelBinaryRetained: 0,
      callerOwnedWaveform: waveform.byteLength,
      diagnosticOnlyParityOutput: actual.byteLength,
    },
  };
  Object.assign(globalThis, { __senkoPyannoteFrontendDiagnostic: result });
  report(`summary ${JSON.stringify(result)}`);
  frontend.destroy();
  report("raw resources released");
}

void main().catch((error: unknown) => {
  report(`ERROR ${error instanceof Error ? `${error.name}: ${error.message}` : String(error)}`);
  throw error;
});
