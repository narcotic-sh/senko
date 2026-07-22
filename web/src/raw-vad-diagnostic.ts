/// <reference types="@webgpu/types" />

import { requestMaximumPerformanceAdapter } from "./pipeline/browser-models";
import { loadModelManifest, selectSegmentationSplit } from "./pipeline/model-manifest";
import { configureOrt, OrtVadBackend } from "./pipeline/ort-backends";
import { PersistentWebGpuLstm } from "./pipeline/persistent-lstm";
import { RawPyannoteFrontendFoundation } from "./pipeline/pyannote-frontend-webgpu";
import { RawPyannoteTail } from "./pipeline/pyannote-tail-webgpu";

const BATCH = 8;
const SAMPLES = 160_000;
const FRAMES = 589;
const CLASSES = 7;

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

async function runRaw(
  device: GPUDevice,
  frontend: RawPyannoteFrontendFoundation,
  lstm: PersistentWebGpuLstm,
  tail: RawPyannoteTail,
): Promise<Float32Array> {
  const encoder = device.createCommandEncoder({ label: "senko-raw-vad" });
  frontend.encode(encoder);
  lstm.encode(encoder);
  tail.encode(encoder, true);
  device.queue.submit([encoder.finish()]);
  return tail.readback();
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
  const manifestUrl = new URL("/models/manifest.json", location.href).toString();
  const manifest = await loadModelManifest(manifestUrl);
  const selected = selectSegmentationSplit(
    manifestUrl,
    manifest.models.segmentation,
    BATCH,
  );
  const waveform = new Float32Array(BATCH * SAMPLES);
  fillDeterministic(waveform);

  const referenceBackend = await OrtVadBackend.create(runtime, selected);
  const device = await runtime.device;
  const referenceStarted = performance.now();
  const reference = await referenceBackend.run(waveform);
  report(`split ORT reference ${(performance.now() - referenceStarted).toFixed(2)} ms`);
  await referenceBackend.release();
  await device.queue.onSubmittedWorkDone();
  report("split ORT reference released");

  const frontend = await RawPyannoteFrontendFoundation.create(
    device,
    "/models/pyannote-segmentation-3.0-frontend-webgpu-f16.json",
  );
  const lstm = await PersistentWebGpuLstm.create(
    device,
    BATCH,
    frontend.frontendOutputBuffer,
    selected.weights,
    selected.metadata,
  );
  const tail = await RawPyannoteTail.create(
    device,
    lstm.outputBuffer,
    "/models/pyannote-segmentation-3.0-tail-webgpu-f16.json",
  );
  frontend.uploadWaveform(waveform);
  report(
    `raw resources frontend=${frontend.gpuBytes.total}, lstm=${lstm.bufferBytes.total}, tail=${tail.gpuBytes.total}`,
  );

  const actual = await runRaw(device, frontend, lstm, tail);
  const comparison = parity(reference, actual);
  const outputSha256 = await floatSha256(actual);
  report(
    `parity max_abs=${comparison.maxAbsolute.toExponential(6)}, rms=${comparison.rms.toExponential(6)}, nonfinite=${comparison.nonFinite}, argmax=${comparison.matchingArgmax}/${comparison.totalFrames}`,
  );

  const timings: number[] = [];
  for (let run = 0; run < 10; run += 1) {
    frontend.uploadWaveform(waveform);
    const runStarted = performance.now();
    await runRaw(device, frontend, lstm, tail);
    const elapsed = performance.now() - runStarted;
    timings.push(elapsed);
    report(`raw VAD run ${run + 1}: ${elapsed.toFixed(3)} ms`);
  }
  const settled = timings.slice(3);
  const settledMeanMs = settled.reduce((sum, value) => sum + value, 0) / settled.length;
  const sorted = [...settled].sort((left, right) => left - right);
  const settledMedianMs = sorted[Math.floor(sorted.length / 2)]!;

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
    totalMs: number;
  }> = [];
  for (let profile = 0; profile < 5; profile += 1) {
    frontend.uploadWaveform(waveform);
    const frontendEncoder = device.createCommandEncoder();
    frontend.encode(frontendEncoder);
    device.queue.submit([frontendEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

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
  }

  const gpuBytes = {
    frontend: frontend.gpuBytes.total,
    lstm: lstm.bufferBytes.total,
    tail: tail.gpuBytes.total,
    total: frontend.gpuBytes.total + lstm.bufferBytes.total + tail.gpuBytes.total,
  };
  const ok =
    comparison.nonFinite === 0 &&
    comparison.matchingArgmax === comparison.totalFrames;
  const result = {
    ok,
    outputSha256,
    comparison,
    timings,
    settledMeanMs,
    settledMedianMs,
    projected47BatchMs: settledMeanMs * 47,
    stageProfiles,
    lstmLayerProfiles,
    gpuBytes,
    retainedCpuBytes: {
      frontendMetadataSerialized: new TextEncoder().encode(
        JSON.stringify(frontend.gpuPackage.metadata),
      ).byteLength,
      tailMetadataSerialized: new TextEncoder().encode(
        JSON.stringify({
          ...tail.metadata,
          sections: [...tail.metadata.sections.values()],
        }),
      ).byteLength,
      packedModelBinariesRetained: 0,
      callerOwnedWaveform: waveform.byteLength,
      diagnosticOnlyReferenceAndActual: reference.byteLength + actual.byteLength,
    },
  };
  tail.destroy();
  lstm.release();
  frontend.destroy();
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
