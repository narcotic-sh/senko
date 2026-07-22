/// <reference types="@webgpu/types" />

import {
  configureOrt,
  OrtEmbeddingBackend,
  OrtMonolithicVadBackend,
  OrtVadBackend,
  type OrtModelAsset,
  type OrtRuntime,
} from "./pipeline/ort-backends";
import { requestMaximumPerformanceAdapter } from "./pipeline/browser-models";
import {
  loadModelManifest,
  selectSegmentationSplit,
} from "./pipeline/model-manifest";

const outputElement = document.querySelector<HTMLPreElement>("#output");
if (outputElement === null) throw new Error("Missing diagnostic output");
const output = outputElement;

const started = performance.now();
function report(message: string): void {
  output.textContent += `\n${(performance.now() - started).toFixed(1)} ms  ${message}`;
  output.dataset.last = message;
}

async function reportMemory(label: string): Promise<void> {
  if (new URL(location.href).searchParams.get("memory") !== "1") return;
  const memoryPerformance = performance as Performance & {
    measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }>;
  };
  if (memoryPerformance.measureUserAgentSpecificMemory === undefined) return;
  try {
    const measurement = await memoryPerformance.measureUserAgentSpecificMemory();
    report(`${label} UA memory: ${(measurement.bytes / (1024 * 1024)).toFixed(1)} MiB`);
  } catch (error: unknown) {
    report(
      `${label} UA memory unavailable: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

async function holdResources(parameters: URLSearchParams): Promise<void> {
  const holdMs = Number(parameters.get("hold") ?? "0");
  if (!Number.isSafeInteger(holdMs) || holdMs < 0 || holdMs > 60_000) {
    throw new Error("hold must be an integer from 0 through 60000 milliseconds");
  }
  if (holdMs === 0) return;
  report(`Holding model resources for ${holdMs} ms`);
  await new Promise<void>((resolve) => setTimeout(resolve, holdMs));
}

async function reportOrtDevice(runtime: OrtRuntime): Promise<void> {
  const device = await runtime.device;
  report(
    `ORT device: buffer=${device.limits.maxBufferSize}, storage=${device.limits.maxStorageBufferBindingSize}, f16=${device.features.has("shader-f16")}`,
  );
}

function reportOutput(label: string, values: Float32Array): void {
  Object.assign(globalThis, { __senkoDiagnosticOutput: values });
  let sum = 0;
  let squareSum = 0;
  let weightedSum = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    sum += value;
    squareSum += value * value;
    weightedSum += value * ((index % 251) + 1);
  }
  report(
    `${label} output: sum=${sum.toPrecision(10)}, weighted=${weightedSum.toPrecision(10)}, l2=${Math.sqrt(squareSum).toPrecision(10)}, first=${values[0]!.toPrecision(10)}`,
  );
}

function reportTimingSummary(label: string, timings: readonly number[]): void {
  if (timings.length <= 1) return;
  const settledStart = Math.min(3, timings.length - 1);
  const steady = timings.slice(settledStart);
  const mean = steady.reduce((sum, value) => sum + value, 0) / steady.length;
  const sorted = [...steady].sort((left, right) => left - right);
  const median = sorted[Math.floor(sorted.length / 2)]!;
  report(
    `${label} summary: first=${timings[0]!.toFixed(1)} ms, settled mean=${mean.toFixed(1)} ms, median=${median.toFixed(1)} ms (${steady.length} runs)`,
  );
}

function reportComparison(
  reference: Float32Array,
  actual: Float32Array,
  batchSize: number,
): void {
  if (reference.length !== actual.length) {
    throw new Error(`Comparison length mismatch: ${reference.length} versus ${actual.length}`);
  }
  let maxAbsolute = 0;
  let squaredError = 0;
  let nonFinite = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const difference = Math.abs(reference[index]! - actual[index]!);
    if (!Number.isFinite(difference)) nonFinite += 1;
    else {
      maxAbsolute = Math.max(maxAbsolute, difference);
      squaredError += difference * difference;
    }
  }
  let argmaxMismatches = 0;
  const frames = batchSize * 589;
  for (let frame = 0; frame < frames; frame += 1) {
    const offset = frame * 7;
    let referenceClass = 0;
    let actualClass = 0;
    for (let candidate = 1; candidate < 7; candidate += 1) {
      if (reference[offset + candidate]! > reference[offset + referenceClass]!) {
        referenceClass = candidate;
      }
      if (actual[offset + candidate]! > actual[offset + actualClass]!) {
        actualClass = candidate;
      }
    }
    if (referenceClass !== actualClass) argmaxMismatches += 1;
  }
  report(
    `Elementwise parity: max_abs=${maxAbsolute.toExponential(6)}, rms=${Math.sqrt(squaredError / reference.length).toExponential(6)}, nonfinite=${nonFinite}, argmax=${frames - argmaxMismatches}/${frames} (${argmaxMismatches} mismatches)`,
  );
}

function fillDeterministicInput(values: Float32Array): void {
  let state = 0x5e4b0;
  for (let index = 0; index < values.length; index += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    values[index] = (state / 0xffff_ffff - 0.5) * 2;
  }
}

function formatMib(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

async function main(): Promise<void> {
  output.textContent = "0.0 ms  Starting";
  for (const delay of [1_000, 5_000, 15_000]) {
    setTimeout(() => {
      const resources = performance
        .getEntriesByType("resource")
        .filter((entry) => /(?:onnx|wasm)/.test(entry.name))
        .map((entry) => `${entry.name.split("/").at(-1)}:${entry.duration.toFixed(0)}ms`)
        .join(", ");
      report(`heartbeat ${delay} ms; resources=[${resources}]`);
    }, delay);
  }
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await requestMaximumPerformanceAdapter(navigator.gpu);
  const parameters = new URL(location.href).searchParams;
  const model = parameters.get("model") ?? "vad-monolithic";
  const splitModel = model === "vad-split" || model === "vad-compare";
  const splitCapture = splitModel && parameters.get("capture") === "1";
  report(
    `GPU adapter: buffer=${adapter.limits.maxBufferSize}, storage=${adapter.limits.maxStorageBufferBindingSize}, f16=${adapter.features.has("shader-f16")}`,
  );
  const runtime = configureOrt({
    adapter,
    graphCapture: splitCapture || parameters.get("capture") === "1",
    graphOptimizationLevel:
      parameters.get("opt") === "disabled"
        ? "disabled"
        : parameters.get("opt") === "basic"
          ? "basic"
          : "all",
    logLevel: parameters.get("verbose") === "1" ? "verbose" : "warning",
    strictWebGpu: splitModel || parameters.get("fallback") !== "1",
  });
  report(
    `ORT configured; graph capture=${runtime.graphCapture ? "on" : "off"}, optimization=${parameters.get("opt") ?? "all"}`,
  );
  await reportMemory("baseline");
  const measuredRuns = Number(parameters.get("runs") ?? "8");
  if (!Number.isSafeInteger(measuredRuns) || measuredRuns < 1) {
    throw new Error("runs must be a positive integer");
  }
  if (model === "cam") {
    const batchSize = Number(parameters.get("batch") ?? "32");
    if (![32, 64, 128].includes(batchSize)) throw new Error("Invalid CAM++ batch");
    const embedding = await OrtEmbeddingBackend.create(
      runtime,
      {
        url: `/models/campplus-t150-b${batchSize}${parameters.get("fp16") === "1" ? "-fp16" : ""}.onnx`,
        inputName: "features",
        outputName: "embeddings",
      },
      batchSize,
      report,
    );
    try {
      await reportOrtDevice(runtime);
      await reportMemory("model loaded");
      const input = new Float32Array(
        embedding.batchSize * embedding.frames * embedding.featureDim,
      );
      if (parameters.get("verify") === "1") {
        let state = 0x5e4b0;
        for (let index = 0; index < input.length; index += 1) {
          state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
          input[index] = (state / 0xffff_ffff - 0.5) * 2;
        }
      }
      const timings: number[] = [];
      let lastOutput: Float32Array | undefined;
      for (let run = 0; run < measuredRuns; run += 1) {
        // Exercise the persistent upload rather than replaying constant data.
        input[0] = run;
        const runStarted = performance.now();
        lastOutput = await embedding.run(input);
        const elapsed = performance.now() - runStarted;
        timings.push(elapsed);
        report(`CAM++ run ${run + 1}: ${elapsed.toFixed(1)} ms`);
      }
      if (lastOutput !== undefined) {
        reportOutput("CAM++", lastOutput);
      }
      if (timings.length > 1) {
        // Capture/build and the first couple of replays are intentionally kept
        // out of the settled statistic. On Apple Silicon those replays ramp from
        // very short scheduling-only work to the repeatable GPU duration.
        const settledStart = Math.min(3, timings.length - 1);
        const steady = timings.slice(settledStart);
        const mean = steady.reduce((sum, value) => sum + value, 0) / steady.length;
        const sorted = [...steady].sort((left, right) => left - right);
        const median = sorted[Math.floor(sorted.length / 2)]!;
        report(
          `CAM++ summary: first=${timings[0]!.toFixed(1)} ms, settled mean=${mean.toFixed(1)} ms, median=${median.toFixed(1)} ms (${steady.length} runs)`,
        );
      }
      await reportMemory("after runs");
      await holdResources(parameters);
    } finally {
      await embedding.release();
      report("CAM++ session and buffers released");
      await reportMemory("released");
    }
    return;
  }
  if (splitModel) {
    const batchSize = Number(parameters.get("batch") ?? "1");
    if (![1, 8, 16, 32].includes(batchSize)) throw new Error("Invalid split VAD batch");
    const manifestUrl = new URL("/models/manifest.json", location.href).toString();
    const manifest = await loadModelManifest(manifestUrl);
    const baselineSelected = selectSegmentationSplit(
      manifestUrl,
      manifest.models.segmentation,
      batchSize,
    );
    const lstmPrecision = parameters.get("lstm") ?? "f16";
    if (lstmPrecision !== "f32" && lstmPrecision !== "f16") {
      throw new Error("lstm must be f32 or f16");
    }
    const selected =
      lstmPrecision === "f16"
        ? baselineSelected
        : {
            ...baselineSelected,
            weights: {
              url: new URL(
                "pyannote-segmentation-3.0-lstm-f32.bin",
                manifestUrl,
              ).toString(),
            },
            metadata: {
              url: new URL(
                "pyannote-segmentation-3.0-lstm.json",
                manifestUrl,
              ).toString(),
            },
          };
    report(`Split VAD LSTM weights: ${lstmPrecision}`);
    const input = new Float32Array(batchSize * 160_000);
    if (parameters.get("verify") === "1") fillDeterministicInput(input);
    let reference: Float32Array | undefined;
    if (model === "vad-compare") {
      const referenceRuntime = configureOrt({
        adapter,
        graphCapture: false,
        graphOptimizationLevel:
          parameters.get("opt") === "disabled"
            ? "disabled"
            : parameters.get("opt") === "basic"
              ? "basic"
              : "all",
        strictWebGpu: false,
      });
      const monolithic = await OrtMonolithicVadBackend.create(
        referenceRuntime,
        {
          url: `/models/pyannote-segmentation-3.0-logits-b${batchSize}.onnx`,
          inputName: "waveform",
          outputName: "logits",
        },
        batchSize,
        (message) => report(`Reference: ${message}`),
      );
      try {
        const referenceStarted = performance.now();
        reference = await monolithic.run(input);
        report(`Monolithic reference B${batchSize}: ${(performance.now() - referenceStarted).toFixed(1)} ms`);
        reportOutput(`Monolithic reference B${batchSize}`, reference);
      } finally {
        await monolithic.release();
        report("Monolithic reference released before loading split model");
      }
    }
    const vad = await OrtVadBackend.create(runtime, selected, report);
    try {
      await reportOrtDevice(runtime);
      report(
        `Split buffers: owned=${formatMib(vad.gpuBufferBytes.totalOwned)}, declared first-conv=${formatMib(vad.declaredBufferBytes.first_convolution_activation_bytes)}`,
      );
      await reportMemory("model loaded");
      const timings: number[] = [];
      let lastOutput: Float32Array | undefined;
      for (let run = 0; run < measuredRuns; run += 1) {
        const runStarted = performance.now();
        lastOutput = await vad.run(input);
        const elapsed = performance.now() - runStarted;
        timings.push(elapsed);
        report(`Split VAD B${batchSize} run ${run + 1}: ${elapsed.toFixed(1)} ms`);
      }
      if (lastOutput !== undefined) {
        reportOutput(`Split VAD B${batchSize}`, lastOutput);
        if (reference !== undefined) reportComparison(reference, lastOutput, batchSize);
      }
      if (parameters.get("stages") === "1") {
        const fingerprints = await vad.debugStageFingerprints();
        for (const [stage, fingerprint] of Object.entries(fingerprints)) {
          reportGpuFingerprint(`${stage} GPU buffer`, fingerprint);
        }
      }
      if (parameters.get("layers") === "1") {
        const fingerprints = await vad.debugLayerFingerprints();
        for (let layer = 0; layer < fingerprints.length; layer += 1) {
          reportGpuFingerprint(`LSTM layer ${layer}`, fingerprints[layer]!);
        }
      }
      reportTimingSummary(`Split VAD B${batchSize}`, timings);
      if (parameters.get("profile") === "1") {
        const profile = await vad.debugProfileRun(input);
        report(
          `Split VAD B${batchSize} profile: upload=${profile.uploadMs.toFixed(1)} ms, frontend=${profile.frontendMs.toFixed(1)} ms, LSTM=${profile.lstmMs.toFixed(1)} ms, tail+readback=${profile.tailAndReadbackMs.toFixed(1)} ms, total=${profile.totalMs.toFixed(1)} ms`,
        );
        reportOutput(`Profiled split VAD B${batchSize}`, profile.output);
      }
      await reportMemory("after runs");
      await holdResources(parameters);
    } finally {
      await vad.release();
      report("Split VAD sessions and buffers released");
      await reportMemory("released");
    }
    return;
  }
  const asset: OrtModelAsset = {
    url: "/models/pyannote-segmentation-3.0-logits-b1.onnx",
    inputName: "waveform",
    outputName: "logits",
  };
  const vad = await OrtMonolithicVadBackend.create(runtime, asset, 1, report);
  try {
    await reportOrtDevice(runtime);
    report("Running warmup");
    await vad.run(new Float32Array(vad.chunkSamples));
    report("Warmup complete");
  } finally {
    await vad.release();
    report("VAD session and buffers released");
  }
}

function reportGpuFingerprint(
  label: string,
  fingerprint: {
    length: number;
    finite: number;
    nonzero: number;
    minimum: number;
    maximum: number;
    sum: number;
    l2: number;
    first: readonly number[];
  },
): void {
  report(
    `${label}: finite=${fingerprint.finite}/${fingerprint.length}, nonzero=${fingerprint.nonzero}, min=${fingerprint.minimum.toPrecision(8)}, max=${fingerprint.maximum.toPrecision(8)}, sum=${fingerprint.sum.toPrecision(10)}, l2=${fingerprint.l2.toPrecision(10)}, first=[${fingerprint.first.map((value) => value.toPrecision(6)).join(",")}]`,
  );
}

void main().catch((error: unknown) => {
  report(`ERROR: ${error instanceof Error ? error.stack ?? error.message : String(error)}`);
  output.dataset.failed = "true";
});
