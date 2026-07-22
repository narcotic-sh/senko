/// <reference types="@webgpu/types" />

import {
  requestMaximumPerformanceAdapter,
  requestMaximumPerformanceDevice,
} from "./pipeline/browser-models";
import {
  loadModelManifest,
  selectCampPlusDirect,
  selectSegmentationSplit,
} from "./pipeline/model-manifest";
import { RawCampPlusEmbeddingBackend } from "./pipeline/raw-campplus-backend";
import { RawWebGpuVadBackend } from "./pipeline/raw-vad-backend";

const DEFAULT_ROUNDS = 12;
const DEFAULT_WARMUPS = 3;
const VAD_BATCH = 8;
const CAM_BATCH = 16;

interface Timed<T> {
  readonly wallMs: number;
  readonly value: T;
}

interface CamGroup {
  readonly outputs: readonly Float32Array[];
  readonly runMs: readonly number[];
}

interface GroupMeasurement {
  readonly combinedMs: number;
  readonly vadMs: number;
  readonly camGroupMs: number;
  readonly camRunMs: readonly number[];
  readonly vadOutput: Float32Array;
  readonly camOutputs: readonly Float32Array[];
}

interface Summary {
  readonly count: number;
  readonly minMs: number;
  readonly medianMs: number;
  readonly meanMs: number;
  readonly p90Ms: number;
  readonly maxMs: number;
  readonly coefficientOfVariation: number;
}

interface ExactParity {
  comparisons: number;
  mismatchedValues: number;
  mismatchedRuns: number;
  maxAbsoluteDifference: number;
}

export async function runDualDeviceConcurrencyDiagnostic(
  root: HTMLElement,
): Promise<void> {
  root.innerHTML = `<section><h1>Dual-device VAD/CAM++ concurrency diagnostic</h1><pre id="dual-device-concurrency-result">Requesting WebGPU…</pre></section>`;
  const output = root.querySelector<HTMLElement>(
    "#dual-device-concurrency-result",
  );
  if (output === null) throw new Error("Missing concurrency diagnostic output");

  try {
    const result = await execute((message) => {
      output.textContent = message;
    });
    output.textContent = JSON.stringify(result, null, 2);
    output.dataset.status = result.ok ? "passed" : "failed";
    globalThis.dispatchEvent(
      new CustomEvent("senko-dual-device-concurrency-diagnostic", {
        detail: result,
      }),
    );
  } catch (error) {
    output.textContent =
      error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    output.dataset.status = "error";
    throw error;
  }
}

async function execute(
  report: (message: string) => void,
): Promise<Record<string, unknown> & { readonly ok: boolean }> {
  if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
  const parameters = new URLSearchParams(location.search);
  const rounds = parseCount(parameters.get("rounds"), DEFAULT_ROUNDS, 6, 30);
  const warmups = parseCount(parameters.get("warmups"), DEFAULT_WARMUPS, 1, 10);
  const camRuns = parseCount(parameters.get("cam-runs"), 2, 1, 4);
  const camInFlight = parseCount(parameters.get("cam-inflight"), 1, 1, 2);

  report("Requesting two handles to the same high-performance adapter…");
  // Chrome/Dawn consumes a GPUAdapter handle after requestDevice(). Request
  // two handles and prove below that both identify the same physical adapter.
  const vadAdapter = await requestMaximumPerformanceAdapter(navigator.gpu);
  const camAdapter = await requestMaximumPerformanceAdapter(navigator.gpu);
  const vadDevice = await requestMaximumPerformanceDevice(vadAdapter);
  let camDevice: GPUDevice | undefined;
  let vad: RawWebGpuVadBackend | undefined;
  let cam: RawCampPlusEmbeddingBackend | undefined;
  const uncapturedErrors: string[] = [];
  try {
    camDevice = await requestMaximumPerformanceDevice(camAdapter);
    collectUncapturedErrors(vadDevice, "vad", uncapturedErrors);
    collectUncapturedErrors(camDevice, "cam", uncapturedErrors);

    const manifestUrl = new URL("/models/manifest.json", location.href).toString();
    const manifest = await loadModelManifest(manifestUrl);
    const vadVariant = selectSegmentationSplit(
      manifestUrl,
      manifest.models.segmentation,
      VAD_BATCH,
    );
    const camVariant = selectCampPlusDirect(
      manifestUrl,
      manifest.models.campplus,
      CAM_BATCH,
    );

    report("Loading the production B8 VAD and B16 CAM++ graphs concurrently…");
    [vad, cam] = await Promise.all([
      RawWebGpuVadBackend.create(
        vadDevice,
        vadVariant,
        {
          frontendMetadata: vadVariant.directWebGpu.frontendMetadata,
          tailMetadata: vadVariant.directWebGpu.tailMetadata,
        },
      ),
      RawCampPlusEmbeddingBackend.create(camDevice, camVariant),
    ]);

    const vadInput = deterministicVadInput(vad.batchSize * vad.chunkSamples);
    const camInput = deterministicCamInput(
      cam.batchSize * cam.frames * cam.featureDim,
    );

    report(`Warming VAD and CAM++ independently with ${warmups} work groups…`);
    let referenceVad: Float32Array | undefined;
    let referenceCam: Float32Array | undefined;
    const vadParity: ExactParity = emptyParity();
    const camParity: ExactParity = emptyParity();
    for (let index = 0; index < warmups; index += 1) {
      const vadOutput = await vad.run(vadInput);
      const camGroup = await runCamGroup(cam, camInput, camRuns, camInFlight);
      const firstCamOutput = camGroup.outputs[0];
      if (firstCamOutput === undefined) {
        throw new Error("CAM++ warmup did not produce an output");
      }
      referenceVad ??= vadOutput;
      referenceCam ??= firstCamOutput;
      accumulateParity(vadParity, referenceVad, vadOutput);
      for (const camOutput of camGroup.outputs) {
        accumulateParity(camParity, referenceCam, camOutput);
      }
    }
    await Promise.all([
      vadDevice.queue.onSubmittedWorkDone(),
      camDevice.queue.onSubmittedWorkDone(),
    ]);
    if (referenceVad === undefined || referenceCam === undefined) {
      throw new Error("Warmup did not produce reference outputs");
    }

    const sequential: GroupMeasurement[] = [];
    const concurrent: GroupMeasurement[] = [];

    report(`Measuring ${rounds} interleaved rounds per mode…`);
    for (let round = 0; round < rounds; round += 1) {
      const launchVadFirst = round % 2 === 0;
      const modes = round % 2 === 0
        ? (["sequential", "concurrent"] as const)
        : (["concurrent", "sequential"] as const);
      for (const mode of modes) {
        const measured =
          mode === "sequential"
            ? await measureSequential(
                vad,
                cam,
                vadInput,
                camInput,
                camRuns,
                camInFlight,
                launchVadFirst,
              )
            : await measureConcurrent(
                vad,
                cam,
                vadInput,
                camInput,
                camRuns,
                camInFlight,
                launchVadFirst,
              );
        (mode === "sequential" ? sequential : concurrent).push(measured);
        accumulateParity(vadParity, referenceVad, measured.vadOutput);
        for (const camOutput of measured.camOutputs) {
          accumulateParity(camParity, referenceCam, camOutput);
        }
      }
    }

    const seqCombined = summarize(sequential.map((run) => run.combinedMs));
    const seqVad = summarize(sequential.map((run) => run.vadMs));
    const seqCam = summarize(sequential.map((run) => run.camGroupMs));
    const conCombined = summarize(concurrent.map((run) => run.combinedMs));
    const conVad = summarize(concurrent.map((run) => run.vadMs));
    const conCam = summarize(concurrent.map((run) => run.camGroupMs));
    const speedup = seqCombined.medianMs / conCombined.medianMs;
    const savedMs = seqCombined.medianMs - conCombined.medianMs;
    const idealConcurrentMs = Math.max(seqVad.medianMs, seqCam.medianMs);
    const overlapOpportunityMs = seqCombined.medianMs - idealConcurrentMs;
    const overlapEfficiency =
      overlapOpportunityMs <= 0 ? 0 : savedMs / overlapOpportunityMs;
    const exact =
      vadParity.mismatchedValues === 0 && camParity.mismatchedValues === 0;
    const verdict = classify(speedup, savedMs / seqCombined.medianMs, exact);
    const sequentialDrift = drift(sequential.map((run) => run.combinedMs));
    const concurrentDrift = drift(concurrent.map((run) => run.combinedMs));
    const pageMemory = await measurePageMemory();

    const result = {
      ok: exact && uncapturedErrors.length === 0,
      verdict,
      configuration: {
        roundsPerMode: rounds,
        warmupIndependentGroups: warmups,
        camRunsPerVad: camRuns,
        camMaxInFlight: camInFlight,
        sequentialWork: `one B8 VAD run plus ${camRuns} B16 CAM++ run${camRuns === 1 ? "" : "s"}, without cross-device overlap`,
        concurrentWork: "the same two sides on separate GPUDevice queues",
        interleavedModeOrder: true,
        alternatingQueueLaunchOrder: true,
      },
      adapters: {
        apiHandles: 2,
        chromeConstraint:
          "A GPUAdapter handle is consumed by its first requestDevice call",
        sameReportedPhysicalAdapter:
          adapterIdentity(vadAdapter) === adapterIdentity(camAdapter),
        vad: adapterInfo(vadAdapter),
        cam: adapterInfo(camAdapter),
      },
      devices: {
        count: 2,
        vadRequiredFeatures: [...vadDevice.features].sort(),
        camRequiredFeatures: [...camDevice.features].sort(),
        vadMaximumRequestedLimits: maximumRequestedLimits(vadAdapter),
        camMaximumRequestedLimits: maximumRequestedLimits(camAdapter),
      },
      timings: {
        sequential: groupStats(sequential, seqCombined, seqVad, seqCam),
        concurrent: groupStats(concurrent, conCombined, conVad, conCam),
        speedup,
        medianSavedMs: savedMs,
        medianSavedFraction: savedMs / seqCombined.medianMs,
        idealConcurrentMs,
        overlapEfficiency,
        vadContentionFactor: conVad.medianMs / seqVad.medianMs,
        camGroupContentionFactor: conCam.medianMs / seqCam.medianMs,
        balancedSequentialRatio: seqVad.medianMs / seqCam.medianMs,
      },
      parity: {
        exact,
        vad: vadParity,
        cam: camParity,
        fingerprints: {
          vad: await fingerprint(referenceVad),
          cam: await fingerprint(referenceCam),
        },
      },
      memory: {
        explicitGpuBufferBytes: {
          vad: vad.gpuBufferBytes.totalOwned,
          cam: cam.gpuBufferBytes.total,
          summedResident: vad.gpuBufferBytes.totalOwned + cam.gpuBufferBytes.total,
        },
        hostInputsBytes: vadInput.byteLength + camInput.byteLength,
        pageAgentClusterBytes: pageMemory.bytes,
        pageMemoryError: pageMemory.error,
      },
      stability: {
        sequentialCombinedDrift: sequentialDrift,
        concurrentCombinedDrift: concurrentDrift,
        sequentialCombinedCoefficientOfVariation:
          seqCombined.coefficientOfVariation,
        concurrentCombinedCoefficientOfVariation:
          conCombined.coefficientOfVariation,
        uncapturedGpuErrors: uncapturedErrors,
        crossOriginIsolated,
        userAgent: navigator.userAgent,
      },
    };
    return result;
  } finally {
    await Promise.allSettled([vad?.release(), cam?.release()]);
    vadDevice.destroy();
    camDevice?.destroy();
  }
}

async function measureSequential(
  vad: RawWebGpuVadBackend,
  cam: RawCampPlusEmbeddingBackend,
  vadInput: Float32Array,
  camInput: Float32Array,
  camRuns: number,
  camInFlight: number,
  vadFirst: boolean,
): Promise<GroupMeasurement> {
  const started = performance.now();
  let vadRun: Timed<Float32Array>;
  let camGroup: Timed<CamGroup>;
  if (vadFirst) {
    vadRun = await timed(() => vad.run(vadInput));
    camGroup = await timed(() =>
      runCamGroup(cam, camInput, camRuns, camInFlight),
    );
  } else {
    camGroup = await timed(() =>
      runCamGroup(cam, camInput, camRuns, camInFlight),
    );
    vadRun = await timed(() => vad.run(vadInput));
  }
  return groupMeasurement(performance.now() - started, vadRun, camGroup);
}

async function measureConcurrent(
  vad: RawWebGpuVadBackend,
  cam: RawCampPlusEmbeddingBackend,
  vadInput: Float32Array,
  camInput: Float32Array,
  camRuns: number,
  camInFlight: number,
  vadFirst: boolean,
): Promise<GroupMeasurement> {
  const started = performance.now();
  const first = vadFirst
    ? timed(() => vad.run(vadInput))
    : timed(() => runCamGroup(cam, camInput, camRuns, camInFlight));
  const second = vadFirst
    ? timed(() => runCamGroup(cam, camInput, camRuns, camInFlight))
    : timed(() => vad.run(vadInput));
  const [firstResult, secondResult] = await Promise.all([first, second]);
  const vadRun = (vadFirst ? firstResult : secondResult) as Timed<Float32Array>;
  const camGroup = (vadFirst ? secondResult : firstResult) as Timed<CamGroup>;
  return groupMeasurement(performance.now() - started, vadRun, camGroup);
}

function groupMeasurement(
  combinedMs: number,
  vad: Timed<Float32Array>,
  cam: Timed<CamGroup>,
): GroupMeasurement {
  return {
    combinedMs,
    vadMs: vad.wallMs,
    camGroupMs: cam.wallMs,
    camRunMs: cam.value.runMs,
    vadOutput: vad.value,
    camOutputs: cam.value.outputs,
  };
}

async function runCamGroup(
  cam: RawCampPlusEmbeddingBackend,
  input: Float32Array,
  runs: number,
  maxInFlight: number,
): Promise<CamGroup> {
  const outputs: Float32Array[] = [];
  const runMs: number[] = [];
  for (let index = 0; index < runs; index += maxInFlight) {
    const count = Math.min(maxInFlight, runs - index);
    const measured = await Promise.all(
      Array.from({ length: count }, () => timed(() => cam.run(input))),
    );
    for (const run of measured) {
      outputs.push(run.value);
      runMs.push(run.wallMs);
    }
  }
  return { outputs, runMs };
}

async function timed<T>(action: () => Promise<T>): Promise<Timed<T>> {
  const started = performance.now();
  const value = await action();
  return { wallMs: performance.now() - started, value };
}

function groupStats(
  runs: readonly GroupMeasurement[],
  combined: Summary,
  vad: Summary,
  camGroup: Summary,
): Record<string, unknown> {
  return {
    combined,
    vad,
    camGroup,
    camSingle: summarize(runs.flatMap((run) => [...run.camRunMs])),
    samples: runs.map((run) => ({
      combinedMs: run.combinedMs,
      vadMs: run.vadMs,
      camGroupMs: run.camGroupMs,
      camRunMs: run.camRunMs,
    })),
  };
}

export function summarize(values: readonly number[]): Summary {
  if (values.length === 0) throw new RangeError("Cannot summarize no values");
  const sorted = [...values].sort((left, right) => left - right);
  const meanMs = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - meanMs) ** 2, 0) /
    values.length;
  return {
    count: values.length,
    minMs: sorted[0]!,
    medianMs: quantile(sorted, 0.5),
    meanMs,
    p90Ms: quantile(sorted, 0.9),
    maxMs: sorted[sorted.length - 1]!,
    coefficientOfVariation: meanMs === 0 ? 0 : Math.sqrt(variance) / meanMs,
  };
}

function drift(values: readonly number[]): Record<string, number> {
  const midpoint = Math.floor(values.length / 2);
  const earlyMedianMs = summarize(values.slice(0, midpoint)).medianMs;
  const lateMedianMs = summarize(values.slice(midpoint)).medianMs;
  return {
    earlyMedianMs,
    lateMedianMs,
    lateToEarlyRatio: lateMedianMs / earlyMedianMs,
  };
}

function quantile(sorted: readonly number[], fraction: number): number {
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower]! * (1 - weight) + sorted[upper]! * weight;
}

export function classify(
  speedup: number,
  savedFraction: number,
  exact: boolean,
): "strong" | "material" | "immaterial" | "parity-failed" {
  if (!exact) return "parity-failed";
  if (speedup >= 1.25 && savedFraction >= 0.2) return "strong";
  if (speedup >= 1.1 && savedFraction >= 0.09) return "material";
  return "immaterial";
}

function emptyParity(): ExactParity {
  return {
    comparisons: 0,
    mismatchedValues: 0,
    mismatchedRuns: 0,
    maxAbsoluteDifference: 0,
  };
}

function accumulateParity(
  parity: ExactParity,
  reference: Float32Array,
  actual: Float32Array,
): void {
  if (reference.length !== actual.length) {
    throw new Error(`Output length differs: ${reference.length}/${actual.length}`);
  }
  parity.comparisons += 1;
  let runMismatch = false;
  const referenceBits = new Uint32Array(
    reference.buffer,
    reference.byteOffset,
    reference.length,
  );
  const actualBits = new Uint32Array(
    actual.buffer,
    actual.byteOffset,
    actual.length,
  );
  for (let index = 0; index < reference.length; index += 1) {
    if (referenceBits[index] !== actualBits[index]) {
      parity.mismatchedValues += 1;
      runMismatch = true;
    }
    const difference = Math.abs(reference[index]! - actual[index]!);
    if (Number.isFinite(difference)) {
      parity.maxAbsoluteDifference = Math.max(
        parity.maxAbsoluteDifference,
        difference,
      );
    }
  }
  if (runMismatch) parity.mismatchedRuns += 1;
}

async function fingerprint(values: Float32Array): Promise<Record<string, unknown>> {
  const bytes = new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  ).slice();
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  let sha256 = "";
  for (const byte of digest) sha256 += byte.toString(16).padStart(2, "0");
  let sum = 0;
  let squared = 0;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    sum += value;
    squared += value * value;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  return {
    sha256,
    length: values.length,
    sum,
    l2: Math.sqrt(squared),
    minimum,
    maximum,
    first: [...values.slice(0, 8)],
  };
}

function deterministicVadInput(length: number): Float32Array<ArrayBuffer> {
  const values = new Float32Array(length);
  let state = 0x5e4b0;
  for (let index = 0; index < values.length; index += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    values[index] = (state / 0xffff_ffff - 0.5) * 0.4;
  }
  return values;
}

function deterministicCamInput(length: number): Float32Array<ArrayBuffer> {
  const values = new Float32Array(length);
  for (let index = 0; index < values.length; index += 1) {
    values[index] =
      Math.sin(index * 0.013) * 0.65 + Math.cos(index * 0.009) * 0.3;
  }
  return values;
}

function parseCount(
  source: string | null,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  if (source === null) return fallback;
  const value = Number(source);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`Expected integer ${minimum}..${maximum}, received ${source}`);
  }
  return value;
}

function collectUncapturedErrors(
  device: GPUDevice,
  label: string,
  destination: string[],
): void {
  device.addEventListener("uncapturederror", (event) => {
    destination.push(`${label}: ${event.error.message}`);
  });
}

function adapterInfo(adapter: GPUAdapter): Record<string, unknown> {
  return {
    vendor: adapter.info.vendor,
    architecture: adapter.info.architecture,
    device: adapter.info.device,
    description: adapter.info.description,
    isFallbackAdapter: adapter.info.isFallbackAdapter,
    features: [...adapter.features].sort(),
  };
}

function adapterIdentity(adapter: GPUAdapter): string {
  return JSON.stringify({
    vendor: adapter.info.vendor,
    architecture: adapter.info.architecture,
    device: adapter.info.device,
    description: adapter.info.description,
    isFallbackAdapter: adapter.info.isFallbackAdapter,
    features: [...adapter.features].sort(),
    limits: maximumRequestedLimits(adapter),
  });
}

function maximumRequestedLimits(adapter: GPUAdapter): Record<string, number> {
  return {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize:
      adapter.limits.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup:
      adapter.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupsPerDimension:
      adapter.limits.maxComputeWorkgroupsPerDimension,
  };
}

async function measurePageMemory(): Promise<{
  readonly bytes?: number;
  readonly error?: string;
}> {
  const source = performance as Performance & {
    measureUserAgentSpecificMemory?: () => Promise<{ readonly bytes: number }>;
  };
  if (source.measureUserAgentSpecificMemory === undefined) {
    return { error: "measureUserAgentSpecificMemory unavailable" };
  }
  try {
    return { bytes: (await source.measureUserAgentSpecificMemory()).bytes };
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}
