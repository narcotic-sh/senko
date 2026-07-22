/// <reference types="@webgpu/types" />

import {
  type DenseBottleneckAccumulation,
  type DenseBottleneckWeightSource,
  type DenseCamDispatch,
} from "./pipeline/campplus-webgpu/dense-cam";
import {
  evaluateDenseBottleneckReference,
  evaluateDenseLocalCamReference,
} from "./pipeline/campplus-webgpu/dense-cam-reference";
import type { CamDenseLayerMetadata, CampPlusPackedSection } from "./pipeline/campplus-webgpu/metadata";
import { float16BitsToFloat32, float32ToFloat16Bits } from "./pipeline/campplus-webgpu/reference";
import {
  RAW_CAMPPLUS_REQUIRED_LIMITS,
  RawCampPlusFoundation,
  requireRawCampPlusAdapterLimits,
} from "./pipeline/campplus-webgpu/runtime";

const METADATA_URL = "/models/campplus-t150-webgpu-fp16.json";
const BATCH_SIZE = 32;
const FRAMES = 75;
const SLAB_CHANNELS = 1024;
const INPUT_CHANNELS = 992;
const APPEND_CHANNEL = 992;
const SLAB_BYTES = BATCH_SIZE * SLAB_CHANNELS * FRAMES * 2;
const SCRATCH_OFFSET = SLAB_BYTES;
const SCRATCH_BYTES = BATCH_SIZE * 128 * FRAMES * 2;
const MEAN_OFFSET = SCRATCH_OFFSET + SCRATCH_BYTES;
const MEAN_BYTES = BATCH_SIZE * 128 * 2;
const APPEND_BYTES = BATCH_SIZE * 32 * FRAMES * 2;
const BOTTLENECK_MACS = BATCH_SIZE * FRAMES * 128 * INPUT_CHANNELS;
const LOCAL_CAM_MACS =
  BATCH_SIZE * FRAMES * 32 * 128 * 3 + BATCH_SIZE * (128 * 64 + 64 * 32);
interface HalfComparison {
  readonly maxAbsoluteError: number;
  readonly meanAbsoluteError: number;
  readonly cosineSimilarity: number;
  readonly exactHalfMatches: number;
}

interface TimedRun {
  readonly gpuMs: number;
  readonly wallMs: number;
  readonly wallMsPerIteration: number;
  readonly iterations: number;
  readonly timestampQuery: boolean;
}

export async function runRawCampPlusDenseDiagnostic(root: HTMLElement): Promise<void> {
  root.innerHTML = `<section><h1>Raw CAM++ B32 dense diagnostic</h1><pre id="raw-campplus-result">Requesting WebGPU…</pre></section>`;
  const output = root.querySelector<HTMLElement>("#raw-campplus-result");
  if (output === null) throw new Error("Missing raw CAM++ dense diagnostic output");
  try {
    const result = await execute((message) => {
      output.textContent = message;
    });
    output.textContent = JSON.stringify(result, null, 2);
    output.dataset.status = result.ok ? "passed" : "failed";
    globalThis.dispatchEvent(new CustomEvent("senko-raw-campplus-dense-diagnostic", { detail: result }));
  } catch (error) {
    output.textContent = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    output.dataset.status = "error";
    throw error;
  }
}

async function execute(report: (message: string) => void): Promise<Record<string, unknown>> {
  if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null || !adapter.features.has("shader-f16")) {
    throw new Error("A shader-f16 WebGPU adapter is required");
  }
  const timestampQuery = adapter.features.has("timestamp-query");
  requireRawCampPlusAdapterLimits(adapter);
  const requiredFeatures: GPUFeatureName[] = ["shader-f16"];
  if (timestampQuery) requiredFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: RAW_CAMPPLUS_REQUIRED_LIMITS,
  });
  let foundation: RawCampPlusFoundation | undefined;
  const bottlenecks: DenseCamDispatch[] = [];
  let localCam: DenseCamDispatch | undefined;
  let denseReadback: GPUBuffer | undefined;
  let appendReadback: GPUBuffer | undefined;
  try {
    report("Streaming weights and compiling B32 dense kernels…");
    const loadStart = performance.now();
    foundation = await RawCampPlusFoundation.create(device, METADATA_URL);
    const loadAndCompileMs = performance.now() - loadStart;
    const layer = selectWorstCaseLayer(foundation);
    const sections = resolveLayerSections(foundation, layer);
    report("Streaming only the selected real layer sections for the CPU oracle…");
    const cpuSections = await fetchSectionBytes(
      foundation.gpuPackage.binaryUrl,
      Object.values(sections),
    );

    const slab = deterministicSlab();
    const slabSlice = foundation.arena.slice("dense-b32-slab", 0, SLAB_BYTES);
    const scratchSlice = foundation.arena.slice(
      "dense-b32-bottleneck-scratch",
      SCRATCH_OFFSET,
      SCRATCH_BYTES,
    );
    const meanSlice = foundation.arena.slice(
      "dense-b32-doubled-mean",
      MEAN_OFFSET,
      MEAN_BYTES,
    );
    foundation.arena.upload(device, slabSlice, slab);
    const variants = [
      {
        id: "tile1-wg128-cache-float32",
        accumulation: "float32",
        outputTile: 1,
        workgroupSize: 128,
        weightSource: "workgroup-cache",
      },
      {
        id: "tile1-wg128-direct-float32",
        accumulation: "float32",
        outputTile: 1,
        workgroupSize: 128,
        weightSource: "direct",
      },
    ] as const satisfies readonly {
      readonly id: string;
      readonly accumulation: DenseBottleneckAccumulation;
      readonly outputTile: 1;
      readonly workgroupSize: 128;
      readonly weightSource: DenseBottleneckWeightSource;
    }[];
    report("Compiling matched cached-weight and direct-weight FP32 variants…");
    await Promise.all(
      variants.map((variant) =>
        foundation!.denseCam.prepareBottleneckVariant(
          variant.accumulation,
          variant.outputTile,
          variant.workgroupSize,
          variant.weightSource,
        ),
      ),
    );
    for (const variant of variants) {
      bottlenecks.push(
        foundation.denseCam.createBottleneckDispatch({
          label: `senko-campplus-block2-layer24-bottleneck-b32-${variant.id}`,
          layer,
          slab: slabSlice,
          slabChannels: SLAB_CHANNELS,
          scratch: scratchSlice,
          doubledMean: meanSlice,
          batchSize: BATCH_SIZE,
          accumulation: variant.accumulation,
          outputTile: variant.outputTile,
          workgroupSize: variant.workgroupSize,
          weightSource: variant.weightSource,
        }),
      );
    }
    localCam = foundation.denseCam.createLocalCamDispatch({
      label: "senko-campplus-block2-layer24-local-cam-b32",
      layer,
      slab: slabSlice,
      slabChannels: SLAB_CHANNELS,
      scratch: scratchSlice,
      doubledMean: meanSlice,
      batchSize: BATCH_SIZE,
    });
    denseReadback = device.createBuffer({
      label: "senko-campplus-dense-b32-readback",
      size: SCRATCH_BYTES + MEAN_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    appendReadback = device.createBuffer({
      label: "senko-campplus-local-cam-b32-readback",
      size: APPEND_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const denseActual = new Map<
      string,
      { readonly timing: TimedRun; readonly scratch: Uint16Array; readonly mean: Uint16Array }
    >();
    for (let index = 0; index < variants.length; index += 1) {
      const variant = variants[index]!;
      const dispatch = bottlenecks[index]!;
      report(`Warming and timing B32 bottleneck ${variant.id}…`);
      await warmDispatch(device, dispatch, 3);
      const timing = await timeDispatch(
        device,
        dispatch,
        denseReadback,
        (encoder) => {
          encoder.copyBufferToBuffer(
            foundation!.arena.buffer,
            SCRATCH_OFFSET,
            denseReadback!,
            0,
            SCRATCH_BYTES,
          );
          encoder.copyBufferToBuffer(
            foundation!.arena.buffer,
            MEAN_OFFSET,
            denseReadback!,
            SCRATCH_BYTES,
            MEAN_BYTES,
          );
        },
        timestampQuery,
        20,
      );
      const denseMapped = new Uint16Array(denseReadback.getMappedRange());
      denseActual.set(variant.id, {
        timing,
        scratch: denseMapped.subarray(0, SCRATCH_BYTES / 2).slice(),
        mean: denseMapped
          .subarray(SCRATCH_BYTES / 2, (SCRATCH_BYTES + MEAN_BYTES) / 2)
          .slice(),
      });
      denseReadback.unmap();
    }

    // Local/CAM correctness remains anchored to the verified FP32 bottleneck.
    await warmDispatch(device, bottlenecks[0]!, 1);

    report("Warming and timing 256 B32 local/CAM workgroups…");
    await warmDispatch(device, localCam, 3);
    const localTiming = await timeDispatch(
      device,
      localCam,
      appendReadback,
      (encoder) => encodeAppendCopies(encoder, foundation!.arena.buffer, appendReadback!),
      timestampQuery,
      50,
    );
    const actualAppend = new Uint16Array(appendReadback.getMappedRange()).slice();
    appendReadback.unmap();

    report("Evaluating the worst-case B32 layer on the independent CPU oracle…");
    const cpuStart = performance.now();
    const expectedDense = evaluateDenseBottleneckReference(
      slab,
      asUint16(cpuSections, sections.bottleneckWeight),
      asUint16(cpuSections, sections.bottleneckBias),
      asFloat32(cpuSections, sections.affine),
      {
        batchSize: BATCH_SIZE,
        inputChannels: INPUT_CHANNELS,
        slabChannels: SLAB_CHANNELS,
        frames: FRAMES,
      },
    );
    const expectedAppend = evaluateDenseLocalCamReference(
      expectedDense.scratch,
      expectedDense.doubledMean,
      asUint16(cpuSections, sections.localWeight),
      asUint16(cpuSections, sections.localBias),
      asUint16(cpuSections, sections.attention1Weight),
      asUint16(cpuSections, sections.attention1Bias),
      asUint16(cpuSections, sections.attention2Weight),
      asUint16(cpuSections, sections.attention2Bias),
      {
        batchSize: BATCH_SIZE,
        slabChannels: SLAB_CHANNELS,
        frames: FRAMES,
        appendChannel: APPEND_CHANNEL,
        dilation: layer.localDilation,
      },
    );
    const cpuMs = performance.now() - cpuStart;
    const denseVariants = variants.map((variant) => {
      const actual = denseActual.get(variant.id)!;
      const parity = compareHalf(actual.scratch, expectedDense.scratch);
      const doubledMeanParity = compareHalf(actual.mean, expectedDense.doubledMean);
      return {
        ...variant,
        workgroups: 1024 / variant.outputTile,
        ...actual.timing,
        gmacPerSecond: BOTTLENECK_MACS / (actual.timing.gpuMs * 1_000_000),
        parity,
        doubledMeanParity,
      };
    });
    const scratchParity = denseVariants[0]!.parity;
    const meanParity = denseVariants[0]!.doubledMeanParity;
    const appendParity = compareHalf(actualAppend, expectedAppend);
    const ok =
      scratchParity.maxAbsoluteError <= 0.02 &&
      scratchParity.cosineSimilarity >= 0.999_999 &&
      meanParity.maxAbsoluteError <= 0.005 &&
      appendParity.maxAbsoluteError <= 0.02 &&
      appendParity.cosineSimilarity >= 0.999_999;
    return {
      ok,
      layer: layer.id,
      limits: {
        adapterMaxComputeWorkgroupStorageSize:
          adapter.limits.maxComputeWorkgroupStorageSize,
        deviceMaxComputeWorkgroupStorageSize:
          device.limits.maxComputeWorkgroupStorageSize,
        requestedMaxComputeWorkgroupStorageSize:
          RAW_CAMPPLUS_REQUIRED_LIMITS.maxComputeWorkgroupStorageSize,
      },
      loadAndCompileMs,
      cpuMs,
      dense: {
        macs: BOTTLENECK_MACS,
        variants: denseVariants,
      },
      localCam: {
        workgroups: 256,
        macs: LOCAL_CAM_MACS,
        ...localTiming,
        gmacPerSecond: LOCAL_CAM_MACS / (localTiming.gpuMs * 1_000_000),
        parity: appendParity,
      },
      explicitGpuBytes: {
        persistent: foundation.gpuBytes.total,
        dispatchUniforms:
          bottlenecks.reduce((sum, dispatch) => sum + dispatch.gpuBufferBytes, 0) +
          localCam.gpuBufferBytes,
        diagnosticReadbacks: SCRATCH_BYTES + MEAN_BYTES + APPEND_BYTES,
      },
      retainedCpuBytes: {
        metadataApproximate: JSON.stringify(foundation.gpuPackage.metadata).length,
        diagnosticLayerSections: [...cpuSections.values()].reduce(
          (sum, value) => sum + value.byteLength,
          0,
        ),
        diagnosticInputSlab: slab.byteLength,
        productionBinaryAfterUpload: 0,
      },
    };
  } finally {
    if (denseReadback?.mapState === "mapped") denseReadback.unmap();
    if (appendReadback?.mapState === "mapped") appendReadback.unmap();
    denseReadback?.destroy();
    appendReadback?.destroy();
    for (const dispatch of bottlenecks) dispatch.destroy();
    localCam?.destroy();
    foundation?.destroy();
    device.destroy();
  }
}

function selectWorstCaseLayer(foundation: RawCampPlusFoundation): CamDenseLayerMetadata {
  const layer = foundation.gpuPackage.metadata.fusedProgram.blocks[1]?.layers[23];
  if (
    layer === undefined ||
    layer.id !== "block2.layer24" ||
    layer.inputChannels !== INPUT_CHANNELS ||
    layer.appendChannel !== APPEND_CHANNEL
  ) {
    throw new Error("Packed metadata does not contain expected block2.layer24");
  }
  return layer;
}

function resolveLayerSections(
  foundation: RawCampPlusFoundation,
  layer: CamDenseLayerMetadata,
): Record<
  | "bottleneckWeight"
  | "bottleneckBias"
  | "affine"
  | "localWeight"
  | "localBias"
  | "attention1Weight"
  | "attention1Bias"
  | "attention2Weight"
  | "attention2Bias",
  CampPlusPackedSection
> {
  const section = (id: string) => foundation.gpuPackage.section(id);
  return {
    bottleneckWeight: section(layer.bottleneck.weight),
    bottleneckBias: section(layer.bottleneck.bias),
    affine: section(layer.preactivationAffine),
    localWeight: section(layer.local.weight),
    localBias: section(layer.local.bias),
    attention1Weight: section(layer.attention1.weight),
    attention1Bias: section(layer.attention1.bias),
    attention2Weight: section(layer.attention2.weight),
    attention2Bias: section(layer.attention2.bias),
  };
}

async function warmDispatch(
  device: GPUDevice,
  dispatch: DenseCamDispatch,
  iterations: number,
): Promise<void> {
  const encoder = device.createCommandEncoder();
  for (let index = 0; index < iterations; index += 1) dispatch.encode(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function timeDispatch(
  device: GPUDevice,
  dispatch: DenseCamDispatch,
  output: GPUBuffer,
  encodeCopies: (encoder: GPUCommandEncoder) => void,
  timestampQuery: boolean,
  iterations: number,
): Promise<TimedRun> {
  const querySet = timestampQuery
    ? device.createQuerySet({ type: "timestamp", count: 2 })
    : undefined;
  const resolve = querySet === undefined
    ? undefined
    : device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
  const timestampReadback = querySet === undefined
    ? undefined
    : device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
  try {
    const encoder = device.createCommandEncoder();
    for (let index = 0; index < iterations; index += 1) {
      let timestampWrites: GPUComputePassTimestampWrites | undefined;
      if (querySet !== undefined) {
        if (iterations === 1) {
          timestampWrites = {
            querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
          };
        } else if (index === 0) {
          timestampWrites = { querySet, beginningOfPassWriteIndex: 0 };
        } else if (index === iterations - 1) {
          timestampWrites = { querySet, endOfPassWriteIndex: 1 };
        }
      }
      dispatch.encode(encoder, timestampWrites);
    }
    encodeCopies(encoder);
    if (querySet !== undefined && resolve !== undefined && timestampReadback !== undefined) {
      encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, timestampReadback, 0, 16);
    }
    const wallStart = performance.now();
    device.queue.submit([encoder.finish()]);
    await Promise.all([
      output.mapAsync(GPUMapMode.READ),
      timestampReadback?.mapAsync(GPUMapMode.READ),
    ]);
    const wallMs = performance.now() - wallStart;
    if (timestampReadback === undefined) {
      return {
        gpuMs: wallMs / iterations,
        wallMs,
        wallMsPerIteration: wallMs / iterations,
        iterations,
        timestampQuery: false,
      };
    }
    const timestamps = new BigUint64Array(timestampReadback.getMappedRange());
    const gpuMs = Number(timestamps[1]! - timestamps[0]!) / 1_000_000 / iterations;
    timestampReadback.unmap();
    return {
      gpuMs,
      wallMs,
      wallMsPerIteration: wallMs / iterations,
      iterations,
      timestampQuery: true,
    };
  } finally {
    if (timestampReadback?.mapState === "mapped") timestampReadback.unmap();
    timestampReadback?.destroy();
    resolve?.destroy();
    querySet?.destroy();
  }
}

function encodeAppendCopies(
  encoder: GPUCommandEncoder,
  arena: GPUBuffer,
  readback: GPUBuffer,
): void {
  const bytesPerBatch = 32 * FRAMES * 2;
  for (let batch = 0; batch < BATCH_SIZE; batch += 1) {
    const sourceOffset = ((batch * SLAB_CHANNELS + APPEND_CHANNEL) * FRAMES) * 2;
    encoder.copyBufferToBuffer(
      arena,
      sourceOffset,
      readback,
      batch * bytesPerBatch,
      bytesPerBatch,
    );
  }
}

function deterministicSlab(): Uint16Array<ArrayBuffer> {
  const slab = new Uint16Array(BATCH_SIZE * SLAB_CHANNELS * FRAMES);
  for (let batch = 0; batch < BATCH_SIZE; batch += 1) {
    for (let channel = 0; channel < INPUT_CHANNELS; channel += 1) {
      for (let frame = 0; frame < FRAMES; frame += 1) {
        const value =
          Math.sin(batch * 0.17 + channel * 0.013 + frame * 0.041) * 0.65 +
          Math.cos(batch * 0.07 - channel * 0.009 + frame * 0.023) * 0.3;
        slab[(batch * SLAB_CHANNELS + channel) * FRAMES + frame] =
          float32ToFloat16Bits(value);
      }
    }
  }
  return slab;
}

async function fetchSectionBytes(
  binaryUrl: string,
  requested: readonly CampPlusPackedSection[],
): Promise<ReadonlyMap<string, Uint8Array<ArrayBuffer>>> {
  const response = await fetch(binaryUrl);
  if (!response.ok || response.body === null) throw new Error("Unable to stream CPU oracle sections");
  const buffers = new Map(
    requested.map((section) => [section.id, new Uint8Array(section.byteLength)]),
  );
  const reader = response.body.getReader();
  let absoluteOffset = 0;
  try {
    while (true) {
      const item = await reader.read();
      if (item.done) break;
      const chunkStart = absoluteOffset;
      const chunkEnd = chunkStart + item.value.byteLength;
      for (const section of requested) {
        const overlapStart = Math.max(chunkStart, section.byteOffset);
        const overlapEnd = Math.min(chunkEnd, section.byteOffset + section.byteLength);
        if (overlapStart >= overlapEnd) continue;
        buffers.get(section.id)!.set(
          item.value.subarray(overlapStart - chunkStart, overlapEnd - chunkStart),
          overlapStart - section.byteOffset,
        );
      }
      absoluteOffset = chunkEnd;
    }
    return buffers;
  } finally {
    reader.releaseLock();
  }
}

function asUint16(
  sections: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
  section: CampPlusPackedSection,
): Uint16Array<ArrayBuffer> {
  const bytes = sections.get(section.id);
  if (bytes === undefined) throw new Error(`Missing CPU section ${section.id}`);
  return new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
}

function asFloat32(
  sections: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
  section: CampPlusPackedSection,
): Float32Array<ArrayBuffer> {
  const bytes = sections.get(section.id);
  if (bytes === undefined) throw new Error(`Missing CPU section ${section.id}`);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function compareHalf(actual: Uint16Array, expected: Uint16Array): HalfComparison {
  if (actual.length !== expected.length) throw new Error("Dense diagnostic output lengths differ");
  let maxAbsoluteError = 0;
  let absoluteError = 0;
  let dot = 0;
  let actualNorm = 0;
  let expectedNorm = 0;
  let exactHalfMatches = 0;
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] === expected[index]) exactHalfMatches += 1;
    const left = float16BitsToFloat32(actual[index]!);
    const right = float16BitsToFloat32(expected[index]!);
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
    exactHalfMatches,
  };
}
