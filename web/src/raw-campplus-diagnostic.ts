/// <reference types="@webgpu/types" />

import {
  float16BitsToFloat32,
  float32ToFloat16Bits,
  evaluatePackedBctConvReference,
} from "./pipeline/campplus-webgpu/reference";
import { RawCampPlusFoundation } from "./pipeline/campplus-webgpu/runtime";
import type { CampPlusPackedSection } from "./pipeline/campplus-webgpu/metadata";

const METADATA_URL = "/models/campplus-t150-webgpu-fp16.json";
const INPUT_CHANNELS = 320;
const INPUT_FRAMES = 150;
const OUTPUT_CHANNELS = 128;
const OUTPUT_FRAMES = 75;
const INPUT_BYTES = INPUT_CHANNELS * INPUT_FRAMES * 2;
const OUTPUT_OFFSET = 96_000;
const OUTPUT_BYTES = OUTPUT_CHANNELS * OUTPUT_FRAMES * 2;

export interface RawCampPlusDiagnosticResult {
  readonly ok: boolean;
  readonly loadAndCompileMs: number;
  readonly gpuMs: number;
  readonly cpuMs: number;
  readonly comparedValues: number;
  readonly maxAbsoluteError: number;
  readonly meanAbsoluteError: number;
  readonly cosineSimilarity: number;
  readonly exactHalfMatches: number;
  readonly gpuFingerprint: NumericFingerprint;
  readonly cpuFingerprint: NumericFingerprint;
  readonly ownedGpuBytes: number;
  readonly retainedCpuWeightBytes: number;
}

interface NumericFingerprint {
  readonly sum: number;
  readonly l2: number;
  readonly max: number;
  readonly nonzero: number;
}

export async function runRawCampPlusDiagnostic(root: HTMLElement): Promise<void> {
  root.innerHTML = `<section><h1>Raw CAM++ WebGPU diagnostic</h1><pre id="raw-campplus-result">Requesting WebGPU…</pre></section>`;
  const output = root.querySelector<HTMLElement>("#raw-campplus-result");
  if (output === null) throw new Error("Missing raw CAM++ diagnostic output");
  try {
    const result = await executeDiagnostic((message) => {
      output.textContent = message;
    });
    output.textContent = JSON.stringify(result, null, 2);
    output.dataset.status = result.ok ? "passed" : "failed";
    globalThis.dispatchEvent(
      new CustomEvent("senko-raw-campplus-diagnostic", { detail: result }),
    );
  } catch (error) {
    output.textContent = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    output.dataset.status = "error";
    throw error;
  }
}

async function executeDiagnostic(
  report: (message: string) => void,
): Promise<RawCampPlusDiagnosticResult> {
  if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
  if (!isLittleEndian()) throw new Error("The packed CAM++ diagnostic requires little-endian typed arrays");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("No WebGPU adapter is available");
  if (!adapter.features.has("shader-f16")) throw new Error("Adapter lacks shader-f16");
  const device = await adapter.requestDevice({ requiredFeatures: ["shader-f16"] });
  let foundation: RawCampPlusFoundation | undefined;
  let dispatch: ReturnType<RawCampPlusFoundation["createPackedConvolution"]> | undefined;
  let readback: GPUBuffer | undefined;
  try {
    report("Streaming and validating the real 13.2 MiB packed model…");
    const loadStart = performance.now();
    foundation = await RawCampPlusFoundation.create(device, METADATA_URL, {
      onProgress(progress) {
        if (progress.stage === "weights") {
          report(
            `Streaming weights: ${formatMiB(progress.loadedBytes)} / ${formatMiB(progress.totalBytes)} MiB`,
          );
        } else {
          report(`${progress.stage[0]!.toUpperCase()}${progress.stage.slice(1)}…`);
        }
      },
    });
    const loadAndCompileMs = performance.now() - loadStart;
    const weight = foundation.gpuPackage.section(
      foundation.gpuPackage.metadata.fusedProgram.tdnn.weight,
    );
    const bias = foundation.gpuPackage.section(
      foundation.gpuPackage.metadata.fusedProgram.tdnn.bias,
    );
    if (
      weight.logicalShape[0] !== OUTPUT_CHANNELS ||
      weight.logicalShape[1] !== INPUT_CHANNELS ||
      weight.logicalShape[2] !== 5
    ) {
      throw new Error("Unexpected packed initial TDNN shape");
    }

    report("Extracting the two TDNN sections for the independent CPU oracle…");
    const sections = await fetchSectionBytes(foundation.gpuPackage.binaryUrl, [weight, bias]);
    const weightBytes = sections.get(weight.id);
    const biasBytes = sections.get(bias.id);
    if (weightBytes === undefined || biasBytes === undefined) {
      throw new Error("Failed to extract TDNN reference sections");
    }
    const packedWeight = toUint16(weightBytes);
    const packedBias = toUint16(biasBytes);
    const input = deterministicInput();
    const inputSlice = foundation.arena.slice("diagnostic-tdnn-input", 0, INPUT_BYTES);
    const outputSlice = foundation.arena.slice(
      "diagnostic-tdnn-output",
      OUTPUT_OFFSET,
      OUTPUT_BYTES,
    );
    foundation.arena.upload(device, inputSlice, input);
    dispatch = foundation.createPackedConvolution({
      label: "senko-campplus-real-tdnn-diagnostic",
      convolution: foundation.gpuPackage.metadata.fusedProgram.tdnn,
      input: inputSlice,
      output: outputSlice,
      batchSize: 1,
      inputChannels: INPUT_CHANNELS,
      inputFrames: INPUT_FRAMES,
      outputFrames: OUTPUT_FRAMES,
      stride: 2,
      dilation: 1,
      padLeft: 2,
      padRight: 2,
      outputRelu: true,
    });
    readback = device.createBuffer({
      label: "senko-campplus-real-tdnn-diagnostic-readback",
      size: OUTPUT_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    report("Running the real packed TDNN on WebGPU…");
    const encoder = device.createCommandEncoder({ label: "senko-campplus-tdnn-diagnostic" });
    dispatch.encode(encoder);
    encoder.copyBufferToBuffer(
      foundation.arena.buffer,
      outputSlice.byteOffset,
      readback,
      0,
      OUTPUT_BYTES,
    );
    const gpuStart = performance.now();
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Uint16Array(readback.getMappedRange()).slice();
    readback.unmap();
    const gpuMs = performance.now() - gpuStart;

    report("Evaluating 15.36 million reference MACs on the CPU…");
    const cpuStart = performance.now();
    const expected = evaluatePackedBctConvReference(
      input,
      packedWeight,
      packedBias,
      {
        batchSize: 1,
        inputChannels: INPUT_CHANNELS,
        outputChannels: OUTPUT_CHANNELS,
        inputFrames: INPUT_FRAMES,
        outputFrames: OUTPUT_FRAMES,
        kernelElements: 5,
        stride: 2,
        dilation: 1,
        padLeft: 2,
        outputRelu: true,
      },
    );
    const cpuMs = performance.now() - cpuStart;
    const comparison = compareHalfOutputs(actual, expected);
    return {
      ok: comparison.maxAbsoluteError <= 0.015 && comparison.cosineSimilarity >= 0.999_999,
      loadAndCompileMs,
      gpuMs,
      cpuMs,
      comparedValues: actual.length,
      ...comparison,
      gpuFingerprint: fingerprint(actual),
      cpuFingerprint: fingerprint(expected),
      ownedGpuBytes: foundation.gpuBytes.total + OUTPUT_BYTES + 64,
      retainedCpuWeightBytes: weightBytes.byteLength + biasBytes.byteLength,
    };
  } finally {
    if (readback?.mapState === "mapped") readback.unmap();
    readback?.destroy();
    dispatch?.destroy();
    foundation?.destroy();
    device.destroy();
  }
}

async function fetchSectionBytes(
  binaryUrl: string,
  requested: readonly CampPlusPackedSection[],
): Promise<ReadonlyMap<string, Uint8Array>> {
  const response = await fetch(binaryUrl);
  if (!response.ok || response.body === null) {
    throw new Error(`Unable to stream diagnostic TDNN sections (${response.status})`);
  }
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
        const sectionStart = section.byteOffset;
        const sectionEnd = section.byteOffset + section.byteLength;
        const overlapStart = Math.max(chunkStart, sectionStart);
        const overlapEnd = Math.min(chunkEnd, sectionEnd);
        if (overlapStart >= overlapEnd) continue;
        buffers
          .get(section.id)!
          .set(
            item.value.subarray(overlapStart - chunkStart, overlapEnd - chunkStart),
            overlapStart - sectionStart,
          );
      }
      absoluteOffset = chunkEnd;
    }
    return buffers;
  } finally {
    reader.releaseLock();
  }
}

function deterministicInput(): Uint16Array<ArrayBuffer> {
  const result = new Uint16Array(INPUT_CHANNELS * INPUT_FRAMES);
  for (let channel = 0; channel < INPUT_CHANNELS; channel += 1) {
    for (let frame = 0; frame < INPUT_FRAMES; frame += 1) {
      const value =
        Math.sin(channel * 0.071 + frame * 0.037) * 0.7 +
        Math.cos(channel * 0.019 - frame * 0.053) * 0.35;
      result[channel * INPUT_FRAMES + frame] = float32ToFloat16Bits(value);
    }
  }
  return result;
}

function compareHalfOutputs(actual: Uint16Array, expected: Uint16Array): {
  readonly maxAbsoluteError: number;
  readonly meanAbsoluteError: number;
  readonly cosineSimilarity: number;
  readonly exactHalfMatches: number;
} {
  if (actual.length !== expected.length) throw new Error("Diagnostic output lengths differ");
  let maxAbsoluteError = 0;
  let absoluteError = 0;
  let dot = 0;
  let actualNorm = 0;
  let expectedNorm = 0;
  let exactHalfMatches = 0;
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] === expected[index]) exactHalfMatches += 1;
    const actualValue = float16BitsToFloat32(actual[index]!);
    const expectedValue = float16BitsToFloat32(expected[index]!);
    const error = Math.abs(actualValue - expectedValue);
    maxAbsoluteError = Math.max(maxAbsoluteError, error);
    absoluteError += error;
    dot += actualValue * expectedValue;
    actualNorm += actualValue * actualValue;
    expectedNorm += expectedValue * expectedValue;
  }
  return {
    maxAbsoluteError,
    meanAbsoluteError: absoluteError / actual.length,
    cosineSimilarity: dot / Math.sqrt(actualNorm * expectedNorm),
    exactHalfMatches,
  };
}

function fingerprint(values: Uint16Array): NumericFingerprint {
  let sum = 0;
  let squared = 0;
  let max = -Infinity;
  let nonzero = 0;
  for (const bits of values) {
    const value = float16BitsToFloat32(bits);
    sum += value;
    squared += value * value;
    max = Math.max(max, value);
    if (value !== 0) nonzero += 1;
  }
  return { sum, l2: Math.sqrt(squared), max, nonzero };
}

function toUint16(bytes: Uint8Array): Uint16Array {
  if (bytes.byteLength % 2 !== 0) throw new Error("FP16 section has an odd byte length");
  return new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
}

function isLittleEndian(): boolean {
  const bytes = new Uint8Array(new Uint16Array([0x1234]).buffer);
  return bytes[0] === 0x34;
}

function formatMiB(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1);
}
