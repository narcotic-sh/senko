/// <reference types="@webgpu/types" />

import { Pcm16WavReader } from "./audio/wav";
import { BrowserModelSet } from "./pipeline/browser-models";
import type { MonoPcmSource } from "./pipeline/types";
import { createVadChunks, runVad } from "./pipeline/vad";
import type { PageMemoryPerformanceSource } from "./runtime/page-memory";

interface ChromiumPerformanceMemory {
  readonly usedJSHeapSize: number;
  readonly totalJSHeapSize: number;
  readonly jsHeapSizeLimit: number;
}

const inputElement = document.querySelector<HTMLInputElement>("#audio");
const outputElement = document.querySelector<HTMLPreElement>("#output");
if (inputElement === null || outputElement === null) {
  throw new Error("Missing diagnostic controls");
}
const input = inputElement;
const output = outputElement;

const started = performance.now();
function report(message: string): void {
  output.textContent += `\n${(performance.now() - started).toFixed(1)} ms  ${message}`;
}

function heap(): ChromiumPerformanceMemory | undefined {
  return (
    performance as Performance & {
      readonly memory?: ChromiumPerformanceMemory;
    }
  ).memory;
}

input.addEventListener("change", () => {
  const file = input.files?.[0];
  if (file === undefined) return;
  input.disabled = true;
  void run(file).catch((error: unknown) => {
    report(`ERROR ${error instanceof Error ? `${error.name}: ${error.message}` : String(error)}`);
    throw error;
  });
});

async function run(file: File): Promise<void> {
  output.textContent = "0.0 ms  Starting file run";
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const initialHeap = heap();
  const reader = await Pcm16WavReader.open(file);
  const chunks = createVadChunks(reader.sampleCount);
  report(
    `WAV file=${file.name}, bytes=${file.size}, duration=${reader.info.durationSeconds.toFixed(3)} s, chunks=${chunks.length}`,
  );

  const manifestUrl = new URL("/models/manifest.json", location.href).toString();
  const loadStarted = performance.now();
  const models = await BrowserModelSet.load(manifestUrl, navigator.gpu, {
    vadBatchSize: 8,
    warmupRuns: 1,
    onProgress: ({ message }) => report(message),
  });
  const loadMs = performance.now() - loadStarted;
  const exactGpuBytes = models.knownGpuBufferBytes;
  report(
    `dual-resident raw models ready in ${loadMs.toFixed(3)} ms, exact GPU=${exactGpuBytes}`,
  );

  const source: MonoPcmSource = {
    sampleRate: reader.sampleRate,
    sampleCount: reader.sampleCount,
    async readInto(sampleOffset, sampleCount, destination, destinationOffset = 0) {
      const written = await reader.readSamplesInto(
        sampleOffset,
        destination,
        destinationOffset,
        sampleCount,
      );
      if (written !== sampleCount) {
        throw new Error(`Short PCM read: ${written}/${sampleCount}`);
      }
    },
  };

  let peakUsedJsHeapBytes = initialHeap?.usedJSHeapSize ?? 0;
  const inferenceStarted = performance.now();
  let lastCompleted = 0;
  const segments = await runVad(source, models.vad, (completed, total) => {
    lastCompleted = completed;
    peakUsedJsHeapBytes = Math.max(
      peakUsedJsHeapBytes,
      heap()?.usedJSHeapSize ?? 0,
    );
    if (completed === total || completed % 80 === 0) {
      report(`VAD progress ${completed}/${total}`);
    }
  });
  const inferenceMs = performance.now() - inferenceStarted;
  const immediateEndingHeap = heap();
  await models.release();
  await new Promise<void>((resolve) => setTimeout(resolve, 5_000));
  const settledHeap = heap();
  const pageMemorySource = performance as Performance & PageMemoryPerformanceSource;
  const measurePageMemory = pageMemorySource.measureUserAgentSpecificMemory;
  let pageAgentClusterBytes: number | undefined;
  let pageMemoryElapsedMs: number | undefined;
  let pageMemoryError: string | undefined;
  if (measurePageMemory !== undefined) {
    report("requesting page-agent-cluster memory");
    const pageMemoryStarted = performance.now();
    try {
      pageAgentClusterBytes = (await measurePageMemory.call(pageMemorySource)).bytes;
    } catch (error) {
      pageMemoryError = error instanceof Error ? error.message : String(error);
    }
    pageMemoryElapsedMs = performance.now() - pageMemoryStarted;
  }

  const result = {
    file: {
      bytes: file.size,
      durationSeconds: reader.info.durationSeconds,
      samples: reader.sampleCount,
      chunks: chunks.length,
      batches: Math.ceil(chunks.length / 8),
    },
    loadAndWarmMs: loadMs,
    vadMs: inferenceMs,
    chunksPerSecond: chunks.length / (inferenceMs / 1_000),
    audioRealtimeFactor: inferenceMs / (reader.info.durationSeconds * 1_000),
    completedChunks: lastCompleted,
    segments: segments.length,
    speechSeconds: segments.reduce(
      (sum, segment) => sum + segment.end - segment.start,
      0,
    ),
    memory: {
      exactGpuBytes,
      fileBytesDiskBacked: file.size,
      vadInputBatchBytes: 8 * 160_000 * 4,
      vadOutputBatchBytes: 8 * 589 * 7 * 4,
      reusableWavReadBufferBytes: reader.reusableReadBufferBytes,
      exactLogicalLiveCpuBytes:
        reader.reusableReadBufferBytes + 8 * 160_000 * 4 + 8 * 589 * 7 * 4,
      initialUsedJsHeapBytes: initialHeap?.usedJSHeapSize,
      peakObservedUsedJsHeapBytes: peakUsedJsHeapBytes,
      immediateEndingUsedJsHeapBytes: immediateEndingHeap?.usedJSHeapSize,
      settledFiveSecondUsedJsHeapBytes: settledHeap?.usedJSHeapSize,
      jsHeapSizeLimit: settledHeap?.jsHeapSizeLimit,
      pageMemorySupported: measurePageMemory !== undefined,
      pageAgentClusterBytes,
      pageMemoryElapsedMs,
      pageMemoryError,
    },
  };
  Object.assign(globalThis, { __senkoRawVadFileDiagnostic: result });
  report(`summary ${JSON.stringify(result)}`);
}
