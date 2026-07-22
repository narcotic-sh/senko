import { open, readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const webDirectory = fileURLToPath(new URL("../..", import.meta.url));
const repositoryDirectory = fileURLToPath(new URL("../../..", import.meta.url));
const wasmPath = `${webDirectory}/src/audio/wasm/senko-fbank.wasm`;
const wavPath = process.argv[2] ?? `${repositoryDirectory}/test_audio.wav`;
const windowCount = Number.parseInt(process.argv[3] ?? "6161", 10);
const reuseEnabled = process.env.FBANK_REUSE !== "0";
const windowSamples = 24_000;
const shiftSamples = 9_600;
const shiftFrames = shiftSamples / 160;

if (!Number.isSafeInteger(windowCount) || windowCount < 1) {
  throw new Error("window count must be a positive integer");
}

const wasmBytes = await readFile(wasmPath);
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const wasm = instance.exports;
wasm._initialize();
if (wasm.fbank_init() !== 1) throw new Error("fbank_init failed");

const input = new Float32Array(
  wasm.memory.buffer,
  wasm.fbank_input_ptr(),
  wasm.fbank_max_samples(),
);
const output = new Float32Array(
  wasm.memory.buffer,
  wasm.fbank_output_ptr(),
  wasm.fbank_max_frames() * wasm.fbank_bins(),
);
const pcmBytes = Buffer.allocUnsafe(windowSamples * 2);
const wav = await open(wavPath, "r");

try {
  const wavInfo = await parseWav(wav);
  const consecutiveWindows =
    Math.floor((wavInfo.sampleCount - windowSamples) / shiftSamples) + 1;

  if (globalThis.gc) globalThis.gc();
  const memoryBefore = process.memoryUsage();
  let peakRss = memoryBefore.rss;
  let peakHeapUsed = memoryBefore.heapUsed;
  let checksum = 0;
  let decodedSamples = 0;
  let reusedWindows = 0;
  let computedRawFrames = 0;
  let previousStart = -1;
  let overlapCopyMs = 0;
  let fileReadMs = 0;
  let pcmDecodeMs = 0;
  let fbankComputeMs = 0;

  const start = performance.now();
  for (let window = 0; window < windowCount; window += 1) {
    // Wrap only when the requested benchmark count exceeds the contiguous
    // windows available in the fixture. The one-hour file fits 6,158 of the
    // 6,161 requested windows, so this adds just one discontinuity.
    const windowStart = (window % consecutiveWindows) * shiftSamples;
    const canReuse =
      reuseEnabled && windowStart === previousStart + shiftSamples;
    let targetSampleOffset = 0;
    let samplesToRead = windowSamples;
    if (canReuse) {
      const overlapCopyStart = performance.now();
      input.copyWithin(0, shiftSamples, windowSamples);
      overlapCopyMs += performance.now() - overlapCopyStart;
      targetSampleOffset = windowSamples - shiftSamples;
      samplesToRead = shiftSamples;
      reusedWindows += 1;
    }

    const sourceSample = windowStart + targetSampleOffset;
    const byteCount = samplesToRead * 2;
    const fileReadStart = performance.now();
    const { bytesRead } = await wav.read(
      pcmBytes,
      0,
      byteCount,
      wavInfo.dataOffset + sourceSample * 2,
    );
    fileReadMs += performance.now() - fileReadStart;
    if (bytesRead !== byteCount) throw new Error(`short PCM read at window ${window}`);
    const pcmDecodeStart = performance.now();
    for (let i = 0; i < samplesToRead; ++i) {
      input[targetSampleOffset + i] = pcmBytes.readInt16LE(i * 2) / 32_768;
    }
    pcmDecodeMs += performance.now() - pcmDecodeStart;
    decodedSamples += samplesToRead;

    const fbankComputeStart = performance.now();
    const frames = wasm.fbank_compute(
      windowSamples,
      canReuse ? shiftFrames : 0,
    );
    fbankComputeMs += performance.now() - fbankComputeStart;
    if (frames !== 148) throw new Error(`FBank failed at window ${window}: ${frames}`);
    computedRawFrames += canReuse ? shiftFrames : frames;
    checksum += output[(window * 977) % output.length];
    previousStart = windowStart;

    if ((window & 127) === 0) {
      const usage = process.memoryUsage();
      peakRss = Math.max(peakRss, usage.rss);
      peakHeapUsed = Math.max(peakHeapUsed, usage.heapUsed);
    }
  }
  const elapsedMs = performance.now() - start;
  const memoryAfter = process.memoryUsage();
  peakRss = Math.max(peakRss, memoryAfter.rss);
  peakHeapUsed = Math.max(peakHeapUsed, memoryAfter.heapUsed);

  console.log(
    JSON.stringify(
      {
        wasmPath,
        wavPath,
        windowCount,
        reuseEnabled,
        frames: windowCount * 148,
        computedRawFrames,
        decodedSamples,
        reusedWindows,
        elapsedMs,
        stageMs: {
          overlapCopy: overlapCopyMs,
          fileRead: fileReadMs,
          pcmDecode: pcmDecodeMs,
          fbankCompute: fbankComputeMs,
          unclassified:
            elapsedMs - overlapCopyMs - fileReadMs - pcmDecodeMs - fbankComputeMs,
        },
        windowsPerSecond: (windowCount * 1000) / elapsedMs,
        realTimeFactor: elapsedMs / 1000 / wavInfo.durationSeconds,
        wasmHeapBytes: wasm.memory.buffer.byteLength,
        processMemory: {
          rssBaselineBytes: memoryBefore.rss,
          rssPeakBytes: peakRss,
          rssPeakDeltaBytes: peakRss - memoryBefore.rss,
          heapUsedBaselineBytes: memoryBefore.heapUsed,
          heapUsedPeakBytes: peakHeapUsed,
          heapUsedEndBytes: memoryAfter.heapUsed,
          externalEndBytes: memoryAfter.external,
          arrayBuffersEndBytes: memoryAfter.arrayBuffers,
        },
        checksum,
      },
      null,
      2,
    ),
  );
} finally {
  wasm.fbank_dispose();
  await wav.close();
}

async function parseWav(file) {
  const header = Buffer.allocUnsafe(12);
  await readExactly(file, header, 0);
  if (header.toString("ascii", 0, 4) !== "RIFF" || header.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error("expected RIFF/WAVE");
  }

  let offset = 12;
  let format;
  let dataOffset;
  let dataLength;
  const chunkHeader = Buffer.allocUnsafe(8);
  while (format === undefined || dataOffset === undefined) {
    await readExactly(file, chunkHeader, offset);
    const id = chunkHeader.toString("ascii", 0, 4);
    const length = chunkHeader.readUInt32LE(4);
    if (id === "fmt ") {
      const bytes = Buffer.allocUnsafe(Math.max(16, length));
      await readExactly(file, bytes, offset + 8);
      format = {
        encoding: bytes.readUInt16LE(0),
        channels: bytes.readUInt16LE(2),
        sampleRate: bytes.readUInt32LE(4),
        bits: bytes.readUInt16LE(14),
      };
    } else if (id === "data") {
      dataOffset = offset + 8;
      dataLength = length;
    }
    offset += 8 + length + (length & 1);
  }
  if (
    format.encoding !== 1 ||
    format.channels !== 1 ||
    format.sampleRate !== 16_000 ||
    format.bits !== 16
  ) {
    throw new Error(`unsupported WAV format: ${JSON.stringify(format)}`);
  }
  const sampleCount = dataLength / 2;
  return {
    dataOffset,
    sampleCount,
    durationSeconds: sampleCount / format.sampleRate,
  };
}

async function readExactly(file, target, position) {
  const { bytesRead } = await file.read(target, 0, target.length, position);
  if (bytesRead !== target.length) throw new Error(`short read at byte ${position}`);
}
