import fbankWasmUrl from "./wasm/senko-fbank.wasm?url";

import {
  SENKO_FBANK_BINS,
  frameCountForSamples,
  type FbankComputeHint,
  type FbankComputer,
  type FbankMatrix,
} from "./fbank";

let compiledModule: Promise<WebAssembly.Module> | undefined;

interface SenkoFbankWasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;
  readonly _initialize: () => void;
  readonly fbank_init: () => number;
  readonly fbank_input_ptr: () => number;
  readonly fbank_output_ptr: () => number;
  readonly fbank_compute: (
    sampleCount: number,
    reusableFrameShift: number,
  ) => number;
  readonly fbank_reset_reuse: () => void;
  readonly fbank_dispose: () => void;
  readonly fbank_max_samples: () => number;
  readonly fbank_max_frames: () => number;
  readonly fbank_bins: () => number;
}

export interface WasmSenkoFbankMemoryStats {
  readonly heapBytes: number;
  readonly inputCapacitySamples: number;
  readonly outputCapacityFrames: number;
}

/**
 * SIMD WebAssembly implementation of Senko's Kaldi-style FBank.
 *
 * The returned matrix aliases the module's reusable output arena and remains
 * valid only until the next `compute` call. Consumers should upload/copy it
 * before advancing the stream. One instance uses a fixed 512 KiB heap which
 * cannot grow.
 */
export class WasmSenkoFbank implements FbankComputer {
  private exports: SenkoFbankWasmExports | undefined;
  private input: Float32Array | undefined;
  private output: Float32Array | undefined;
  readonly memoryStats: WasmSenkoFbankMemoryStats;

  private constructor(exports: SenkoFbankWasmExports) {
    exports._initialize();
    if (exports.fbank_init() !== 1) {
      throw new Error("Failed to initialize Senko FBank WebAssembly module");
    }

    const inputCapacitySamples = exports.fbank_max_samples();
    const outputCapacityFrames = exports.fbank_max_frames();
    const binCount = exports.fbank_bins();
    if (binCount !== SENKO_FBANK_BINS) {
      throw new Error(
        `FBank WebAssembly module exposes ${binCount} bins; expected ${SENKO_FBANK_BINS}`,
      );
    }

    const inputPointer = exports.fbank_input_ptr();
    const outputPointer = exports.fbank_output_ptr();
    this.input = checkedFloat32View(
      exports.memory,
      inputPointer,
      inputCapacitySamples,
      "input",
    );
    this.output = checkedFloat32View(
      exports.memory,
      outputPointer,
      outputCapacityFrames * binCount,
      "output",
    );
    this.exports = exports;
    this.memoryStats = {
      heapBytes: exports.memory.buffer.byteLength,
      inputCapacitySamples,
      outputCapacityFrames,
    };
  }

  static async create(): Promise<WasmSenkoFbank> {
    compiledModule ??= WebAssembly.compileStreaming(fetch(fbankWasmUrl));
    const instance = await WebAssembly.instantiate(await compiledModule, {});
    return WasmSenkoFbank.fromInstance(instance);
  }

  /** Instantiate supplied bytes, primarily for Node/Vitest verification. */
  static async fromBytes(bytes: BufferSource): Promise<WasmSenkoFbank> {
    const instantiated = await WebAssembly.instantiate(bytes, {});
    const instance =
      instantiated instanceof WebAssembly.Instance
        ? instantiated
        : instantiated.instance;
    return WasmSenkoFbank.fromInstance(instance);
  }

  compute(samples: Float32Array, hint?: FbankComputeHint): FbankMatrix {
    const exports = this.requireExports();
    const input = this.input;
    const output = this.output;
    if (input === undefined || output === undefined) {
      throw new Error("WasmSenkoFbank has been disposed");
    }
    if (samples.length > this.memoryStats.inputCapacitySamples) {
      throw new RangeError(
        `FBank input has ${samples.length} samples; fixed WASM capacity is ${this.memoryStats.inputCapacitySamples}`,
      );
    }

    const expectedFrameCount = frameCountForSamples(samples.length);
    if (expectedFrameCount > this.memoryStats.outputCapacityFrames) {
      throw new RangeError(
        `FBank output needs ${expectedFrameCount} frames; fixed WASM capacity is ${this.memoryStats.outputCapacityFrames}`,
      );
    }
    const reusableFrameShift = hint?.reusableFrameShift ?? 0;
    if (!Number.isInteger(reusableFrameShift) || reusableFrameShift < 0) {
      throw new RangeError("reusableFrameShift must be a non-negative integer");
    }

    input.set(samples);
    const frameCount = exports.fbank_compute(
      samples.length,
      reusableFrameShift,
    );
    if (frameCount !== expectedFrameCount) {
      throw new Error(
        `FBank WebAssembly computation failed (${frameCount}); expected ${expectedFrameCount} frames`,
      );
    }
    return {
      data: output.subarray(0, frameCount * SENKO_FBANK_BINS),
      frameCount,
      binCount: SENKO_FBANK_BINS,
    };
  }

  resetReuse(): void {
    this.requireExports().fbank_reset_reuse();
  }

  dispose(): void {
    if (this.exports === undefined) return;
    this.exports.fbank_dispose();
    this.input = undefined;
    this.output = undefined;
    this.exports = undefined;
  }

  private static fromInstance(instance: WebAssembly.Instance): WasmSenkoFbank {
    return new WasmSenkoFbank(instance.exports as SenkoFbankWasmExports);
  }

  private requireExports(): SenkoFbankWasmExports {
    if (this.exports === undefined) {
      throw new Error("WasmSenkoFbank has been disposed");
    }
    return this.exports;
  }
}

function checkedFloat32View(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
  label: string,
): Float32Array {
  if (pointer % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error(`Unaligned FBank WASM ${label} pointer ${pointer}`);
  }
  const byteEnd = pointer + length * Float32Array.BYTES_PER_ELEMENT;
  if (pointer < 0 || length < 0 || byteEnd > memory.buffer.byteLength) {
    throw new Error(`FBank WASM ${label} arena is outside linear memory`);
  }
  return new Float32Array(memory.buffer, pointer, length);
}
