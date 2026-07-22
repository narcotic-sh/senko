import * as ort from "onnxruntime-web";
// The browser entry uses ORT's mature JSEP WebGPU backend. Do not pair this
// binary with `onnxruntime-web/webgpu`: that entry is the newer native WebGPU
// EP and requires the incompatible asyncify binary.
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url";

import type {
  SelectedSegmentationSplit,
  VadBufferBytes,
} from "./model-manifest";
import {
  PersistentWebGpuLstm,
  type PersistentLstmBufferBytes,
} from "./persistent-lstm";
import type { EmbeddingBatchBackend, VadBatchBackend } from "./types";
import {
  VAD_CHUNK_SAMPLES,
  VAD_OUTPUT_CLASSES,
  VAD_OUTPUT_FRAMES,
} from "./vad";

export interface OrtModelAsset {
  url: string;
  inputName?: string;
  outputName?: string;
  sha256?: string;
  byteLength?: number;
}

export type OrtLoadProgress = (message: string) => void;

export interface OrtRuntimeOptions {
  adapter?: GPUAdapter;
  device?: GPUDevice;
  graphCapture?: boolean;
  graphOptimizationLevel?: ort.InferenceSession.SessionOptions["graphOptimizationLevel"];
  logLevel?: "verbose" | "info" | "warning" | "error" | "fatal";
  profile?: boolean;
  strictWebGpu?: boolean;
}

export interface OrtRuntime {
  readonly adapter: GPUAdapter | undefined;
  readonly device: Promise<GPUDevice>;
  readonly graphCapture: boolean;
  readonly strictWebGpu: boolean;
  readonly sessionOptions: ort.InferenceSession.SessionOptions;
}

let configured = false;
let configuredAdapter: GPUAdapter | undefined;
let configuredDevice: GPUDevice | undefined;

/** Configure ORT once, before the first inference session is constructed. */
export function configureOrt(options: OrtRuntimeOptions = {}): OrtRuntime {
  if (!configured) {
    ort.env.logLevel = options.logLevel ?? "warning";
    ort.env.wasm.proxy = false;
    // Inference is WebGPU-only. Nested ORT pthread workers add startup cost and
    // can deadlock when ORT itself runs inside our dedicated module worker.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = { wasm: new URL(ortWasmUrl, globalThis.location.href) };
    configuredAdapter = options.adapter;
    configuredDevice = options.device;
    if (options.device !== undefined) ort.env.webgpu.device = options.device;
    else if (options.adapter !== undefined) ort.env.webgpu.adapter = options.adapter;
    if (options.profile) {
      ort.env.webgpu.profiling = { mode: "default" };
    }
    configured = true;
  } else if (options.adapter !== undefined && configuredAdapter !== options.adapter) {
    throw new Error("ONNX Runtime Web was already initialized with another GPUAdapter");
  } else if (options.device !== undefined && configuredDevice !== options.device) {
    throw new Error("ONNX Runtime Web was already initialized with another GPUDevice");
  }

  const webgpuBase: ort.InferenceSession.WebGpuExecutionProviderOption = {
    name: "webgpu",
    preferredLayout: "NCHW",
    validationMode: "wgpuOnly",
  };
  const graphCapture = options.graphCapture ?? true;
  if (graphCapture && options.adapter === undefined && ort.env.webgpu.adapter === undefined) {
    throw new Error("ORT graph capture requires an explicit GPUAdapter");
  }
  const strictWebGpu = options.strictWebGpu !== false;
  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: [webgpuBase],
    graphOptimizationLevel: options.graphOptimizationLevel ?? "all",
    enableGraphCapture: graphCapture,
    // executionProviders alone does not prevent ORT from registering its CPU
    // EP. Strict mode turns any accidental unsupported node into a hard failure.
    ...(!strictWebGpu
      ? {}
      : { extra: { session: { disable_cpu_ep_fallback: "1" } } }),
  };
  return {
    adapter: configuredAdapter,
    get device(): Promise<GPUDevice> {
      return configuredDevice === undefined
        ? ort.env.webgpu.device
        : Promise.resolve(configuredDevice);
    },
    graphCapture,
    strictWebGpu,
    sessionOptions,
  };
}

async function loadSession(
  runtime: OrtRuntime,
  asset: OrtModelAsset,
  onProgress?: OrtLoadProgress,
): Promise<ort.InferenceSession> {
  onProgress?.("Fetching ONNX graph");
  const model = await fetchVerifiedAsset(asset);
  onProgress?.("Creating ORT session");
  const session = await ort.InferenceSession.create(model, runtime.sessionOptions);
  onProgress?.("ORT session ready");
  return session;
}

async function fetchVerifiedAsset(asset: OrtModelAsset): Promise<ArrayBuffer> {
  const response = await fetch(asset.url);
  if (!response.ok) {
    throw new Error(`Failed to load model ${asset.url}: HTTP ${response.status}`);
  }
  const data = await response.arrayBuffer();
  if (asset.byteLength !== undefined && data.byteLength !== asset.byteLength) {
    throw new Error(
      `Model ${asset.url} has ${data.byteLength} bytes; expected ${asset.byteLength}`,
    );
  }
  if (asset.sha256 !== undefined) {
    const actual = bytesToHex(await crypto.subtle.digest("SHA-256", data));
    if (actual !== asset.sha256.toLowerCase()) {
      throw new Error(`SHA-256 mismatch for ${asset.url}: ${actual}`);
    }
  }
  return data;
}

function bytesToHex(buffer: ArrayBuffer): string {
  return [...new Uint8Array(buffer)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function selectName(
  requested: string | undefined,
  available: readonly string[],
  kind: "input" | "output",
): string {
  if (requested !== undefined) {
    if (!available.includes(requested)) {
      throw new Error(`Model ${kind} '${requested}' is not one of: ${available.join(", ")}`);
    }
    return requested;
  }
  if (available.length !== 1) {
    throw new Error(`Model has ${available.length} ${kind}s; specify one explicitly`);
  }
  return available[0]!;
}

function copyFloatOutput(tensor: ort.Tensor): Float32Array {
  if (!(tensor.data instanceof Float32Array)) {
    throw new Error(`Expected float32 output, received ${tensor.type}`);
  }
  // ORT owns tensor storage and may recycle it on the next run.
  return tensor.data.slice();
}

/**
 * Persistent external I/O buffers required by ORT WebGPU graph capture.
 *
 * ORT records the buffer bindings on the first run and rejects replacement
 * buffers afterwards. Keeping both tensors and buffers alive also avoids a
 * CPU tensor allocation/upload and a GPU output allocation on every batch.
 */
class CapturedFloat32Io {
  readonly inputTensor: ort.Tensor;
  readonly outputTensor: ort.Tensor;
  readonly bufferBytes: {
    readonly input: number;
    readonly output: number;
    readonly readback: number;
    readonly total: number;
  };

  private readonly inputBuffer: GPUBuffer;
  private readonly outputBuffer: GPUBuffer;
  private readonly readbackBuffer: GPUBuffer;
  private readonly outputLength: number;
  private running = false;

  constructor(
    private readonly device: GPUDevice,
    inputDims: readonly number[],
    outputDims: readonly number[],
  ) {
    const inputLength = elementCount(inputDims);
    this.outputLength = elementCount(outputDims);
    const inputBytes = inputLength * Float32Array.BYTES_PER_ELEMENT;

    this.inputBuffer = device.createBuffer({
      label: "senko-ort-captured-input",
      size: inputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBytes = this.outputLength * Float32Array.BYTES_PER_ELEMENT;
    this.outputBuffer = device.createBuffer({
      label: "senko-ort-captured-output",
      size: outputBytes,
      // ORT can either dispatch directly into or copy into a preallocated
      // output. COPY_DST is required by the latter JSEP path.
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.readbackBuffer = device.createBuffer({
      label: "senko-ort-captured-readback",
      size: outputBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.inputTensor = ort.Tensor.fromGpuBuffer(this.inputBuffer, {
      dataType: "float32",
      dims: inputDims,
    });
    this.outputTensor = ort.Tensor.fromGpuBuffer(this.outputBuffer, {
      dataType: "float32",
      dims: outputDims,
    });
    this.bufferBytes = {
      input: inputBytes,
      output: outputBytes,
      readback: outputBytes,
      total: inputBytes + outputBytes * 2,
    };
  }

  async run(
    session: ort.InferenceSession,
    inputName: string,
    outputName: string,
    input: Float32Array,
  ): Promise<Float32Array> {
    if (this.running) {
      throw new Error("Concurrent runs are not supported by captured ORT buffers");
    }
    if (input.byteLength !== this.inputBuffer.size) {
      throw new RangeError(
        `Captured ORT input has ${input.byteLength} bytes; expected ${this.inputBuffer.size}`,
      );
    }

    this.running = true;
    try {
      let upload: Float32Array<ArrayBuffer>;
      if (input.buffer instanceof ArrayBuffer) {
        upload = new Float32Array(input.buffer, input.byteOffset, input.length);
      } else {
        upload = new Float32Array(input);
      }
      this.device.queue.writeBuffer(this.inputBuffer, 0, upload);
      const outputs = await session.run(
        { [inputName]: this.inputTensor },
        { [outputName]: this.outputTensor },
      );
      if (outputs[outputName] !== this.outputTensor) {
        throw new Error(`ORT output '${outputName}' is missing`);
      }
      return await readFloat32Buffer(
        this.device,
        this.outputBuffer,
        this.readbackBuffer,
        this.outputLength,
      );
    } finally {
      this.running = false;
    }
  }

  dispose(): void {
    this.inputTensor.dispose();
    this.outputTensor.dispose();
    this.inputBuffer.destroy();
    this.outputBuffer.destroy();
    this.readbackBuffer.destroy();
  }
}

function elementCount(dims: readonly number[]): number {
  return dims.reduce((count, dimension) => count * dimension, 1);
}

async function readFloat32Buffer(
  device: GPUDevice,
  source: GPUBuffer,
  readback: GPUBuffer,
  elementLength: number,
): Promise<Float32Array> {
  const encoder = device.createCommandEncoder({ label: "senko-ort-readback" });
  encoder.copyBufferToBuffer(source, 0, readback, 0, elementLength * 4);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  try {
    return new Float32Array(readback.getMappedRange()).slice();
  } finally {
    readback.unmap();
  }
}

/** The original full ONNX graph, retained only for diagnostic parity checks. */
export class OrtMonolithicVadBackend implements VadBatchBackend {
  readonly chunkSamples = VAD_CHUNK_SAMPLES;
  readonly outputFrames = VAD_OUTPUT_FRAMES;
  readonly outputClasses = VAD_OUTPUT_CLASSES;
  private released = false;

  private constructor(
    readonly batchSize: number,
    private readonly session: ort.InferenceSession,
    private readonly inputName: string,
    private readonly outputName: string,
    private readonly capturedIo: CapturedFloat32Io | undefined,
  ) {}

  static async create(
    runtime: OrtRuntime,
    asset: OrtModelAsset,
    batchSize: number,
    onProgress?: OrtLoadProgress,
  ): Promise<OrtMonolithicVadBackend> {
    const session = await loadSession(runtime, asset, onProgress);
    const inputName = selectName(asset.inputName, session.inputNames, "input");
    const outputName = selectName(asset.outputName, session.outputNames, "output");
    return new OrtMonolithicVadBackend(
      batchSize,
      session,
      inputName,
      outputName,
      runtime.graphCapture
        ? new CapturedFloat32Io(
            await runtime.device,
            [batchSize, 1, VAD_CHUNK_SAMPLES],
            [batchSize, VAD_OUTPUT_FRAMES, VAD_OUTPUT_CLASSES],
          )
        : undefined,
    );
  }

  async run(audio: Float32Array): Promise<Float32Array> {
    if (this.released) throw new Error("VAD backend has been released");
    const expected = this.batchSize * this.chunkSamples;
    if (audio.length !== expected) {
      throw new RangeError(`VAD input has ${audio.length} values; expected ${expected}`);
    }
    const result =
      this.capturedIo === undefined
        ? await this.runUncaptured(audio)
        : await this.capturedIo.run(
            this.session,
            this.inputName,
            this.outputName,
            audio,
          );
    const outputExpected =
      this.batchSize * this.outputFrames * this.outputClasses;
    if (result.length !== outputExpected) {
      throw new Error(`VAD output has ${result.length} values; expected ${outputExpected}`);
    }
    return result;
  }

  private async runUncaptured(audio: Float32Array): Promise<Float32Array> {
    const input = new ort.Tensor("float32", audio, [this.batchSize, 1, this.chunkSamples]);
    try {
      const outputs = await this.session.run({ [this.inputName]: input });
      const output = outputs[this.outputName];
      if (output === undefined) throw new Error(`VAD output '${this.outputName}' is missing`);
      try {
        return copyFloatOutput(output);
      } finally {
        output.dispose();
      }
    } finally {
      input.dispose();
    }
  }

  async release(): Promise<void> {
    if (this.released) return;
    this.released = true;
    try {
      await this.session.release();
    } finally {
      this.capturedIo?.dispose();
    }
  }
}

export interface OrtVadGpuBufferBytes {
  readonly audioInput: number;
  readonly frontendOutput: number;
  readonly lstm: PersistentLstmBufferBytes;
  /** Zero in production; ORT owns its uncaptured final output allocation. */
  readonly tailOutput: number;
  /** Zero in production; ORT performs the sole final download. */
  readonly readback: number;
  /** Buffers directly owned by this backend; excludes ORT session internals. */
  readonly totalOwned: number;
}

export interface GpuStageFingerprint {
  readonly length: number;
  readonly finite: number;
  readonly nonzero: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly sum: number;
  readonly l2: number;
  readonly first: readonly number[];
}

/** Diagnostic wall times with an explicit GPU queue drain at each boundary. */
export interface VadGpuProfile {
  readonly uploadMs: number;
  readonly frontendMs: number;
  readonly lstmMs: number;
  readonly tailAndReadbackMs: number;
  readonly totalMs: number;
  readonly output: Float32Array;
}

/**
 * Production pyannote segmentation backend.
 *
 * The frontend and tail are strict WebGPU ORT sessions. Their fixed external
 * tensors sandwich a hand-written persistent bidirectional LSTM. ORT owns and
 * downloads the uncaptured tail result because JSEP 1.27 does not reliably
 * write a preallocated GPU output for this graph. The only host transfers per
 * batch remain the waveform upload and final seven-logit readback.
 */
export class OrtVadBackend implements VadBatchBackend {
  readonly chunkSamples = VAD_CHUNK_SAMPLES;
  readonly outputFrames = VAD_OUTPUT_FRAMES;
  readonly outputClasses = VAD_OUTPUT_CLASSES;
  readonly declaredBufferBytes: VadBufferBytes;
  readonly gpuBufferBytes: OrtVadGpuBufferBytes;

  private running = false;
  private released = false;

  private constructor(
    readonly batchSize: number,
    private readonly device: GPUDevice,
    private readonly frontendSession: ort.InferenceSession,
    private readonly tailSession: ort.InferenceSession,
    private readonly frontendInputName: string,
    private readonly frontendOutputName: string,
    private readonly tailInputName: string,
    private readonly tailOutputName: string,
    private readonly audioInputBuffer: GPUBuffer,
    private readonly frontendOutputBuffer: GPUBuffer,
    private readonly frontendInputTensor: ort.Tensor,
    private readonly frontendOutputTensor: ort.Tensor,
    private readonly tailInputTensor: ort.Tensor,
    private readonly lstm: PersistentWebGpuLstm,
    declaredBufferBytes: VadBufferBytes,
    gpuBufferBytes: OrtVadGpuBufferBytes,
  ) {
    this.declaredBufferBytes = declaredBufferBytes;
    this.gpuBufferBytes = gpuBufferBytes;
  }

  static async create(
    runtime: OrtRuntime,
    selected: SelectedSegmentationSplit,
    onProgress?: OrtLoadProgress,
  ): Promise<OrtVadBackend> {
    if (!runtime.strictWebGpu) {
      throw new Error("Production pyannote segmentation requires strict WebGPU execution");
    }
    if (runtime.graphCapture) {
      throw new Error(
        "Production pyannote segmentation requires graph capture off for reliable ORT tail output",
      );
    }
    let frontendSession: ort.InferenceSession | undefined;
    let tailSession: ort.InferenceSession | undefined;
    let audioInputBuffer: GPUBuffer | undefined;
    let frontendOutputBuffer: GPUBuffer | undefined;
    let frontendInputTensor: ort.Tensor | undefined;
    let frontendOutputTensor: ort.Tensor | undefined;
    let tailInputTensor: ort.Tensor | undefined;
    let lstm: PersistentWebGpuLstm | undefined;

    try {
      frontendSession = await loadSession(runtime, selected.frontend.asset, (message) =>
        onProgress?.(`Frontend: ${message}`),
      );
      // JSEP creates a device from env.webgpu.adapter during the first session
      // initialization. The execution-provider `device` option belongs to the
      // native EP build and is ignored by JSEP, so external buffers must use
      // this exact device.
      const device = await runtime.device;

      const frontendInputName = selectName(
        selected.frontend.asset.inputName,
        frontendSession.inputNames,
        "input",
      );
      const frontendOutputName = selectName(
        selected.frontend.asset.outputName,
        frontendSession.outputNames,
        "output",
      );
      const audioInputBytes = selected.batchSize * VAD_CHUNK_SAMPLES * 4;
      const frontendOutputBytes = selected.batchSize * VAD_OUTPUT_FRAMES * 60 * 4;

      audioInputBuffer = device.createBuffer({
        label: "senko-pyannote-audio-input",
        size: audioInputBytes,
        usage:
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      frontendOutputBuffer = device.createBuffer({
        label: "senko-pyannote-frontend-output",
        size: frontendOutputBytes,
        usage:
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      frontendInputTensor = ort.Tensor.fromGpuBuffer(audioInputBuffer, {
        dataType: "float32",
        dims: [selected.batchSize, 1, VAD_CHUNK_SAMPLES],
      });
      frontendOutputTensor = ort.Tensor.fromGpuBuffer(frontendOutputBuffer, {
        dataType: "float32",
        dims: [selected.batchSize, VAD_OUTPUT_FRAMES, 60],
      });

      lstm = await PersistentWebGpuLstm.create(
        device,
        selected.batchSize,
        frontendOutputBuffer,
        selected.weights,
        selected.metadata,
        onProgress,
      );
      tailSession = await loadSession(runtime, selected.tail.asset, (message) =>
        onProgress?.(`Tail: ${message}`),
      );
      const tailInputName = selectName(
        selected.tail.asset.inputName,
        tailSession.inputNames,
        "input",
      );
      const tailOutputName = selectName(
        selected.tail.asset.outputName,
        tailSession.outputNames,
        "output",
      );
      tailInputTensor = ort.Tensor.fromGpuBuffer(lstm.outputBuffer, {
        dataType: "float32",
        dims: [selected.batchSize, VAD_OUTPUT_FRAMES, 256],
      });

      const gpuBufferBytes = {
        audioInput: audioInputBytes,
        frontendOutput: frontendOutputBytes,
        lstm: lstm.bufferBytes,
        tailOutput: 0,
        readback: 0,
        totalOwned:
          audioInputBytes +
          frontendOutputBytes +
          lstm.bufferBytes.total,
      } satisfies OrtVadGpuBufferBytes;

      return new OrtVadBackend(
        selected.batchSize,
        device,
        frontendSession,
        tailSession,
        frontendInputName,
        frontendOutputName,
        tailInputName,
        tailOutputName,
        audioInputBuffer,
        frontendOutputBuffer,
        frontendInputTensor,
        frontendOutputTensor,
        tailInputTensor,
        lstm,
        selected.declaredBufferBytes,
        gpuBufferBytes,
      );
    } catch (error) {
      frontendInputTensor?.dispose();
      frontendOutputTensor?.dispose();
      tailInputTensor?.dispose();
      lstm?.release();
      audioInputBuffer?.destroy();
      frontendOutputBuffer?.destroy();
      await Promise.allSettled([frontendSession?.release(), tailSession?.release()]);
      throw error;
    }
  }

  async run(audio: Float32Array): Promise<Float32Array> {
    if (this.released) throw new Error("VAD backend has been released");
    if (this.running) throw new Error("Concurrent VAD runs are not supported");
    const expected = this.batchSize * this.chunkSamples;
    if (audio.length !== expected) {
      throw new RangeError(`VAD input has ${audio.length} values; expected ${expected}`);
    }

    this.running = true;
    try {
      const upload =
        audio.buffer instanceof ArrayBuffer
          ? new Float32Array(audio.buffer, audio.byteOffset, audio.length)
          : new Float32Array(audio);
      this.device.queue.writeBuffer(this.audioInputBuffer, 0, upload);

      const frontendOutputs = await this.frontendSession.run(
        { [this.frontendInputName]: this.frontendInputTensor },
        { [this.frontendOutputName]: this.frontendOutputTensor },
      );
      if (frontendOutputs[this.frontendOutputName] !== this.frontendOutputTensor) {
        throw new Error(`ORT frontend output '${this.frontendOutputName}' is missing`);
      }

      const recurrentEncoder = this.device.createCommandEncoder({
        label: "senko-pyannote-lstm",
      });
      this.lstm.encode(recurrentEncoder);
      this.device.queue.submit([recurrentEncoder.finish()]);

      const tailOutputs = await this.tailSession.run({
        [this.tailInputName]: this.tailInputTensor,
      });
      const tailOutput = tailOutputs[this.tailOutputName];
      if (tailOutput === undefined) {
        throw new Error(`ORT tail output '${this.tailOutputName}' is missing`);
      }
      try {
        const data = await tailOutput.getData();
        if (!(data instanceof Float32Array)) {
          throw new Error(`Expected float32 tail output, received ${tailOutput.type}`);
        }
        const result = data.slice();
        const expectedOutput =
          this.batchSize * this.outputFrames * this.outputClasses;
        if (result.length !== expectedOutput) {
          throw new Error(
            `ORT tail output has ${result.length} values; expected ${expectedOutput}`,
          );
        }
        return result;
      } finally {
        tailOutput.dispose();
      }
    } finally {
      this.running = false;
    }
  }

  /**
   * Diagnostic-only stage profile. Explicit queue drains make these values
   * unsuitable for production inference, but assign deferred WebGPU work to
   * the stage that submitted it instead of the later tail readback.
   */
  async debugProfileRun(audio: Float32Array): Promise<VadGpuProfile> {
    if (this.released) throw new Error("VAD backend has been released");
    if (this.running) throw new Error("Concurrent VAD runs are not supported");
    const expected = this.batchSize * this.chunkSamples;
    if (audio.length !== expected) {
      throw new RangeError(`VAD input has ${audio.length} values; expected ${expected}`);
    }

    this.running = true;
    const totalStarted = performance.now();
    try {
      const uploadStarted = performance.now();
      const upload =
        audio.buffer instanceof ArrayBuffer
          ? new Float32Array(audio.buffer, audio.byteOffset, audio.length)
          : new Float32Array(audio);
      this.device.queue.writeBuffer(this.audioInputBuffer, 0, upload);
      await this.device.queue.onSubmittedWorkDone();
      const uploadMs = performance.now() - uploadStarted;

      const frontendStarted = performance.now();
      const frontendOutputs = await this.frontendSession.run(
        { [this.frontendInputName]: this.frontendInputTensor },
        { [this.frontendOutputName]: this.frontendOutputTensor },
      );
      if (frontendOutputs[this.frontendOutputName] !== this.frontendOutputTensor) {
        throw new Error(`ORT frontend output '${this.frontendOutputName}' is missing`);
      }
      await this.device.queue.onSubmittedWorkDone();
      const frontendMs = performance.now() - frontendStarted;

      const lstmStarted = performance.now();
      const recurrentEncoder = this.device.createCommandEncoder({
        label: "senko-pyannote-lstm-profile",
      });
      this.lstm.encode(recurrentEncoder);
      this.device.queue.submit([recurrentEncoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const lstmMs = performance.now() - lstmStarted;

      const tailStarted = performance.now();
      const tailOutputs = await this.tailSession.run({
        [this.tailInputName]: this.tailInputTensor,
      });
      const tailOutput = tailOutputs[this.tailOutputName];
      if (tailOutput === undefined) {
        throw new Error(`VAD tail output '${this.tailOutputName}' is missing`);
      }
      let output: Float32Array;
      try {
        const data = await tailOutput.getData();
        if (!(data instanceof Float32Array)) {
          throw new Error(`Expected float32 tail output, received ${tailOutput.type}`);
        }
        output = data.slice();
      } finally {
        tailOutput.dispose();
      }
      const tailAndReadbackMs = performance.now() - tailStarted;
      return {
        uploadMs,
        frontendMs,
        lstmMs,
        tailAndReadbackMs,
        totalMs: performance.now() - totalStarted,
        output,
      };
    } finally {
      this.running = false;
    }
  }

  /** Expensive, query-gated diagnostic readback of every split boundary. */
  async debugStageFingerprints(): Promise<
    Readonly<Record<"audio" | "frontend" | "lstm", GpuStageFingerprint>>
  > {
    if (this.released) throw new Error("VAD backend has been released");
    if (this.running) throw new Error("Cannot inspect VAD buffers during inference");
    const entries = [
      {
        name: "audio" as const,
        buffer: this.audioInputBuffer,
        length: this.batchSize * this.chunkSamples,
      },
      {
        name: "frontend" as const,
        buffer: this.frontendOutputBuffer,
        length: this.batchSize * this.outputFrames * 60,
      },
      {
        name: "lstm" as const,
        buffer: this.lstm.outputBuffer,
        length: this.batchSize * this.outputFrames * 256,
      },
    ];
    const staging = entries.map((entry) =>
      this.device.createBuffer({
        label: `senko-pyannote-debug-${entry.name}`,
        size: entry.length * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
    );
    try {
      const encoder = this.device.createCommandEncoder({
        label: "senko-pyannote-debug-readback",
      });
      for (let index = 0; index < entries.length; index += 1) {
        const entry = entries[index]!;
        encoder.copyBufferToBuffer(
          entry.buffer,
          0,
          staging[index]!,
          0,
          entry.length * 4,
        );
      }
      this.device.queue.submit([encoder.finish()]);
      await Promise.all(staging.map((buffer) => buffer.mapAsync(GPUMapMode.READ)));
      return Object.fromEntries(
        entries.map((entry, index) => [
          entry.name,
          fingerprintFloat32(new Float32Array(staging[index]!.getMappedRange())),
        ]),
      ) as Readonly<
        Record<"audio" | "frontend" | "lstm", GpuStageFingerprint>
      >;
    } finally {
      for (const buffer of staging) {
        if (buffer.mapState === "mapped") buffer.unmap();
        buffer.destroy();
      }
    }
  }

  /** Replays the recurrent stack one layer at a time and reads each output. */
  async debugLayerFingerprints(): Promise<readonly GpuStageFingerprint[]> {
    if (this.released) throw new Error("VAD backend has been released");
    if (this.running) throw new Error("Cannot inspect VAD buffers during inference");
    const length = this.batchSize * this.outputFrames * 256;
    const fingerprints: GpuStageFingerprint[] = [];
    for (let layerIndex = 0; layerIndex < 4; layerIndex += 1) {
      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({
        label: `senko-pyannote-debug-layer-${layerIndex}`,
      });
      this.lstm.encodeLayer(encoder, layerIndex);
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) {
        throw new Error(
          `LSTM layer ${layerIndex} WebGPU validation error: ${validationError.message}`,
        );
      }
      fingerprints.push(
        await readBufferFingerprint(
          this.device,
          this.lstm.layerOutputBuffer(layerIndex),
          length,
          `senko-pyannote-debug-layer-${layerIndex}-readback`,
        ),
      );
    }
    return fingerprints;
  }

  async release(): Promise<void> {
    if (this.released) return;
    if (this.running) throw new Error("Cannot release VAD backend during inference");
    this.released = true;
    try {
      await Promise.allSettled([
        this.frontendSession.release(),
        this.tailSession.release(),
      ]);
    } finally {
      this.frontendInputTensor.dispose();
      this.frontendOutputTensor.dispose();
      this.tailInputTensor.dispose();
      this.lstm.release();
      this.audioInputBuffer.destroy();
      this.frontendOutputBuffer.destroy();
    }
  }
}

function fingerprintFloat32(values: Float32Array): GpuStageFingerprint {
  let finite = 0;
  let nonzero = 0;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let squareSum = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    finite += 1;
    if (value !== 0) nonzero += 1;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    sum += value;
    squareSum += value * value;
  }
  return {
    length: values.length,
    finite,
    nonzero,
    minimum,
    maximum,
    sum,
    l2: Math.sqrt(squareSum),
    first: Array.from(values.subarray(0, 8)),
  };
}

async function readBufferFingerprint(
  device: GPUDevice,
  source: GPUBuffer,
  length: number,
  label: string,
): Promise<GpuStageFingerprint> {
  const staging = device.createBuffer({
    label,
    size: length * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({ label });
    encoder.copyBufferToBuffer(source, 0, staging, 0, length * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return fingerprintFloat32(new Float32Array(staging.getMappedRange()));
  } finally {
    if (staging.mapState === "mapped") staging.unmap();
    staging.destroy();
  }
}

export class OrtEmbeddingBackend implements EmbeddingBatchBackend {
  readonly frames = 150;
  readonly featureDim = 80;
  readonly embeddingDim = 192;
  readonly gpuBufferBytes: number;
  private released = false;

  private constructor(
    readonly batchSize: number,
    private readonly session: ort.InferenceSession,
    private readonly inputName: string,
    private readonly outputName: string,
    private readonly capturedIo: CapturedFloat32Io | undefined,
  ) {
    this.gpuBufferBytes = capturedIo?.bufferBytes.total ?? 0;
  }

  static async create(
    runtime: OrtRuntime,
    asset: OrtModelAsset,
    batchSize: number,
    onProgress?: OrtLoadProgress,
  ): Promise<OrtEmbeddingBackend> {
    const session = await loadSession(runtime, asset, onProgress);
    const inputName = selectName(asset.inputName, session.inputNames, "input");
    const outputName = selectName(asset.outputName, session.outputNames, "output");
    return new OrtEmbeddingBackend(
      batchSize,
      session,
      inputName,
      outputName,
      runtime.graphCapture
        ? new CapturedFloat32Io(
            await runtime.device,
            [batchSize, 150, 80],
            [batchSize, 192],
          )
        : undefined,
    );
  }

  async run(features: Float32Array): Promise<Float32Array> {
    if (this.released) throw new Error("Embedding backend has been released");
    const expected = this.batchSize * this.frames * this.featureDim;
    if (features.length !== expected) {
      throw new RangeError(
        `Embedding input has ${features.length} values; expected ${expected}`,
      );
    }
    const result =
      this.capturedIo === undefined
        ? await this.runUncaptured(features)
        : await this.capturedIo.run(
            this.session,
            this.inputName,
            this.outputName,
            features,
          );
    const outputExpected = this.batchSize * this.embeddingDim;
    if (result.length !== outputExpected) {
      throw new Error(
        `Embedding output has ${result.length} values; expected ${outputExpected}`,
      );
    }
    return result;
  }

  private async runUncaptured(features: Float32Array): Promise<Float32Array> {
    const input = new ort.Tensor("float32", features, [
      this.batchSize,
      this.frames,
      this.featureDim,
    ]);
    try {
      const outputs = await this.session.run({ [this.inputName]: input });
      const output = outputs[this.outputName];
      if (output === undefined) {
        throw new Error(`Embedding output '${this.outputName}' is missing`);
      }
      try {
        return copyFloatOutput(output);
      } finally {
        output.dispose();
      }
    } finally {
      input.dispose();
    }
  }

  async release(): Promise<void> {
    if (this.released) return;
    this.released = true;
    try {
      await this.session.release();
    } finally {
      this.capturedIo?.dispose();
    }
  }
}
