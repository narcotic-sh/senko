/// <reference types="@webgpu/types" />

import type { OrtModelAsset } from "./ort-backends";
import type { SelectedSegmentationSplit } from "./model-manifest";
import { PersistentWebGpuLstm, type PersistentLstmBufferBytes } from "./persistent-lstm";
import {
  RawPyannoteFrontendFoundation,
  type RawPyannoteFrontendGpuBytes,
} from "./pyannote-frontend-webgpu";
import {
  RawPyannoteTail,
  type RawPyannoteTailGpuBytes,
} from "./pyannote-tail-webgpu";
import type { VadBatchBackend } from "./types";
import {
  VAD_CHUNK_SAMPLES,
  VAD_OUTPUT_CLASSES,
  VAD_OUTPUT_FRAMES,
} from "./vad";

export interface RawVadModelAssets {
  readonly frontendMetadata: OrtModelAsset;
  readonly tailMetadata: OrtModelAsset;
}

export interface RawVadGpuBufferBytes {
  readonly frontend: RawPyannoteFrontendGpuBytes;
  readonly lstm: PersistentLstmBufferBytes;
  readonly tail: RawPyannoteTailGpuBytes;
  /** Exact sum of every directly owned GPUBuffer allocation. */
  readonly totalOwned: number;
}

/**
 * Fully direct-WebGPU pyannote segmentation backend.
 *
 * The only host transfers per batch are the waveform upload and the final
 * seven-logit readback. All intermediate tensors stay on one GPUDevice and
 * the frontend's two activation slots are overwritten according to their
 * statically proven lifetimes.
 */
export class RawWebGpuVadBackend implements VadBatchBackend {
  readonly chunkSamples = VAD_CHUNK_SAMPLES;
  readonly outputFrames = VAD_OUTPUT_FRAMES;
  readonly outputClasses = VAD_OUTPUT_CLASSES;
  readonly gpuBufferBytes: RawVadGpuBufferBytes;

  private running = false;
  private released = false;

  private constructor(
    readonly batchSize: number,
    private readonly device: GPUDevice,
    private readonly frontend: RawPyannoteFrontendFoundation,
    private readonly lstm: PersistentWebGpuLstm,
    private readonly tail: RawPyannoteTail,
  ) {
    this.gpuBufferBytes = {
      frontend: frontend.gpuBytes,
      lstm: lstm.bufferBytes,
      tail: tail.gpuBytes,
      totalOwned:
        frontend.gpuBytes.total + lstm.bufferBytes.total + tail.gpuBytes.total,
    };
  }

  static async create(
    device: GPUDevice,
    selected: SelectedSegmentationSplit,
    assets: RawVadModelAssets,
    onProgress?: (message: string) => void,
  ): Promise<RawWebGpuVadBackend> {
    let frontend: RawPyannoteFrontendFoundation | undefined;
    let lstm: PersistentWebGpuLstm | undefined;
    let tail: RawPyannoteTail | undefined;
    try {
      onProgress?.("Loading direct WebGPU frontend");
      frontend = await RawPyannoteFrontendFoundation.create(
        device,
        assets.frontendMetadata.url,
        { fetch: verifiedMetadataFetch(assets.frontendMetadata) },
      );
      const frontendBatch = frontend.gpuPackage.metadata.contract.inputShape[0];
      if (frontendBatch !== selected.batchSize) {
        throw new Error(
          `Direct WebGPU frontend is B${frontendBatch}; selected B${selected.batchSize}`,
        );
      }

      lstm = await PersistentWebGpuLstm.create(
        device,
        selected.batchSize,
        frontend.frontendOutputBuffer,
        selected.weights,
        selected.metadata,
        onProgress,
      );

      onProgress?.("Loading direct WebGPU tail");
      tail = await RawPyannoteTail.create(
        device,
        lstm.outputBuffer,
        assets.tailMetadata.url,
        verifiedMetadataFetch(assets.tailMetadata),
      );
      if (tail.metadata.batch !== selected.batchSize) {
        throw new Error(
          `Direct WebGPU tail is B${tail.metadata.batch}; selected B${selected.batchSize}`,
        );
      }
      const frontendMetadata = frontend.gpuPackage.metadata.contract;
      if (
        frontendMetadata.intermediateDtype !== selected.precision ||
        frontendMetadata.weightDtype !== selected.precision ||
        lstm.weightPrecision !== selected.precision ||
        tail.metadata.weightPrecision !== selected.precision
      ) {
        throw new Error(
          `Direct WebGPU VAD package precision does not match selected ${selected.precision}`,
        );
      }
      const backend = new RawWebGpuVadBackend(
        selected.batchSize,
        device,
        frontend,
        lstm,
        tail,
      );
      if (backend.gpuBufferBytes.totalOwned !== selected.directWebGpu.explicitGpuBytes) {
        throw new Error(
          `Direct WebGPU VAD owns ${backend.gpuBufferBytes.totalOwned} GPU bytes; manifest declares ${selected.directWebGpu.explicitGpuBytes}`,
        );
      }
      onProgress?.("Direct WebGPU pyannote ready");
      return backend;
    } catch (error) {
      tail?.destroy();
      lstm?.release();
      frontend?.destroy();
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
      this.frontend.uploadWaveform(audio);
      const encoder = this.device.createCommandEncoder({
        label: "senko-pyannote-direct-webgpu",
      });
      this.frontend.encode(encoder);
      this.lstm.encode(encoder);
      this.tail.encode(encoder, true);
      this.device.queue.submit([encoder.finish()]);
      const output = await this.tail.readback();
      const outputExpected =
        this.batchSize * this.outputFrames * this.outputClasses;
      if (output.length !== outputExpected) {
        throw new Error(
          `Direct WebGPU VAD output has ${output.length} values; expected ${outputExpected}`,
        );
      }
      return output;
    } finally {
      this.running = false;
    }
  }

  async release(): Promise<void> {
    if (this.released) return;
    if (this.running) {
      throw new Error("Cannot release VAD while inference is running");
    }
    this.released = true;
    await this.device.queue.onSubmittedWorkDone();
    this.tail.destroy();
    this.lstm.release();
    this.frontend.destroy();
  }
}

function verifiedMetadataFetch(asset: OrtModelAsset): typeof fetch {
  const expectedUrl = new URL(
    asset.url,
    globalThis.location?.href ?? "http://localhost/",
  ).href;
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const response = await fetch(input, init);
    const requestedUrl = new URL(
      input instanceof Request ? input.url : input.toString(),
      globalThis.location?.href ?? "http://localhost/",
    ).href;
    if (!response.ok || requestedUrl !== expectedUrl) return response;

    const bytes = await response.arrayBuffer();
    if (asset.byteLength !== undefined && bytes.byteLength !== asset.byteLength) {
      throw new Error(
        `Direct WebGPU metadata has ${bytes.byteLength} bytes; expected ${asset.byteLength}`,
      );
    }
    if (asset.sha256 !== undefined) {
      const actual = bytesToHex(await crypto.subtle.digest("SHA-256", bytes));
      if (actual !== asset.sha256.toLowerCase()) {
        throw new Error(`Direct WebGPU metadata SHA-256 mismatch: ${actual}`);
      }
    }
    return new Response(bytes, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };
}

function bytesToHex(buffer: ArrayBuffer): string {
  let result = "";
  for (const byte of new Uint8Array(buffer)) {
    result += byte.toString(16).padStart(2, "0");
  }
  return result;
}
