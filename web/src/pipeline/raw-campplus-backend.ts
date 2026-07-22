/// <reference types="@webgpu/types" />

import {
  CAMPPLUS_RAW_MAX_IN_FLIGHT_RUNS,
  CampPlusRawGraph,
  type CampPlusRawGraphGpuBytes,
} from "./campplus-webgpu";
import type { SelectedCampPlusDirect } from "./model-manifest";
import type { OrtModelAsset } from "./ort-backends";
import type { EmbeddingBatchBackend } from "./types";

export class RawCampPlusEmbeddingBackend implements EmbeddingBatchBackend {
  readonly frames = 150;
  readonly featureDim = 80;
  readonly embeddingDim = 192;
  readonly maxInFlightRuns = CAMPPLUS_RAW_MAX_IN_FLIGHT_RUNS;
  readonly gpuBufferBytes: CampPlusRawGraphGpuBytes;
  private activeRuns = 0;
  private released = false;

  private constructor(
    readonly batchSize: 4 | 8 | 16 | 32,
    private readonly device: GPUDevice,
    private readonly graph: CampPlusRawGraph,
  ) {
    this.gpuBufferBytes = graph.gpuBytes;
  }

  static async create(
    device: GPUDevice,
    selected: SelectedCampPlusDirect,
    onProgress?: (message: string) => void,
  ): Promise<RawCampPlusEmbeddingBackend> {
    onProgress?.(`Loading direct WebGPU CAM++ B${selected.batchSize}`);
    let graph: CampPlusRawGraph | undefined;
    try {
      graph = await CampPlusRawGraph.create(device, selected.metadata.url, {
        batchSize: selected.batchSize,
        fetch: verifiedMetadataFetch(selected.metadata),
      });
      const binary = graph.foundation.gpuPackage.metadata.binary;
      const binaryUrl = new URL(
        graph.foundation.gpuPackage.binaryUrl,
        globalThis.location?.href ?? "http://localhost/",
      ).href;
      const expectedBinaryUrl = new URL(
        selected.weights.url,
        globalThis.location?.href ?? "http://localhost/",
      ).href;
      if (
        binaryUrl !== expectedBinaryUrl ||
        binary.byteLength !== selected.weights.byteLength ||
        binary.sha256.toLowerCase() !== selected.weights.sha256?.toLowerCase()
      ) {
        throw new Error("Direct WebGPU CAM++ binary does not match the pinned manifest");
      }
      if (graph.gpuBytes.total !== selected.explicitGpuBufferBytes) {
        throw new Error(
          `Direct WebGPU CAM++ owns ${graph.gpuBytes.total} GPUBuffer bytes; manifest declares ${selected.explicitGpuBufferBytes}`,
        );
      }
      const backend = new RawCampPlusEmbeddingBackend(
        selected.batchSize,
        device,
        graph,
      );
      onProgress?.("Direct WebGPU CAM++ ready");
      return backend;
    } catch (error) {
      graph?.destroy();
      throw error;
    }
  }

  async run(features: Float32Array): Promise<Float32Array> {
    if (this.released) throw new Error("Direct WebGPU CAM++ has been released");
    if (this.activeRuns >= this.maxInFlightRuns) {
      throw new Error(
        `Direct WebGPU CAM++ allows at most ${this.maxInFlightRuns} concurrent runs`,
      );
    }
    if (!(features.buffer instanceof ArrayBuffer)) {
      throw new Error("Direct WebGPU CAM++ requires ArrayBuffer-backed features");
    }
    this.activeRuns += 1;
    try {
      const result = await this.graph.run(
        features as Float32Array<ArrayBuffer>,
      );
      return result.embeddings;
    } finally {
      this.activeRuns -= 1;
    }
  }

  async release(): Promise<void> {
    if (this.released) return;
    if (this.activeRuns > 0) {
      throw new Error("Cannot release CAM++ while inference is running");
    }
    this.released = true;
    await this.device.queue.onSubmittedWorkDone();
    this.graph.destroy();
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
        `Direct WebGPU CAM++ metadata has ${bytes.byteLength} bytes; expected ${asset.byteLength}`,
      );
    }
    if (asset.sha256 !== undefined) {
      const actual = bytesToHex(await crypto.subtle.digest("SHA-256", bytes));
      if (actual !== asset.sha256.toLowerCase()) {
        throw new Error(`Direct WebGPU CAM++ metadata SHA-256 mismatch: ${actual}`);
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
