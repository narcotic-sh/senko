import { beforeEach, describe, expect, it, vi } from "vitest";

const fakes = vi.hoisted(() => {
  type PendingRun = {
    readonly resolve: (result: {
      embeddings: Float32Array;
      wallMs: number;
    }) => void;
  };
  const pending: PendingRun[] = [];
  const graph = {
    gpuBytes: {
      weights: 1,
      activationArena: 1,
      input: 1,
      output: 1,
      readback: 2,
      timestampBuffers: 0,
      dispatchUniforms: 1,
      total: 7,
    },
    foundation: {
      gpuPackage: {
        binaryUrl: "https://example.test/campplus.bin",
        metadata: {
          binary: { byteLength: 13, sha256: "a".repeat(64) },
        },
      },
    },
    run() {
      return new Promise<{
        embeddings: Float32Array;
        wallMs: number;
      }>((resolve) => pending.push({ resolve }));
    },
    destroy: vi.fn(),
  };
  const device = {
    queue: { onSubmittedWorkDone: vi.fn(async () => {}) },
  };
  const createGraph = vi.fn(
    async (_device: unknown, _metadataUrl: string, _options: unknown) => graph,
  );
  return { createGraph, device, graph, pending };
});

vi.mock("./campplus-webgpu", () => ({
  CAMPPLUS_RAW_MAX_IN_FLIGHT_RUNS: 2,
  CampPlusRawGraph: {
    create: fakes.createGraph,
  },
}));

import type { SelectedCampPlusDirect } from "./model-manifest";
import { RawCampPlusEmbeddingBackend } from "./raw-campplus-backend";

const SELECTED: SelectedCampPlusDirect = {
  precision: "float16",
  batchSize: 16,
  metadata: { url: "https://example.test/campplus.json" },
  weights: {
    url: "https://example.test/campplus.bin",
    byteLength: 13,
    sha256: "a".repeat(64),
  },
  explicitGpuBufferBytes: 7,
};

describe("RawCampPlusEmbeddingBackend in-flight lifecycle", () => {
  beforeEach(() => {
    fakes.pending.length = 0;
    fakes.createGraph.mockClear();
    fakes.graph.destroy.mockClear();
    fakes.device.queue.onSubmittedWorkDone.mockClear();
  });

  it("allows two runs, rejects a third, and cannot release an active graph", async () => {
    const backend = await RawCampPlusEmbeddingBackend.create(
      fakes.device as unknown as GPUDevice,
      SELECTED,
    );
    expect(fakes.createGraph).toHaveBeenCalledWith(
      fakes.device,
      "https://example.test/campplus.json",
      expect.objectContaining({ batchSize: 16, storageDtype: "float16" }),
    );
    const features = new Float32Array(16 * 150 * 80);
    const first = backend.run(features);
    const second = backend.run(features);

    expect(backend.maxInFlightRuns).toBe(2);
    expect(fakes.pending).toHaveLength(2);
    await expect(backend.run(features)).rejects.toThrow(/at most 2 concurrent/);
    await expect(backend.release()).rejects.toThrow(/inference is running/);

    fakes.pending[1]!.resolve({
      embeddings: new Float32Array(16 * 192).fill(2),
      wallMs: 1,
    });
    await expect(second).resolves.toHaveLength(16 * 192);
    await expect(backend.release()).rejects.toThrow(/inference is running/);

    fakes.pending[0]!.resolve({
      embeddings: new Float32Array(16 * 192).fill(1),
      wallMs: 1,
    });
    await expect(first).resolves.toHaveLength(16 * 192);
    await backend.release();

    expect(fakes.device.queue.onSubmittedWorkDone).toHaveBeenCalledOnce();
    expect(fakes.graph.destroy).toHaveBeenCalledOnce();
    await expect(backend.run(features)).rejects.toThrow(/has been released/);
  });
});
