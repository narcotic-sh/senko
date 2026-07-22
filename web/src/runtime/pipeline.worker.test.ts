import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_PIPELINE_OPTIONS } from "./types";

const fakes = vi.hoisted(() => ({
  loadModels: vi.fn(),
  createClustering: vi.fn(),
  runPipeline: vi.fn(),
}));

vi.mock("../pipeline/browser-models", () => ({
  BrowserModelSet: {
    load: (...arguments_: unknown[]) => fakes.loadModels(...arguments_),
  },
}));

vi.mock("../clustering", () => ({
  WasmClusteringKernels: {
    create: (...arguments_: unknown[]) => fakes.createClustering(...arguments_),
  },
}));

vi.mock("../pipeline/browser-pipeline", () => {
  class BrowserPipelineCancelledError extends Error {}
  class BrowserPipelineStageError extends Error {
    readonly stage = "vad" as const;
  }
  return {
    BrowserPipelineCancelledError,
    BrowserPipelineStageError,
    runBrowserPipeline: (...arguments_: unknown[]) =>
      fakes.runPipeline(...arguments_),
  };
});

interface Deferred<T> {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

describe("pipeline worker dual-device loss handling", () => {
  let messageListener: ((event: MessageEvent<unknown>) => void) | undefined;
  let messages: unknown[];
  let gpu: GPU;
  let vadLost: Deferred<GPUDeviceLostInfo>;
  let embeddingLost: Deferred<GPUDeviceLostInfo>;
  let release: ReturnType<typeof vi.fn>;
  let dispose: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    vi.resetModules();
    messageListener = undefined;
    messages = [];
    gpu = {} as GPU;
    vadLost = deferred<GPUDeviceLostInfo>();
    embeddingLost = deferred<GPUDeviceLostInfo>();
    release = vi.fn(async () => undefined);
    dispose = vi.fn();
    const models = {
      vadDevice: { lost: vadLost.promise },
      embeddingDevice: { lost: embeddingLost.promise },
      release,
    };
    const clustering = { warmup: vi.fn(), dispose };
    fakes.loadModels.mockReset();
    fakes.loadModels.mockResolvedValue(models);
    fakes.createClustering.mockReset();
    fakes.createClustering.mockResolvedValue(clustering);
    fakes.runPipeline.mockReset();

    vi.stubGlobal("self", {
      navigator: { gpu },
      location: { href: "https://example.test/worker.js" },
      crossOriginIsolated: true,
      postMessage(message: unknown) {
        messages.push(message);
      },
      addEventListener(
        type: string,
        listener: (event: MessageEvent<unknown>) => void,
      ) {
        if (type === "message") messageListener = listener;
      },
    });
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    await import("./pipeline.worker");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it.each([
    ["VAD", () => vadLost],
    ["CAM++", () => embeddingLost],
  ] as const)("invalidates the worker and releases both models on %s device loss", async (_role, loss) => {
    initializeWorker();
    await vi.waitFor(() => {
      expect(messages).toContainEqual(
        expect.objectContaining({ type: "initialized", requestId: "init-1" }),
      );
    });

    expect(fakes.loadModels).toHaveBeenCalledWith(
      "https://example.test/models/manifest.json",
      gpu,
      expect.objectContaining({ vadBatchSize: 8, embeddingBatchSize: 16 }),
    );
    loss().resolve({
      reason: "unknown",
      message: "test loss",
    } as GPUDeviceLostInfo);
    await vi.waitFor(() => expect(release).toHaveBeenCalledOnce());
    expect(dispose).toHaveBeenCalledOnce();

    messageListener?.({
      data: {
        type: "diarize",
        requestId: "job-after-loss",
        audio: new Blob(),
        fileName: "audio.wav",
      },
    } as MessageEvent<unknown>);
    expect(messages).toContainEqual(
      expect.objectContaining({
        type: "pipeline-failed",
        requestId: "job-after-loss",
        code: "NOT_INITIALIZED",
      }),
    );
  });

  function initializeWorker(): void {
    messageListener?.({
      data: {
        type: "initialize",
        requestId: "init-1",
        manifest: {
          schemaVersion: 1,
          pipelineVersion: "test",
          assets: [
            {
              id: "model-manifest",
              role: "runtime-data",
              format: "json",
              url: "./models/manifest.json",
              byteLength: 100,
              sha256: "a".repeat(64),
            },
          ],
        },
        options: DEFAULT_PIPELINE_OPTIONS,
      },
    } as MessageEvent<unknown>);
  }
});
