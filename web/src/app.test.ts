import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  formatPageMemorySummary,
  formatPipelineMemorySummary,
  type PageExitEvent,
  type PageExitListener,
  type PageLifecycleTarget,
  SenkoBrowserApp,
} from "./app";
import type { PipelineMemorySummary } from "./runtime/types";
import type { RuntimeCapabilities } from "./capabilities";
import type { PipelineWorkerClient } from "./runtime/worker-client";

class TestPageLifecycle implements PageLifecycleTarget {
  readonly listeners = new Map<PageExitEvent, Set<PageExitListener>>();

  addEventListener(type: PageExitEvent, listener: PageExitListener): void {
    let listeners = this.listeners.get(type);
    if (listeners === undefined) {
      listeners = new Set();
      this.listeners.set(type, listeners);
    }
    listeners.add(listener);
  }

  removeEventListener(type: PageExitEvent, listener: PageExitListener): void {
    this.listeners.get(type)?.delete(listener);
  }

  emit(type: PageExitEvent): void {
    for (const listener of [...(this.listeners.get(type) ?? [])]) listener();
  }

  count(type: PageExitEvent): number {
    return this.listeners.get(type)?.size ?? 0;
  }
}

class TestElement {
  innerHTML = "";
  textContent: string | null = "";
  readonly dataset: Record<string, string> = {};
  disabled = false;
  hidden = false;

  addEventListener(): void {}
}

class TestRoot {
  innerHTML = "";
  readonly elements = new Map<string, TestElement>(
    [
      "#audio-file",
      "#run-pipeline",
      "#status",
      "#capabilities",
      "#runtime-notes",
      "#timing-body",
      "#timing-total",
    ].map((selector) => [selector, new TestElement()]),
  );

  querySelector<T>(selector: string): T | null {
    return (this.elements.get(selector) as T | undefined) ?? null;
  }
}

const runnableCapabilities: RuntimeCapabilities = {
  secureContext: true,
  crossOriginIsolated: true,
  dedicatedWorker: true,
  sharedArrayBuffer: true,
  wasm: true,
  wasmSimd: true,
  wasmThreads: true,
  webgpu: {
    available: true,
    features: ["shader-f16"],
  },
};

beforeEach(() => {
  vi.stubGlobal("document", {
    createElement: () => new TestElement(),
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SenkoBrowserApp lifecycle", () => {
  it.each(["pagehide", "beforeunload"] as const)(
    "disposes once on %s and unregisters both page-exit listeners",
    async (event) => {
      const lifecycle = new TestPageLifecycle();
      const app = new SenkoBrowserApp({} as HTMLElement, lifecycle);
      expect(lifecycle.count("pagehide")).toBe(1);
      expect(lifecycle.count("beforeunload")).toBe(1);

      lifecycle.emit(event);
      expect(lifecycle.count("pagehide")).toBe(0);
      expect(lifecycle.count("beforeunload")).toBe(0);
      app.dispose();
      await expect(app.start()).rejects.toThrow("has been disposed");
    },
  );

  it("coalesces concurrent and repeated start calls into one worker initialization", async () => {
    const root = new TestRoot();
    let finishInitialization!: () => void;
    const initialization = new Promise<void>((resolve) => {
      finishInitialization = resolve;
    });
    const initialize = vi.fn(() => initialization);
    const dispose = vi.fn();
    const worker = { initialize, dispose } as unknown as PipelineWorkerClient;
    const detectCapabilities = vi.fn(async () => runnableCapabilities);
    const createWorkerClient = vi.fn(() => worker);
    const app = new SenkoBrowserApp(
      root as unknown as HTMLElement,
      undefined,
      false,
      { detectCapabilities, createWorkerClient },
    );

    const first = app.start();
    const second = app.start();
    expect(second).toBe(first);
    await vi.waitFor(() => expect(initialize).toHaveBeenCalledOnce());
    expect(detectCapabilities).toHaveBeenCalledOnce();
    expect(createWorkerClient).toHaveBeenCalledOnce();

    finishInitialization();
    await first;
    expect(app.start()).toBe(first);
    expect(dispose).not.toHaveBeenCalled();
    app.dispose();
  });

  it("disposes and detaches a worker whose initialization fails", async () => {
    const root = new TestRoot();
    const failure = new Error("manifest rejected");
    const dispose = vi.fn();
    const worker = {
      initialize: vi.fn(async () => {
        throw failure;
      }),
      dispose,
    } as unknown as PipelineWorkerClient;
    const app = new SenkoBrowserApp(
      root as unknown as HTMLElement,
      undefined,
      false,
      {
        detectCapabilities: async () => runnableCapabilities,
        createWorkerClient: () => worker,
      },
    );

    await app.start();
    expect(dispose).toHaveBeenCalledOnce();
    expect(root.elements.get("#status")).toMatchObject({
      textContent: failure.message,
      dataset: { kind: "error" },
    });
    app.dispose();
    expect(dispose).toHaveBeenCalledOnce();
  });
});

describe("formatPipelineMemorySummary", () => {
  it("shows available peaks and identifies the externally backed input", () => {
    const memory: PipelineMemorySummary = {
      knownCpuPeakBytes: 5 * 1024 * 1024,
      knownGpuBufferBytes: 8 * 1024 * 1024,
      wasmHeapBytes: 512 * 1024,
      jsHeapPeakBytes: 20 * 1024 * 1024,
      allocations: {
        audioBlobBytes: 2 * 1024 * 1024,
        audioBlobCopied: false,
        wavReadBufferBytes: 0,
        vadInputBatchBytes: 0,
        vadLogitsBatchBytes: 0,
        pcmCachePeakBytes: 0,
        camInputBatchBytes: 0,
        camOutputBatchBytes: 0,
        retainedEmbeddingsBytes: 0,
        clusterLabelsBytes: 0,
        clusteringPeakWorkingBytes: 0,
      },
      checkpoints: [],
    };

    expect(formatPipelineMemorySummary(memory)).toBe(
      "Known CPU peak ≥ 5.0 MiB · Known GPU buffers 8.0 MiB · " +
        "WASM heap 512.0 KiB · JS heap peak 20.0 MiB · " +
        "Input Blob 2.0 MiB (external, not copied)",
    );
  });

  it("shows page-agent current/peak values with their sample labels", () => {
    expect(
      formatPageMemorySummary({
        supported: true,
        active: false,
        pending: false,
        currentBytes: 96 * 1024 * 1024,
        currentLabel: "pipeline:complete",
        peakBytes: 128 * 1024 * 1024,
        peakLabel: "embedding:start",
        samples: [
          { label: "embedding:start", bytes: 128 * 1024 * 1024 },
          { label: "pipeline:complete", bytes: 96 * 1024 * 1024 },
        ],
      }),
    ).toBe(
      "Senko page + worker UA memory current 96.0 MiB @ pipeline:complete; " +
        "peak 128.0 MiB @ embedding:start; 2 samples",
    );
  });
});
