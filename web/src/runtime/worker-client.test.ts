import { describe, expect, it, vi } from "vitest";
import type { PipelineWorkerRequest } from "./protocol";
import { PipelineWorkerClient, PipelineWorkerError } from "./worker-client";
import {
  DEFAULT_PIPELINE_OPTIONS,
  type PipelineAssetManifest,
} from "./types";

class FakeWorker {
  readonly messages: PipelineWorkerRequest[] = [];
  readonly #listeners = new Set<(event: { data: unknown }) => void>();
  terminated = false;

  public postMessage(message: PipelineWorkerRequest): void {
    this.messages.push(message);
  }

  public addEventListener(
    _type: "message",
    listener: (event: { data: unknown }) => void,
  ): void {
    this.#listeners.add(listener);
  }

  public removeEventListener(
    _type: "message",
    listener: (event: { data: unknown }) => void,
  ): void {
    this.#listeners.delete(listener);
  }

  public terminate(): void {
    this.terminated = true;
  }

  public emit(data: unknown): void {
    for (const listener of this.#listeners) {
      listener({ data });
    }
  }
}

const manifest: PipelineAssetManifest = {
  schemaVersion: 1,
  pipelineVersion: "test",
  assets: [],
};

describe("PipelineWorkerClient", () => {
  it("correlates initialization responses", async () => {
    const worker = new FakeWorker();
    const client = new PipelineWorkerClient(worker, () => "initialize-1");
    const pending = client.initialize(manifest, DEFAULT_PIPELINE_OPTIONS);

    expect(worker.messages[0]).toMatchObject({
      type: "initialize",
      requestId: "initialize-1",
    });
    worker.emit({
      type: "initialized",
      requestId: "initialize-1",
      runtime: {
        crossOriginIsolated: true,
        sharedArrayBuffer: true,
        webgpu: true,
        modelPrecision: "float16",
      },
    });

    await expect(pending).resolves.toMatchObject({ type: "initialized" });
  });

  it("routes progress and rejects typed worker errors", async () => {
    const worker = new FakeWorker();
    const onProgress = vi.fn();
    const client = new PipelineWorkerClient(worker, () => "job-1");
    const pending = client.diarize(
      new Blob([new Uint8Array(16)]),
      "audio.wav",
      onProgress,
    );

    worker.emit({
      type: "stage-started",
      requestId: "job-1",
      stage: "decode",
    });
    expect(onProgress).toHaveBeenCalledOnce();

    worker.emit({
      type: "pipeline-failed",
      requestId: "job-1",
      code: "NOT_IMPLEMENTED",
      message: "pending backend",
    });
    await expect(pending).rejects.toBeInstanceOf(PipelineWorkerError);
  });

  it("detaches and rejects work on disposal", async () => {
    const worker = new FakeWorker();
    const client = new PipelineWorkerClient(worker, () => "job-1");
    const pending = client.diarize(new Blob(), "audio.wav");

    client.dispose();

    expect(worker.terminated).toBe(true);
    await expect(pending).rejects.toThrow("disposed");
  });
});
