import {
  isPipelineWorkerResponse,
  type InitializedResponse,
  type InitializationProgressResponse,
  type PipelineCompletedResponse,
  type PipelineFailedResponse,
  type PipelineProgressResponse,
  type PipelineWorkerRequest,
  type PipelineWorkerResponse,
} from "./protocol";
import type {
  PipelineAssetManifest,
  PipelineOptions,
  PipelineResult,
} from "./types";

export interface WorkerMessageEvent {
  readonly data: unknown;
}

export interface PipelineWorkerPort {
  postMessage(message: PipelineWorkerRequest): void;
  addEventListener(
    type: "message",
    listener: (event: WorkerMessageEvent) => void,
  ): void;
  removeEventListener(
    type: "message",
    listener: (event: WorkerMessageEvent) => void,
  ): void;
  terminate(): void;
}

type PendingRequest =
  | {
      readonly kind: "initialize";
      readonly resolve: (response: InitializedResponse) => void;
      readonly reject: (error: Error) => void;
      readonly onProgress?: (response: InitializationProgressResponse) => void;
    }
  | {
      readonly kind: "diarize";
      readonly resolve: (response: PipelineCompletedResponse) => void;
      readonly reject: (error: Error) => void;
      readonly onProgress?: (event: PipelineProgressResponse) => void;
    };

export class PipelineWorkerError extends Error {
  public readonly response: PipelineFailedResponse;

  public constructor(response: PipelineFailedResponse) {
    super(response.message);
    this.name = "PipelineWorkerError";
    this.response = response;
  }
}

export class PipelineWorkerClient {
  readonly #worker: PipelineWorkerPort;
  readonly #idFactory: () => string;
  readonly #pending = new Map<string, PendingRequest>();
  readonly #handleMessage = (event: WorkerMessageEvent): void => {
    if (!isPipelineWorkerResponse(event.data)) {
      console.error("[senko] Ignoring malformed worker response", event.data);
      return;
    }
    this.#receive(event.data);
  };

  public constructor(
    worker: PipelineWorkerPort,
    idFactory: () => string = () => crypto.randomUUID(),
  ) {
    this.#worker = worker;
    this.#idFactory = idFactory;
    worker.addEventListener("message", this.#handleMessage);
  }

  public initialize(
    manifest: PipelineAssetManifest,
    options: PipelineOptions,
    onProgress?: (response: InitializationProgressResponse) => void,
  ): Promise<InitializedResponse> {
    const requestId = this.#idFactory();
    return new Promise((resolve, reject) => {
      this.#pending.set(requestId, {
        kind: "initialize",
        resolve,
        reject,
        ...(onProgress === undefined ? {} : { onProgress }),
      });
      this.#worker.postMessage({
        type: "initialize",
        requestId,
        manifest,
        options,
      });
    });
  }

  public diarize(
    audio: Blob,
    fileName: string,
    onProgress?: (event: PipelineProgressResponse) => void,
  ): Promise<PipelineResult> {
    const requestId = this.#idFactory();
    return new Promise((resolve, reject) => {
      const pending: PendingRequest = {
        kind: "diarize",
        resolve: (response) => resolve(response.result),
        reject,
        ...(onProgress === undefined ? {} : { onProgress }),
      };
      this.#pending.set(requestId, pending);
      this.#worker.postMessage({
        type: "diarize",
        requestId,
        audio,
        fileName,
      });
    });
  }

  public cancel(targetRequestId: string): void {
    this.#worker.postMessage({
      type: "cancel",
      requestId: this.#idFactory(),
      targetRequestId,
    });
  }

  public dispose(): void {
    this.#worker.removeEventListener("message", this.#handleMessage);
    this.#worker.terminate();
    for (const request of this.#pending.values()) {
      request.reject(new Error("Pipeline worker was disposed"));
    }
    this.#pending.clear();
  }

  #receive(response: PipelineWorkerResponse): void {
    const pending = this.#pending.get(response.requestId);
    if (pending === undefined) {
      return;
    }

    if (response.type === "pipeline-failed") {
      this.#pending.delete(response.requestId);
      pending.reject(new PipelineWorkerError(response));
      return;
    }

    if (response.type === "pipeline-cancelled") {
      this.#pending.delete(response.requestId);
      pending.reject(new DOMException("Pipeline cancelled", "AbortError"));
      return;
    }

    if (pending.kind === "initialize") {
      if (response.type === "initialization-progress") {
        pending.onProgress?.(response);
      } else if (response.type === "initialized") {
        this.#pending.delete(response.requestId);
        pending.resolve(response);
      }
      return;
    }

    if (pending.kind === "diarize") {
      if (
        response.type === "pipeline-started" ||
        response.type === "stage-started" ||
        response.type === "stage-completed"
      ) {
        pending.onProgress?.(response);
      } else if (response.type === "pipeline-completed") {
        this.#pending.delete(response.requestId);
        pending.resolve(response);
      }
    }
  }
}

export function createPipelineWorkerClient(): PipelineWorkerClient {
  const worker = new Worker(new URL("./pipeline.worker.ts", import.meta.url), {
    name: "senko-pipeline",
    type: "module",
  });
  return new PipelineWorkerClient(worker);
}
