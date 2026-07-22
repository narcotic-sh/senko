/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />

import type {
  PipelineFailedResponse,
  PipelineWorkerRequest,
  PipelineWorkerResponse,
} from "./protocol";
import { isPipelineWorkerRequest } from "./protocol";
import { BrowserModelSet } from "../pipeline/browser-models";
import {
  BrowserPipelineCancelledError,
  BrowserPipelineStageError,
  runBrowserPipeline,
} from "../pipeline/browser-pipeline";
import { WasmClusteringKernels } from "../clustering";
import type { PipelineOptions } from "./types";
import { loadWorkerResources } from "./worker-resources";

const workerScope = self as unknown as DedicatedWorkerGlobalScope;

let initialized = false;
let initializing = false;
let modelSet: BrowserModelSet | undefined;
let clusteringKernels: WasmClusteringKernels | undefined;
let pipelineOptions: PipelineOptions | undefined;
let activeJob:
  | { readonly requestId: string; readonly controller: AbortController }
  | undefined;

function send(response: PipelineWorkerResponse): void {
  workerScope.postMessage(response);
}

function fail(
  requestId: string,
  code: PipelineFailedResponse["code"],
  message: string,
  stage?: PipelineFailedResponse["stage"],
): void {
  send({
    type: "pipeline-failed",
    requestId,
    code,
    message,
    ...(stage === undefined ? {} : { stage }),
  });
}

async function initialize(
  request: Extract<PipelineWorkerRequest, { type: "initialize" }>,
): Promise<void> {
  if (workerScope.navigator.gpu === undefined) {
    fail(request.requestId, "UNSUPPORTED_RUNTIME", "WebGPU is unavailable in this worker.");
    return;
  }
  if (initializing || initialized || activeJob !== undefined) {
    fail(
      request.requestId,
      "INVALID_REQUEST",
      initializing
        ? "Worker initialization is already in progress."
        : "Worker initialization has already completed.",
    );
    return;
  }

  const manifestAsset = request.manifest.assets.find(
    (asset) => asset.id === "model-manifest" && asset.format === "json",
  );
  if (manifestAsset === undefined) {
    fail(request.requestId, "INVALID_REQUEST", "The model manifest asset is missing.");
    return;
  }

  initializing = true;
  try {
    const manifestUrl = new URL(manifestAsset.url, workerScope.location.href).toString();
    const loaded = await loadWorkerResources(
      () => BrowserModelSet.load(manifestUrl, workerScope.navigator.gpu!, {
        manifestIntegrity: {
          byteLength: manifestAsset.byteLength,
          sha256: manifestAsset.sha256,
        },
        warmupRuns: 1,
        vadBatchSize: 8,
        // Direct WebGPU B16 is the measured CAM++ throughput/memory sweet spot.
        // Together with B8 VAD, both resident devices own 64.7 MB explicitly.
        embeddingBatchSize: 16,
        onProgress: ({ message }) => {
          console.info(`[senko] ${message}`);
          send({
            type: "initialization-progress",
            requestId: request.requestId,
            message,
          });
        },
      }),
      () => WasmClusteringKernels.create(),
    );
    modelSet = loaded.models;
    clusteringKernels = loaded.clustering;
    pipelineOptions = request.options;
    initialized = true;
    observeDeviceLoss(modelSet);
  } catch (error) {
    const failedModels = modelSet;
    modelSet = undefined;
    pipelineOptions = undefined;
    initialized = false;
    clusteringKernels?.dispose();
    clusteringKernels = undefined;
    if (failedModels !== undefined) {
      await Promise.allSettled([failedModels.release()]);
    }
    fail(
      request.requestId,
      "ASSET_LOAD_FAILED",
      error instanceof Error ? error.message : "Failed to initialize WebGPU models.",
    );
    return;
  } finally {
    initializing = false;
  }

  send({
    type: "initialized",
    requestId: request.requestId,
    runtime: {
      crossOriginIsolated: workerScope.crossOriginIsolated,
      sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
      webgpu: true,
    },
  });
}

function observeDeviceLoss(models: BrowserModelSet): void {
  const devices = [
    ["VAD", models.vadDevice],
    ["CAM++", models.embeddingDevice],
  ] as const;
  for (const [role, device] of devices) {
    void device.lost
      .then((info) => handleDeviceLoss(models, role, info))
      .catch((error: unknown) => {
        console.error(
          `[senko] Failed to clean up after ${role} device loss: ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      });
  }
}

async function handleDeviceLoss(
  models: BrowserModelSet,
  role: "VAD" | "CAM++",
  info: GPUDeviceLostInfo,
): Promise<void> {
  if (modelSet !== models) return;
  console.error(
    `[senko] ${role} WebGPU device lost: ${info.reason}: ${info.message}`,
  );
  initialized = false;
  modelSet = undefined;
  pipelineOptions = undefined;
  activeJob?.controller.abort();
  const kernels = clusteringKernels;
  clusteringKernels = undefined;
  try {
    kernels?.dispose();
  } finally {
    await Promise.allSettled([models.release()]);
  }
}

async function diarize(
  request: Extract<PipelineWorkerRequest, { type: "diarize" }>,
): Promise<void> {
  if (
    !initialized ||
    modelSet === undefined ||
    clusteringKernels === undefined ||
    pipelineOptions === undefined
  ) {
    fail(request.requestId, "NOT_INITIALIZED", "Initialize the worker first.");
    return;
  }
  if (activeJob !== undefined) {
    fail(
      request.requestId,
      "INVALID_REQUEST",
      `Pipeline request ${activeJob.requestId} is already running.`,
    );
    return;
  }

  const controller = new AbortController();
  activeJob = { requestId: request.requestId, controller };

  send({
    type: "pipeline-started",
    requestId: request.requestId,
    fileName: request.fileName,
    byteLength: request.audio.size,
  });

  try {
    const result = await runBrowserPipeline(
      request.audio,
      modelSet,
      pipelineOptions,
      {
        signal: controller.signal,
        clusteringKernels,
        onStageStarted: (stage) => {
          send({ type: "stage-started", requestId: request.requestId, stage });
        },
        onStageCompleted: (result) => {
          send({ type: "stage-completed", requestId: request.requestId, result });
        },
      },
    );
    send({ type: "pipeline-completed", requestId: request.requestId, result });
  } catch (error) {
    if (error instanceof BrowserPipelineCancelledError) {
      send({ type: "pipeline-cancelled", requestId: request.requestId });
    } else {
      fail(
        request.requestId,
        "PIPELINE_FAILED",
        error instanceof Error ? error.message : "Browser pipeline failed.",
        error instanceof BrowserPipelineStageError ? error.stage : undefined,
      );
    }
  } finally {
    if (activeJob?.requestId === request.requestId) activeJob = undefined;
  }
}

function cancel(request: Extract<PipelineWorkerRequest, { type: "cancel" }>): void {
  if (activeJob?.requestId === request.targetRequestId) {
    activeJob.controller.abort();
  } else {
    send({ type: "pipeline-cancelled", requestId: request.targetRequestId });
  }
}

workerScope.addEventListener("message", (event: MessageEvent<unknown>) => {
  if (!isPipelineWorkerRequest(event.data)) {
    fail("unknown", "INVALID_REQUEST", "Malformed worker request.");
    return;
  }

  switch (event.data.type) {
    case "initialize":
      void initialize(event.data);
      break;
    case "diarize":
      void diarize(event.data);
      break;
    case "cancel":
      cancel(event.data);
      break;
  }
});
