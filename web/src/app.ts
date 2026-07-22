import {
  assessRuntimeCapabilities,
  detectRuntimeCapabilities,
  type CapabilityAssessment,
  type RuntimeCapabilities,
} from "./capabilities";
import type { PipelineProgressResponse } from "./runtime/protocol";
import {
  isPageMemoryDiagnosticsEnabled,
  PageMemorySampler,
  type PageMemorySummary,
} from "./runtime/page-memory";
import { PipelineTimingLedger } from "./runtime/timing";
import {
  DEFAULT_PIPELINE_OPTIONS,
  PIPELINE_STAGE_LABELS,
  type PipelineAssetManifest,
  type PipelineMemorySummary,
  type PipelineOptions,
  type PipelineResult,
} from "./runtime/types";
import {
  createPipelineWorkerClient,
  PipelineWorkerClient,
} from "./runtime/worker-client";

const BROWSER_MANIFEST: PipelineAssetManifest = {
  schemaVersion: 1,
  pipelineVersion: "browser-direct-vad-campplus-v3",
  assets: [
    {
      id: "model-manifest",
      role: "runtime-data",
      format: "json",
      url: "/models/manifest.json",
      byteLength: 31_493,
      sha256: "7466593fa840665199f6d5a896ac36a6c527288aba4a17b40d8a27bb1eec4c42",
    },
  ],
};

function escapeHtml(value: string): string {
  const node = document.createElement("span");
  node.textContent = value;
  return node.innerHTML;
}

function formatBytes(bytes: number): string {
  const units = ["B", "KiB", "MiB", "GiB"] as const;
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatMilliseconds(milliseconds: number): string {
  if (milliseconds < 1_000) {
    return `${milliseconds.toFixed(1)} ms`;
  }
  return `${(milliseconds / 1_000).toFixed(2)} s`;
}

export function formatPipelineMemorySummary(
  memory: PipelineMemorySummary,
  pageMemory?: PageMemorySummary,
): string {
  const parts = [
    `Known CPU peak ≥ ${formatBytes(memory.knownCpuPeakBytes)}`,
    ...(memory.knownGpuBufferBytes === undefined
      ? []
      : [`Known GPU buffers ${formatBytes(memory.knownGpuBufferBytes)}`]),
    ...(memory.wasmHeapBytes === undefined
      ? []
      : [`WASM heap ${formatBytes(memory.wasmHeapBytes)}`]),
    ...(memory.jsHeapPeakBytes === undefined
      ? []
      : [`JS heap peak ${formatBytes(memory.jsHeapPeakBytes)}`]),
    `Input Blob ${formatBytes(memory.allocations.audioBlobBytes)} (external, not copied)`,
    ...(pageMemory === undefined ? [] : [formatPageMemorySummary(pageMemory)]),
  ];
  return parts.join(" · ");
}

export function formatPageMemorySummary(memory: PageMemorySummary): string {
  const scope = "Senko page + worker UA memory";
  if (!memory.supported) {
    return `${scope} unavailable`;
  }
  if (memory.error !== undefined) {
    return `${scope} failed: ${memory.error}`;
  }
  if (
    memory.currentBytes === undefined ||
    memory.currentLabel === undefined ||
    memory.peakBytes === undefined ||
    memory.peakLabel === undefined
  ) {
    return `${scope} measurement pending (coarse Chrome cadence)`;
  }
  return (
    `${scope} current ${formatBytes(memory.currentBytes)} @ ${memory.currentLabel}; ` +
    `peak ${formatBytes(memory.peakBytes)} @ ${memory.peakLabel}; ` +
    `${memory.samples.length} sample${memory.samples.length === 1 ? "" : "s"}` +
    (memory.pending ? "; final sample pending" : "")
  );
}

export type PageExitEvent = "pagehide" | "beforeunload";
export type PageExitListener = () => void;

/** Narrow event target shape, injectable so page-exit cleanup is testable. */
export interface PageLifecycleTarget {
  addEventListener(type: PageExitEvent, listener: PageExitListener): void;
  removeEventListener(type: PageExitEvent, listener: PageExitListener): void;
}

export interface SenkoBrowserAppDependencies {
  readonly detectCapabilities?: typeof detectRuntimeCapabilities;
  readonly createWorkerClient?: typeof createPipelineWorkerClient;
  readonly pipelineOptions?: PipelineOptions;
}

/** `?precision=float32` forces the fallback even on shader-f16 adapters. */
export function pipelineOptionsFromSearch(search: string): PipelineOptions {
  const forceFloat32 =
    new URLSearchParams(search).get("precision") === "float32";
  return forceFloat32
    ? { ...DEFAULT_PIPELINE_OPTIONS, preferFloat16: false }
    : DEFAULT_PIPELINE_OPTIONS;
}

export class SenkoBrowserApp {
  readonly #root: HTMLElement;
  readonly #lifecycleTarget: PageLifecycleTarget | undefined;
  readonly #timings = new PipelineTimingLedger();
  readonly #handlePageExit = (): void => this.dispose();
  readonly #pageMemorySampler: PageMemorySampler | undefined;
  readonly #detectCapabilities: typeof detectRuntimeCapabilities;
  readonly #createWorkerClient: typeof createPipelineWorkerClient;
  readonly #pipelineOptions: PipelineOptions;
  #worker: PipelineWorkerClient | undefined;
  #startPromise: Promise<void> | undefined;
  #file: File | undefined;
  #lastResult: PipelineResult | undefined;
  #canRun = false;
  #disposed = false;

  public constructor(
    root: HTMLElement,
    lifecycleTarget: PageLifecycleTarget | undefined = defaultPageLifecycleTarget(),
    memoryDiagnosticsEnabled = defaultMemoryDiagnosticsEnabled(),
    dependencies: SenkoBrowserAppDependencies = {},
  ) {
    this.#root = root;
    this.#lifecycleTarget = lifecycleTarget;
    lifecycleTarget?.addEventListener("pagehide", this.#handlePageExit);
    lifecycleTarget?.addEventListener("beforeunload", this.#handlePageExit);
    this.#pageMemorySampler = memoryDiagnosticsEnabled
      ? new PageMemorySampler(undefined, (summary) => {
          if (!this.#disposed) this.#renderPageMemory(summary);
        })
      : undefined;
    this.#detectCapabilities =
      dependencies.detectCapabilities ?? detectRuntimeCapabilities;
    this.#createWorkerClient =
      dependencies.createWorkerClient ?? createPipelineWorkerClient;
    this.#pipelineOptions =
      dependencies.pipelineOptions ??
      pipelineOptionsFromSearch(globalThis.location?.search ?? "");
  }

  public start(): Promise<void> {
    if (this.#disposed) {
      return Promise.reject(new Error("SenkoBrowserApp has been disposed"));
    }
    this.#startPromise ??= this.#startOnce();
    return this.#startPromise;
  }

  async #startOnce(): Promise<void> {
    this.#renderShell();

    try {
      const capabilities = await this.#detectCapabilities();
      if (this.#disposed) return;
      const assessment = assessRuntimeCapabilities(capabilities);
      this.#renderCapabilities(capabilities, assessment);
      if (!assessment.canRun) {
        this.#setStatus("This browser cannot run the WebGPU pipeline.", "error");
        return;
      }

      this.#worker = this.#createWorkerClient();
      this.#setStatus("Loading and compiling WebGPU models…", "loading");
      const initialized = await this.#worker.initialize(
        BROWSER_MANIFEST,
        this.#pipelineOptions,
        ({ message }) => this.#setStatus(`${message}…`, "loading"),
      );
      if (this.#disposed) return;
      this.#root.dataset.modelPrecision = initialized.runtime.modelPrecision;
      this.#canRun = true;
      this.#setStatus(
        `WebGPU ${initialized.runtime.modelPrecision === "float16" ? "FP16" : "FP32"} models ready.`,
        "ready",
      );
      this.#updateRunButton();
    } catch (error) {
      this.#worker?.dispose();
      this.#worker = undefined;
      if (this.#disposed) return;
      this.#setStatus(
        error instanceof Error ? error.message : "Initialization failed.",
        "error",
      );
    }
  }

  public dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#lifecycleTarget?.removeEventListener(
      "pagehide",
      this.#handlePageExit,
    );
    this.#lifecycleTarget?.removeEventListener(
      "beforeunload",
      this.#handlePageExit,
    );
    this.#canRun = false;
    this.#file = undefined;
    this.#pageMemorySampler?.stop("page:disposed");
    this.#worker?.dispose();
    this.#worker = undefined;
  }

  #renderShell(): void {
    this.#root.innerHTML = `
      <section class="shell">
        <header class="hero">
          <div>
            <p class="eyebrow">Browser performance harness</p>
            <h1>Senko diarization</h1>
            <p class="lede">WebGPU inference, WASM clustering, one dedicated worker.</p>
          </div>
          <div class="goal"><strong>&lt;30 s</strong><span>one hour of audio</span></div>
        </header>

        <section class="panel controls" aria-labelledby="input-title">
          <div>
            <h2 id="input-title">Input</h2>
            <p id="file-summary" class="secondary">Choose a mono 16 kHz WAV file.</p>
          </div>
          <label class="file-picker">
            <span>Choose audio</span>
            <input id="audio-file" type="file" accept="audio/wav,.wav" />
          </label>
          <button id="run-pipeline" type="button" disabled>Run pipeline</button>
        </section>

        <section class="panel" aria-labelledby="runtime-title">
          <div class="section-heading">
            <h2 id="runtime-title">Runtime</h2>
            <p id="status" class="status" data-kind="loading">Detecting capabilities…</p>
          </div>
          <div id="capabilities" class="capability-grid" aria-live="polite"></div>
          <ul id="runtime-notes" class="runtime-notes"></ul>
        </section>

        <section class="panel" aria-labelledby="timing-title">
          <div class="section-heading">
            <h2 id="timing-title">Stage timings</h2>
            <p id="timing-total" class="timing-total">0.0 ms</p>
          </div>
          <table>
            <thead><tr><th>Stage</th><th>Status</th><th>Elapsed</th></tr></thead>
            <tbody id="timing-body"></tbody>
          </table>
          <p id="page-memory-live" class="memory-summary" hidden></p>
        </section>

        <section id="result-panel" class="panel result-panel" hidden>
          <h2>Result</h2>
          <p id="memory-summary" class="memory-summary"></p>
          <pre id="result"></pre>
        </section>
      </section>
    `;

    this.#root.querySelector<HTMLInputElement>("#audio-file")?.addEventListener(
      "change",
      (event) => {
        const input = event.currentTarget as HTMLInputElement;
        this.#file = input.files?.[0];
        const summary = this.#requiredElement("#file-summary");
        summary.textContent =
          this.#file === undefined
            ? "Choose a mono 16 kHz WAV file."
            : `${this.#file.name} · ${formatBytes(this.#file.size)}`;
        this.#updateRunButton();
      },
    );
    this.#root.querySelector<HTMLButtonElement>("#run-pipeline")?.addEventListener(
      "click",
      () => void this.#runPipeline(),
    );
    this.#renderTimings();
    if (this.#pageMemorySampler !== undefined) {
      this.#renderPageMemory(this.#pageMemorySampler.summary());
    }
  }

  #renderCapabilities(
    capabilities: RuntimeCapabilities,
    assessment: CapabilityAssessment,
  ): void {
    const f16 = capabilities.webgpu.features.includes("shader-f16");
    const values = [
      ["WebGPU", capabilities.webgpu.available],
      ["Worker", capabilities.dedicatedWorker],
      ["WASM SIMD", capabilities.wasmSimd],
      ["WASM threads", capabilities.wasmThreads],
      ["shader-f16", f16],
      ["Cross-origin isolated", capabilities.crossOriginIsolated],
    ] as const;

    this.#requiredElement("#capabilities").innerHTML = values
      .map(
        ([label, value]) => `
          <div class="capability" data-supported="${String(value)}">
            <span>${escapeHtml(label)}</span><strong>${value ? "Yes" : "No"}</strong>
          </div>`,
      )
      .join("");

    const notes = [...assessment.errors, ...assessment.warnings];
    this.#requiredElement("#runtime-notes").innerHTML = notes
      .map((note) => `<li>${escapeHtml(note)}</li>`)
      .join("");
  }

  async #runPipeline(): Promise<void> {
    const worker = this.#worker;
    const file = this.#file;
    if (worker === undefined || file === undefined || this.#disposed) {
      return;
    }

    const button = this.#requiredElement<HTMLButtonElement>("#run-pipeline");
    button.disabled = true;
    this.#lastResult = undefined;
    this.#timings.reset();
    this.#renderTimings();
    this.#setStatus("Running…", "loading");

    const pageMemory = this.#pageMemorySampler;
    pageMemory?.start("pipeline:start");
    const onProgress =
      pageMemory === undefined
        ? (event: PipelineProgressResponse) => this.#recordProgress(event)
        : (event: PipelineProgressResponse) => {
            this.#recordProgress(event);
            switch (event.type) {
              case "pipeline-started":
                pageMemory.mark("pipeline:start");
                break;
              case "stage-started":
                pageMemory.mark(`${event.stage}:start`);
                break;
              case "stage-completed":
                pageMemory.mark(`${event.result.stage}:complete`);
                break;
            }
          };

    try {
      const result = await worker.diarize(file, file.name, onProgress);
      if (this.#disposed) return;
      pageMemory?.stop("pipeline:complete");
      this.#renderResult(result);
      this.#setStatus("Pipeline complete.", "ready");
    } catch (error) {
      if (this.#disposed) return;
      pageMemory?.stop("pipeline:failed");
      this.#setStatus(
        error instanceof Error ? error.message : "Pipeline failed.",
        "error",
      );
    } finally {
      if (!this.#disposed) button.disabled = false;
    }
  }

  #recordProgress(event: PipelineProgressResponse): void {
    switch (event.type) {
      case "pipeline-started":
        console.info(
          `[senko] Started ${event.fileName} (${formatBytes(event.byteLength)})`,
        );
        break;
      case "stage-started":
        this.#timings.start(event.stage);
        console.info(`[senko] ${PIPELINE_STAGE_LABELS[event.stage]} started`);
        break;
      case "stage-completed":
        this.#timings.complete(event.result);
        console.info(
          `[senko] ${PIPELINE_STAGE_LABELS[event.result.stage]}: ${formatMilliseconds(event.result.elapsedMs)}`,
          event.result.metrics,
        );
        break;
    }
    this.#renderTimings();
  }

  #renderTimings(): void {
    const body = this.#root.querySelector("#timing-body");
    if (body === null) {
      return;
    }

    body.innerHTML = this.#timings
      .snapshot()
      .map(
        (timing) => `
          <tr>
            <td>${PIPELINE_STAGE_LABELS[timing.stage]}</td>
            <td><span class="stage-status" data-status="${timing.status}">${timing.status}</span></td>
            <td>${timing.elapsedMs === undefined ? "—" : formatMilliseconds(timing.elapsedMs)}</td>
          </tr>`,
      )
      .join("");
    this.#requiredElement("#timing-total").textContent = formatMilliseconds(
      this.#timings.completedTotalMs(),
    );
  }

  #renderResult(result: PipelineResult): void {
    const panel = this.#requiredElement<HTMLElement>("#result-panel");
    panel.hidden = false;
    this.#lastResult = result;
    // FBank extraction and CAM++ inference overlap, so summing stage-attributed
    // durations double-counts concurrent work. Once the run completes, show
    // the worker's actual end-to-end wall clock beside the stage table.
    this.#requiredElement("#timing-total").textContent =
      `${formatMilliseconds(result.totalElapsedMs)} wall`;
    this.#renderPageMemory(this.#pageMemorySampler?.summary());
    console.table(
      result.stages.map((stage) => ({
        stage: PIPELINE_STAGE_LABELS[stage.stage],
        milliseconds: stage.elapsedMs,
      })),
    );
    console.info(`[senko] Total: ${formatMilliseconds(result.totalElapsedMs)}`);
  }

  #renderPageMemory(pageMemory: PageMemorySummary | undefined): void {
    if (pageMemory !== undefined) {
      const live = this.#root.querySelector<HTMLElement>("#page-memory-live");
      if (live !== null) {
        live.hidden = false;
        live.textContent = formatPageMemorySummary(pageMemory);
      }
    }

    const result = this.#lastResult;
    if (result === undefined) return;
    this.#requiredElement("#memory-summary").textContent =
      formatPipelineMemorySummary(result.memory, pageMemory);
    this.#requiredElement("#result").textContent = JSON.stringify(
      pageMemory === undefined ? result : { ...result, pageMemory },
      null,
      2,
    );
  }

  #setStatus(message: string, kind: "loading" | "ready" | "error"): void {
    const status = this.#requiredElement("#status");
    status.textContent = message;
    status.dataset.kind = kind;
  }

  #updateRunButton(): void {
    const button = this.#root.querySelector<HTMLButtonElement>("#run-pipeline");
    if (button !== null) {
      button.disabled = !this.#canRun || this.#file === undefined;
    }
  }

  #requiredElement<T extends Element = HTMLElement>(selector: string): T {
    const element = this.#root.querySelector<T>(selector);
    if (element === null) {
      throw new Error(`Missing UI element: ${selector}`);
    }
    return element;
  }
}

function defaultPageLifecycleTarget(): PageLifecycleTarget | undefined {
  if (typeof window === "undefined") return undefined;
  return window as unknown as PageLifecycleTarget;
}

function defaultMemoryDiagnosticsEnabled(): boolean {
  if (typeof location === "undefined") return false;
  return isPageMemoryDiagnosticsEnabled(location.search);
}
