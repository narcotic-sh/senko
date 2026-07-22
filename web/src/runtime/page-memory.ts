/**
 * A page-scoped memory sample returned by Chromium. The total covers the
 * current page's agent cluster, including its dedicated workers, rather than
 * every Chrome renderer on the machine.
 */
export interface PageMemoryMeasurement {
  readonly bytes: number;
}

/** Minimal injectable shape for Performance.measureUserAgentSpecificMemory. */
export interface PageMemoryPerformanceSource {
  measureUserAgentSpecificMemory?: () => Promise<PageMemoryMeasurement>;
}

export interface PageMemorySample {
  /** Pipeline boundary that was current when Chromium resolved the sample. */
  readonly label: string;
  readonly bytes: number;
}

export interface PageMemorySummary {
  readonly supported: boolean;
  readonly active: boolean;
  readonly pending: boolean;
  readonly currentBytes?: number;
  readonly currentLabel?: string;
  readonly peakBytes?: number;
  readonly peakLabel?: string;
  readonly samples: readonly PageMemorySample[];
  readonly error?: string;
}

export type PageMemoryUpdateListener = (summary: PageMemorySummary) => void;

/** `?memory=1` is deliberately strict so ordinary benchmark URLs do no work. */
export function isPageMemoryDiagnosticsEnabled(search: string): boolean {
  return new URLSearchParams(search).get("memory") === "1";
}

/**
 * Runs at most one Chromium page-memory request at a time.
 *
 * Chrome intentionally resolves this privacy-sensitive API at a coarse
 * cadence (a call can take many seconds), so pipeline boundaries update a
 * label instead of starting more requests. Each result is attributed to the
 * boundary current at resolution, and another request starts only while the
 * sampler remains active. Nothing in the pipeline awaits these requests.
 */
export class PageMemorySampler {
  readonly #source: PageMemoryPerformanceSource | undefined;
  readonly #onUpdate: PageMemoryUpdateListener | undefined;
  readonly #samples: PageMemorySample[] = [];
  #active = false;
  #currentBoundary = "pipeline:start";
  #error: string | undefined;
  #inFlight: Promise<void> | undefined;
  #generation = 0;

  public constructor(
    source: PageMemoryPerformanceSource | undefined = defaultPerformanceSource(),
    onUpdate?: PageMemoryUpdateListener,
  ) {
    this.#source = source;
    this.#onUpdate = onUpdate;
  }

  public get supported(): boolean {
    return typeof this.#source?.measureUserAgentSpecificMemory === "function";
  }

  /** Begin a run without waiting for the first coarse-cadence measurement. */
  public start(label = "pipeline:start"): void {
    this.#generation += 1;
    this.#samples.length = 0;
    this.#error = undefined;
    this.#currentBoundary = label;
    this.#active = this.supported;
    this.#notify();
    this.#requestIfNeeded();
  }

  /** Mark a stage boundary; this has no measurement or allocation side effect. */
  public mark(label: string): void {
    this.#currentBoundary = label;
  }

  /** Stop after the already-running request resolves; never blocks the caller. */
  public stop(label: string): void {
    this.#currentBoundary = label;
    this.#active = false;
    this.#notify();
  }

  public summary(): PageMemorySummary {
    const current = this.#samples.at(-1);
    let peak: PageMemorySample | undefined;
    for (const sample of this.#samples) {
      if (peak === undefined || sample.bytes > peak.bytes) peak = sample;
    }
    return {
      supported: this.supported,
      active: this.#active,
      pending: this.#inFlight !== undefined,
      ...(current === undefined
        ? {}
        : { currentBytes: current.bytes, currentLabel: current.label }),
      ...(peak === undefined
        ? {}
        : { peakBytes: peak.bytes, peakLabel: peak.label }),
      samples: [...this.#samples],
      ...(this.#error === undefined ? {} : { error: this.#error }),
    };
  }

  /** Test/diagnostic hook; production pipeline execution never awaits this. */
  public whenIdle(): Promise<void> {
    return this.#inFlight ?? Promise.resolve();
  }

  #requestIfNeeded(): void {
    const measure = this.#source?.measureUserAgentSpecificMemory;
    if (!this.#active || this.#inFlight !== undefined || measure === undefined) {
      return;
    }

    const generation = this.#generation;
    this.#inFlight = (async () => {
      try {
        const measurement = await measure.call(this.#source);
        if (!isByteCount(measurement?.bytes)) {
          throw new TypeError("Chromium returned an invalid page-memory byte count");
        }
        // A prior run can finish while Chrome is holding a rate-limited call.
        // Discard that stale result rather than attributing it to the new run.
        if (generation === this.#generation) {
          this.#samples.push({
            label: this.#currentBoundary,
            bytes: measurement.bytes,
          });
        }
      } catch (error) {
        if (generation === this.#generation) {
          this.#error = error instanceof Error ? error.message : String(error);
          this.#active = false;
        }
      }
    })();

    void this.#inFlight.then(() => {
      this.#inFlight = undefined;
      this.#notify();
      this.#requestIfNeeded();
    });
  }

  #notify(): void {
    this.#onUpdate?.(this.summary());
  }
}

function defaultPerformanceSource(): PageMemoryPerformanceSource | undefined {
  if (typeof performance === "undefined") return undefined;
  return performance as Performance & PageMemoryPerformanceSource;
}

function isByteCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}
