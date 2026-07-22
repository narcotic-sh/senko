import { describe, expect, it, vi } from "vitest";

import {
  isPageMemoryDiagnosticsEnabled,
  PageMemorySampler,
  type PageMemoryMeasurement,
} from "./page-memory";

interface DeferredMeasurement {
  readonly promise: Promise<PageMemoryMeasurement>;
  readonly resolve: (measurement: PageMemoryMeasurement) => void;
}

function deferredMeasurement(): DeferredMeasurement {
  let resolve!: (measurement: PageMemoryMeasurement) => void;
  const promise = new Promise<PageMemoryMeasurement>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

describe("PageMemorySampler", () => {
  it("strictly gates diagnostics on ?memory=1", () => {
    expect(isPageMemoryDiagnosticsEnabled("?memory=1")).toBe(true);
    expect(isPageMemoryDiagnosticsEnabled("?profile=1&memory=1")).toBe(true);
    expect(isPageMemoryDiagnosticsEnabled("?memory=0")).toBe(false);
    expect(isPageMemoryDiagnosticsEnabled("?memory=true")).toBe(false);
    expect(isPageMemoryDiagnosticsEnabled("")).toBe(false);
  });

  it("keeps one request in flight and labels samples at resolution", async () => {
    const first = deferredMeasurement();
    const second = deferredMeasurement();
    const measure = vi
      .fn<() => Promise<PageMemoryMeasurement>>()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const updates: number[] = [];
    const sampler = new PageMemorySampler(
      { measureUserAgentSpecificMemory: measure },
      (summary) => updates.push(summary.samples.length),
    );

    sampler.start();
    sampler.mark("vad:start");
    sampler.mark("vad:complete");
    expect(measure).toHaveBeenCalledTimes(1);

    first.resolve({ bytes: 120 * 1024 * 1024 });
    await first.promise;
    await Promise.resolve();
    expect(measure).toHaveBeenCalledTimes(2);
    expect(sampler.summary().samples).toEqual([
      { label: "vad:complete", bytes: 120 * 1024 * 1024 },
    ]);

    sampler.mark("embedding:start");
    sampler.stop("pipeline:complete");
    second.resolve({ bytes: 96 * 1024 * 1024 });
    await sampler.whenIdle();
    await Promise.resolve();

    expect(measure).toHaveBeenCalledTimes(2);
    expect(sampler.summary()).toMatchObject({
      supported: true,
      active: false,
      pending: false,
      currentBytes: 96 * 1024 * 1024,
      currentLabel: "pipeline:complete",
      peakBytes: 120 * 1024 * 1024,
      peakLabel: "vad:complete",
      samples: [
        { label: "vad:complete", bytes: 120 * 1024 * 1024 },
        { label: "pipeline:complete", bytes: 96 * 1024 * 1024 },
      ],
    });
    expect(updates.length).toBeGreaterThanOrEqual(3);
  });

  it("does no work when the API is unavailable", () => {
    const sampler = new PageMemorySampler({});
    sampler.start();
    sampler.mark("embedding:start");
    sampler.stop("pipeline:complete");

    expect(sampler.summary()).toEqual({
      supported: false,
      active: false,
      pending: false,
      samples: [],
    });
  });

  it("stops safely after a failed or malformed measurement", async () => {
    const sampler = new PageMemorySampler({
      measureUserAgentSpecificMemory: async () => ({ bytes: -1 }),
    });
    sampler.start();
    await sampler.whenIdle();
    await Promise.resolve();

    expect(sampler.summary()).toMatchObject({
      active: false,
      pending: false,
      samples: [],
      error: "Chromium returned an invalid page-memory byte count",
    });
  });
});
