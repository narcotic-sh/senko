import { describe, expect, it } from "vitest";
import { PipelineTimingLedger } from "./timing";

describe("PipelineTimingLedger", () => {
  it("records ordered stage timings and their total", () => {
    const ledger = new PipelineTimingLedger();
    ledger.start("vad");
    ledger.complete({
      stage: "vad",
      elapsedMs: 150.25,
      metrics: { modelRuns: 370, regionCount: 166, speechSeconds: 3_525 },
    });
    ledger.start("embedding");
    ledger.complete({
      stage: "embedding",
      elapsedMs: 400.5,
      metrics: { batchCount: 45, embeddingCount: 5_713, dimensions: 192 },
    });

    expect(ledger.completedTotalMs()).toBe(550.75);
    expect(ledger.snapshot().find(({ stage }) => stage === "vad")).toEqual({
      stage: "vad",
      status: "complete",
      elapsedMs: 150.25,
    });
  });

  it("rejects an out-of-order completion", () => {
    const ledger = new PipelineTimingLedger();
    expect(() =>
      ledger.complete({
        stage: "postprocess",
        elapsedMs: 1,
        metrics: { segmentCount: 1, speakerCount: 1 },
      }),
    ).toThrow(/Cannot complete postprocess/);
  });

  it("resets all stages to pending", () => {
    const ledger = new PipelineTimingLedger();
    ledger.start("decode");
    ledger.reset();
    expect(ledger.snapshot().every(({ status }) => status === "pending")).toBe(
      true,
    );
  });
});
