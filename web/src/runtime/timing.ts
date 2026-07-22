import {
  PIPELINE_STAGES,
  type AnyStageResult,
  type PipelineStage,
} from "./types";

export type StageTimingStatus = "pending" | "running" | "complete";

export interface StageTimingSnapshot {
  readonly stage: PipelineStage;
  readonly status: StageTimingStatus;
  readonly elapsedMs?: number;
}

/** Tracks stage transitions independently of rendering and worker clock origins. */
export class PipelineTimingLedger {
  readonly #states = new Map<PipelineStage, StageTimingSnapshot>();

  public constructor() {
    this.reset();
  }

  public reset(): void {
    this.#states.clear();
    for (const stage of PIPELINE_STAGES) {
      this.#states.set(stage, { stage, status: "pending" });
    }
  }

  public start(stage: PipelineStage): void {
    const current = this.#states.get(stage);
    if (current?.status !== "pending") {
      throw new Error(`Cannot start ${stage} from ${current?.status ?? "unknown"}`);
    }

    this.#states.set(stage, { stage, status: "running" });
  }

  public complete(result: AnyStageResult): void {
    const current = this.#states.get(result.stage);
    if (current?.status !== "running") {
      throw new Error(
        `Cannot complete ${result.stage} from ${current?.status ?? "unknown"}`,
      );
    }
    if (!Number.isFinite(result.elapsedMs) || result.elapsedMs < 0) {
      throw new Error(`Invalid elapsed time for ${result.stage}`);
    }

    this.#states.set(result.stage, {
      stage: result.stage,
      status: "complete",
      elapsedMs: result.elapsedMs,
    });
  }

  public snapshot(): readonly StageTimingSnapshot[] {
    return PIPELINE_STAGES.map((stage) => {
      const snapshot = this.#states.get(stage);
      if (snapshot === undefined) {
        throw new Error(`Missing timing state for ${stage}`);
      }
      return snapshot;
    });
  }

  public completedTotalMs(): number {
    return this.snapshot().reduce(
      (total, state) => total + (state.elapsedMs ?? 0),
      0,
    );
  }
}
