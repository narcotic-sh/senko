import { describe, expect, it } from "vitest";

import {
  ThreadedUmapLayoutPool,
  type ThreadedUmapLayoutInput,
} from "./threaded-umap-layout";

const CONTROL_WORD_OFFSET = 104;
const CONTROL_CANCELLED = 2;
const CONTROL_STATUS = 3;
const CONTROL_COMPLETED_EPOCHS = 4;
const SUCCESS_STATUS = 1;
const CANCELLED_STATUS = -2;
const MINIMUM_SHARED_BYTES = 16 * 1024 * 1024;

interface FakeRunRequest {
  readonly type: "run";
  readonly jobId: number;
  readonly workerId: number;
  readonly module?: WebAssembly.Module;
  readonly memory: WebAssembly.Memory;
  readonly headerPtr: number;
  readonly planOffsets: readonly number[];
}

type FakeBehavior =
  | "success"
  | "hold"
  | "stale-then-success"
  | "fail-first"
  | "incomplete"
  | "non-finite"
  | "bad-status";

type FakeEventListener = (event: { readonly data?: unknown }) => void;

class FakeWorker {
  readonly requests: FakeRunRequest[] = [];
  terminateCount = 0;

  private readonly listeners = new Map<string, Set<FakeEventListener>>();

  constructor(
    readonly workerId: number,
    private readonly controller: FakeWorkerController,
  ) {}

  postMessage(value: unknown): void {
    const request = value as FakeRunRequest;
    this.requests.push(request);
    this.controller.handleRun(this, request);
  }

  addEventListener(type: string, listener: FakeEventListener): void {
    let listeners = this.listeners.get(type);
    if (listeners === undefined) {
      listeners = new Set();
      this.listeners.set(type, listeners);
    }
    listeners.add(listener);
  }

  removeEventListener(type: string, listener: FakeEventListener): void {
    this.listeners.get(type)?.delete(listener);
  }

  terminate(): void {
    this.terminateCount += 1;
  }

  emitMessage(data: unknown): void {
    for (const listener of this.listeners.get("message") ?? []) {
      listener({ data });
    }
  }
}

class FakeWorkerController {
  readonly workers: FakeWorker[] = [];
  readonly heldRequests: Array<{
    readonly worker: FakeWorker;
    readonly request: FakeRunRequest;
  }> = [];
  readonly requestedWorkerIds: number[] = [];
  behavior: FakeBehavior = "success";

  readonly workerFactory = (workerId: number): Worker => {
    this.requestedWorkerIds.push(workerId);
    const worker = new FakeWorker(workerId, this);
    this.workers.push(worker);
    return worker as unknown as Worker;
  };

  handleRun(worker: FakeWorker, request: FakeRunRequest): void {
    expect(request.workerId).toBe(worker.workerId);
    expect(request.memory.buffer.byteLength).toBe(
      request.planOffsets[request.planOffsets.length - 1],
    );
    expect(() => request.memory.grow(1)).toThrow();

    if (request.jobId === 1 || this.behavior === "success") {
      queueMicrotask(() => this.completeSuccess(worker, request));
      return;
    }
    if (this.behavior === "hold") {
      this.heldRequests.push({ worker, request });
      return;
    }
    if (this.behavior === "stale-then-success") {
      queueMicrotask(() => {
        worker.emitMessage({
          type: "complete",
          jobId: request.jobId - 1,
          workerId: request.workerId,
          status: -999,
          durationMs: 0,
        });
        this.completeSuccess(worker, request);
      });
      return;
    }
    if (this.behavior === "fail-first") {
      queueMicrotask(() => {
        if (worker.workerId === 0) {
          worker.emitMessage({
            type: "error",
            jobId: request.jobId,
            workerId: request.workerId,
            error: "synthetic leaf failure",
          });
        } else {
          expect(this.cancelled(request)).toBe(true);
          this.complete(worker, request, CANCELLED_STATUS);
        }
      });
      return;
    }
    queueMicrotask(() => {
      if (this.behavior === "incomplete") {
        this.writeSuccessfulControl(request, -1);
        this.complete(worker, request, SUCCESS_STATUS);
      } else if (this.behavior === "non-finite") {
        this.writeSuccessfulControl(request);
        if (worker.workerId === 0) {
          new Float32Array(
            request.memory.buffer,
            request.planOffsets[1],
            1,
          )[0] = Number.NaN;
        }
        this.complete(worker, request, SUCCESS_STATUS);
      } else {
        this.writeSuccessfulControl(request);
        this.complete(worker, request, -3);
      }
    });
  }

  completeHeldAsCancelled(): void {
    for (const { worker, request } of this.heldRequests.splice(0)) {
      this.complete(worker, request, CANCELLED_STATUS);
    }
  }

  private completeSuccess(
    worker: FakeWorker,
    request: FakeRunRequest,
  ): void {
    this.writeSuccessfulControl(request);
    if (request.jobId > 1 && worker.workerId === 0) {
      new Float32Array(
        request.memory.buffer,
        request.planOffsets[1],
        1,
      )[0] = 42;
    }
    this.complete(worker, request, SUCCESS_STATUS);
  }

  private complete(
    worker: FakeWorker,
    request: FakeRunRequest,
    status: number,
  ): void {
    worker.emitMessage({
      type: "complete",
      jobId: request.jobId,
      workerId: request.workerId,
      status,
      durationMs: 0.25,
    });
  }

  private writeSuccessfulControl(
    request: FakeRunRequest,
    epochAdjustment = 0,
  ): void {
    const epochCount = new DataView(
      request.memory.buffer,
      request.headerPtr,
      128,
    ).getUint32(28, true);
    const control = this.control(request);
    Atomics.store(control, CONTROL_STATUS, SUCCESS_STATUS);
    Atomics.store(
      control,
      CONTROL_COMPLETED_EPOCHS,
      epochCount + epochAdjustment,
    );
  }

  cancelled(request: FakeRunRequest): boolean {
    return (
      Atomics.load(this.control(request), CONTROL_CANCELLED) === 1
    );
  }

  private control(request: FakeRunRequest): Int32Array {
    return new Int32Array(
      request.memory.buffer,
      request.headerPtr + CONTROL_WORD_OFFSET,
      6,
    );
  }
}

const emptyModule = new WebAssembly.Module(
  new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]),
);

function fixture(): ThreadedUmapLayoutInput {
  return {
    embedding: new Float32Array([1, 2, 3, 4]),
    rngState: new BigInt64Array([42n, 43n, 44n]),
    head: new Int32Array([0, 1]),
    tail: new Int32Array([1, 0]),
    epochsPerSample: new Float64Array([1, 1.5]),
    vertexCount: 2,
    dimension: 2,
    epochCount: 5,
    a: 1.932808397545408,
    b: 0.7904949735905139,
  };
}

async function createPool(
  controller: FakeWorkerController,
  workerCount = 3,
  loaderCalls?: { value: number },
): Promise<ThreadedUmapLayoutPool> {
  return ThreadedUmapLayoutPool.create({
    workerCount,
    workerFactory: controller.workerFactory,
    moduleLoader: async () => {
      if (loaderCalls !== undefined) loaderCalls.value += 1;
      return emptyModule;
    },
  });
}

describe("ThreadedUmapLayoutPool", () => {
  it("warms every persistent worker and reuses one compiled module", async () => {
    const controller = new FakeWorkerController();
    const loaderCalls = { value: 0 };
    const pool = await createPool(controller, 3, loaderCalls);
    try {
      expect(loaderCalls.value).toBe(1);
      expect(controller.requestedWorkerIds).toEqual([0, 1, 2]);
      expect(controller.workers).toHaveLength(3);
      for (const worker of controller.workers) {
        expect(worker.requests.map((request) => request.jobId)).toEqual([1]);
      }
      expect(pool.memoryStats).toEqual({
        activeSharedBytes: 0,
        peakSharedBytes: MINIMUM_SHARED_BYTES,
      });
      pool.resetMemoryStats();
      expect(pool.memoryStats).toEqual({
        activeSharedBytes: 0,
        peakSharedBytes: 0,
      });

      const result = await pool.optimize(fixture());
      expect(result.projection).toEqual(new Float32Array([42, 2, 3, 4]));
      expect(result.workerCount).toBe(3);
      expect(result.sharedMemoryBytes).toBe(MINIMUM_SHARED_BYTES);
      expect(result.layoutMs).toBeGreaterThanOrEqual(0);
      expect(pool.memoryStats).toEqual({
        activeSharedBytes: 0,
        peakSharedBytes: MINIMUM_SHARED_BYTES,
      });
      pool.resetMemoryStats();
      expect(pool.memoryStats).toEqual({
        activeSharedBytes: 0,
        peakSharedBytes: 0,
      });
      for (const worker of controller.workers) {
        expect(worker.requests.map((request) => request.jobId)).toEqual([
          1, 2,
        ]);
        expect(worker.requests[0]!.module).toBe(emptyModule);
        expect(worker.requests[1]!.module).toBeUndefined();
        expect(worker.requests[0]!.memory).not.toBe(
          worker.requests[1]!.memory,
        );
        expect(worker.requests[1]!.memory).toBe(
          controller.workers[0]!.requests[1]!.memory,
        );
      }
    } finally {
      pool.dispose();
    }
    expect(controller.workers.map((worker) => worker.terminateCount)).toEqual([
      1, 1, 1,
    ]);
  });

  it("ignores stale job messages", async () => {
    const controller = new FakeWorkerController();
    const pool = await createPool(controller, 2);
    try {
      controller.behavior = "stale-then-success";
      const result = await pool.optimize(fixture());
      expect(result.projection[0]).toBe(42);
      expect(pool.memoryStats.activeSharedBytes).toBe(0);
    } finally {
      pool.dispose();
    }
  });

  it("atomically cancels, settles, and remains reusable after abort", async () => {
    const controller = new FakeWorkerController();
    const pool = await createPool(controller, 3);
    try {
      controller.behavior = "hold";
      const abortController = new AbortController();
      const pending = pool.optimize(fixture(), abortController.signal);
      expect(controller.heldRequests).toHaveLength(3);
      expect(pool.memoryStats.activeSharedBytes).toBe(MINIMUM_SHARED_BYTES);
      expect(() => pool.resetMemoryStats()).toThrow(/active job/);
      await expect(pool.optimize(fixture())).rejects.toThrow(/active job/);

      abortController.abort(new Error("stop layout"));
      expect(
        controller.heldRequests.every(({ request }) =>
          controller.cancelled(request),
        ),
      ).toBe(true);
      controller.completeHeldAsCancelled();
      await expect(pending).rejects.toMatchObject({
        name: "AbortError",
        message: "stop layout",
      });
      expect(pool.memoryStats.activeSharedBytes).toBe(0);
      expect(
        controller.workers.every((worker) => worker.terminateCount === 0),
      ).toBe(true);

      controller.behavior = "success";
      await expect(pool.optimize(fixture())).resolves.toMatchObject({
        workerCount: 3,
      });
    } finally {
      pool.dispose();
    }
  });

  it("cancels siblings and permanently breaks the pool after leaf failure", async () => {
    const controller = new FakeWorkerController();
    const pool = await createPool(controller, 3);
    controller.behavior = "fail-first";
    const pending = pool.optimize(fixture());
    await expect(pending).rejects.toThrow(/synthetic leaf failure/);
    const failedRequest = controller.workers[0]!.requests[1]!;
    expect(controller.cancelled(failedRequest)).toBe(true);
    expect(controller.workers.map((worker) => worker.terminateCount)).toEqual([
      1, 1, 1,
    ]);
    expect(pool.memoryStats.activeSharedBytes).toBe(0);
    expect(() => pool.optimize(fixture())).toThrow(/broken/);
    pool.dispose();
    expect(controller.workers.map((worker) => worker.terminateCount)).toEqual([
      1, 1, 1,
    ]);
  });

  it("idempotently disposes and rejects an active job", async () => {
    const controller = new FakeWorkerController();
    const pool = await createPool(controller, 2);
    controller.behavior = "hold";
    const pending = pool.optimize(fixture());
    expect(pool.memoryStats.activeSharedBytes).toBe(MINIMUM_SHARED_BYTES);
    pool.dispose();
    pool.dispose();
    await expect(pending).rejects.toThrow(/disposed/);
    expect(pool.memoryStats.activeSharedBytes).toBe(0);
    expect(controller.workers.map((worker) => worker.terminateCount)).toEqual([
      1, 1,
    ]);
    expect(() => pool.optimize(fixture())).toThrow(/disposed/);
  });

  it.each([
    ["worker status", "bad-status", /workers \[-3, -3\]/],
    ["completed epochs", "incomplete", /4\/5 epochs/],
    ["finite output", "non-finite", /not finite/],
  ] as const)(
    "rejects and breaks on invalid %s",
    async (_name, behavior, expected) => {
      const controller = new FakeWorkerController();
      const pool = await createPool(controller, 2);
      controller.behavior = behavior;
      await expect(pool.optimize(fixture())).rejects.toThrow(expected);
      expect(() => pool.optimize(fixture())).toThrow(/broken/);
      expect(
        controller.workers.every((worker) => worker.terminateCount === 1),
      ).toBe(true);
      pool.dispose();
    },
  );

  it("rejects unsupported explicit worker counts", async () => {
    await expect(
      ThreadedUmapLayoutPool.create({
        workerCount: 0,
        workerFactory: () => {
          throw new Error("should not construct");
        },
        moduleLoader: async () => emptyModule,
      }),
    ).rejects.toThrow(/worker count/);
    await expect(
      ThreadedUmapLayoutPool.create({
        workerCount: 9,
        workerFactory: () => {
          throw new Error("should not construct");
        },
        moduleLoader: async () => emptyModule,
      }),
    ).rejects.toThrow(/worker count/);
  });
});
