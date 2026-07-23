/// <reference lib="webworker" />

const PLAN_SECTION_COUNT = 11;

interface RunRequest {
  readonly type: "run";
  readonly jobId: number;
  readonly workerId: number;
  readonly module?: WebAssembly.Module;
  readonly memory: WebAssembly.Memory;
  readonly stackTop: number;
  readonly headerPtr: number;
  readonly workerCount: number;
  readonly vertexCount: number;
  readonly dimension: number;
  readonly edgeCount: number;
  readonly planOffsets: readonly number[];
}

interface ThreadedLayoutExports extends WebAssembly.Exports {
  readonly __stack_pointer: WebAssembly.Global;
  readonly _initialize: () => void;
  readonly umap_layout_threaded_plan_offset: (
    section: number,
    workerCount: number,
    vertexCount: number,
    dimension: number,
    edgeCount: number,
  ) => number;
  readonly umap_layout_threaded_stack_top: (workerId: number) => number;
  readonly umap_layout_threaded_run: (
    workerId: number,
    headerPtr: number,
  ) => number;
}

const workerScope = self as unknown as DedicatedWorkerGlobalScope;
let activeJobId: number | undefined;
let compiledModule: WebAssembly.Module | undefined;

function isRunRequest(value: unknown): value is RunRequest {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Partial<RunRequest>;
  return (
    candidate.type === "run" &&
    Number.isSafeInteger(candidate.jobId) &&
    Number.isSafeInteger(candidate.workerId) &&
    candidate.memory instanceof WebAssembly.Memory
  );
}

function readExports(instance: WebAssembly.Instance): ThreadedLayoutExports {
  const exports = instance.exports as Partial<ThreadedLayoutExports>;
  if (
    !(exports.__stack_pointer instanceof WebAssembly.Global) ||
    typeof exports._initialize !== "function" ||
    typeof exports.umap_layout_threaded_plan_offset !== "function" ||
    typeof exports.umap_layout_threaded_stack_top !== "function" ||
    typeof exports.umap_layout_threaded_run !== "function"
  ) {
    throw new Error("Threaded UMAP layout Wasm has an invalid ABI");
  }
  return exports as ThreadedLayoutExports;
}

function validatePlan(exports: ThreadedLayoutExports, request: RunRequest): void {
  if (
    request.planOffsets.length !== PLAN_SECTION_COUNT ||
    request.headerPtr !== request.planOffsets[0]
  ) {
    throw new Error("Threaded UMAP layout plan has an invalid shape");
  }
  for (let section = 0; section < PLAN_SECTION_COUNT; section += 1) {
    const actual = exports.umap_layout_threaded_plan_offset(
      section,
      request.workerCount,
      request.vertexCount,
      request.dimension,
      request.edgeCount,
    );
    if (actual !== request.planOffsets[section]) {
      throw new Error(
        `Threaded UMAP layout plan mismatch in section ${section}`,
      );
    }
  }
  const expectedStackTop = exports.umap_layout_threaded_stack_top(
    request.workerId,
  );
  if (expectedStackTop !== request.stackTop) {
    throw new Error("Threaded UMAP layout stack plan mismatch");
  }
  const allocatedBytes = request.memory.buffer.byteLength;
  if (
    !((request.memory.buffer as unknown) instanceof SharedArrayBuffer) ||
    allocatedBytes !== request.planOffsets[PLAN_SECTION_COUNT - 1]
  ) {
    throw new Error("Threaded UMAP layout memory is not exact and shared");
  }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.stack ?? error.message : String(error);
}

workerScope.addEventListener("message", async (event: MessageEvent<unknown>) => {
  if (!isRunRequest(event.data)) return;
  const request = event.data;
  if (activeJobId !== undefined) {
    workerScope.postMessage({
      type: "error",
      jobId: request.jobId,
      workerId: request.workerId,
      error: `Worker is already running job ${activeJobId}`,
    });
    return;
  }
  activeJobId = request.jobId;
  try {
    if (compiledModule === undefined) {
      if (!(request.module instanceof WebAssembly.Module)) {
        throw new Error("Threaded UMAP layout worker was not initialized");
      }
      compiledModule = request.module;
    }
    const instance = await WebAssembly.instantiate(compiledModule, {
      env: { memory: request.memory },
    });
    const exports = readExports(instance);
    exports.__stack_pointer.value = request.stackTop;
    exports._initialize();
    validatePlan(exports, request);
    const startedAt = performance.now();
    const status = exports.umap_layout_threaded_run(
      request.workerId,
      request.headerPtr,
    );
    workerScope.postMessage({
      type: "complete",
      jobId: request.jobId,
      workerId: request.workerId,
      status,
      durationMs: performance.now() - startedAt,
    });
  } catch (error) {
    workerScope.postMessage({
      type: "error",
      jobId: request.jobId,
      workerId: request.workerId,
      error: errorText(error),
    });
  } finally {
    activeJobId = undefined;
  }
});

export {};
