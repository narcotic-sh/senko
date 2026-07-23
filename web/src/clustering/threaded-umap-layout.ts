import threadedUmapLayoutWasmUrl from "../../scripts/clustering-wasm/umap_layout_threaded.wasm?url";

const PAGE_BYTES = 65_536;
const MINIMUM_MEMORY_BYTES = 16 * 1024 * 1024;
const MAXIMUM_MEMORY_BYTES = 2 * 1024 * 1024 * 1024;
const MAXIMUM_WORKERS = 8;
const HEADER_BYTES = 128;
const HEADER_MAGIC = 0x534b554d;
const HEADER_VERSION = 1;
const STACK_REGION_OFFSET = 131_072;
const WORKER_STACK_BYTES = 65_536;
const ALIGNMENT = 64;
const CONTROL_WORD_OFFSET = 104;
const CONTROL_WORD_COUNT = 6;
const SUCCESS_STATUS = 1;
const CANCELLATION_SETTLE_TIMEOUT_MS = 5_000;

const CONTROL_ARRIVED = 0;
const CONTROL_GENERATION = 1;
const CONTROL_CANCELLED = 2;
const CONTROL_STATUS = 3;
const CONTROL_COMPLETED_EPOCHS = 4;

const PLAN_SECTION_COUNT = 11;

export interface ThreadedUmapLayoutInput {
  readonly embedding: Float32Array;
  readonly rngState: BigInt64Array;
  readonly head: Int32Array;
  readonly tail: Int32Array;
  readonly epochsPerSample: Float64Array;
  readonly vertexCount: number;
  readonly dimension: number;
  readonly epochCount: number;
  readonly a: number;
  readonly b: number;
  readonly gamma?: number;
  readonly negativeSampleRate?: number;
}

export interface ThreadedUmapLayoutResult {
  readonly projection: Float32Array;
  readonly layoutMs: number;
  readonly workerCount: number;
  readonly sharedMemoryBytes: number;
}

export interface ThreadedUmapLayoutMemoryStats {
  readonly activeSharedBytes: number;
  readonly peakSharedBytes: number;
}

export interface ThreadedUmapLayoutPoolOptions {
  readonly workerCount?: number;
  readonly workerFactory?: (workerId: number) => Worker;
  readonly moduleLoader?: () => Promise<WebAssembly.Module>;
}

interface LayoutPlan {
  readonly offsets: readonly number[];
  readonly header: number;
  readonly embedding: number;
  readonly head: number;
  readonly tail: number;
  readonly epochsPerSample: number;
  readonly rngSeed: number;
  readonly totalBytes: number;
  readonly pageCount: number;
}

interface ResolvedLayoutInput {
  readonly embedding: Float32Array;
  readonly rngState: BigInt64Array;
  readonly head: Int32Array;
  readonly tail: Int32Array;
  readonly epochsPerSample: Float64Array;
  readonly vertexCount: number;
  readonly dimension: number;
  readonly edgeCount: number;
  readonly epochCount: number;
  readonly a: number;
  readonly b: number;
  readonly gamma: number;
  readonly negativeSampleRate: number;
}

interface LeafRunRequest {
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

interface LeafCompleteMessage {
  readonly type: "complete";
  readonly jobId: number;
  readonly workerId: number;
  readonly status: number;
  readonly durationMs: number;
}

interface LeafErrorMessage {
  readonly type: "error";
  readonly jobId: number;
  readonly workerId: number;
  readonly error: string;
}

type LeafMessage = LeafCompleteMessage | LeafErrorMessage;

interface WorkerSlot {
  readonly worker: Worker;
  readonly messageListener: (event: MessageEvent<unknown>) => void;
  readonly errorListener: (event: ErrorEvent) => void;
  readonly messageErrorListener: (event: MessageEvent<unknown>) => void;
}

interface ActiveJob {
  readonly id: number;
  readonly expectedEpochs: number;
  readonly outputLength: number;
  readonly startedAt: number;
  readonly pendingWorkers: Set<number>;
  readonly workerStatuses: number[];
  readonly resolve: (result: ThreadedUmapLayoutResult) => void;
  readonly reject: (reason: unknown) => void;
  readonly signal: AbortSignal | undefined;
  readonly abortListener: (() => void) | undefined;
  memory: WebAssembly.Memory | undefined;
  plan: LayoutPlan | undefined;
  failure: Error | undefined;
  abortRequested: boolean;
  settleTimer: ReturnType<typeof setTimeout> | undefined;
}

let defaultModulePromise: Promise<WebAssembly.Module> | undefined;

async function loadDefaultModule(): Promise<WebAssembly.Module> {
  defaultModulePromise ??= (async () => {
    const response = await fetch(threadedUmapLayoutWasmUrl);
    if (!response.ok) {
      throw new Error(
        `HTTP ${response.status} fetching threaded UMAP layout Wasm`,
      );
    }
    try {
      return await WebAssembly.compileStreaming(response.clone());
    } catch {
      return WebAssembly.compile(await response.arrayBuffer());
    }
  })();
  return defaultModulePromise;
}

function createDefaultWorker(workerId: number): Worker {
  return new Worker(
    new URL("./threaded-umap-layout.worker.ts", import.meta.url),
    { type: "module", name: `senko-umap-layout-${workerId}` },
  );
}

function defaultWorkerCount(): number {
  const reported =
    typeof navigator === "undefined" ? 1 : navigator.hardwareConcurrency;
  const finiteReported =
    Number.isFinite(reported) && reported > 0 ? Math.floor(reported) : 1;
  return Math.min(finiteReported, MAXIMUM_WORKERS);
}

function resolveWorkerCount(value: number | undefined): number {
  const workerCount = value ?? defaultWorkerCount();
  if (
    !Number.isSafeInteger(workerCount) ||
    workerCount < 1 ||
    workerCount > MAXIMUM_WORKERS
  ) {
    throw new RangeError(
      `Threaded UMAP worker count must be an integer from 1 to ${MAXIMUM_WORKERS}`,
    );
  }
  return workerCount;
}

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function appendArray(
  offsets: number[],
  cursor: number,
  count: number,
  itemBytes: number,
): number {
  const offset = align(cursor, ALIGNMENT);
  const byteLength = count * itemBytes;
  const end = offset + byteLength;
  if (
    !Number.isSafeInteger(byteLength) ||
    !Number.isSafeInteger(end) ||
    end > 0xffff_ffff
  ) {
    throw new RangeError("Threaded UMAP layout memory plan is too large");
  }
  offsets.push(offset);
  return end;
}

/**
 * Mirrors the allocation-free native planner. Every leaf validates these
 * offsets against the module export before entering the kernel.
 */
function buildLayoutPlan(
  workerCount: number,
  vertexCount: number,
  dimension: number,
  edgeCount: number,
): LayoutPlan {
  let cursor = align(
    STACK_REGION_OFFSET + workerCount * WORKER_STACK_BYTES,
    ALIGNMENT,
  );
  const offsets: number[] = [cursor];
  cursor += HEADER_BYTES;
  cursor = appendArray(
    offsets,
    cursor,
    vertexCount * dimension,
    Float32Array.BYTES_PER_ELEMENT,
  );
  cursor = appendArray(
    offsets,
    cursor,
    edgeCount,
    Int32Array.BYTES_PER_ELEMENT,
  );
  cursor = appendArray(
    offsets,
    cursor,
    edgeCount,
    Int32Array.BYTES_PER_ELEMENT,
  );
  cursor = appendArray(
    offsets,
    cursor,
    edgeCount,
    Float64Array.BYTES_PER_ELEMENT,
  );
  cursor = appendArray(
    offsets,
    cursor,
    3,
    BigInt64Array.BYTES_PER_ELEMENT,
  );
  cursor = appendArray(
    offsets,
    cursor,
    edgeCount,
    Float64Array.BYTES_PER_ELEMENT,
  );
  cursor = appendArray(
    offsets,
    cursor,
    edgeCount,
    Float64Array.BYTES_PER_ELEMENT,
  );
  cursor = appendArray(
    offsets,
    cursor,
    edgeCount,
    Float64Array.BYTES_PER_ELEMENT,
  );
  cursor = appendArray(
    offsets,
    cursor,
    vertexCount * 3,
    BigInt64Array.BYTES_PER_ELEMENT,
  );
  const totalBytes = align(
    Math.max(cursor, MINIMUM_MEMORY_BYTES),
    PAGE_BYTES,
  );
  if (
    offsets.length !== PLAN_SECTION_COUNT - 1 ||
    totalBytes > MAXIMUM_MEMORY_BYTES
  ) {
    throw new RangeError("Threaded UMAP layout memory plan is too large");
  }
  offsets.push(totalBytes);
  return {
    offsets,
    header: offsets[0]!,
    embedding: offsets[1]!,
    head: offsets[2]!,
    tail: offsets[3]!,
    epochsPerSample: offsets[4]!,
    rngSeed: offsets[5]!,
    totalBytes,
    pageCount: totalBytes / PAGE_BYTES,
  };
}

function resolveInput(input: ThreadedUmapLayoutInput): ResolvedLayoutInput {
  const {
    embedding,
    rngState,
    head,
    tail,
    epochsPerSample,
    vertexCount,
    dimension,
    epochCount,
    a,
    b,
  } = input;
  const gamma = input.gamma ?? 1;
  const negativeSampleRate = input.negativeSampleRate ?? 5;
  if (
    !Number.isSafeInteger(vertexCount) ||
    vertexCount < 1 ||
    vertexCount > 0x7fff_ffff ||
    !Number.isSafeInteger(dimension) ||
    dimension < 1 ||
    dimension > 0x7fff_ffff ||
    !Number.isSafeInteger(epochCount) ||
    epochCount < 1 ||
    epochCount > 0xffff_ffff
  ) {
    throw new RangeError("Invalid threaded UMAP layout dimensions");
  }
  const valueCount = vertexCount * dimension;
  if (
    !Number.isSafeInteger(valueCount) ||
    embedding.length !== valueCount
  ) {
    throw new RangeError(
      `Expected ${valueCount} embedding values, received ${embedding.length}`,
    );
  }
  const edgeCount = head.length;
  if (
    edgeCount < 1 ||
    edgeCount > 0x7fff_ffff ||
    tail.length !== edgeCount ||
    epochsPerSample.length !== edgeCount
  ) {
    throw new RangeError("Threaded UMAP edge arrays have mismatched lengths");
  }
  if (rngState.length !== 3) {
    throw new RangeError(
      `Expected three UMAP RNG words, received ${rngState.length}`,
    );
  }
  if (
    !Number.isFinite(a) ||
    a <= 0 ||
    !Number.isFinite(b) ||
    b <= 0 ||
    !Number.isFinite(gamma) ||
    gamma < 0 ||
    !Number.isFinite(negativeSampleRate) ||
    negativeSampleRate <= 0
  ) {
    throw new RangeError("Invalid threaded UMAP layout numeric parameters");
  }
  for (let index = 0; index < embedding.length; index += 1) {
    if (!Number.isFinite(embedding[index])) {
      throw new RangeError(`Embedding value ${index} is not finite`);
    }
  }
  for (let edge = 0; edge < edgeCount; edge += 1) {
    const headIndex = head[edge]!;
    const tailIndex = tail[edge]!;
    const epochs = epochsPerSample[edge]!;
    if (
      headIndex < 0 ||
      headIndex >= vertexCount ||
      tailIndex < 0 ||
      tailIndex >= vertexCount ||
      !Number.isFinite(epochs) ||
      epochs <= 0
    ) {
      throw new RangeError(`Invalid threaded UMAP edge ${edge}`);
    }
  }
  return {
    embedding,
    rngState,
    head,
    tail,
    epochsPerSample,
    vertexCount,
    dimension,
    edgeCount,
    epochCount,
    a,
    b,
    gamma,
    negativeSampleRate,
  };
}

function writeRun(
  memory: WebAssembly.Memory,
  plan: LayoutPlan,
  workerCount: number,
  input: ResolvedLayoutInput,
): void {
  const header = new DataView(memory.buffer, plan.header, HEADER_BYTES);
  const setU32 = (offset: number, value: number): void => {
    header.setUint32(offset, value, true);
  };
  setU32(0, HEADER_MAGIC);
  setU32(4, HEADER_VERSION);
  setU32(8, plan.totalBytes);
  setU32(12, workerCount);
  setU32(16, input.vertexCount);
  setU32(20, input.dimension);
  setU32(24, input.edgeCount);
  setU32(28, input.epochCount);
  for (let index = 1; index <= 9; index += 1) {
    setU32(28 + index * 4, plan.offsets[index]!);
  }
  setU32(68, 0);
  header.setFloat64(72, input.a, true);
  header.setFloat64(80, input.b, true);
  header.setFloat64(88, input.gamma, true);
  header.setFloat64(96, input.negativeSampleRate, true);
  new Int32Array(
    memory.buffer,
    plan.header + CONTROL_WORD_OFFSET,
    CONTROL_WORD_COUNT,
  ).fill(0);

  new Float32Array(
    memory.buffer,
    plan.embedding,
    input.embedding.length,
  ).set(input.embedding);
  new Int32Array(memory.buffer, plan.head, input.head.length).set(input.head);
  new Int32Array(memory.buffer, plan.tail, input.tail.length).set(input.tail);
  new Float64Array(
    memory.buffer,
    plan.epochsPerSample,
    input.epochsPerSample.length,
  ).set(input.epochsPerSample);
  new BigInt64Array(memory.buffer, plan.rngSeed, 3).set(input.rngState);
}

function abortError(reason?: unknown): DOMException {
  const message =
    reason instanceof Error ? reason.message : "Threaded UMAP layout aborted";
  return new DOMException(message, "AbortError");
}

function errorFromUnknown(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

function buildWarmupInput(): ThreadedUmapLayoutInput {
  const vertexCount = 512;
  const dimension = 60;
  const edgesPerVertex = 16;
  const edgeCount = vertexCount * edgesPerVertex;
  const embedding = new Float32Array(vertexCount * dimension);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    for (let coordinate = 0; coordinate < dimension; coordinate += 1) {
      embedding[vertex * dimension + coordinate] = Math.fround(
        Math.sin(vertex * 0.071 + coordinate * 0.113) * 0.5 +
          Math.cos(vertex * 0.037 - coordinate * 0.053) * 0.25,
      );
    }
  }
  const head = new Int32Array(edgeCount);
  const tail = new Int32Array(edgeCount);
  const epochsPerSample = new Float64Array(edgeCount);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    for (let neighbor = 0; neighbor < edgesPerVertex; neighbor += 1) {
      const edge = vertex * edgesPerVertex + neighbor;
      head[edge] = vertex;
      tail[edge] =
        (vertex + 1 + neighbor * 17 + (vertex % 7)) % vertexCount;
      epochsPerSample[edge] = 1 + (edge % 7) * 0.5;
    }
  }
  return {
    embedding,
    rngState: new BigInt64Array([42n, 43n, 44n]),
    head,
    tail,
    epochsPerSample,
    vertexCount,
    dimension,
    epochCount: 40,
    a: 1.932808397545408,
    b: 0.7904949735905139,
  };
}

function isLeafMessage(value: unknown): value is LeafMessage {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Partial<LeafMessage>;
  return (
    (candidate.type === "complete" || candidate.type === "error") &&
    Number.isSafeInteger(candidate.jobId) &&
    Number.isSafeInteger(candidate.workerId)
  );
}

/**
 * Persistent nested-worker pool for the standalone shared-memory UMAP kernel.
 *
 * A fresh, fixed-size shared memory and a fresh module instance in every leaf
 * are used for each job. The workers and compiled module are reused.
 */
export class ThreadedUmapLayoutPool {
  readonly workerCount: number;

  private readonly module: WebAssembly.Module;
  private readonly workers: WorkerSlot[] = [];
  private nextJobId = 1;
  private activeJob: ActiveJob | undefined;
  private warmupPromise: Promise<void> | undefined;
  private warmed = false;
  private disposed = false;
  private brokenError: Error | undefined;
  private workersTerminated = false;
  private activeSharedBytesValue = 0;
  private peakSharedBytesValue = 0;

  private constructor(
    module: WebAssembly.Module,
    workerCount: number,
    workerFactory: (workerId: number) => Worker,
  ) {
    this.module = module;
    this.workerCount = workerCount;
    try {
      for (let workerId = 0; workerId < workerCount; workerId += 1) {
        const worker = workerFactory(workerId);
        const messageListener = (event: MessageEvent<unknown>): void => {
          this.handleWorkerMessage(workerId, event.data);
        };
        const errorListener = (event: ErrorEvent): void => {
          event.preventDefault();
          this.handleWorkerFailure(
            workerId,
            new Error(event.message || `UMAP layout worker ${workerId} failed`),
          );
        };
        const messageErrorListener = (): void => {
          this.handleWorkerFailure(
            workerId,
            new Error(`UMAP layout worker ${workerId} sent an invalid message`),
          );
        };
        worker.addEventListener("message", messageListener);
        worker.addEventListener("error", errorListener);
        worker.addEventListener("messageerror", messageErrorListener);
        this.workers.push({
          worker,
          messageListener,
          errorListener,
          messageErrorListener,
        });
      }
    } catch (error) {
      this.terminateWorkers();
      throw error;
    }
  }

  static async create(
    options: ThreadedUmapLayoutPoolOptions = {},
  ): Promise<ThreadedUmapLayoutPool> {
    if (
      typeof SharedArrayBuffer === "undefined" ||
      typeof WebAssembly.Memory !== "function"
    ) {
      throw new Error(
        "Threaded UMAP layout requires shared WebAssembly memory",
      );
    }
    const [module, workerCount] = await Promise.all([
      (options.moduleLoader ?? loadDefaultModule)(),
      Promise.resolve(resolveWorkerCount(options.workerCount)),
    ]);
    const pool = new ThreadedUmapLayoutPool(
      module,
      workerCount,
      options.workerFactory ?? createDefaultWorker,
    );
    try {
      await pool.warmup();
      return pool;
    } catch (error) {
      pool.dispose();
      throw error;
    }
  }

  get memoryStats(): ThreadedUmapLayoutMemoryStats {
    return {
      activeSharedBytes: this.activeSharedBytesValue,
      peakSharedBytes: this.peakSharedBytesValue,
    };
  }

  resetMemoryStats(): void {
    this.assertUsable();
    if (this.activeJob !== undefined) {
      throw new Error(
        "Cannot reset threaded UMAP memory stats during an active job",
      );
    }
    this.peakSharedBytesValue = this.activeSharedBytesValue;
  }

  async warmup(): Promise<void> {
    this.assertUsable();
    if (this.warmed) return;
    this.warmupPromise ??= this.execute(
      buildWarmupInput(),
      undefined,
    ).then(() => {
      this.warmed = true;
    });
    return this.warmupPromise;
  }

  optimize(
    input: ThreadedUmapLayoutInput,
    signal?: AbortSignal,
  ): Promise<ThreadedUmapLayoutResult> {
    this.assertUsable();
    if (!this.warmed) {
      throw new Error("Threaded UMAP layout pool has not finished warming up");
    }
    return this.execute(input, signal);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const job = this.activeJob;
    if (job !== undefined) {
      this.cancelJob(job);
      this.finalizeRejectedJob(
        job,
        new Error("Threaded UMAP layout pool was disposed"),
      );
    }
    this.terminateWorkers();
  }

  private execute(
    unresolvedInput: ThreadedUmapLayoutInput,
    signal: AbortSignal | undefined,
  ): Promise<ThreadedUmapLayoutResult> {
    this.assertUsable();
    if (this.activeJob !== undefined) {
      return Promise.reject(
        new Error("Threaded UMAP layout already has an active job"),
      );
    }
    if (signal?.aborted === true) {
      return Promise.reject(abortError(signal.reason));
    }
    const input = resolveInput(unresolvedInput);
    const plan = buildLayoutPlan(
      this.workerCount,
      input.vertexCount,
      input.dimension,
      input.edgeCount,
    );
    const memory = new WebAssembly.Memory({
      initial: plan.pageCount,
      maximum: plan.pageCount,
      shared: true,
    });
    const allocatedBytes = memory.buffer.byteLength;
    if (
      !((memory.buffer as unknown) instanceof SharedArrayBuffer) ||
      allocatedBytes !== plan.totalBytes
    ) {
      throw new Error("Could not allocate exact shared UMAP layout memory");
    }
    writeRun(memory, plan, this.workerCount, input);

    const promise = new Promise<ThreadedUmapLayoutResult>(
      (resolve, reject) => {
        const id = this.nextJobId;
        this.nextJobId += 1;
        const abortListener =
          signal === undefined
            ? undefined
            : (): void => {
                const active = this.activeJob;
                if (active?.id !== id) return;
                active.abortRequested = true;
                this.cancelJob(active);
                this.startSettleTimer(active);
              };
        const job: ActiveJob = {
          id,
          expectedEpochs: input.epochCount,
          outputLength: input.embedding.length,
          startedAt: performance.now(),
          pendingWorkers: new Set(
            Array.from({ length: this.workerCount }, (_, index) => index),
          ),
          workerStatuses: Array.from(
            { length: this.workerCount },
            () => 0,
          ),
          resolve,
          reject,
          signal,
          abortListener,
          memory,
          plan,
          failure: undefined,
          abortRequested: false,
          settleTimer: undefined,
        };
        this.activeJob = job;
        this.activeSharedBytesValue = plan.totalBytes;
        this.peakSharedBytesValue = Math.max(
          this.peakSharedBytesValue,
          plan.totalBytes,
        );
        signal?.addEventListener("abort", abortListener!, { once: true });
        if (signal?.aborted === true) {
          abortListener!();
        }

        let postedWorkers = 0;
        for (; postedWorkers < this.workerCount; postedWorkers += 1) {
          const request: LeafRunRequest = {
            type: "run",
            jobId: id,
            workerId: postedWorkers,
            memory,
            stackTop:
              STACK_REGION_OFFSET + (postedWorkers + 1) * WORKER_STACK_BYTES,
            headerPtr: plan.header,
            workerCount: this.workerCount,
            vertexCount: input.vertexCount,
            dimension: input.dimension,
            edgeCount: input.edgeCount,
            planOffsets: plan.offsets,
            ...(id === 1 ? { module: this.module } : {}),
          };
          try {
            this.workers[postedWorkers]!.worker.postMessage(request);
          } catch (error) {
            job.pendingWorkers.delete(postedWorkers);
            this.recordWorkerFailure(
              job,
              postedWorkers,
              errorFromUnknown(error),
            );
            postedWorkers += 1;
            break;
          }
        }
        for (
          let unposted = postedWorkers;
          unposted < this.workerCount;
          unposted += 1
        ) {
          job.pendingWorkers.delete(unposted);
        }
        this.maybeFinishJob(job);
      },
    );
    return promise;
  }

  private handleWorkerMessage(workerId: number, value: unknown): void {
    const job = this.activeJob;
    if (job === undefined) {
      return;
    }
    if (!isLeafMessage(value)) {
      if (job.pendingWorkers.has(workerId)) {
        job.pendingWorkers.delete(workerId);
        this.recordWorkerFailure(
          job,
          workerId,
          new Error(`UMAP layout worker ${workerId} sent an invalid response`),
        );
        this.maybeFinishJob(job);
      }
      return;
    }
    if (value.jobId !== job.id) return;
    if (
      value.workerId !== workerId ||
      !job.pendingWorkers.has(workerId)
    ) {
      job.pendingWorkers.delete(workerId);
      this.recordWorkerFailure(
        job,
        workerId,
        new Error(`Invalid completion from UMAP layout worker ${workerId}`),
      );
      this.maybeFinishJob(job);
      return;
    }
    job.pendingWorkers.delete(workerId);
    if (value.type === "error") {
      this.recordWorkerFailure(
        job,
        workerId,
        new Error(
          `UMAP layout worker ${workerId} failed: ${value.error}`,
        ),
      );
    } else if (!Number.isFinite(value.status)) {
      this.recordWorkerFailure(
        job,
        workerId,
        new Error(`UMAP layout worker ${workerId} returned no status`),
      );
    } else {
      job.workerStatuses[workerId] = value.status;
    }
    this.maybeFinishJob(job);
  }

  private handleWorkerFailure(workerId: number, error: Error): void {
    const job = this.activeJob;
    if (job === undefined) {
      this.breakPool(error);
      return;
    }
    if (job.pendingWorkers.has(workerId)) {
      job.pendingWorkers.delete(workerId);
    }
    this.recordWorkerFailure(job, workerId, error);
    this.maybeFinishJob(job);
  }

  private recordWorkerFailure(
    job: ActiveJob,
    workerId: number,
    error: Error,
  ): void {
    job.failure ??= new Error(
      `Threaded UMAP layout worker ${workerId} failed: ${error.message}`,
    );
    this.cancelJob(job);
    this.startSettleTimer(job);
  }

  private cancelJob(job: ActiveJob): void {
    const memory = job.memory;
    const plan = job.plan;
    if (memory === undefined || plan === undefined) return;
    const control = new Int32Array(
      memory.buffer,
      plan.header + CONTROL_WORD_OFFSET,
      CONTROL_WORD_COUNT,
    );
    Atomics.store(control, CONTROL_CANCELLED, 1);
    Atomics.notify(control, CONTROL_GENERATION);
  }

  private startSettleTimer(job: ActiveJob): void {
    job.settleTimer ??= setTimeout(() => {
      if (this.activeJob?.id !== job.id) return;
      const error =
        job.failure ??
        new Error("Threaded UMAP layout workers did not settle after cancel");
      this.breakPool(error);
      this.finalizeRejectedJob(
        job,
        job.abortRequested ? abortError(job.signal?.reason) : error,
      );
    }, CANCELLATION_SETTLE_TIMEOUT_MS);
  }

  private maybeFinishJob(job: ActiveJob): void {
    if (this.activeJob?.id !== job.id || job.pendingWorkers.size !== 0) {
      return;
    }
    if (job.failure !== undefined) {
      this.breakPool(job.failure);
      this.finalizeRejectedJob(job, job.failure);
      return;
    }
    if (job.abortRequested) {
      this.finalizeRejectedJob(job, abortError(job.signal?.reason));
      return;
    }
    const memory = job.memory;
    const plan = job.plan;
    if (memory === undefined || plan === undefined) {
      const error = new Error("Threaded UMAP layout job lost its memory");
      this.breakPool(error);
      this.finalizeRejectedJob(job, error);
      return;
    }
    const control = new Int32Array(
      memory.buffer,
      plan.header + CONTROL_WORD_OFFSET,
      CONTROL_WORD_COUNT,
    );
    const status = Atomics.load(control, CONTROL_STATUS);
    const completedEpochs = Atomics.load(
      control,
      CONTROL_COMPLETED_EPOCHS,
    );
    const arrived = Atomics.load(control, CONTROL_ARRIVED);
    const cancelled = Atomics.load(control, CONTROL_CANCELLED);
    if (
      job.workerStatuses.some((workerStatus) => workerStatus !== SUCCESS_STATUS) ||
      status !== SUCCESS_STATUS ||
      completedEpochs !== job.expectedEpochs ||
      arrived !== 0 ||
      cancelled !== 0
    ) {
      const error = new Error(
        `Threaded UMAP layout returned status ${status}, ` +
          `${completedEpochs}/${job.expectedEpochs} epochs, ` +
          `workers [${job.workerStatuses.join(", ")}]`,
      );
      this.breakPool(error);
      this.finalizeRejectedJob(job, error);
      return;
    }
    const outputView = new Float32Array(
      memory.buffer,
      plan.embedding,
      job.outputLength,
    );
    for (let index = 0; index < outputView.length; index += 1) {
      if (!Number.isFinite(outputView[index])) {
        const error = new Error(
          `Threaded UMAP layout output ${index} is not finite`,
        );
        this.breakPool(error);
        this.finalizeRejectedJob(job, error);
        return;
      }
    }
    const result: ThreadedUmapLayoutResult = {
      projection: outputView.slice(),
      layoutMs: performance.now() - job.startedAt,
      workerCount: this.workerCount,
      sharedMemoryBytes: plan.totalBytes,
    };
    this.releaseJob(job);
    job.resolve(result);
  }

  private finalizeRejectedJob(job: ActiveJob, error: unknown): void {
    if (this.activeJob?.id !== job.id) return;
    this.releaseJob(job);
    job.reject(error);
  }

  private releaseJob(job: ActiveJob): void {
    if (job.settleTimer !== undefined) {
      clearTimeout(job.settleTimer);
      job.settleTimer = undefined;
    }
    if (job.abortListener !== undefined) {
      job.signal?.removeEventListener("abort", job.abortListener);
    }
    job.pendingWorkers.clear();
    job.memory = undefined;
    job.plan = undefined;
    this.activeJob = undefined;
    this.activeSharedBytesValue = 0;
  }

  private breakPool(error: Error): void {
    this.brokenError ??= error;
    this.terminateWorkers();
  }

  private terminateWorkers(): void {
    if (this.workersTerminated) return;
    this.workersTerminated = true;
    for (const {
      worker,
      messageListener,
      errorListener,
      messageErrorListener,
    } of this.workers) {
      worker.removeEventListener("message", messageListener);
      worker.removeEventListener("error", errorListener);
      worker.removeEventListener("messageerror", messageErrorListener);
      worker.terminate();
    }
  }

  private assertUsable(): void {
    if (this.disposed) {
      throw new Error("Threaded UMAP layout pool is disposed");
    }
    if (this.brokenError !== undefined) {
      throw new Error(
        `Threaded UMAP layout pool is broken: ${this.brokenError.message}`,
      );
    }
  }
}
