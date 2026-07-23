import {
  buildPlan,
  cancelRun,
  copyRunInputs,
  readRunControl,
  resetRunControl,
  writeRunHeader,
} from "./umap_layout_threaded_host.js";

const wasmUrl = new URL("./umap_layout_threaded.wasm", import.meta.url);
const leafUrl = new URL(
  "./umap_layout_threaded_browser_leaf.worker.js",
  import.meta.url,
);

let modulePromise;
let fixturePromise;
let activeRun;

function once(worker, expectedType) {
  return new Promise((resolvePromise, rejectPromise) => {
    const onMessage = (event) => {
      const message = event.data;
      if (message?.type === "error") {
        cleanup();
        rejectPromise(new Error(message.error));
      } else if (message?.type === expectedType) {
        cleanup();
        resolvePromise(message);
      }
    };
    const onError = (event) => {
      cleanup();
      rejectPromise(event.error ?? new Error(event.message));
    };
    const cleanup = () => {
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
    };
    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", onError);
  });
}

async function compileModule() {
  if (modulePromise === undefined) {
    modulePromise = fetch(wasmUrl).then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} fetching ${wasmUrl}`);
      }
      try {
        return await WebAssembly.compileStreaming(response.clone());
      } catch {
        return WebAssembly.compile(await response.arrayBuffer());
      }
    });
  }
  return modulePromise;
}

async function fetchTypedArray(url, Constructor) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} fetching ${url}`);
  }
  return new Constructor(await response.arrayBuffer());
}

async function loadFixture(fixtureUrl) {
  if (fixturePromise === undefined) {
    fixturePromise = Promise.all([
      fetchTypedArray(
        new URL("umap-layout-initial-embedding.f32", fixtureUrl),
        Float32Array,
      ),
      fetchTypedArray(
        new URL("umap-layout-head.i32", fixtureUrl),
        Int32Array,
      ),
      fetchTypedArray(
        new URL("umap-layout-tail.i32", fixtureUrl),
        Int32Array,
      ),
      fetchTypedArray(
        new URL("umap-layout-epochs-per-sample.f64", fixtureUrl),
        Float64Array,
      ),
      fetchTypedArray(
        new URL("umap-layout-rng-state.i64", fixtureUrl),
        BigInt64Array,
      ),
    ]).then(
      ([embedding, head, tail, epochsPerSample, rngSeed]) => ({
        embedding,
        head,
        tail,
        epochsPerSample,
        rngSeed,
      }),
    );
  }
  return fixturePromise;
}

async function runOnce(workers, memory, plan) {
  const completions = workers.map((worker) =>
    once(worker, "complete"),
  );
  const startedAt = performance.now();
  workers.forEach((worker, workerId) => {
    worker.postMessage({
      type: "run",
      workerId,
      headerPtr: plan.header,
    });
  });
  const timeout = setTimeout(() => cancelRun(memory, plan), 60_000);
  const results = await Promise.all(completions);
  clearTimeout(timeout);
  return {
    durationMs: performance.now() - startedAt,
    slowestWorkerMs: Math.max(
      ...results.map((result) => result.durationMs),
    ),
    statuses: results.map((result) => result.status),
    control: readRunControl(memory, plan),
  };
}

async function runFixture({
  fixtureUrl,
  workerCount,
  repeatCount,
}) {
  const [module, fixture] = await Promise.all([
    compileModule(),
    loadFixture(fixtureUrl),
  ]);
  const plannerMemory = new WebAssembly.Memory({
    initial: 256,
    maximum: 256,
    shared: true,
  });
  const planner = await WebAssembly.instantiate(module, {
    env: { memory: plannerMemory },
  });
  planner.exports._initialize();

  const vertexCount = 5_713;
  const dimension = 60;
  const edgeCount = fixture.head.length;
  const plan = buildPlan(
    planner.exports,
    workerCount,
    vertexCount,
    dimension,
    edgeCount,
  );
  const memory = new WebAssembly.Memory({
    initial: plan.pageCount,
    maximum: plan.pageCount,
    shared: true,
  });
  writeRunHeader(memory, plan, {
    workerCount,
    vertexCount,
    dimension,
    edgeCount,
    epochCount: 500,
    a: 1.932808397545408,
    b: 0.7904949735905139,
  });
  copyRunInputs(memory, plan, fixture);

  const workers = Array.from(
    { length: workerCount },
    () => new Worker(leafUrl, { type: "module" }),
  );
  const state = {
    memory,
    plan,
    workers,
    cancel() {
      cancelRun(memory, plan);
    },
  };
  activeRun = state;
  try {
    await Promise.all(
      workers.map(async (worker, workerId) => {
        const ready = once(worker, "ready");
        worker.postMessage({
          type: "initialize",
          module,
          memory,
          stackTop:
            planner.exports.umap_layout_threaded_stack_top(workerId),
        });
        await ready;
      }),
    );

    const trials = [];
    for (let trial = 0; trial < repeatCount; trial += 1) {
      if (trial > 0) {
        resetRunControl(memory, plan);
        copyRunInputs(memory, plan, fixture);
      }
      const timing = await runOnce(workers, memory, plan);
      const projection = new Float32Array(
        memory.buffer,
        plan.embedding,
        fixture.embedding.length,
      ).slice();
      trials.push({ ...timing, projection });
    }
    return {
      workerCount,
      sharedMemoryBytes: memory.buffer.byteLength,
      trials,
    };
  } finally {
    if (activeRun === state) activeRun = undefined;
    workers.forEach((worker) => worker.terminate());
  }
}

self.onmessage = async (event) => {
  const message = event.data;
  if (message?.type === "cancel") {
    activeRun?.cancel();
    return;
  }
  if (message?.type !== "run") return;

  try {
    const result = await runFixture(message);
    const transfer = result.trials.map(
      (trial) => trial.projection.buffer,
    );
    self.postMessage({ type: "complete", result }, transfer);
  } catch (error) {
    activeRun?.cancel();
    self.postMessage({
      type: "error",
      error: error instanceof Error ? error.stack : String(error),
    });
  }
};
