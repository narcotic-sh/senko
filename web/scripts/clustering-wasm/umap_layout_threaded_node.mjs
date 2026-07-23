import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";

import {
  buildPlan,
  copyRunInputs,
  readRunControl,
  resetRunControl,
  writeRunHeader,
} from "./umap_layout_threaded_host.js";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptDirectory, "../../..");
const fixtureDirectory = resolve(
  repositoryRoot,
  ".research/native-reference/clustering-parity/test-audio/seed-42",
);
const workerUrl = new URL(
  "./umap_layout_threaded_node_leaf.mjs",
  import.meta.url,
);

function parseWorkerCount() {
  const flag = process.argv.find((argument) =>
    argument.startsWith("--workers="),
  );
  const value = flag === undefined ? 1 : Number(flag.slice(10));
  if (!Number.isSafeInteger(value) || value <= 0 || value > 32) {
    throw new RangeError("--workers must be an integer from 1 through 32");
  }
  return value;
}

function parseOutputPath() {
  const flag = process.argv.find((argument) =>
    argument.startsWith("--output="),
  );
  return flag === undefined
    ? undefined
    : resolve(process.cwd(), flag.slice("--output=".length));
}

function parseWasmPath() {
  const flag = process.argv.find((argument) =>
    argument.startsWith("--wasm="),
  );
  return flag === undefined
    ? resolve(scriptDirectory, "umap_layout_threaded.wasm")
    : resolve(process.cwd(), flag.slice("--wasm=".length));
}

function once(worker, expectedType) {
  return new Promise((resolvePromise, rejectPromise) => {
    const onMessage = (message) => {
      if (message?.type === "error") {
        cleanup();
        rejectPromise(new Error(message.error));
      } else if (message?.type === expectedType) {
        cleanup();
        resolvePromise(message);
      }
    };
    const onError = (error) => {
      cleanup();
      rejectPromise(error);
    };
    const cleanup = () => {
      worker.off("message", onMessage);
      worker.off("error", onError);
    };
    worker.on("message", onMessage);
    worker.on("error", onError);
  });
}

async function loadTypedArray(name, Constructor) {
  const bytes = await readFile(resolve(fixtureDirectory, name));
  const copied = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  return new Constructor(copied);
}

function sampledPairDistanceRelativeError(
  reference,
  candidate,
  count,
  dimension,
  sampleCount,
) {
  let state = 0x243f6a88;
  let numerator = 0;
  let denominator = 0;
  for (let sample = 0; sample < sampleCount; sample += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    const left = state % count;
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    let right = state % count;
    if (right === left) right = (right + 1) % count;
    let referenceSquared = 0;
    let candidateSquared = 0;
    for (let column = 0; column < dimension; column += 1) {
      const referenceDifference =
        reference[left * dimension + column] -
        reference[right * dimension + column];
      const candidateDifference =
        candidate[left * dimension + column] -
        candidate[right * dimension + column];
      referenceSquared += referenceDifference * referenceDifference;
      candidateSquared += candidateDifference * candidateDifference;
    }
    const referenceDistance = Math.sqrt(referenceSquared);
    const difference = Math.sqrt(candidateSquared) - referenceDistance;
    numerator += difference * difference;
    denominator += referenceDistance * referenceDistance;
  }
  return Math.sqrt(numerator / denominator);
}

const workerCount = parseWorkerCount();
const outputPath = parseOutputPath();
const wasmPath = parseWasmPath();
const [wasmBytes, embedding, head, tail, epochsPerSample, rngSeed, reference] =
  await Promise.all([
    readFile(wasmPath),
    loadTypedArray("umap-layout-initial-embedding.f32", Float32Array),
    loadTypedArray("umap-layout-head.i32", Int32Array),
    loadTypedArray("umap-layout-tail.i32", Int32Array),
    loadTypedArray("umap-layout-epochs-per-sample.f64", Float64Array),
    loadTypedArray("umap-layout-rng-state.i64", BigInt64Array),
    loadTypedArray("umap-projection.f32", Float32Array),
  ]);
const module = await WebAssembly.compile(wasmBytes);
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
const edgeCount = head.length;
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
copyRunInputs(memory, plan, {
  embedding,
  head,
  tail,
  epochsPerSample,
  rngSeed,
});

const workers = Array.from(
  { length: workerCount },
  () => new Worker(workerUrl, { type: "module" }),
);
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
  const results = await Promise.all(completions);
  const durationMs = performance.now() - startedAt;
  const projection = new Float32Array(
    memory.buffer,
    plan.embedding,
    embedding.length,
  );
  console.log({
    workerCount,
    durationMs,
    slowestWorkerMs: Math.max(
      ...results.map((result) => result.durationMs),
    ),
    statuses: results.map((result) => result.status),
    control: readRunControl(memory, plan),
    sharedMemoryBytes: memory.buffer.byteLength,
    pairDistanceError: sampledPairDistanceRelativeError(
      reference,
      projection,
      vertexCount,
      dimension,
      8_192,
    ),
  });
  if (outputPath !== undefined) {
    await writeFile(
      outputPath,
      new Uint8Array(
        projection.buffer,
        projection.byteOffset,
        projection.byteLength,
      ),
    );
  }

  // Leave this reset logic exercised for the browser's warmed second run.
  resetRunControl(memory, plan);
} finally {
  await Promise.all(workers.map((worker) => worker.terminate()));
}
