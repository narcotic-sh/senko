export const HEADER_MAGIC = 0x534b554d;
export const HEADER_VERSION = 1;
export const MINIMUM_MEMORY_PAGES = 256;

export const PLAN_SECTION = Object.freeze({
  header: 0,
  embedding: 1,
  head: 2,
  tail: 3,
  epochsPerSample: 4,
  rngSeed: 5,
  epochsPerNegativeSample: 6,
  epochOfNextNegativeSample: 7,
  epochOfNextSample: 8,
  rngStatePerVertex: 9,
  totalBytes: 10,
});

const LITTLE_ENDIAN = true;
const PAGE_BYTES = 65_536;

export function buildPlan(
  exports,
  workerCount,
  vertexCount,
  dimension,
  edgeCount,
) {
  const values = {};
  for (const [name, section] of Object.entries(PLAN_SECTION)) {
    values[name] = exports.umap_layout_threaded_plan_offset(
      section,
      workerCount,
      vertexCount,
      dimension,
      edgeCount,
    );
    if (values[name] === 0) {
      throw new RangeError(`Could not plan UMAP layout section ${name}`);
    }
  }
  if (
    values.totalBytes % PAGE_BYTES !== 0 ||
    values.totalBytes < MINIMUM_MEMORY_PAGES * PAGE_BYTES
  ) {
    throw new RangeError(
      `Invalid planned memory size ${values.totalBytes}`,
    );
  }
  return Object.freeze({
    ...values,
    pageCount: values.totalBytes / PAGE_BYTES,
  });
}

export function writeRunHeader(
  memory,
  plan,
  {
    workerCount,
    vertexCount,
    dimension,
    edgeCount,
    epochCount,
    a,
    b,
    gamma = 1,
    negativeSampleRate = 5,
  },
) {
  const header = new DataView(memory.buffer, plan.header, 128);
  const u32 = (offset, value) =>
    header.setUint32(offset, value, LITTLE_ENDIAN);
  const f64 = (offset, value) =>
    header.setFloat64(offset, value, LITTLE_ENDIAN);

  u32(0, HEADER_MAGIC);
  u32(4, HEADER_VERSION);
  u32(8, plan.totalBytes);
  u32(12, workerCount);
  u32(16, vertexCount);
  u32(20, dimension);
  u32(24, edgeCount);
  u32(28, epochCount);
  u32(32, plan.embedding);
  u32(36, plan.head);
  u32(40, plan.tail);
  u32(44, plan.epochsPerSample);
  u32(48, plan.rngSeed);
  u32(52, plan.epochsPerNegativeSample);
  u32(56, plan.epochOfNextNegativeSample);
  u32(60, plan.epochOfNextSample);
  u32(64, plan.rngStatePerVertex);
  u32(68, 0);
  f64(72, a);
  f64(80, b);
  f64(88, gamma);
  f64(96, negativeSampleRate);
  new Int32Array(memory.buffer, plan.header + 104, 6).fill(0);
}

export function copyRunInputs(
  memory,
  plan,
  { embedding, head, tail, epochsPerSample, rngSeed },
) {
  new Float32Array(
    memory.buffer,
    plan.embedding,
    embedding.length,
  ).set(embedding);
  new Int32Array(memory.buffer, plan.head, head.length).set(head);
  new Int32Array(memory.buffer, plan.tail, tail.length).set(tail);
  new Float64Array(
    memory.buffer,
    plan.epochsPerSample,
    epochsPerSample.length,
  ).set(epochsPerSample);
  new BigInt64Array(memory.buffer, plan.rngSeed, rngSeed.length).set(
    rngSeed,
  );
}

export function resetRunControl(memory, plan) {
  new Int32Array(memory.buffer, plan.header + 104, 6).fill(0);
}

export function readRunControl(memory, plan) {
  const words = new Int32Array(
    memory.buffer,
    plan.header + 104,
    6,
  );
  return {
    arrived: Atomics.load(words, 0),
    generation: Atomics.load(words, 1),
    cancelled: Atomics.load(words, 2),
    status: Atomics.load(words, 3),
    completedEpochs: Atomics.load(words, 4),
  };
}

export function cancelRun(memory, plan) {
  const words = new Int32Array(
    memory.buffer,
    plan.header + 104,
    6,
  );
  Atomics.store(words, 2, 1);
  Atomics.notify(words, 1);
}
