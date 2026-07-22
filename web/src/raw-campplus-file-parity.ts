/// <reference types="@webgpu/types" />

import { clusterEmbeddings, WasmClusteringKernels } from "./clustering";
import { BrowserModelSet } from "./pipeline/browser-models";
import { runBrowserPipeline, type BrowserPipelineModels } from "./pipeline/browser-pipeline";
import { configureOrt, OrtEmbeddingBackend } from "./pipeline/ort-backends";
import { postprocessClustering } from "./pipeline/postprocess";
import type { EmbeddingBatchBackend, Subsegment } from "./pipeline/types";
import { DEFAULT_PIPELINE_OPTIONS, type PipelineResult } from "./runtime/types";

const MANIFEST_URL = "/models/manifest.json";
const RAW_BATCH = 16;
const ORT_BATCH = 32;
const FRAMES = 150;
const FEATURES = 80;
const EMBEDDING_DIM = 192;

interface RowError {
  readonly index: number;
  readonly cosine: number;
  readonly maxAbsoluteError: number;
  readonly meanAbsoluteError: number;
  readonly rawNorm: number;
  readonly ortNorm: number;
}

interface EmbeddingComparison {
  readonly maxAbsoluteError: number;
  readonly meanAbsoluteError: number;
  readonly rmsError: number;
  readonly cosine: {
    readonly minimum: number;
    readonly p01: number;
    readonly p05: number;
    readonly median: number;
    readonly mean: number;
    readonly p95: number;
  };
  readonly worstRows: readonly RowError[];
}

/**
 * Query-gated correctness harness. It feeds each real FBank batch to the raw
 * B16 graph and the pinned ORT-fp16 B32 reference before the host buffer can be
 * reused, while retaining only the two final embedding matrices.
 */
export async function runRawCampPlusFileParityDiagnostic(
  root: HTMLElement,
): Promise<void> {
  root.innerHTML = `
    <section class="shell">
      <header class="hero"><div><p class="eyebrow">Correctness harness</p>
        <h1>Raw CAM++ real-window parity</h1>
        <p class="lede">Identical streamed FBank windows, raw B16 versus ORT-fp16 B32.</p>
      </div></header>
      <section class="panel controls">
        <label class="file-picker"><span>Choose audio</span>
          <input id="audio-file" type="file" accept="audio/wav,.wav" />
        </label>
        <button id="run-pipeline" type="button" disabled>Run parity diagnostic</button>
      </section>
      <section class="panel"><p id="status" class="status" data-kind="loading">Loading direct WebGPU VAD…</p></section>
      <section id="result-panel" class="panel result-panel" hidden>
        <h2>Result</h2><pre id="result"></pre>
      </section>
    </section>`;

  const input = required<HTMLInputElement>(root, "#audio-file");
  const button = required<HTMLButtonElement>(root, "#run-pipeline");
  const status = required<HTMLElement>(root, "#status");
  const resultElement = required<HTMLElement>(root, "#result");
  const resultPanel = required<HTMLElement>(root, "#result-panel");
  let file: File | undefined;
  let models: BrowserModelSet | undefined;
  let kernels: WasmClusteringKernels | undefined;
  let comparator: ComparingEmbeddingBackend | undefined;

  const setStatus = (message: string, kind: "loading" | "ready" | "error") => {
    status.textContent = message;
    status.dataset.kind = kind;
  };
  const updateButton = () => {
    button.disabled = file === undefined || models === undefined || kernels === undefined;
  };
  input.addEventListener("change", () => {
    file = input.files?.[0];
    updateButton();
  });

  try {
    if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
    [models, kernels] = await Promise.all([
      BrowserModelSet.load(new URL(MANIFEST_URL, location.href).href, navigator.gpu, {
        vadBatchSize: 8,
        embeddingBatchSize: RAW_BATCH,
        warmupRuns: 1,
        onProgress: ({ message }) => setStatus(`${message}…`, "loading"),
      }),
      WasmClusteringKernels.create(),
    ]);
    kernels.warmup();
    comparator = new ComparingEmbeddingBackend(models.embedding, models);
    setStatus("WebGPU models ready.", "ready");
    updateButton();
  } catch (error) {
    setStatus(errorMessage(error), "error");
    await models?.release();
    kernels?.dispose();
    return;
  }

  const release = () => {
    void comparator?.release();
    void models?.release();
    kernels?.dispose();
  };
  globalThis.addEventListener("pagehide", release, { once: true });

  button.addEventListener("click", () => {
    if (file === undefined || models === undefined || kernels === undefined || comparator === undefined) {
      return;
    }
    button.disabled = true;
    setStatus("Running raw and ORT CAM++ on identical real windows…", "loading");
    void execute(file, models, kernels, comparator)
      .then((result) => {
        resultPanel.hidden = false;
        resultElement.textContent = JSON.stringify(result, null, 2);
        setStatus("Pipeline complete.", "ready");
      })
      .catch((error) => setStatus(errorMessage(error), "error"));
  });
}

async function execute(
  audio: File,
  models: BrowserModelSet,
  kernels: WasmClusteringKernels,
  comparator: ComparingEmbeddingBackend,
): Promise<PipelineResult & { readonly diagnostic: Record<string, unknown> }> {
  const facade: BrowserPipelineModels = {
    get vad() {
      return models.vad;
    },
    embedding: comparator,
    get knownGpuBufferBytes() {
      return models.knownGpuBufferBytes;
    },
  };
  await comparator.prepare();
  let browserSubsegments: readonly Subsegment[] = [];
  let pipeline: PipelineResult;
  try {
    pipeline = await runBrowserPipeline(
      audio,
      facade,
      DEFAULT_PIPELINE_OPTIONS,
      {
        clusteringKernels: kernels,
        onSubsegmentsCreated: (subsegments) => {
          browserSubsegments = subsegments.map((item) => ({ ...item }));
        },
      },
    );
  } finally {
    await comparator.finish();
  }
  const embeddingStage = pipeline.stages.find((stage) => stage.stage === "embedding");
  if (embeddingStage?.stage !== "embedding") {
    throw new Error("Embedding stage metrics are missing");
  }
  const count = embeddingStage.metrics.embeddingCount;
  if (browserSubsegments.length !== count) {
    throw new Error(`Captured ${browserSubsegments.length}/${count} browser subsegments`);
  }
  const { raw, ort } = comparator.embeddings(count);

  const rawClusteringStart = performance.now();
  const rawLabels = clusterEmbeddings(raw, count, EMBEDDING_DIM, {}, kernels);
  const rawClusteringMs = performance.now() - rawClusteringStart;
  const ortClusteringStart = performance.now();
  const ortLabels = clusterEmbeddings(ort, count, EMBEDDING_DIM, {}, kernels);
  const ortClusteringMs = performance.now() - ortClusteringStart;
  const rawPostprocess = postprocessClustering(raw, rawLabels, browserSubsegments);
  const ortPostprocess = postprocessClustering(ort, ortLabels, browserSubsegments);
  const native = await loadNativeReference();
  const nativeDiagnostic =
    native === undefined
      ? undefined
      : buildNativeDiagnostic(
          raw,
          ort,
          rawLabels,
          ortLabels,
          browserSubsegments,
          native,
          rawPostprocess.centroids,
          ortPostprocess.centroids,
        );
  const diagnostic = {
    embeddingCount: count,
    rawBatchSize: RAW_BATCH,
    ortBatchSize: ORT_BATCH,
    ortModel: "campplus-t150-b32-fp16.onnx",
    embeddingComparison: compareEmbeddings(raw, ort, count),
    adjustedRandIndex: adjustedRandIndex(rawLabels, ortLabels),
    rawClusterSizes: clusterSizes(rawLabels),
    ortClusterSizes: clusterSizes(ortLabels),
    rawToOrtContingency: contingency(rawLabels, ortLabels, raw, ort),
    ortOnlinePairing: comparator.onlinePairingDiagnostics(),
    rawPostprocessSpeakerCount: rawPostprocess.speakerCount,
    ortPostprocessSpeakerCount: ortPostprocess.speakerCount,
    ...(nativeDiagnostic === undefined ? {} : { nativeReference: nativeDiagnostic }),
    rawLabels: Array.from(rawLabels),
    ortLabels: Array.from(ortLabels),
    repeatedClusteringMs: { raw: rawClusteringMs, ort: ortClusteringMs },
    pipelineSpeakerCount: pipeline.speakerCount,
    ...(new URLSearchParams(location.search).get("captureBuffers") === "1"
      ? {
          capturedBuffers: {
            encoding: "base64",
            rawEmbeddingsF32: toBase64(raw),
            ortEmbeddingsF32: toBase64(ort),
            browserSubsegmentsF64: toBase64(
              encodeSubsegments(browserSubsegments),
            ),
          },
        }
      : {}),
  };
  Object.assign(globalThis, {
    __senkoRawCampPlusEmbeddings: raw,
    __senkoOrtCampPlusEmbeddings: ort,
    __senkoRawCampPlusLabels: rawLabels,
    __senkoOrtCampPlusLabels: ortLabels,
  });
  return {
    ...pipeline,
    diagnostic,
    // The generic isolated-Chrome runner uses page-memory mode to permit a
    // development-only /@fs native oracle. This harness owns no memory sampler;
    // mark that auxiliary channel settled so the runner can capture the JSON.
    ...(new URLSearchParams(location.search).has("memory")
      ? { pageMemory: { supported: false, pending: false, samples: [] } }
      : {}),
  };
}

interface NativeReference {
  readonly embeddings: Float32Array;
  readonly labels: Int32Array;
  readonly browserReferenceLabels?: Int32Array;
  readonly subsegments: Float64Array;
}

interface SubsegmentPair {
  readonly browserIndex: number;
  readonly nativeIndex: number;
  readonly startErrorSeconds: number;
  readonly endErrorSeconds: number;
}

async function loadNativeReference(): Promise<NativeReference | undefined> {
  const parameters = new URLSearchParams(location.search);
  const embeddingsUrl = parameters.get("nativeEmbeddings");
  const labelsUrl = parameters.get("nativeLabels");
  const subsegmentsUrl = parameters.get("nativeSubsegments");
  if (embeddingsUrl === null && labelsUrl === null && subsegmentsUrl === null) return undefined;
  if (embeddingsUrl === null || labelsUrl === null || subsegmentsUrl === null) {
    throw new Error(
      "nativeEmbeddings, nativeLabels, and nativeSubsegments must be supplied together",
    );
  }
  const [embeddingBytes, labelBytes, browserLabelBytes, subsegmentBytes] =
    await Promise.all([
      fetchBytes(embeddingsUrl),
      fetchBytes(labelsUrl),
      fetchOptionalBytes(parameters.get("browserReferenceLabels")),
      fetchBytes(subsegmentsUrl),
    ]);
  const rowBytes = EMBEDDING_DIM * Float32Array.BYTES_PER_ELEMENT;
  if (embeddingBytes.byteLength % rowBytes !== 0) {
    throw new Error("native embeddings do not contain complete rows");
  }
  const count = embeddingBytes.byteLength / rowBytes;
  requireByteLength(
    labelBytes,
    count * Int32Array.BYTES_PER_ELEMENT,
    "native labels",
  );
  if (browserLabelBytes !== undefined) {
    requireByteLength(
      browserLabelBytes,
      count * Int32Array.BYTES_PER_ELEMENT,
      "browser/native-embedding labels",
    );
  }
  requireByteLength(
    subsegmentBytes,
    count * 2 * Float64Array.BYTES_PER_ELEMENT,
    "native subsegments",
  );
  return {
    embeddings: new Float32Array(embeddingBytes),
    labels: new Int32Array(labelBytes),
    ...(browserLabelBytes === undefined
      ? {}
      : { browserReferenceLabels: new Int32Array(browserLabelBytes) }),
    subsegments: new Float64Array(subsegmentBytes),
  };
}

function buildNativeDiagnostic(
  raw: Float32Array,
  ort: Float32Array,
  rawLabels: Int32Array,
  ortLabels: Int32Array,
  browserSubsegments: readonly Subsegment[],
  native: NativeReference,
  rawCentroids: Readonly<Record<string, Float32Array>>,
  ortCentroids: Readonly<Record<string, Float32Array>>,
): Record<string, unknown> {
  const alignment = alignSubsegments(browserSubsegments, native.subsegments);
  const browserIndices = alignment.pairs.map((pair) => pair.browserIndex);
  const nativeIndices = alignment.pairs.map((pair) => pair.nativeIndex);
  const alignedRaw = selectRows(raw, browserIndices);
  const alignedOrt = selectRows(ort, browserIndices);
  const alignedNative = selectRows(native.embeddings, nativeIndices);
  const alignedRawLabels = selectLabels(rawLabels, browserIndices);
  const alignedOrtLabels = selectLabels(ortLabels, browserIndices);
  const alignedNativeLabels = selectLabels(native.labels, nativeIndices);
  const rawVsNative = compareEmbeddings(alignedRaw, alignedNative, alignment.pairs.length);
  const ortVsNative = compareEmbeddings(alignedOrt, alignedNative, alignment.pairs.length);
  const exactPairs = alignment.pairs.filter(
    (pair) => pair.startErrorSeconds <= 1e-9 && pair.endErrorSeconds <= 1e-9,
  );
  const exactBrowserIndices = exactPairs.map((pair) => pair.browserIndex);
  const exactNativeIndices = exactPairs.map((pair) => pair.nativeIndex);
  const exactRaw = selectRows(raw, exactBrowserIndices);
  const exactOrt = selectRows(ort, exactBrowserIndices);
  const exactNative = selectRows(native.embeddings, exactNativeIndices);
  const nativeSubsegments = nativeSubsegmentObjects(native.subsegments);
  const nativePostprocess = postprocessClustering(
    native.embeddings,
    // The native HDBSCAN file intentionally retains 16 noise rows. The
    // browser-on-native labels are the seven-speaker postprocessed oracle and
    // therefore the compatible source for ranked speaker centroids.
    native.browserReferenceLabels ?? native.labels,
    nativeSubsegments,
  );
  return {
    counts: {
      browser: browserSubsegments.length,
      native: native.labels.length,
      matched: alignment.pairs.length,
    },
    alignment: {
      exactMatches: exactPairs.length,
      maxStartErrorSeconds: Math.max(
        0,
        ...alignment.pairs.map((pair) => pair.startErrorSeconds),
      ),
      maxEndErrorSeconds: Math.max(
        0,
        ...alignment.pairs.map((pair) => pair.endErrorSeconds),
      ),
      unmatchedBrowser: alignment.unmatchedBrowser.map((index) => ({
        index,
        start: browserSubsegments[index]!.start,
        end: browserSubsegments[index]!.end,
      })),
      unmatchedNative: alignment.unmatchedNative.map((index) => ({
        index,
        start: native.subsegments[index * 2]!,
        end: native.subsegments[index * 2 + 1]!,
      })),
    },
    embeddingComparison: {
      allTimeAlignedRows: { rawVsNative, ortVsNative },
      exactBoundaryRows: {
        count: exactPairs.length,
        rawVsNative: compareEmbeddings(exactRaw, exactNative, exactPairs.length),
        ortVsNative: compareEmbeddings(exactOrt, exactNative, exactPairs.length),
      },
      interpretation:
        "Raw-vs-ORT uses identical FBank tensors and isolates inference. Native comparisons additionally include model/backend and FBank implementation differences; non-exact boundary rows also differ in audio content.",
    },
    labelComparison: {
      rawVsNativeAri: adjustedRandIndex(alignedRawLabels, alignedNativeLabels),
      ortVsNativeAri: adjustedRandIndex(alignedOrtLabels, alignedNativeLabels),
      rawToNativeContingency: contingency(
        alignedRawLabels,
        alignedNativeLabels,
        alignedRaw,
        alignedNative,
      ),
      nativeClusterSizes: clusterSizes(native.labels),
      ...(native.browserReferenceLabels === undefined
        ? {}
        : {
            rawVsBrowserNativeEmbeddingLabelsAri: adjustedRandIndex(
              alignedRawLabels,
              selectLabels(native.browserReferenceLabels, nativeIndices),
            ),
            ortVsBrowserNativeEmbeddingLabelsAri: adjustedRandIndex(
              alignedOrtLabels,
              selectLabels(native.browserReferenceLabels, nativeIndices),
            ),
            browserNativeEmbeddingClusterSizes: clusterSizes(
              native.browserReferenceLabels,
            ),
          }),
    },
    rareSpeakerCentroids: rareSpeakerCentroidDiagnostics(
      rawCentroids,
      ortCentroids,
      nativePostprocess.centroids,
    ),
    worstRawRowsWithTimes: attachAlignedTimes(
      rawVsNative.worstRows,
      alignment.pairs,
      browserSubsegments,
      native.subsegments,
    ),
  };
}

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Unable to fetch native oracle ${url}: HTTP ${response.status}`);
  return await response.arrayBuffer();
}

async function fetchOptionalBytes(url: string | null): Promise<ArrayBuffer | undefined> {
  return url === null ? undefined : await fetchBytes(url);
}

function requireByteLength(bytes: ArrayBuffer, expected: number, label: string): void {
  if (bytes.byteLength !== expected) {
    throw new Error(`${label} has ${bytes.byteLength} bytes; expected ${expected}`);
  }
}

function attachAlignedTimes(
  rows: readonly RowError[],
  pairs: readonly SubsegmentPair[],
  browser: readonly Subsegment[],
  native: Float64Array,
): Array<Record<string, number>> {
  return rows.map((row) => {
    const pair = pairs[row.index]!;
    const browserSegment = browser[pair.browserIndex]!;
    return {
      ...row,
      browserIndex: pair.browserIndex,
      nativeIndex: pair.nativeIndex,
      browserStart: browserSegment.start,
      browserEnd: browserSegment.end,
      nativeStart: native[pair.nativeIndex * 2]!,
      nativeEnd: native[pair.nativeIndex * 2 + 1]!,
    };
  });
}

function alignSubsegments(
  browser: readonly Subsegment[],
  native: Float64Array,
): {
  readonly pairs: readonly SubsegmentPair[];
  readonly unmatchedBrowser: readonly number[];
  readonly unmatchedNative: readonly number[];
} {
  const nativeCount = native.length / 2;
  const pairs: SubsegmentPair[] = [];
  const unmatchedBrowser: number[] = [];
  const unmatchedNative: number[] = [];
  let browserIndex = 0;
  let nativeIndex = 0;
  const matchToleranceSeconds = 0.2;
  while (browserIndex < browser.length && nativeIndex < nativeCount) {
    const current = subsegmentError(browser[browserIndex]!, native, nativeIndex);
    if (current.total <= matchToleranceSeconds * 2) {
      pairs.push({
        browserIndex,
        nativeIndex,
        startErrorSeconds: current.start,
        endErrorSeconds: current.end,
      });
      browserIndex += 1;
      nativeIndex += 1;
      continue;
    }
    const afterSkippingBrowser =
      browserIndex + 1 < browser.length
        ? subsegmentError(browser[browserIndex + 1]!, native, nativeIndex).total
        : Number.POSITIVE_INFINITY;
    const afterSkippingNative =
      nativeIndex + 1 < nativeCount
        ? subsegmentError(browser[browserIndex]!, native, nativeIndex + 1).total
        : Number.POSITIVE_INFINITY;
    if (
      afterSkippingNative <= matchToleranceSeconds * 2 &&
      afterSkippingNative <= afterSkippingBrowser
    ) {
      unmatchedNative.push(nativeIndex);
      nativeIndex += 1;
    } else if (afterSkippingBrowser <= matchToleranceSeconds * 2) {
      unmatchedBrowser.push(browserIndex);
      browserIndex += 1;
    } else if (browser[browserIndex]!.start < native[nativeIndex * 2]!) {
      unmatchedBrowser.push(browserIndex);
      browserIndex += 1;
    } else {
      unmatchedNative.push(nativeIndex);
      nativeIndex += 1;
    }
  }
  while (browserIndex < browser.length) unmatchedBrowser.push(browserIndex++);
  while (nativeIndex < nativeCount) unmatchedNative.push(nativeIndex++);
  return { pairs, unmatchedBrowser, unmatchedNative };
}

function subsegmentError(
  browser: Subsegment,
  native: Float64Array,
  nativeIndex: number,
): { readonly start: number; readonly end: number; readonly total: number } {
  const start = Math.abs(browser.start - native[nativeIndex * 2]!);
  const end = Math.abs(browser.end - native[nativeIndex * 2 + 1]!);
  return { start, end, total: start + end };
}

function selectRows(values: Float32Array, indices: readonly number[]): Float32Array {
  const output = new Float32Array(indices.length * EMBEDDING_DIM);
  for (let row = 0; row < indices.length; row += 1) {
    const source = indices[row]! * EMBEDDING_DIM;
    output.set(
      values.subarray(source, source + EMBEDDING_DIM),
      row * EMBEDDING_DIM,
    );
  }
  return output;
}

function selectLabels(values: Int32Array, indices: readonly number[]): Int32Array {
  const output = new Int32Array(indices.length);
  for (let row = 0; row < indices.length; row += 1) output[row] = values[indices[row]!]!;
  return output;
}

function nativeSubsegmentObjects(values: Float64Array): Subsegment[] {
  return Array.from({ length: values.length / 2 }, (_, index) => ({
    index,
    start: values[index * 2]!,
    end: values[index * 2 + 1]!,
  }));
}

function rareSpeakerCentroidDiagnostics(
  raw: Readonly<Record<string, Float32Array>>,
  ort: Readonly<Record<string, Float32Array>>,
  native: Readonly<Record<string, Float32Array>>,
): Record<string, unknown> {
  const rawS05 = raw.SPEAKER_05;
  const rawS08 = raw.SPEAKER_08;
  return {
    ...(rawS05 === undefined || rawS08 === undefined
      ? {}
      : { rawS05ToRawS08Cosine: cosine(rawS05, rawS08) }),
    ...(rawS05 === undefined ? {} : { rawS05ToOrt: centroidMatches(rawS05, ort) }),
    ...(rawS08 === undefined ? {} : { rawS08ToOrt: centroidMatches(rawS08, ort) }),
    ...(rawS05 === undefined ? {} : { rawS05ToNative: centroidMatches(rawS05, native) }),
    ...(rawS08 === undefined ? {} : { rawS08ToNative: centroidMatches(rawS08, native) }),
    ...(ort.SPEAKER_05 === undefined
      ? {}
      : { ortS05ToNative: centroidMatches(ort.SPEAKER_05, native) }),
  };
}

function centroidMatches(
  source: Float32Array,
  targets: Readonly<Record<string, Float32Array>>,
): Array<{ readonly speaker: string; readonly cosine: number }> {
  return Object.entries(targets)
    .map(([speaker, target]) => ({ speaker, cosine: cosine(source, target) }))
    .sort((left, right) => right.cosine - left.cosine);
}

class ComparingEmbeddingBackend implements EmbeddingBatchBackend {
  readonly batchSize = RAW_BATCH;
  readonly frames = FRAMES;
  readonly featureDim = FEATURES;
  readonly embeddingDim = EMBEDDING_DIM;

  private readonly rawChunks: Float32Array[] = [];
  private readonly ortChunks: Float32Array[] = [];
  private readonly ortInput = new Float32Array(ORT_BATCH * FRAMES * FEATURES);
  private readonly onlineDirectCosines: number[] = [];
  private readonly onlineSwappedCosines: number[] = [];
  private readonly onlinePreviousShiftCosines: number[] = [];
  private readonly onlineNextShiftCosines: number[] = [];
  private ort: OrtEmbeddingBackend | undefined;
  private halfPending = false;
  private released = false;

  constructor(
    private readonly raw: EmbeddingBatchBackend,
    private readonly models: BrowserModelSet,
  ) {}

  async prepare(): Promise<void> {
    if (this.ort !== undefined) return;
    if (this.released) throw new Error("Parity backend has been released");
    const precision = this.models.manifest.models.campplus.precision_variants?.float16;
    const variant = precision?.batches[String(ORT_BATCH)];
    if (precision === undefined || variant === undefined) {
      throw new Error("Pinned ORT-fp16 CAM++ B32 reference is missing");
    }
    const runtime = configureOrt({
      // Give ORT the CAM++ device so both embedding graphs share one ordered
      // queue without contending with the separately resident VAD device.
      device: this.models.embeddingDevice,
      graphCapture: false,
      graphOptimizationLevel:
        precision.ort_web?.required_graph_optimization_level ?? "basic",
      strictWebGpu: true,
    });
    this.ort = await OrtEmbeddingBackend.create(
      runtime,
      {
        url: new URL(variant.file, new URL(MANIFEST_URL, location.href)).href,
        inputName: this.models.manifest.models.campplus.input.name,
        outputName: this.models.manifest.models.campplus.output.name,
        byteLength: variant.bytes,
        sha256: variant.sha256,
      },
      ORT_BATCH,
    );
    await this.ort.run(this.ortInput);
  }

  async run(features: Float32Array): Promise<Float32Array> {
    if (this.released) throw new Error("Parity backend has been released");
    const ort = this.ort;
    if (ort === undefined) throw new Error("ORT parity backend is not prepared");
    const expected = RAW_BATCH * FRAMES * FEATURES;
    if (features.length !== expected) throw new Error(`Expected ${expected} CAM++ features`);
    const rawOutput = await this.raw.run(features);
    this.rawChunks.push(rawOutput);
    const offset = this.halfPending ? expected : 0;
    this.ortInput.set(features, offset);
    if (this.halfPending) {
      const ortOutput = await ort.run(this.ortInput);
      const previousRaw = this.rawChunks[this.rawChunks.length - 2]!;
      observePairing(
        previousRaw,
        rawOutput,
        ortOutput,
        this.onlineDirectCosines,
        this.onlineSwappedCosines,
        this.onlinePreviousShiftCosines,
        this.onlineNextShiftCosines,
      );
      this.ortChunks.push(ortOutput);
      this.halfPending = false;
    } else {
      this.halfPending = true;
    }
    return rawOutput;
  }

  async finish(): Promise<void> {
    const ort = this.ort;
    if (ort === undefined) return;
    if (this.halfPending) {
      this.ortInput.fill(0, RAW_BATCH * FRAMES * FEATURES);
      this.ortChunks.push(await ort.run(this.ortInput));
      this.halfPending = false;
    }
    this.ort = undefined;
    await ort.release();
  }

  embeddings(count: number): { readonly raw: Float32Array; readonly ort: Float32Array } {
    const raw = flattenChunks(this.rawChunks, RAW_BATCH, count);
    const ort = flattenChunks(this.ortChunks, ORT_BATCH, count);
    if (raw.length !== ort.length) throw new Error("Raw and ORT embedding captures differ in size");
    return { raw, ort };
  }

  onlinePairingDiagnostics(): Record<string, unknown> {
    return {
      pairedRows: this.onlineDirectCosines.length,
      direct: summarizeCosines(this.onlineDirectCosines),
      swappedB16Halves: summarizeCosines(this.onlineSwappedCosines),
      previousRowWithinHalf: summarizeCosines(this.onlinePreviousShiftCosines),
      nextRowWithinHalf: summarizeCosines(this.onlineNextShiftCosines),
      rawChunkCount: this.rawChunks.length,
      ortChunkCount: this.ortChunks.length,
      uniqueRawArrayBuffers: new Set(this.rawChunks.map((chunk) => chunk.buffer)).size,
      uniqueOrtArrayBuffers: new Set(this.ortChunks.map((chunk) => chunk.buffer)).size,
    };
  }

  async release(): Promise<void> {
    if (this.released) return;
    this.released = true;
    const ort = this.ort;
    this.ort = undefined;
    await ort?.release();
  }

}

function flattenChunks(
  chunks: readonly Float32Array[],
  batchSize: number,
  count: number,
): Float32Array {
  const output = new Float32Array(count * EMBEDDING_DIM);
  let rows = 0;
  for (const chunk of chunks) {
    const copiedRows = Math.min(batchSize, count - rows);
    if (copiedRows <= 0) break;
    output.set(chunk.subarray(0, copiedRows * EMBEDDING_DIM), rows * EMBEDDING_DIM);
    rows += copiedRows;
  }
  if (rows !== count) throw new Error(`Captured ${rows}/${count} embedding rows`);
  return output;
}

function observePairing(
  firstRaw: Float32Array,
  secondRaw: Float32Array,
  ort: Float32Array,
  direct: number[],
  swapped: number[],
  previousShift: number[],
  nextShift: number[],
): void {
  for (let row = 0; row < RAW_BATCH; row += 1) {
    direct.push(rowCosine(firstRaw, row, ort, row));
    direct.push(rowCosine(secondRaw, row, ort, RAW_BATCH + row));
    swapped.push(rowCosine(firstRaw, row, ort, RAW_BATCH + row));
    swapped.push(rowCosine(secondRaw, row, ort, row));
    if (row > 0) {
      previousShift.push(rowCosine(firstRaw, row, ort, row - 1));
      previousShift.push(rowCosine(secondRaw, row, ort, RAW_BATCH + row - 1));
    }
    if (row + 1 < RAW_BATCH) {
      nextShift.push(rowCosine(firstRaw, row, ort, row + 1));
      nextShift.push(rowCosine(secondRaw, row, ort, RAW_BATCH + row + 1));
    }
  }
}

function rowCosine(
  left: Float32Array,
  leftRow: number,
  right: Float32Array,
  rightRow: number,
): number {
  let dot = 0;
  let leftSquared = 0;
  let rightSquared = 0;
  const leftOffset = leftRow * EMBEDDING_DIM;
  const rightOffset = rightRow * EMBEDDING_DIM;
  for (let column = 0; column < EMBEDDING_DIM; column += 1) {
    const leftValue = left[leftOffset + column]!;
    const rightValue = right[rightOffset + column]!;
    dot += leftValue * rightValue;
    leftSquared += leftValue * leftValue;
    rightSquared += rightValue * rightValue;
  }
  return dot / Math.sqrt(leftSquared * rightSquared);
}

function summarizeCosines(values: readonly number[]): Record<string, number> {
  if (values.length === 0) return { count: 0 };
  const sorted = [...values].sort((left, right) => left - right);
  return {
    count: values.length,
    first: values[0]!,
    minimum: sorted[0]!,
    p01: quantile(sorted, 0.01),
    median: quantile(sorted, 0.5),
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    p99: quantile(sorted, 0.99),
    maximum: sorted[sorted.length - 1]!,
  };
}

function encodeSubsegments(subsegments: readonly Subsegment[]): Float64Array {
  const output = new Float64Array(subsegments.length * 2);
  for (let index = 0; index < subsegments.length; index += 1) {
    output[index * 2] = subsegments[index]!.start;
    output[index * 2 + 1] = subsegments[index]!.end;
  }
  return output;
}

function toBase64(values: Float32Array | Float64Array): string {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  const chunkBytes = 32_768;
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += chunkBytes) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkBytes));
  }
  return btoa(binary);
}

function compareEmbeddings(
  raw: Float32Array,
  ort: Float32Array,
  count: number,
): EmbeddingComparison {
  let maximum = 0;
  let absoluteSum = 0;
  let squaredSum = 0;
  const rows: RowError[] = [];
  for (let row = 0; row < count; row += 1) {
    let dot = 0;
    let rawSquared = 0;
    let ortSquared = 0;
    let rowMaximum = 0;
    let rowAbsolute = 0;
    const offset = row * EMBEDDING_DIM;
    for (let column = 0; column < EMBEDDING_DIM; column += 1) {
      const left = raw[offset + column]!;
      const right = ort[offset + column]!;
      const difference = Math.abs(left - right);
      maximum = Math.max(maximum, difference);
      rowMaximum = Math.max(rowMaximum, difference);
      absoluteSum += difference;
      rowAbsolute += difference;
      squaredSum += difference * difference;
      dot += left * right;
      rawSquared += left * left;
      ortSquared += right * right;
    }
    rows.push({
      index: row,
      cosine: dot / Math.sqrt(rawSquared * ortSquared),
      maxAbsoluteError: rowMaximum,
      meanAbsoluteError: rowAbsolute / EMBEDDING_DIM,
      rawNorm: Math.sqrt(rawSquared),
      ortNorm: Math.sqrt(ortSquared),
    });
  }
  const sortedCosines = rows.map((row) => row.cosine).sort((a, b) => a - b);
  const cosineMean = sortedCosines.reduce((sum, value) => sum + value, 0) / count;
  return {
    maxAbsoluteError: maximum,
    meanAbsoluteError: absoluteSum / raw.length,
    rmsError: Math.sqrt(squaredSum / raw.length),
    cosine: {
      minimum: sortedCosines[0]!,
      p01: quantile(sortedCosines, 0.01),
      p05: quantile(sortedCosines, 0.05),
      median: quantile(sortedCosines, 0.5),
      mean: cosineMean,
      p95: quantile(sortedCosines, 0.95),
    },
    worstRows: [...rows].sort((a, b) => a.cosine - b.cosine).slice(0, 32),
  };
}

function clusterSizes(labels: Int32Array): Array<{ readonly label: number; readonly count: number }> {
  const counts = new Map<number, number>();
  for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
  return [...counts.entries()]
    .map(([label, count]) => ({ label, count }))
    .sort((left, right) => right.count - left.count || left.label - right.label);
}

function contingency(
  rawLabels: Int32Array,
  ortLabels: Int32Array,
  raw: Float32Array,
  ort: Float32Array,
): Array<Record<string, unknown>> {
  const rawSizes = clusterSizes(rawLabels);
  const ortCentroids = centroids(ort, ortLabels);
  const rawCentroids = centroids(raw, rawLabels);
  return rawSizes.map(({ label, count }) => {
    const overlaps = new Map<number, number>();
    for (let index = 0; index < rawLabels.length; index += 1) {
      if (rawLabels[index] === label) {
        const target = ortLabels[index]!;
        overlaps.set(target, (overlaps.get(target) ?? 0) + 1);
      }
    }
    const ordered = [...overlaps.entries()]
      .map(([ortLabel, overlap]) => ({ ortLabel, overlap }))
      .sort((left, right) => right.overlap - left.overlap || left.ortLabel - right.ortLabel);
    const best = ordered[0];
    const own = rawCentroids.get(label);
    const matched = best === undefined ? undefined : ortCentroids.get(best.ortLabel);
    const nearestOther = [...rawCentroids.entries()]
      .filter(([candidate]) => candidate !== label)
      .map(([candidate, centroid]) => ({
        rawLabel: candidate,
        cosine: own === undefined ? Number.NaN : cosine(own, centroid),
      }))
      .sort((left, right) => right.cosine - left.cosine)[0];
    return {
      rawLabel: label,
      rawCount: count,
      ortOverlaps: ordered,
      ...(own === undefined || matched === undefined
        ? {}
        : { matchedOrtCentroidCosine: cosine(own, matched) }),
      ...(nearestOther === undefined ? {} : { nearestRawCentroid: nearestOther }),
    };
  });
}

function centroids(values: Float32Array, labels: Int32Array): Map<number, Float64Array> {
  const result = new Map<number, Float64Array>();
  const counts = new Map<number, number>();
  for (let row = 0; row < labels.length; row += 1) {
    const label = labels[row]!;
    const centroid = result.get(label) ?? new Float64Array(EMBEDDING_DIM);
    const offset = row * EMBEDDING_DIM;
    for (let column = 0; column < EMBEDDING_DIM; column += 1) {
      centroid[column] = centroid[column]! + values[offset + column]!;
    }
    result.set(label, centroid);
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  for (const [label, centroid] of result) {
    const count = counts.get(label)!;
    for (let column = 0; column < EMBEDDING_DIM; column += 1) {
      centroid[column] = centroid[column]! / count;
    }
  }
  return result;
}

function adjustedRandIndex(left: Int32Array, right: Int32Array): number {
  if (left.length !== right.length) throw new Error("ARI label lengths differ");
  const cells = new Map<string, number>();
  const rowCounts = new Map<number, number>();
  const columnCounts = new Map<number, number>();
  for (let index = 0; index < left.length; index += 1) {
    const row = left[index]!;
    const column = right[index]!;
    const key = `${row}:${column}`;
    cells.set(key, (cells.get(key) ?? 0) + 1);
    rowCounts.set(row, (rowCounts.get(row) ?? 0) + 1);
    columnCounts.set(column, (columnCounts.get(column) ?? 0) + 1);
  }
  const cellPairs = sumPairs(cells.values());
  const rowPairs = sumPairs(rowCounts.values());
  const columnPairs = sumPairs(columnCounts.values());
  const totalPairs = chooseTwo(left.length);
  if (totalPairs === 0) return 1;
  const expected = (rowPairs * columnPairs) / totalPairs;
  const maximum = (rowPairs + columnPairs) / 2;
  return maximum === expected ? 1 : (cellPairs - expected) / (maximum - expected);
}

function sumPairs(values: Iterable<number>): number {
  let total = 0;
  for (const value of values) total += chooseTwo(value);
  return total;
}

function chooseTwo(value: number): number {
  return (value * (value - 1)) / 2;
}

function cosine(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let dot = 0;
  let leftSquared = 0;
  let rightSquared = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index]! * right[index]!;
    leftSquared += left[index]! * left[index]!;
    rightSquared += right[index]! * right[index]!;
  }
  return dot / Math.sqrt(leftSquared * rightSquared);
}

function quantile(sorted: readonly number[], fraction: number): number {
  const index = Math.min(sorted.length - 1, Math.floor(fraction * (sorted.length - 1)));
  return sorted[index]!;
}

function required<T extends Element>(root: HTMLElement, selector: string): T {
  const element = root.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing diagnostic element ${selector}`);
  return element;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}
