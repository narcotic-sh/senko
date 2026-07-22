import assert from "node:assert/strict";
import test from "node:test";

import {
  assessOfflineReferenceScore,
  buildBenchmarkUrl,
  classifyServedSenkoHtml,
  parseBenchmarkArguments,
  summarizePipelineResult,
  timingAcceptanceCapabilityFailures,
  validateCanonicalAcceptanceInput,
  validateCanonicalAcceptanceResult,
} from "./run-browser-pipeline.mjs";

test("timing mode strips page-memory instrumentation from the URL", () => {
  assert.equal(
    buildBenchmarkUrl("http://127.0.0.1:5173/?memory=1&fixture=a", "timing"),
    "http://127.0.0.1:5173/?fixture=a",
  );
});

test("FP32 benchmark mode forces the production query switch", () => {
  assert.equal(
    buildBenchmarkUrl(
      "http://127.0.0.1:4173/?memory=1&fixture=a",
      "timing",
      "float32",
    ),
    "http://127.0.0.1:4173/?fixture=a&precision=float32",
  );
  assert.equal(
    parseBenchmarkArguments(["--precision", "float32"]).options.precision,
    "float32",
  );
  assert.throws(
    () => parseBenchmarkArguments(["--precision", "float16"]),
    /auto or float32/,
  );
});

test("page-memory mode enables only the diagnostic query switch", () => {
  assert.equal(
    buildBenchmarkUrl("http://127.0.0.1:5173/?fixture=a", "page-memory"),
    "http://127.0.0.1:5173/?fixture=a&memory=1",
  );
});

test("correctness mode leaves diagnostic query parameters intact", () => {
  assert.equal(
    buildBenchmarkUrl(
      "http://127.0.0.1:5173/?raw-campplus-file-parity=1",
      "correctness",
    ),
    "http://127.0.0.1:5173/?raw-campplus-file-parity=1",
  );
  assert.equal(
    parseBenchmarkArguments(["--mode", "correctness"]).options.mode,
    "correctness",
  );
});

test("retained-memory mode disables the page sampler it measures around", () => {
  assert.equal(
    buildBenchmarkUrl(
      "http://127.0.0.1:4173/?memory=1&fixture=a",
      "retained-memory",
    ),
    "http://127.0.0.1:4173/?fixture=a",
  );
});

test("served HTML distinguishes a production bundle from Vite development", () => {
  assert.equal(
    classifyServedSenkoHtml(
      '<script type="module" src="/@vite/client"></script><script type="module" src="/src/main.ts"></script>',
    ),
    "vite-development",
  );
  assert.equal(
    classifyServedSenkoHtml(
      '<script type="module" crossorigin src="/assets/index-C0FFEE.js"></script>',
    ),
    "vite-production-build",
  );
  assert.equal(classifyServedSenkoHtml("<main>not Senko</main>"), "unknown");
});

test("argument parser keeps timing and profile dispositions explicit", () => {
  const parsed = parseBenchmarkArguments([
    "--mode",
    "page-memory",
    "--keep-profile",
    "--run-timeout-ms",
    "1234",
  ]);
  assert.equal(parsed.help, false);
  assert.equal(parsed.options.mode, "page-memory");
  assert.equal(parsed.options.keepProfile, true);
  assert.equal(parsed.options.runTimeoutMs, 1234);
  assert.throws(
    () =>
      parseBenchmarkArguments(["--keep-profile", "--remove-profile"]),
    /mutually exclusive/,
  );
  assert.throws(
    () => parseBenchmarkArguments(["--mode", "combined"]),
    /timing, correctness, page-memory, or retained-memory/,
  );
  assert.equal(parseBenchmarkArguments([]).options.url, "http://127.0.0.1:4173/");
});

test("timing acceptance requires shader-f16 only for the FP16 path", () => {
  const runtime = {
    secureContext: true,
    crossOriginIsolated: true,
    modelPrecision: "float16",
    capabilities: {
      WebGPU: true,
      Worker: true,
      "WASM SIMD": true,
      "WASM threads": true,
      "shader-f16": true,
      "Cross-origin isolated": true,
    },
  };
  assert.deepEqual(timingAcceptanceCapabilityFailures(runtime), []);
  assert.deepEqual(
    timingAcceptanceCapabilityFailures({
      ...runtime,
      capabilities: { ...runtime.capabilities, "shader-f16": false },
    }),
    ["shader-f16"],
  );
  assert.deepEqual(
    timingAcceptanceCapabilityFailures({
      ...runtime,
      modelPrecision: "float32",
      capabilities: { ...runtime.capabilities, "shader-f16": false },
    }),
    [],
  );
});

test("timing acceptance pins the canonical one-hour WAV identity", () => {
  assert.doesNotThrow(() =>
    validateCanonicalAcceptanceInput({
      byteLength: 118_273_444,
      sha256: "3144fb24aac5729e923ba0bbcb42672e0eff956a370a0bde3a85ea1874dfd3f5",
    }),
  );
  assert.throws(
    () =>
      validateCanonicalAcceptanceInput({
        byteLength: 118_273_444,
        sha256: "0".repeat(64),
      }),
    /canonical test_audio\.wav/,
  );
});

test("offline correctness thresholds can reject a timing candidate", () => {
  const assessment = assessOfflineReferenceScore({
    // Informational only: the pinned native UMAP run is not speaker-count
    // ground truth, so this must not mask the actual frame-quality failure.
    speakerCountDelta: 2,
    segmentCountDelta: 0,
    timelines: {
      "10ms": {
        speechIntersectionOverUnion: 0.999,
        mappedSpeakerAgreementOnJointSpeech: 0.99,
      },
      "50ms": {
        speechIntersectionOverUnion: 0.99,
        mappedSpeakerAgreementOnJointSpeech: 0.99,
      },
    },
  });
  assert.equal(assessment.passed, false);
  assert.deepEqual(assessment.failures, ["50ms speech IoU"]);
});

test("summary reports wall time separately from overlapping stage attribution", () => {
  const pipelineResult = {
    totalElapsedMs: 21,
    durationSeconds: 3_696.042_687_5,
    speakerCount: 2,
    segments: [
      { startSeconds: 1, endSeconds: 2, speaker: "A" },
      { startSeconds: 3, endSeconds: 4, speaker: "B" },
    ],
    stages: [
      { stage: "decode", elapsedMs: 1, metrics: {} },
      { stage: "vad", elapsedMs: 2, metrics: {} },
      { stage: "fbank", elapsedMs: 3, metrics: {} },
      { stage: "embedding", elapsedMs: 4, metrics: {} },
      { stage: "clustering", elapsedMs: 5, metrics: {} },
      { stage: "postprocess", elapsedMs: 6, metrics: {} },
    ],
    memory: {
      knownCpuPeakBytes: 10,
      knownGpuBufferBytes: 20,
      wasmHeapBytes: 30,
      allocations: { audioBlobBytes: 40 },
    },
  };
  const result = summarizePipelineResult(
    pipelineResult,
    {
      mode: "timing",
      acceptanceValidated: true,
      servedArtifact: "vite-production-build",
      url: "http://127.0.0.1:5173/",
      chrome: { product: "Chrome/150" },
      input: { path: "/audio.wav", byteLength: 40 },
      exactResultCapture: { byteLength: 1, sha256: "a".repeat(64) },
      isolatedProfile: { retained: false },
    },
  );

  assert.equal(result.mode, "timing-acceptance");
  assert.equal(result.timingAcceptanceEligible, true);
  assert.equal(result.wallMs, 21);
  assert.equal(result.stageAttributedTotalMs, 21);
  assert.deepEqual(result.stagesMs, {
    decode: 1,
    vad: 2,
    fbank: 3,
    embedding: 4,
    clustering: 5,
    postprocess: 6,
  });
  assert.equal(result.logicalMemory.knownGpuBufferBytes, 20);
  const forcedFp32 = summarizePipelineResult(pipelineResult, {
    mode: "timing",
    acceptanceValidated: true,
    requestedPrecision: "float32",
    runtime: { capabilities: { "shader-f16": true } },
  });
  assert.equal(forcedFp32.timingAcceptanceEligible, false);
  assert.equal(forcedFp32.mode, "timing-fp32-compatibility-diagnostic");
  const nativeFp32 = summarizePipelineResult(pipelineResult, {
    mode: "timing",
    acceptanceValidated: true,
    requestedPrecision: "auto",
    runtime: { capabilities: { "shader-f16": false } },
  });
  assert.equal(nativeFp32.timingAcceptanceEligible, true);
  assert.equal(nativeFp32.mode, "timing-acceptance");
  assert.doesNotThrow(() => validateCanonicalAcceptanceResult(pipelineResult));
  assert.equal(
    summarizePipelineResult(pipelineResult, { mode: "timing" })
      .timingAcceptanceEligible,
    false,
  );
  const correctnessRejected = summarizePipelineResult(pipelineResult, {
    mode: "timing",
    acceptanceValidated: true,
    offlineReference: { acceptance: { passed: false } },
  });
  assert.equal(correctnessRejected.timingAcceptanceEligible, false);
  assert.equal(correctnessRejected.mode, "timing-correctness-rejected");
  assert.throws(
    () =>
      summarizePipelineResult(
        { ...pipelineResult, stages: pipelineResult.stages.slice(1) },
        { mode: "timing" },
      ),
    /exactly 6 stages/,
  );
  assert.throws(
    () =>
      summarizePipelineResult(
        {
          ...pipelineResult,
          stages: pipelineResult.stages.map((stage, index) =>
            index === 0 ? { ...stage, elapsedMs: -1 } : stage,
          ),
        },
        { mode: "timing" },
      ),
    /malformed stage/,
  );

  const paddedFinalVadResult = {
    ...pipelineResult,
    durationSeconds: 500.877625,
    speakerCount: 1,
    segments: [
      {
        startSeconds: 495.25,
        endSeconds: 500.894375,
        speaker: "A",
      },
    ],
  };
  assert.doesNotThrow(() =>
    summarizePipelineResult(paddedFinalVadResult, { mode: "page-memory" }),
  );
  assert.throws(
    () =>
      summarizePipelineResult(
        {
          ...paddedFinalVadResult,
          segments: [
            {
              startSeconds: 495.25,
              endSeconds: 510.000_002,
              speaker: "A",
            },
          ],
        },
        { mode: "page-memory" },
      ),
    /malformed segment/,
  );
});
