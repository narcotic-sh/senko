#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  access,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { scoreAgainstOfflineSenkoReference } from "./offline-reference-score.mjs";

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const REPOSITORY_ROOT = resolve(SCRIPT_DIRECTORY, "../../..");
const ACCEPTANCE_AUDIO = Object.freeze({
  byteLength: 118_273_444,
  sha256: "3144fb24aac5729e923ba0bbcb42672e0eff956a370a0bde3a85ea1874dfd3f5",
  durationSeconds: 3_696.042_687_5,
});
const REQUIRED_STAGES = Object.freeze([
  "decode",
  "vad",
  "fbank",
  "embedding",
  "clustering",
  "postprocess",
]);
const OFFLINE_ACCEPTANCE_THRESHOLDS = Object.freeze({
  minimumSpeechIntersectionOverUnion: 0.995,
  minimumMappedSpeakerAgreementOnJointSpeech: 0.98,
  maximumAbsoluteSegmentCountDelta: 10,
});

const DEFAULTS = Object.freeze({
  url: "http://127.0.0.1:4173/",
  audio: join(REPOSITORY_ROOT, "test_audio.wav"),
  profileRoot: join(REPOSITORY_ROOT, ".research/chrome-benchmark-runs"),
  mode: "timing",
  keepProfile: false,
  readyTimeoutMs: 120_000,
  runTimeoutMs: 10 * 60_000,
  pageMemoryTimeoutMs: 60_000,
  rawResultPath: undefined,
  offlineReferencePath: undefined,
  chrome: undefined,
});

const HELP = `Usage:
  node web/scripts/benchmark/run-browser-pipeline.mjs [options]

Options:
  --url <url>                    Senko page (default: ${DEFAULTS.url})
  --audio <absolute-or-relative> WAV input (default: test_audio.wav)
  --chrome <path>                Chrome executable (or set CHROME_PATH)
  --mode <mode>                  timing, correctness, page-memory, or retained-memory
  --profile-root <path>          Parent for the unique temporary profile
  --keep-profile                 Retain the unique profile after teardown
  --remove-profile               Remove it after teardown (the default)
  --ready-timeout-ms <ms>        Model initialization timeout
  --run-timeout-ms <ms>          Pipeline timeout
  --page-memory-timeout-ms <ms>  Final coarse page-memory sample timeout
  --raw-result <path>            Save exact result JSON (two indexed files in retained-memory)
  --offline-reference <path>     Score against an offline Senko reference JSON
  --help                         Show this help

Timing and memory diagnostics are intentionally separate. Timing mode requires
a production Vite build and removes memory=1. Both memory modes are explicitly
ineligible for timing acceptance.
`;

export function parseBenchmarkArguments(argv) {
  const options = { ...DEFAULTS };
  let profileDisposition;

  const takeValue = (index, flag) => {
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`${flag} requires a value`);
    }
    return value;
  };

  const takePositiveInteger = (index, flag) => {
    const source = takeValue(index, flag);
    const value = Number(source);
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new Error(`${flag} must be a positive integer, received ${source}`);
    }
    return value;
  };

  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    switch (flag) {
      case "--help":
      case "-h":
        return { help: true, options };
      case "--url":
        options.url = takeValue(index, flag);
        index += 1;
        break;
      case "--audio":
        options.audio = takeValue(index, flag);
        index += 1;
        break;
      case "--chrome":
        options.chrome = takeValue(index, flag);
        index += 1;
        break;
      case "--mode": {
        const mode = takeValue(index, flag);
        if (
          mode !== "timing" &&
          mode !== "correctness" &&
          mode !== "page-memory" &&
          mode !== "retained-memory"
        ) {
          throw new Error(
            `--mode must be timing, correctness, page-memory, or retained-memory, received ${mode}`,
          );
        }
        options.mode = mode;
        index += 1;
        break;
      }
      case "--profile-root":
        options.profileRoot = takeValue(index, flag);
        index += 1;
        break;
      case "--keep-profile":
        if (profileDisposition === "remove") {
          throw new Error("--keep-profile and --remove-profile are mutually exclusive");
        }
        profileDisposition = "keep";
        options.keepProfile = true;
        break;
      case "--remove-profile":
        if (profileDisposition === "keep") {
          throw new Error("--keep-profile and --remove-profile are mutually exclusive");
        }
        profileDisposition = "remove";
        options.keepProfile = false;
        break;
      case "--ready-timeout-ms":
        options.readyTimeoutMs = takePositiveInteger(index, flag);
        index += 1;
        break;
      case "--run-timeout-ms":
        options.runTimeoutMs = takePositiveInteger(index, flag);
        index += 1;
        break;
      case "--page-memory-timeout-ms":
        options.pageMemoryTimeoutMs = takePositiveInteger(index, flag);
        index += 1;
        break;
      case "--raw-result":
        options.rawResultPath = takeValue(index, flag);
        index += 1;
        break;
      case "--offline-reference":
        options.offlineReferencePath = takeValue(index, flag);
        index += 1;
        break;
      default:
        throw new Error(`Unknown argument: ${flag}`);
    }
  }

  return { help: false, options };
}

export function buildBenchmarkUrl(source, mode) {
  const url = new URL(source);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`Benchmark URL must use HTTP(S), received ${url.protocol}`);
  }
  if (mode === "page-memory") {
    url.searchParams.set("memory", "1");
  } else {
    url.searchParams.delete("memory");
  }
  return url.href;
}

export function classifyServedSenkoHtml(html) {
  if (
    html.includes("/@vite/client") ||
    /<script\b[^>]*\bsrc=["']\/src\//i.test(html)
  ) {
    return "vite-development";
  }
  if (/<script\b[^>]*\bsrc=["']\/assets\/[^"']+\.js["']/i.test(html)) {
    return "vite-production-build";
  }
  return "unknown";
}

export function summarizePipelineRun(result, exactResultCapture) {
  if (
    result === null ||
    typeof result !== "object" ||
    !isNonnegativeFinite(result.totalElapsedMs) ||
    !isNonnegativeFinite(result.durationSeconds) ||
    !Number.isSafeInteger(result.speakerCount) ||
    result.speakerCount < 0 ||
    !Array.isArray(result.segments) ||
    !Array.isArray(result.stages) ||
    result.memory === null ||
    typeof result.memory !== "object" ||
    !isNonnegativeSafeInteger(result.memory.knownCpuPeakBytes) ||
    result.memory.allocations === null ||
    typeof result.memory.allocations !== "object"
  ) {
    throw new Error("The page returned a malformed pipeline result");
  }

  const stagesMs = {};
  if (result.stages.length !== REQUIRED_STAGES.length) {
    throw new Error(
      `Pipeline result must contain exactly ${REQUIRED_STAGES.length} stages`,
    );
  }
  for (const stage of result.stages) {
    if (
      stage === null ||
      typeof stage !== "object" ||
      !REQUIRED_STAGES.includes(stage.stage) ||
      !isNonnegativeFinite(stage.elapsedMs) ||
      stage.metrics === null ||
      typeof stage.metrics !== "object" ||
      Object.hasOwn(stagesMs, stage.stage)
    ) {
      throw new Error("The page returned a malformed stage result");
    }
    stagesMs[stage.stage] = stage.elapsedMs;
  }
  if (!REQUIRED_STAGES.every((stage) => Object.hasOwn(stagesMs, stage))) {
    throw new Error("Pipeline result is missing a required stage");
  }

  const segmentSpeakers = new Set();
  let priorStart = Number.NEGATIVE_INFINITY;
  // Native Senko decodes every padded 10-second pyannote chunk as-is. Its
  // final timestamp can therefore extend beyond the physical WAV duration;
  // bound validation by the decoded chunk timeline instead of clipping a
  // result that is intentionally native-compatible.
  const decodedVadTimelineEnd =
    Math.ceil(result.durationSeconds / 10) * 10;
  for (const [index, segment] of result.segments.entries()) {
    if (
      segment === null ||
      typeof segment !== "object" ||
      !isNonnegativeFinite(segment.startSeconds) ||
      !isNonnegativeFinite(segment.endSeconds) ||
      segment.endSeconds < segment.startSeconds ||
      segment.endSeconds > decodedVadTimelineEnd + 1e-6 ||
      segment.startSeconds < priorStart ||
      typeof segment.speaker !== "string" ||
      segment.speaker.length === 0
    ) {
      throw new Error(`The page returned a malformed segment at index ${index}`);
    }
    priorStart = segment.startSeconds;
    segmentSpeakers.add(segment.speaker);
  }
  if (segmentSpeakers.size !== result.speakerCount) {
    throw new Error(
      `speakerCount ${result.speakerCount} disagrees with ${segmentSpeakers.size} segment labels`,
    );
  }

  for (const key of [
    "knownGpuBufferBytes",
    "wasmHeapBytes",
    "jsHeapPeakBytes",
  ]) {
    if (
      result.memory[key] !== undefined &&
      !isNonnegativeSafeInteger(result.memory[key])
    ) {
      throw new Error(`Pipeline memory.${key} is malformed`);
    }
  }

  const memory = result.memory;
  return {
    wallMs: result.totalElapsedMs,
    stagesMs,
    stageAttributedTotalMs: Object.values(stagesMs).reduce(
      (sum, elapsedMs) => sum + elapsedMs,
      0,
    ),
    audioDurationSeconds: result.durationSeconds,
    speakerCount: result.speakerCount,
    segmentCount: result.segments.length,
    logicalMemory: {
      knownCpuPeakBytes: memory.knownCpuPeakBytes,
      knownGpuBufferBytes: memory.knownGpuBufferBytes,
      wasmHeapBytes: memory.wasmHeapBytes,
      jsHeapPeakBytes: memory.jsHeapPeakBytes,
      allocations: memory.allocations,
    },
    exactResultCapture,
  };
}

export function validateCanonicalAcceptanceResult(result) {
  // Reuse the complete structural/semantic validation before applying the
  // canonical one-hour workload invariants.
  summarizePipelineRun(result, undefined);
  if (
    Math.abs(result.durationSeconds - ACCEPTANCE_AUDIO.durationSeconds) > 1e-6
  ) {
    throw new Error(
      `Acceptance audio duration mismatch: ${result.durationSeconds} seconds`,
    );
  }
  if (result.speakerCount < 1 || result.speakerCount > 15) {
    throw new Error(`Acceptance speaker count is implausible: ${result.speakerCount}`);
  }
  if (result.segments.length === 0) {
    throw new Error("Acceptance result contains no diarization segments");
  }
}

function isNonnegativeFinite(value) {
  return Number.isFinite(value) && value >= 0;
}

function isNonnegativeSafeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

export function summarizePipelineResult(result, metadata) {
  const timingAcceptanceEligible =
    metadata.mode === "timing" &&
    metadata.acceptanceValidated === true &&
    metadata.offlineReference?.acceptance?.passed !== false;
  return {
    schemaVersion: 1,
    mode:
      metadata.mode === "timing"
        ? timingAcceptanceEligible
          ? "timing-acceptance"
          : "timing-correctness-rejected"
        : metadata.mode === "correctness"
          ? "correctness-diagnostic"
          : "page-memory-diagnostic",
    timingAcceptanceEligible,
    servedArtifact: metadata.servedArtifact,
    url: metadata.url,
    chrome: metadata.chrome,
    runtime: metadata.runtime,
    input: metadata.input,
    ...summarizePipelineRun(result, metadata.exactResultCapture),
    ...(metadata.pageMemory === undefined
      ? {}
      : { pageMemory: metadata.pageMemory }),
    ...(metadata.offlineReference === undefined
      ? {}
      : { offlineReference: metadata.offlineReference }),
    isolatedProfile: metadata.isolatedProfile,
  };
}

class CdpConnection {
  #socket;
  #nextId = 1;
  #pending = new Map();
  #closed = false;

  static async connect(url, timeoutMs = 10_000) {
    if (typeof WebSocket !== "function") {
      throw new Error("This runner requires Node.js with a global WebSocket");
    }
    const socket = new WebSocket(url);
    await new Promise((resolvePromise, rejectPromise) => {
      const timer = setTimeout(() => {
        socket.close();
        rejectPromise(new Error(`Timed out connecting to Chrome CDP at ${url}`));
      }, timeoutMs);
      const finish = (callback, value) => {
        clearTimeout(timer);
        socket.removeEventListener("open", handleOpen);
        socket.removeEventListener("error", handleError);
        callback(value);
      };
      const handleOpen = () => finish(resolvePromise);
      const handleError = (event) =>
        finish(
          rejectPromise,
          new Error(event?.message ?? `Could not connect to Chrome CDP at ${url}`),
        );
      socket.addEventListener("open", handleOpen, { once: true });
      socket.addEventListener("error", handleError, { once: true });
    });
    return new CdpConnection(socket);
  }

  constructor(socket) {
    this.#socket = socket;
    socket.addEventListener("message", (event) => void this.#handleMessage(event));
    socket.addEventListener("close", () => {
      this.#closed = true;
      this.#rejectPending(new Error("Chrome closed the CDP connection"));
    });
    socket.addEventListener("error", (event) => {
      this.#rejectPending(new Error(event?.message ?? "Chrome CDP WebSocket failed"));
    });
  }

  async #handleMessage(event) {
    let source = event.data;
    if (source instanceof ArrayBuffer) {
      source = new TextDecoder().decode(source);
    } else if (ArrayBuffer.isView(source)) {
      source = new TextDecoder().decode(source);
    } else if (typeof Blob === "function" && source instanceof Blob) {
      source = await source.text();
    }
    const message = JSON.parse(String(source));
    if (message.id === undefined) return;
    const pending = this.#pending.get(message.id);
    if (pending === undefined) return;
    this.#pending.delete(message.id);
    clearTimeout(pending.timer);
    if (message.error !== undefined) {
      pending.reject(
        new Error(
          `${pending.method}: ${message.error.message ?? JSON.stringify(message.error)}`,
        ),
      );
    } else {
      pending.resolve(message.result ?? {});
    }
  }

  send(method, params = {}, { sessionId, timeoutMs = 30_000 } = {}) {
    if (this.#closed) {
      return Promise.reject(new Error(`Cannot send ${method}; CDP is closed`));
    }
    const id = this.#nextId;
    this.#nextId += 1;
    return new Promise((resolvePromise, rejectPromise) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id);
        rejectPromise(new Error(`${method} timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      this.#pending.set(id, {
        method,
        resolve: resolvePromise,
        reject: rejectPromise,
        timer,
      });
      this.#socket.send(
        JSON.stringify({
          id,
          method,
          params,
          ...(sessionId === undefined ? {} : { sessionId }),
        }),
      );
    });
  }

  close(reason = new Error("CDP connection closed by benchmark runner")) {
    if (this.#closed) return;
    this.#closed = true;
    this.#rejectPending(reason);
    this.#socket.close();
  }

  #rejectPending(error) {
    for (const pending of this.#pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.#pending.clear();
  }
}

async function evaluate(cdp, sessionId, expression, timeoutMs = 30_000) {
  const response = await cdp.send(
    "Runtime.evaluate",
    {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true,
    },
    { sessionId, timeoutMs },
  );
  if (response.exceptionDetails !== undefined) {
    const details = response.exceptionDetails;
    throw new Error(
      details.exception?.description ??
        details.text ??
        "JavaScript evaluation failed in the Senko page",
    );
  }
  return response.result?.value;
}

async function waitForCondition(check, {
  timeoutMs,
  intervalMs = 100,
  description,
  signal,
}) {
  const deadline = Date.now() + timeoutMs;
  let lastValue;
  while (Date.now() < deadline) {
    signal?.throwIfAborted();
    lastValue = await check();
    if (lastValue?.done) return lastValue.value;
    await abortableDelay(intervalMs, signal);
  }
  throw new Error(
    `Timed out after ${timeoutMs} ms waiting for ${description}` +
      (lastValue?.detail === undefined ? "" : ` (${lastValue.detail})`),
  );
}

function abortableDelay(milliseconds, signal) {
  return new Promise((resolvePromise, rejectPromise) => {
    if (signal?.aborted) {
      rejectPromise(signal.reason);
      return;
    }
    const timer = setTimeout(finish, milliseconds);
    const handleAbort = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", handleAbort);
      rejectPromise(signal.reason);
    };
    function finish() {
      signal?.removeEventListener("abort", handleAbort);
      resolvePromise();
    }
    signal?.addEventListener("abort", handleAbort, { once: true });
  });
}

async function resolveChromeExecutable(explicitPath) {
  const configuredPath = explicitPath ?? process.env.CHROME_PATH;
  if (configuredPath !== undefined) {
    const absolute = resolve(configuredPath);
    try {
      await access(absolute, fsConstants.X_OK);
      return absolute;
    } catch {
      throw new Error(`Configured Chrome executable is not runnable: ${absolute}`);
    }
  }

  const candidates = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
  ];
  for (const candidate of candidates) {
    const absolute = resolve(candidate);
    try {
      await access(absolute, fsConstants.X_OK);
      return absolute;
    } catch {
      // Try the next well-known executable.
    }
  }
  throw new Error(
    "Chrome was not found; pass --chrome <path> or set CHROME_PATH",
  );
}

async function launchIsolatedChrome(chrome, profileDirectory) {
  const args = [
    `--user-data-dir=${profileDirectory}`,
    "--remote-debugging-address=127.0.0.1",
    "--remote-debugging-port=0",
    "--remote-allow-origins=*",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-component-extensions-with-background-pages",
    "--disable-default-apps",
    "--disable-background-networking",
    "--disable-background-mode",
    "--disable-sync",
    "--disable-client-side-phishing-detection",
    "--disable-breakpad",
    "--disable-crash-reporter",
    "--disable-renderer-backgrounding",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--metrics-recording-only",
    "--password-store=basic",
    "--use-mock-keychain",
    "--window-size=1280,900",
    "about:blank",
  ];
  const child = spawn(chrome, args, {
    detached: process.platform !== "win32",
    stdio: "ignore",
  });
  await new Promise((resolvePromise, rejectPromise) => {
    const handleSpawn = () => {
      child.removeListener("error", handleError);
      resolvePromise();
    };
    const handleError = (error) => {
      child.removeListener("spawn", handleSpawn);
      rejectPromise(error);
    };
    child.once("spawn", handleSpawn);
    child.once("error", handleError);
  });
  return child;
}

async function waitForDevTools(profileDirectory, child, timeoutMs, signal) {
  const activePortPath = join(profileDirectory, "DevToolsActivePort");
  return waitForCondition(
    async () => {
      if (child.exitCode !== null || child.signalCode !== null) {
        throw new Error(
          `Isolated Chrome exited before CDP became ready (${child.exitCode ?? child.signalCode})`,
        );
      }
      try {
        const [portLine, browserPath] = (await readFile(activePortPath, "utf8"))
          .trim()
          .split(/\r?\n/);
        const port = Number(portLine);
        if (!Number.isSafeInteger(port) || port <= 0 || !browserPath) {
          return { done: false, detail: "DevToolsActivePort is incomplete" };
        }
        const response = await fetch(`http://127.0.0.1:${port}/json/version`, {
          signal,
        });
        if (!response.ok) {
          return { done: false, detail: `CDP HTTP ${response.status}` };
        }
        const version = await response.json();
        return {
          done: true,
          value: {
            port,
            webSocketUrl:
              version.webSocketDebuggerUrl ??
              `ws://127.0.0.1:${port}${browserPath}`,
          },
        };
      } catch (error) {
        if (signal?.aborted) throw error;
        return { done: false, detail: error instanceof Error ? error.message : String(error) };
      }
    },
    {
      timeoutMs,
      intervalMs: 100,
      description: "isolated Chrome DevTools",
      signal,
    },
  );
}

async function createSinglePageSession(cdp) {
  const { targetInfos } = await cdp.send("Target.getTargets");
  const pageTargets = targetInfos.filter((target) => target.type === "page");
  let targetId = pageTargets[0]?.targetId;
  if (targetId === undefined) {
    ({ targetId } = await cdp.send("Target.createTarget", { url: "about:blank" }));
  }
  for (const target of pageTargets) {
    if (target.targetId !== targetId) {
      await cdp.send("Target.closeTarget", { targetId: target.targetId });
    }
  }
  await cdp.send("Target.activateTarget", { targetId });
  const { sessionId } = await cdp.send("Target.attachToTarget", {
    targetId,
    flatten: true,
  });
  await Promise.all([
    cdp.send("Page.enable", {}, { sessionId }),
    cdp.send("Runtime.enable", {}, { sessionId }),
    cdp.send("DOM.enable", {}, { sessionId }),
  ]);
  return { targetId, sessionId };
}

async function waitForModelsReady(cdp, sessionId, timeoutMs, signal) {
  return waitForCondition(
    async () => {
      let state;
      try {
        state = await evaluate(
          cdp,
          sessionId,
          `(() => {
            const status = document.querySelector("#status");
            return {
              text: status?.textContent?.trim() ?? "",
              kind: status?.getAttribute("data-kind") ?? "",
            };
          })()`,
        );
      } catch (error) {
        // Page.navigate can return just before Chrome swaps execution contexts.
        // Only that narrow navigation race is retryable; real page errors fail.
        const message = error instanceof Error ? error.message : String(error);
        if (
          /execution context|cannot find context|target navigated/i.test(message)
        ) {
          return { done: false, detail: "navigation context is settling" };
        }
        throw error;
      }
      if (state?.kind === "error") {
        throw new Error(`Senko initialization failed: ${state.text}`);
      }
      return {
        done: state?.text === "WebGPU models ready.",
        value: state,
        detail: state?.text,
      };
    },
    {
      timeoutMs,
      intervalMs: 100,
      description: "WebGPU models ready",
      signal,
    },
  );
}

async function readRuntimeFingerprint(cdp, sessionId) {
  return evaluate(
    cdp,
    sessionId,
    `(async () => {
      const capabilities = Object.fromEntries(
        [...document.querySelectorAll("#capabilities .capability")].map((node) => [
          node.querySelector("span")?.textContent?.trim() ?? "",
          node.querySelector("strong")?.textContent?.trim() === "Yes",
        ]),
      );
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
      return {
        secureContext: globalThis.isSecureContext,
        crossOriginIsolated: globalThis.crossOriginIsolated,
        hardwareConcurrency: navigator.hardwareConcurrency,
        capabilities,
        webgpu: adapter === null || adapter === undefined
          ? { available: false, features: [] }
          : {
              available: true,
              adapterInfo: {
                architecture: adapter.info.architecture,
                description: adapter.info.description,
                device: adapter.info.device,
                vendor: adapter.info.vendor,
              },
              features: [...adapter.features].sort(),
              maxBufferSize: adapter.limits.maxBufferSize,
              maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            },
      };
    })()`,
  );
}

export function timingAcceptanceCapabilityFailures(runtime) {
  const failures = [];
  if (runtime?.secureContext !== true) failures.push("secure context");
  if (runtime?.crossOriginIsolated !== true) {
    failures.push("cross-origin isolation");
  }
  for (const label of [
    "WebGPU",
    "Worker",
    "WASM SIMD",
    "WASM threads",
    "shader-f16",
  ]) {
    if (runtime?.capabilities?.[label] !== true) failures.push(label);
  }
  return [...new Set(failures)];
}

async function attachAudio(cdp, sessionId, audioPath, audioBytes, signal) {
  const { root } = await cdp.send(
    "DOM.getDocument",
    { depth: -1, pierce: true },
    { sessionId },
  );
  const { nodeId } = await cdp.send(
    "DOM.querySelector",
    { nodeId: root.nodeId, selector: "input#audio-file[type=file]" },
    { sessionId },
  );
  if (!nodeId) throw new Error("Could not find the Senko WAV file input");
  await cdp.send(
    "DOM.setFileInputFiles",
    { files: [audioPath], nodeId },
    { sessionId },
  );

  await waitForCondition(
    async () => {
      const state = await evaluate(
        cdp,
        sessionId,
        `(() => {
          const input = document.querySelector("#audio-file");
          const button = document.querySelector("#run-pipeline");
          return {
            byteLength: input?.files?.[0]?.size,
            buttonEnabled: button instanceof HTMLButtonElement && !button.disabled,
          };
        })()`,
      );
      return {
        done: state?.byteLength === audioBytes && state?.buttonEnabled === true,
        value: state,
        detail: JSON.stringify(state),
      };
    },
    {
      timeoutMs: 10_000,
      intervalMs: 50,
      description: "the WAV input to become runnable",
      signal,
    },
  );
}

async function runAndCapture(cdp, sessionId, timeoutMs) {
  await evaluate(
    cdp,
    sessionId,
    `(() => {
      const status = document.querySelector("#status");
      if (status === null) throw new Error("Missing #status");
      globalThis.__senkoBenchmarkCompletion = new Promise((resolve) => {
        let observer;
        const inspect = () => {
          const text = status.textContent?.trim() ?? "";
          const kind = status.getAttribute("data-kind") ?? "";
          if (text === "Pipeline complete.") {
            observer?.disconnect();
            resolve({ ok: true, resultJson: document.querySelector("#result")?.textContent ?? "" });
          } else if (kind === "error") {
            observer?.disconnect();
            resolve({ ok: false, error: text || "Pipeline failed" });
          }
        };
        observer = new MutationObserver(inspect);
        observer.observe(status, { attributes: true, childList: true, subtree: true });
      });
      return true;
    })()`,
  );
  const click = await evaluate(
    cdp,
    sessionId,
    `(() => {
      const button = document.querySelector("#run-pipeline");
      if (!(button instanceof HTMLButtonElement)) return { ok: false, error: "Missing run button" };
      if (button.disabled) return { ok: false, error: "Run button is disabled" };
      button.click();
      return { ok: true };
    })()`,
  );
  if (!click?.ok) throw new Error(click?.error ?? "Could not start the pipeline");

  const capture = await evaluate(
    cdp,
    sessionId,
    "globalThis.__senkoBenchmarkCompletion",
    timeoutMs,
  );
  if (!capture?.ok) throw new Error(capture?.error ?? "Pipeline failed");
  if (typeof capture.resultJson !== "string" || capture.resultJson.length === 0) {
    throw new Error("Pipeline completed without result JSON");
  }
  return capture.resultJson;
}

async function waitForFinalPageMemory(cdp, sessionId, timeoutMs, signal) {
  return waitForCondition(
    async () => {
      const pageMemory = await evaluate(
        cdp,
        sessionId,
        `(() => {
          try {
            return JSON.parse(document.querySelector("#result")?.textContent ?? "null")?.pageMemory ?? null;
          } catch {
            return null;
          }
        })()`,
      );
      if (pageMemory !== null && pageMemory.pending === false) {
        if (pageMemory.supported !== true) {
          throw new Error(
            "Chrome does not support page-scoped measureUserAgentSpecificMemory",
          );
        }
        if (pageMemory.error !== undefined) {
          throw new Error(`Page-memory measurement failed: ${pageMemory.error}`);
        }
        if (
          !Array.isArray(pageMemory.samples) ||
          pageMemory.samples.length === 0 ||
          !Number.isSafeInteger(pageMemory.currentBytes) ||
          pageMemory.currentBytes < 0 ||
          !Number.isSafeInteger(pageMemory.peakBytes) ||
          pageMemory.peakBytes < 0
        ) {
          throw new Error(
            "Page-memory mode completed without a valid Chrome measurement",
          );
        }
      }
      return {
        done:
          pageMemory !== null &&
          pageMemory.pending === false &&
          pageMemory.supported === true &&
          Array.isArray(pageMemory.samples) &&
          pageMemory.samples.length > 0,
        value: pageMemory,
        detail:
          pageMemory === null
            ? "page-memory summary not rendered"
            : `pending=${String(pageMemory.pending)}`,
      };
    },
    {
      timeoutMs,
      intervalMs: 100,
      description: "Chrome's final page-scoped memory sample",
      signal,
    },
  );
}

async function measurePageAgentClusterMemory(cdp, sessionId, timeoutMs) {
  const measurement = await evaluate(
    cdp,
    sessionId,
    `(async () => {
      const measure = performance.measureUserAgentSpecificMemory;
      if (typeof measure !== "function") {
        throw new Error("measureUserAgentSpecificMemory is unavailable");
      }
      const result = await measure.call(performance);
      return { bytes: result.bytes };
    })()`,
    timeoutMs,
  );
  if (!Number.isSafeInteger(measurement?.bytes) || measurement.bytes < 0) {
    throw new Error("Chrome returned an invalid page-agent-cluster memory result");
  }
  return measurement;
}

function indexedRawResultPath(rawResultPath, runNumber) {
  const extension = extname(rawResultPath);
  const stem =
    extension.length === 0
      ? rawResultPath
      : rawResultPath.slice(0, -extension.length);
  return `${stem}.run-${runNumber}${extension || ".json"}`;
}

async function captureMetadata(exactResultJson, rawResultPath) {
  if (rawResultPath !== undefined) {
    await mkdir(dirname(rawResultPath), { recursive: true });
    await writeFile(rawResultPath, exactResultJson, "utf8");
  }
  return {
    byteLength: Buffer.byteLength(exactResultJson),
    sha256: createHash("sha256").update(exactResultJson).digest("hex"),
    ...(rawResultPath === undefined ? {} : { path: rawResultPath }),
  };
}

async function loadOfflineReference(sourcePath) {
  if (sourcePath === undefined) return undefined;
  const path = resolve(sourcePath);
  const source = await readFile(path);
  let value;
  try {
    value = JSON.parse(source.toString("utf8"));
  } catch (error) {
    throw new Error(
      `Could not parse offline reference ${path}: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  // Validate the oracle shape before launching Chrome.
  scoreAgainstOfflineSenkoReference(
    { speakerCount: 0, segments: [] },
    value,
  );
  return {
    value,
    metadata: {
      path,
      byteLength: source.byteLength,
      sha256: createHash("sha256").update(source).digest("hex"),
    },
  };
}

export function assessOfflineReferenceScore(score) {
  const failures = [];
  for (const resolution of ["10ms", "50ms"]) {
    const timeline = score?.timelines?.[resolution];
    if (
      !Number.isFinite(timeline?.speechIntersectionOverUnion) ||
      timeline.speechIntersectionOverUnion <
        OFFLINE_ACCEPTANCE_THRESHOLDS.minimumSpeechIntersectionOverUnion
    ) {
      failures.push(`${resolution} speech IoU`);
    }
    if (
      !Number.isFinite(timeline?.mappedSpeakerAgreementOnJointSpeech) ||
      timeline.mappedSpeakerAgreementOnJointSpeech <
        OFFLINE_ACCEPTANCE_THRESHOLDS.minimumMappedSpeakerAgreementOnJointSpeech
    ) {
      failures.push(`${resolution} mapped speaker agreement`);
    }
  }
  // Speaker count remains reported, but is not a gate. Native Senko's
  // unseeded UMAP varies across runs, and the pinned seven-speaker reference
  // is independently known to undercount this recording. Extra/split speakers
  // are already penalized by one-to-one mapped frame agreement.
  if (
    !Number.isSafeInteger(score?.segmentCountDelta) ||
    Math.abs(score.segmentCountDelta) >
      OFFLINE_ACCEPTANCE_THRESHOLDS.maximumAbsoluteSegmentCountDelta
  ) {
    failures.push("segment-count delta");
  }
  return {
    passed: failures.length === 0,
    thresholds: OFFLINE_ACCEPTANCE_THRESHOLDS,
    failures,
  };
}

function scoreOfflineReference(result, loadedReference) {
  if (loadedReference === undefined) return undefined;
  const score = scoreAgainstOfflineSenkoReference(
    result,
    loadedReference.value,
  );
  return {
    ...loadedReference.metadata,
    score,
    acceptance: assessOfflineReferenceScore(score),
  };
}

async function sha256File(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

export function validateCanonicalAcceptanceInput(input) {
  if (
    input?.byteLength !== ACCEPTANCE_AUDIO.byteLength ||
    input?.sha256 !== ACCEPTANCE_AUDIO.sha256
  ) {
    throw new Error(
      "Timing acceptance requires the canonical test_audio.wav " +
        `(${ACCEPTANCE_AUDIO.byteLength} bytes, SHA-256 ${ACCEPTANCE_AUDIO.sha256})`,
    );
  }
}

function waitForChildExit(child, timeoutMs) {
  if (child.exitCode !== null || child.signalCode !== null) {
    return Promise.resolve(true);
  }
  return new Promise((resolvePromise) => {
    const timer = setTimeout(() => finish(false), timeoutMs);
    const handleExit = () => finish(true);
    const finish = (exited) => {
      clearTimeout(timer);
      child.removeListener("exit", handleExit);
      resolvePromise(exited);
    };
    child.once("exit", handleExit);
  });
}

async function terminateIsolatedChrome(child) {
  if (child === undefined) return;
  const signalProcessGroup = (signal) => {
    try {
      if (process.platform === "win32") child.kill(signal);
      else process.kill(-child.pid, signal);
    } catch (error) {
      if (error?.code !== "ESRCH") throw error;
    }
  };
  signalProcessGroup("SIGTERM");
  await waitForChildExit(child, 5_000);

  // A crashed browser leader can leave renderer/GPU descendants in its
  // dedicated process group. Probe the group itself before declaring teardown
  // complete; the negative PID can never address the user's other Chrome.
  let groupStillExists = child.exitCode === null && child.signalCode === null;
  if (process.platform !== "win32") {
    await abortableDelay(100);
    try {
      process.kill(-child.pid, 0);
      groupStillExists = true;
    } catch (error) {
      if (error?.code === "ESRCH") groupStillExists = false;
      else throw error;
    }
  }
  if (groupStillExists) {
    signalProcessGroup("SIGKILL");
    await waitForChildExit(child, 2_000);
  }
}

async function preflightServer(url, mode, signal) {
  let response;
  try {
    response = await fetch(url, { signal });
  } catch (error) {
    throw new Error(
      `Could not reach the Senko page at ${url}; start the Vite server first (${error instanceof Error ? error.message : String(error)})`,
    );
  }
  if (!response.ok) {
    await response.body?.cancel();
    throw new Error(`Senko page returned HTTP ${response.status} at ${url}`);
  }
  const html = await response.text();
  const kind = classifyServedSenkoHtml(html);
  if (mode === "timing" && kind !== "vite-production-build") {
    throw new Error(
      kind === "vite-development"
        ? "Timing acceptance refuses the Vite development server; run pnpm build and pnpm preview on port 4173"
        : "Timing acceptance could not verify a production Vite build from the served HTML",
    );
  }
  return {
    kind,
    htmlSha256: createHash("sha256").update(html).digest("hex"),
    scriptUrls: [
      ...html.matchAll(/<script\b[^>]*\bsrc=["']([^"']+)["'][^>]*>/gi),
    ].map((match) => new URL(match[1], url).href),
  };
}

export async function runBrowserBenchmark(options) {
  const mode = options.mode;
  const benchmarkUrl = buildBenchmarkUrl(options.url, mode);
  const audioPath = resolve(options.audio);
  const profileRoot = resolve(options.profileRoot);
  const rawResultPath =
    options.rawResultPath === undefined ? undefined : resolve(options.rawResultPath);
  const offlineReference = await loadOfflineReference(
    options.offlineReferencePath,
  );
  const chrome = await resolveChromeExecutable(options.chrome);
  const audioStats = await stat(audioPath);
  if (!audioStats.isFile()) throw new Error(`Audio input is not a file: ${audioPath}`);
  if (mode === "timing" && audioStats.size !== ACCEPTANCE_AUDIO.byteLength) {
    validateCanonicalAcceptanceInput({
      byteLength: audioStats.size,
      sha256: "",
    });
  }

  const abortController = new AbortController();
  let cdp;
  let chromeProcess;
  let profileDirectory;
  const handleSignal = (signal) => {
    const error = new Error(`Received ${signal}`);
    abortController.abort(error);
    cdp?.close(error);
  };
  const handleSigint = () => handleSignal("SIGINT");
  const handleSigterm = () => handleSignal("SIGTERM");
  process.once("SIGINT", handleSigint);
  process.once("SIGTERM", handleSigterm);

  try {
    const servedArtifact = await preflightServer(
      benchmarkUrl,
      mode,
      abortController.signal,
    );
    await mkdir(profileRoot, { recursive: true });
    profileDirectory = await mkdtemp(join(profileRoot, "senko-chrome-"));
    chromeProcess = await launchIsolatedChrome(chrome, profileDirectory);
    process.stderr.write(
      `[senko-benchmark] isolated Chrome PID ${chromeProcess.pid}; profile ${profileDirectory}\n`,
    );

    const { webSocketUrl } = await waitForDevTools(
      profileDirectory,
      chromeProcess,
      20_000,
      abortController.signal,
    );
    cdp = await CdpConnection.connect(webSocketUrl);
    const browserVersion = await cdp.send("Browser.getVersion");
    const { sessionId } = await createSinglePageSession(cdp);
    const navigation = await cdp.send(
      "Page.navigate",
      { url: benchmarkUrl },
      { sessionId, timeoutMs: 30_000 },
    );
    if (navigation.errorText !== undefined) {
      throw new Error(`Chrome could not navigate to Senko: ${navigation.errorText}`);
    }
    await waitForModelsReady(
      cdp,
      sessionId,
      options.readyTimeoutMs,
      abortController.signal,
    );
    await attachAudio(
      cdp,
      sessionId,
      audioPath,
      audioStats.size,
      abortController.signal,
    );
    const runtime = await readRuntimeFingerprint(cdp, sessionId);
    if (mode === "timing") {
      const capabilityFailures = timingAcceptanceCapabilityFailures(runtime);
      if (capabilityFailures.length > 0) {
        throw new Error(
          `Timing acceptance requires the full accelerated runtime; missing ${capabilityFailures.join(
            ", ",
          )}`,
        );
      }
    }

    const chromeMetadata = {
      executable: chrome,
      product: browserVersion.product,
      revision: browserVersion.revision,
      userAgent: browserVersion.userAgent,
      jsVersion: browserVersion.jsVersion,
    };
    const inputMetadata = {
      path: audioPath,
      byteLength: audioStats.size,
    };
    const isolatedProfile = {
      retained: options.keepProfile,
      ...(options.keepProfile ? { path: profileDirectory } : {}),
    };

    if (mode === "retained-memory") {
      process.stderr.write(
        "[senko-benchmark] models ready; measuring initial dual-resident state\n",
      );
      const initialDualResident = await measurePageAgentClusterMemory(
        cdp,
        sessionId,
        options.pageMemoryTimeoutMs,
      );
      const runs = [];
      const postRunMeasurements = [];
      for (let runNumber = 1; runNumber <= 2; runNumber += 1) {
        process.stderr.write(
          `[senko-benchmark] starting retained-memory run ${runNumber}/2\n`,
        );
        const exactResultJson = await runAndCapture(
          cdp,
          sessionId,
          options.runTimeoutMs,
        );
        const result = JSON.parse(exactResultJson);
        if (Object.hasOwn(result, "pageMemory")) {
          throw new Error(
            "Retained-memory mode must own its measurements; page sampler was unexpectedly enabled",
          );
        }
        postRunMeasurements.push(
          await measurePageAgentClusterMemory(
            cdp,
            sessionId,
            options.pageMemoryTimeoutMs,
          ),
        );
        const indexedPath =
          rawResultPath === undefined
            ? undefined
            : indexedRawResultPath(rawResultPath, runNumber);
        const exactResultCapture = await captureMetadata(
          exactResultJson,
          indexedPath,
        );
        const offlineReferenceScore =
          offlineReference === undefined
            ? undefined
            : scoreAgainstOfflineSenkoReference(
                result,
                offlineReference.value,
              );
        runs.push({
          run: runNumber,
          ...summarizePipelineRun(result, exactResultCapture),
          ...(offlineReferenceScore === undefined
            ? {}
            : {
                offlineReferenceScore: {
                  score: offlineReferenceScore,
                  acceptance: assessOfflineReferenceScore(
                    offlineReferenceScore,
                  ),
                },
              }),
        });
      }
      const postRun1 = postRunMeasurements[0];
      const postRun2 = postRunMeasurements[1];
      const audioSha256 = await sha256File(audioPath);
      return {
        schemaVersion: 1,
        mode: "retained-memory-diagnostic",
        timingAcceptanceEligible: false,
        servedArtifact,
        url: benchmarkUrl,
        chrome: chromeMetadata,
        runtime,
        input: { ...inputMetadata, sha256: audioSha256 },
        ...(offlineReference === undefined
          ? {}
          : { offlineReference: offlineReference.metadata }),
        runs,
        pageAgentClusterMemory: {
          scope: "Senko Window and dedicated worker agent cluster only",
          api: "performance.measureUserAgentSpecificMemory",
          samples: [
            {
              label: "models-ready-dual-resident-context",
              bytes: initialDualResident.bytes,
            },
            {
              label: "post-run-1-dual-resident-baseline",
              bytes: postRun1.bytes,
            },
            {
              label: "post-run-2-dual-resident",
              bytes: postRun2.bytes,
            },
          ],
          deltasBytes: {
            postRun1MinusModelsReady:
              postRun1.bytes - initialDualResident.bytes,
            retainedGrowthPostRun2MinusPostRun1:
              postRun2.bytes - postRun1.bytes,
          },
        },
        isolatedProfile,
      };
    }

    process.stderr.write(
      `[senko-benchmark] models ready; starting ${mode} run\n`,
    );
    const exactResultJson = await runAndCapture(
      cdp,
      sessionId,
      options.runTimeoutMs,
    );
    const result = JSON.parse(exactResultJson);
    if (mode === "timing") {
      if (Object.hasOwn(result, "pageMemory")) {
        throw new Error("Timing run was contaminated by page-memory instrumentation");
      }
      validateCanonicalAcceptanceResult(result);
    }
    const pageMemory =
      mode === "page-memory"
        ? await waitForFinalPageMemory(
            cdp,
            sessionId,
            options.pageMemoryTimeoutMs,
            abortController.signal,
          )
        : undefined;
    // Hash after the pipeline so identity validation cannot warm the WAV in the
    // filesystem cache and make an acceptance run artificially faster.
    const audioSha256 = await sha256File(audioPath);
    if (mode === "timing") {
      validateCanonicalAcceptanceInput({
        byteLength: audioStats.size,
        sha256: audioSha256,
      });
    }
    const exactResultCapture = await captureMetadata(
      exactResultJson,
      rawResultPath,
    );
    return summarizePipelineResult(result, {
      mode,
      acceptanceValidated: mode === "timing",
      servedArtifact,
      url: benchmarkUrl,
      chrome: chromeMetadata,
      runtime,
      input: { ...inputMetadata, sha256: audioSha256 },
      pageMemory,
      offlineReference: scoreOfflineReference(result, offlineReference),
      exactResultCapture,
      isolatedProfile,
    });
  } finally {
    process.removeListener("SIGINT", handleSigint);
    process.removeListener("SIGTERM", handleSigterm);
    cdp?.close();
    await terminateIsolatedChrome(chromeProcess);
    if (profileDirectory !== undefined && !options.keepProfile) {
      await rm(profileDirectory, {
        recursive: true,
        force: true,
        maxRetries: 3,
        retryDelay: 100,
      });
    }
  }
}

async function main() {
  const { help, options } = parseBenchmarkArguments(process.argv.slice(2));
  if (help) {
    process.stdout.write(HELP);
    return;
  }
  const result = await runBrowserBenchmark(options);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

const isMain =
  process.argv[1] !== undefined &&
  pathToFileURL(resolve(process.argv[1])).href === import.meta.url;
if (isMain) {
  main().catch((error) => {
    process.stderr.write(
      `[senko-benchmark] ${error instanceof Error ? error.stack ?? error.message : String(error)}\n`,
    );
    process.exitCode = 1;
  });
}
