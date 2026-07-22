#!/usr/bin/env node

import { spawn } from "node:child_process";
import { constants as fsConstants } from "node:fs";
import { access, mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const REPOSITORY_ROOT = resolve(SCRIPT_DIRECTORY, "../../..");
const CAPTURE_SLOT = "__senkoBrowserDiagnosticCaptureV1";

const DEFAULTS = Object.freeze({
  url: "http://127.0.0.1:4173/?raw-campplus-graph-diagnostic=1",
  eventName: "senko-raw-campplus-graph-diagnostic",
  selector: "#raw-campplus-graph-result",
  statusAttribute: "data-status",
  terminalStatuses: Object.freeze(["passed", "failed", "error"]),
  timeoutMs: 180_000,
  profileRoot: join(REPOSITORY_ROOT, ".research/chrome-diagnostic-runs"),
  keepProfile: false,
  chrome: undefined,
});

const HELP = `Usage:
  node web/scripts/benchmark/run-browser-diagnostic.mjs [options]

Options:
  --url <url>               Diagnostic URL (default: ${DEFAULTS.url})
  --event <name>            CustomEvent carrying the result detail
  --no-event                Disable CustomEvent capture
  --selector <css>          Result element used as a fallback
  --no-selector             Disable DOM fallback
  --status-attribute <name> Terminal-status attribute (default: data-status)
  --status <value>          Terminal attribute value; repeat to replace defaults
  --timeout-ms <ms>         Diagnostic timeout (default: ${DEFAULTS.timeoutMs})
  --chrome <path>           Chrome executable (or set CHROME_PATH)
  --profile-root <path>     Parent for the unique temporary profile
  --keep-profile            Retain the unique profile after teardown
  --remove-profile          Remove it after teardown (the default)
  --help                    Show this help

The default event and selector target the raw CAM++ graph diagnostic. The event
listener is installed before navigation; the selector/data-status path remains
available for diagnostics that do not dispatch an event. Standard output is
only the parsed diagnostic JSON.
`;

export function parseDiagnosticArguments(argv) {
  const options = {
    ...DEFAULTS,
    terminalStatuses: [...DEFAULTS.terminalStatuses],
  };
  let profileDisposition;
  let eventDisposition;
  let selectorDisposition;
  let sawStatus = false;

  const takeValue = (index, flag) => {
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`${flag} requires a value`);
    }
    if (value.length === 0) throw new Error(`${flag} requires a nonempty value`);
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
      case "--event":
        if (eventDisposition === "disabled") {
          throw new Error("--event and --no-event are mutually exclusive");
        }
        eventDisposition = "enabled";
        options.eventName = takeValue(index, flag);
        index += 1;
        break;
      case "--no-event":
        if (eventDisposition === "enabled") {
          throw new Error("--event and --no-event are mutually exclusive");
        }
        eventDisposition = "disabled";
        options.eventName = undefined;
        break;
      case "--selector":
        if (selectorDisposition === "disabled") {
          throw new Error("--selector and --no-selector are mutually exclusive");
        }
        selectorDisposition = "enabled";
        options.selector = takeValue(index, flag);
        index += 1;
        break;
      case "--no-selector":
        if (selectorDisposition === "enabled") {
          throw new Error("--selector and --no-selector are mutually exclusive");
        }
        selectorDisposition = "disabled";
        options.selector = undefined;
        break;
      case "--status-attribute":
        options.statusAttribute = takeValue(index, flag);
        index += 1;
        break;
      case "--status":
        if (!sawStatus) {
          options.terminalStatuses = [];
          sawStatus = true;
        }
        options.terminalStatuses.push(takeValue(index, flag));
        index += 1;
        break;
      case "--timeout-ms":
        options.timeoutMs = takePositiveInteger(index, flag);
        index += 1;
        break;
      case "--chrome":
        options.chrome = takeValue(index, flag);
        index += 1;
        break;
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
      default:
        throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (options.eventName === undefined && options.selector === undefined) {
    throw new Error("At least one of CustomEvent capture or DOM fallback is required");
  }
  return { help: false, options };
}

export function normalizeDiagnosticUrl(source) {
  let url;
  try {
    url = new URL(source);
  } catch {
    throw new Error(`Diagnostic URL must be absolute, received ${source}`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`Diagnostic URL must use HTTP(S), received ${url.protocol}`);
  }
  return url.href;
}

export function decodeDiagnosticState(state, terminalStatuses) {
  if (state?.event?.fired === true) {
    if (typeof state.event.serializationError === "string") {
      throw new Error(
        `Diagnostic CustomEvent detail is not JSON-serializable: ${state.event.serializationError}`,
      );
    }
    return {
      source: "event",
      result: parseResultJson(state.event.json, "CustomEvent detail"),
    };
  }

  if (typeof state?.dom?.selectorError === "string") {
    throw new Error(`Invalid result selector: ${state.dom.selectorError}`);
  }
  if (
    typeof state?.dom?.status === "string" &&
    terminalStatuses.includes(state.dom.status)
  ) {
    return {
      source: "dom",
      status: state.dom.status,
      result: parseResultJson(
        state.dom.text,
        `result element with status ${JSON.stringify(state.dom.status)}`,
      ),
    };
  }
  return undefined;
}

function parseResultJson(source, description) {
  if (typeof source !== "string" || source.length === 0) {
    throw new Error(`${description} did not contain JSON`);
  }
  try {
    return JSON.parse(source);
  } catch (error) {
    const preview = source.length <= 2_000 ? source : `${source.slice(0, 2_000)}…`;
    throw new Error(
      `${description} contained invalid JSON: ${
        error instanceof Error ? error.message : String(error)
      }\n${preview}`,
    );
  }
}

function buildEventCaptureScript(eventName) {
  return `(() => {
    const slot = ${JSON.stringify(CAPTURE_SLOT)};
    const eventName = ${JSON.stringify(eventName)};
    globalThis[slot] = { fired: false, eventName };
    globalThis.addEventListener(eventName, (event) => {
      try {
        const json = JSON.stringify(event.detail);
        if (typeof json !== "string") {
          throw new TypeError("JSON.stringify returned undefined");
        }
        globalThis[slot] = { fired: true, eventName, json };
      } catch (error) {
        globalThis[slot] = {
          fired: true,
          eventName,
          serializationError: error instanceof Error ? error.message : String(error),
        };
      }
    }, { once: true });
  })()`;
}

function buildProbeExpression(options) {
  const selector = JSON.stringify(options.selector ?? null);
  const statusAttribute = JSON.stringify(options.statusAttribute);
  return `(() => {
    const event = globalThis[${JSON.stringify(CAPTURE_SLOT)}] ?? null;
    const selector = ${selector};
    let dom = null;
    if (selector !== null) {
      try {
        const node = document.querySelector(selector);
        dom = node === null
          ? { found: false, readyState: document.readyState }
          : {
              found: true,
              readyState: document.readyState,
              status: node.getAttribute(${statusAttribute}) ?? "",
              text: node.textContent?.trim() ?? "",
            };
      } catch (error) {
        dom = {
          found: false,
          readyState: document.readyState,
          selectorError: error instanceof Error ? error.message : String(error),
        };
      }
    }
    return { event, dom };
  })()`;
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

  close(reason = new Error("CDP connection closed by diagnostic runner")) {
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
        "JavaScript evaluation failed in the diagnostic page",
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
  let lastDetail;
  while (Date.now() < deadline) {
    signal?.throwIfAborted();
    const state = await check();
    if (state.done) return state.value;
    lastDetail = state.detail;
    await abortableDelay(intervalMs, signal);
  }
  throw new Error(
    `Timed out after ${timeoutMs} ms waiting for ${description}` +
      (lastDetail === undefined ? "" : ` (${lastDetail})`),
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
    try {
      await access(candidate, fsConstants.X_OK);
      return candidate;
    } catch {
      // Try the next well-known executable.
    }
  }
  throw new Error("Chrome was not found; pass --chrome <path> or set CHROME_PATH");
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

async function waitForDevTools(profileDirectory, child, signal) {
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
          value:
            version.webSocketDebuggerUrl ??
            `ws://127.0.0.1:${port}${browserPath}`,
        };
      } catch (error) {
        if (signal.aborted) throw error;
        return {
          done: false,
          detail: error instanceof Error ? error.message : String(error),
        };
      }
    },
    {
      timeoutMs: 20_000,
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
  const { sessionId } = await cdp.send("Target.attachToTarget", {
    targetId,
    flatten: true,
  });
  await Promise.all([
    cdp.send("Page.enable", {}, { sessionId }),
    cdp.send("Runtime.enable", {}, { sessionId }),
  ]);
  return { targetId, sessionId };
}

async function waitForDiagnostic(cdp, sessionId, options, signal) {
  const expression = buildProbeExpression(options);
  return waitForCondition(
    async () => {
      let state;
      try {
        state = await evaluate(cdp, sessionId, expression);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (/execution context|cannot find context|target navigated/i.test(message)) {
          return { done: false, detail: "navigation context is settling" };
        }
        throw error;
      }
      const capture = decodeDiagnosticState(state, options.terminalStatuses);
      return {
        done: capture !== undefined,
        value: capture,
        detail: summarizePendingState(state),
      };
    },
    {
      timeoutMs: options.timeoutMs,
      intervalMs: 50,
      description: "diagnostic result",
      signal,
    },
  );
}

function summarizePendingState(state) {
  if (state?.dom?.found === true) {
    const text = state.dom.text ?? "";
    const preview = text.length > 120 ? `${text.slice(0, 117)}...` : text;
    return `status=${JSON.stringify(state.dom.status ?? "")}, text=${JSON.stringify(preview)}`;
  }
  if (state?.dom?.found === false) {
    return `result element not found, document=${state.dom.readyState ?? "unknown"}`;
  }
  return "waiting for CustomEvent";
}

async function preflightServer(url, signal) {
  let response;
  try {
    response = await fetch(url, { signal });
  } catch (error) {
    throw new Error(
      `Could not reach the diagnostic URL ${url}; start the Vite server first (${error instanceof Error ? error.message : String(error)})`,
    );
  }
  if (!response.ok) {
    await response.body?.cancel();
    throw new Error(`Diagnostic page returned HTTP ${response.status} at ${url}`);
  }
  await response.body?.cancel();
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

export async function runBrowserDiagnostic(options) {
  const url = normalizeDiagnosticUrl(options.url);
  const chrome = await resolveChromeExecutable(options.chrome);
  const profileRoot = resolve(options.profileRoot);
  const abortController = new AbortController();
  let profileDirectory;
  let chromeProcess;
  let cdp;

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
    await preflightServer(url, abortController.signal);
    await mkdir(profileRoot, { recursive: true });
    profileDirectory = await mkdtemp(join(profileRoot, "senko-diagnostic-chrome-"));
    chromeProcess = await launchIsolatedChrome(chrome, profileDirectory);
    process.stderr.write(
      `[senko-diagnostic] isolated Chrome PID ${chromeProcess.pid}; profile ${profileDirectory}\n`,
    );

    const webSocketUrl = await waitForDevTools(
      profileDirectory,
      chromeProcess,
      abortController.signal,
    );
    cdp = await CdpConnection.connect(webSocketUrl);
    const { sessionId } = await createSinglePageSession(cdp);
    if (options.eventName !== undefined) {
      await cdp.send(
        "Page.addScriptToEvaluateOnNewDocument",
        { source: buildEventCaptureScript(options.eventName) },
        { sessionId },
      );
    }
    const navigation = await cdp.send(
      "Page.navigate",
      { url },
      { sessionId, timeoutMs: 30_000 },
    );
    if (navigation.errorText !== undefined) {
      throw new Error(`Chrome could not navigate to the diagnostic: ${navigation.errorText}`);
    }

    const capture = await waitForDiagnostic(
      cdp,
      sessionId,
      options,
      abortController.signal,
    );
    process.stderr.write(
      `[senko-diagnostic] captured JSON from ${capture.source}${
        capture.status === undefined ? "" : ` (${options.statusAttribute}=${capture.status})`
      }\n`,
    );
    return capture.result;
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
  const { help, options } = parseDiagnosticArguments(process.argv.slice(2));
  if (help) {
    process.stdout.write(HELP);
    return;
  }
  const result = await runBrowserDiagnostic(options);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

const isMain =
  process.argv[1] !== undefined &&
  pathToFileURL(resolve(process.argv[1])).href === import.meta.url;
if (isMain) {
  main().catch((error) => {
    process.stderr.write(
      `[senko-diagnostic] ${error instanceof Error ? error.stack ?? error.message : String(error)}\n`,
    );
    process.exitCode = 1;
  });
}
