#!/usr/bin/env node

import { spawn } from "node:child_process";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { join, resolve } from "node:path";

const repositoryRoot = resolve(import.meta.dirname, "../../..");
const profileRoot = resolve(
  repositoryRoot,
  ".research/chrome-umap-layout-runs",
);
const resultPath = resolve(
  repositoryRoot,
  ".research/umap-layout-threaded-chrome-result.json",
);
const chrome =
  process.env.CHROME_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const url =
  "http://127.0.0.1:5173/scripts/clustering-wasm/umap_layout_threaded_browser.html";

class CdpConnection {
  #socket;
  #nextId = 1;
  #pending = new Map();

  static async connect(webSocketUrl) {
    const socket = new WebSocket(webSocketUrl);
    await new Promise((resolvePromise, rejectPromise) => {
      const timer = setTimeout(
        () => rejectPromise(new Error("CDP connection timed out")),
        10_000,
      );
      socket.addEventListener(
        "open",
        () => {
          clearTimeout(timer);
          resolvePromise();
        },
        { once: true },
      );
      socket.addEventListener(
        "error",
        (event) => {
          clearTimeout(timer);
          rejectPromise(
            new Error(event.message ?? "CDP connection failed"),
          );
        },
        { once: true },
      );
    });
    return new CdpConnection(socket);
  }

  constructor(socket) {
    this.#socket = socket;
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(String(event.data));
      if (message.id === undefined) return;
      const pending = this.#pending.get(message.id);
      if (pending === undefined) return;
      this.#pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error === undefined) {
        pending.resolve(message.result ?? {});
      } else {
        pending.reject(
          new Error(
            `${pending.method}: ${message.error.message}`,
          ),
        );
      }
    });
  }

  send(method, params = {}, sessionId, timeoutMs = 30_000) {
    const id = this.#nextId++;
    return new Promise((resolvePromise, rejectPromise) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id);
        rejectPromise(new Error(`${method} timed out`));
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

  close() {
    this.#socket.close();
  }
}

function delay(milliseconds) {
  return new Promise((resolvePromise) =>
    setTimeout(resolvePromise, milliseconds),
  );
}

async function waitForDevTools(profileDirectory, child) {
  const activePort = join(profileDirectory, "DevToolsActivePort");
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new Error("Isolated Chrome exited before CDP was ready");
    }
    try {
      const [portSource, browserPath] = (
        await readFile(activePort, "utf8")
      )
        .trim()
        .split(/\r?\n/);
      const port = Number(portSource);
      if (Number.isSafeInteger(port) && browserPath) {
        const response = await fetch(
          `http://127.0.0.1:${port}/json/version`,
        );
        if (response.ok) {
          const version = await response.json();
          return (
            version.webSocketDebuggerUrl ??
            `ws://127.0.0.1:${port}${browserPath}`
          );
        }
      }
    } catch {
      // Chrome has not written its active port yet.
    }
    await delay(100);
  }
  throw new Error("Timed out waiting for isolated Chrome");
}

async function evaluate(cdp, sessionId, expression) {
  const response = await cdp.send(
    "Runtime.evaluate",
    {
      expression,
      awaitPromise: true,
      returnByValue: true,
    },
    sessionId,
  );
  if (response.exceptionDetails !== undefined) {
    throw new Error(
      response.exceptionDetails.exception?.description ??
        response.exceptionDetails.text,
    );
  }
  return response.result?.value;
}

async function terminateProcessGroup(child) {
  if (child === undefined) return;
  const signal = (name) => {
    try {
      process.kill(-child.pid, name);
    } catch (error) {
      if (error.code !== "ESRCH") throw error;
    }
  };
  signal("SIGTERM");
  await Promise.race([
    new Promise((resolvePromise) =>
      child.once("exit", resolvePromise),
    ),
    delay(5_000),
  ]);
  try {
    process.kill(-child.pid, 0);
    signal("SIGKILL");
  } catch (error) {
    if (error.code !== "ESRCH") throw error;
  }
}

await mkdir(profileRoot, { recursive: true });
const profileDirectory = await mkdtemp(
  join(profileRoot, "threaded-layout-"),
);
let child;
let cdp;
try {
  child = spawn(
    chrome,
    [
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
    ],
    {
      detached: true,
      stdio: "ignore",
    },
  );
  const webSocketUrl = await waitForDevTools(profileDirectory, child);
  cdp = await CdpConnection.connect(webSocketUrl);
  const version = await cdp.send("Browser.getVersion");
  const { targetInfos } = await cdp.send("Target.getTargets");
  const targetId = targetInfos.find(
    (target) => target.type === "page",
  )?.targetId;
  if (targetId === undefined) throw new Error("Chrome has no page target");
  const { sessionId } = await cdp.send(
    "Target.attachToTarget",
    { targetId, flatten: true },
  );
  await Promise.all([
    cdp.send("Page.enable", {}, sessionId),
    cdp.send("Runtime.enable", {}, sessionId),
  ]);
  await cdp.send("Page.navigate", { url }, sessionId);

  const deadline = Date.now() + 120_000;
  let previousResultCount = -1;
  let result;
  while (Date.now() < deadline) {
    try {
      const state = await evaluate(
        cdp,
        sessionId,
        `(() => {
          const text = document.querySelector("#output")?.textContent ?? "";
          let report;
          try { report = JSON.parse(text); } catch {}
          return {
            ready: report?.ok !== undefined,
            ok: report?.ok,
            error: report?.error,
            resultCount: report?.results?.length ?? 0,
            text,
          };
        })()`,
      );
      if (state.resultCount !== previousResultCount) {
        previousResultCount = state.resultCount;
        process.stderr.write(
          `[threaded-layout] completed ${state.resultCount}/4 worker-count configurations\n`,
        );
      }
      if (state.ready) {
        result = JSON.parse(state.text);
        break;
      }
    } catch {
      // Navigation may be swapping execution contexts.
    }
    await delay(250);
  }
  if (result === undefined) {
    throw new Error("Timed out waiting for the Chrome layout matrix");
  }
  if (!result.ok) {
    throw new Error(result.error ?? "Chrome layout harness failed");
  }
  result.chrome = {
    product: version.product,
    userAgent: version.userAgent,
    jsVersion: version.jsVersion,
    executable: chrome,
    isolatedProfile: true,
  };
  await writeFile(resultPath, `${JSON.stringify(result, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} finally {
  cdp?.close();
  await terminateProcessGroup(child);
  await rm(profileDirectory, {
    recursive: true,
    force: true,
    maxRetries: 3,
    retryDelay: 100,
  });
}
