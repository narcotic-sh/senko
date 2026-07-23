import { parentPort } from "node:worker_threads";

if (parentPort === null) {
  throw new Error("This module must run inside a worker thread");
}

let run;

parentPort.on("message", async (message) => {
  if (message?.type === "initialize") {
    try {
      const instance = await WebAssembly.instantiate(message.module, {
        env: { memory: message.memory },
      });
      instance.exports.__stack_pointer.value = message.stackTop;
      instance.exports._initialize();
      run = instance.exports.umap_layout_threaded_run;
      parentPort.postMessage({ type: "ready" });
    } catch (error) {
      parentPort.postMessage({
        type: "error",
        error: error instanceof Error ? error.stack : String(error),
      });
    }
    return;
  }
  if (message?.type === "run") {
    try {
      const startedAt = performance.now();
      const status = run(message.workerId, message.headerPtr);
      parentPort.postMessage({
        type: "complete",
        status,
        durationMs: performance.now() - startedAt,
      });
    } catch (error) {
      parentPort.postMessage({
        type: "error",
        error: error instanceof Error ? error.stack : String(error),
      });
    }
  }
});
