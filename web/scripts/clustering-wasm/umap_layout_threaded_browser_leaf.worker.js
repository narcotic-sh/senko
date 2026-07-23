let run;

self.onmessage = async (event) => {
  const message = event.data;
  if (message?.type === "initialize") {
    try {
      const instance = await WebAssembly.instantiate(message.module, {
        env: { memory: message.memory },
      });
      instance.exports.__stack_pointer.value = message.stackTop;
      instance.exports._initialize();
      run = instance.exports.umap_layout_threaded_run;
      self.postMessage({ type: "ready" });
    } catch (error) {
      self.postMessage({
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
      self.postMessage({
        type: "complete",
        status,
        durationMs: performance.now() - startedAt,
      });
    } catch (error) {
      self.postMessage({
        type: "error",
        error: error instanceof Error ? error.stack : String(error),
      });
    }
  }
};
