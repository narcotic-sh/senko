# Senko browser runtime

This package is the browser performance harness for Senko. It keeps pipeline
work in a dedicated module worker, uses a typed request/event protocol, and
reports wall-clock timing for every stage.

## Local development

```sh
cd web
pnpm install
pnpm dev
```

Open `http://127.0.0.1:5173`. Vite serves both development and preview builds
with `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`. These headers are required for
shared-memory WASM. Production hosting must send equivalent headers over HTTPS.

Append `?memory=1` for diagnostic, page-scoped memory sampling. Chromium's
`performance.measureUserAgentSpecificMemory()` measures the Senko page's agent
cluster, including its dedicated pipeline worker, without adding unrelated
Chrome tabs or windows. The API is privacy/rate limited: a single result can
take many seconds. Senko therefore keeps at most one request in flight, never
awaits it from pipeline execution, and labels each result with the pipeline
boundary current when it resolves. The UI reports the current and peak values
plus every coarse-cadence labeled sample. URLs without `?memory=1` do not call
the API and incur no measurement overhead.

The user-agent total is an approximate browser estimate and does not guarantee
coverage of GPU allocations. `knownGpuBufferBytes` remains separate: it is the
exact WebGPU-buffer ownership of Senko's two resident inference backends,
currently 84,001,024 bytes. Opaque browser/driver and ONNX Runtime GPU
allocations are not part of that owned-buffer counter.

The benchmark runner launches a separate Chrome profile with exactly one
Senko tab. Its page-memory result therefore covers that page plus the dedicated
pipeline worker and excludes unrelated Chrome windows and tabs. Process-wide
Chrome totals are not valid Senko memory measurements when other Chrome work is
open. See [`scripts/benchmark/README.md`](scripts/benchmark/README.md) for the
current timing, correctness, and memory snapshot and the isolated measurement
protocol.

Useful checks:

```sh
pnpm check
pnpm test
pnpm build
```

## Layout

- `src/runtime/` owns the main-thread/worker boundary, messages, shared result
  types, cancellation, and timing telemetry.
- `src/pipeline/` contains inference and diarization stages.
- `src/audio/` contains streaming audio and feature extraction code.
- `src/capabilities.ts` performs WebGPU, WASM SIMD/thread, secure-context, and
  adapter feature detection before model loading begins.

Large audio files cross the worker boundary as `Blob` objects so the main
thread does not first create a full-file `ArrayBuffer`. Model assets are
described by URL, byte length, and SHA-256 digest. Pipeline implementations
should keep neural intermediates GPU-resident and send only progress telemetry
and the final diarization result back to the UI.

PCM decoding uses one 320,000-byte BYOB buffer. Chrome transfers that backing
store to each `Blob.stream()` read and returns it under a new `ArrayBuffer`
wrapper; Senko decodes the returned view synchronously and recycles the same
store. A B8 VAD input is fetched as one contiguous random-access range, so an
hour-long file creates 47 short-lived Blob streams rather than 370 and never
allocates file-sized or per-window PCM byte buffers. The scratch capacity is
included as `wavReadBufferBytes` in exact logical CPU memory accounting.

Worker initialization requests two high-performance WebGPU devices and loads
and warms both production model sets concurrently. The B8 pyannote VAD owns
44,145,664 GPU-buffer bytes on one device; B16 CAM++ owns 39,855,360 bytes on
the other, for exact summed ownership of 84,001,024 bytes. Streaming scheduling
submits VAD first, then overlaps up to two B16 CAM++ batches on the second
device as stable speech windows become available. Both models remain resident
during clustering and across subsequent recordings. Their buffers and devices
are released only when the model set or its worker is disposed.
