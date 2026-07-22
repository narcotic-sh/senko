# Senko FBank WebAssembly

This is the production-oriented browser port of Senko's fixed Kaldi-style
80-bin filterbank. It is deliberately a small standalone WebAssembly module,
not an Emscripten JavaScript runtime:

- 19 KiB `.wasm`, compiled with `-O3 -msimd128`
- no filesystem, threads, dynamic allocation, imports, or JavaScript glue
- a fixed 512 KiB linear memory (8 WebAssembly pages), with growth disabled
- one reusable PCM input, raw-feature, and normalized output arena
- capacity for 24,399 samples / 150 frames, covering the float32 timestamp
  drift in native Senko's nominal 1.5-second windows without memory growth
- explicit `dispose()` on both `WasmSenkoFbank` and
  `StreamingFbankExtractor`

Build it with:

```sh
cd web
pnpm build:fbank-wasm
```

The checked-in binary is loaded with Vite's `?url` asset handling. Browser
code initializes it once and injects it into the existing stream:

```ts
const fbank = await WasmSenkoFbank.create();
const extractor = new StreamingFbankExtractor(reader, fbank);
try {
  for await (const window of extractor.extract(requests)) {
    // window.features.data aliases the reusable WASM output arena. Copy or
    // upload it before requesting the next window.
  }
} finally {
  extractor.dispose();
}
```

For ordinary 1.5-second windows advancing by 0.6 seconds, the stream reuses
the 88 overlapping raw log-mel frames and computes only 60 new FFT frames.
Mean normalization still runs independently over every complete window.

Run the hour-long benchmark with:

```sh
cd web
pnpm benchmark:fbank-wasm
```

The benchmark includes bounded WAV range reads, PCM16-to-float decoding,
filterbank computation, and sampled process memory. Its `stageMs` breakdown
separates overlap copies, file reads, PCM conversion, and the WASM kernel, while
`computedRawFrames` makes reuse effectiveness explicit. Native-fixture parity
and reuse equivalence are covered by `src/audio/wasm-fbank.test.ts`.
