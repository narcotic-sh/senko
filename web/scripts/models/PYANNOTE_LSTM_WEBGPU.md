# Pyannote persistent WebGPU LSTM contract

The split export removes all four ONNX `LSTM` operators from the browser model:

```text
waveform [B,1,160000]
  -> frontend ONNX
features [B,589,60]
  -> custom persistent WebGPU LSTM
recurrent [B,589,256]
  -> tail ONNX
logits [B,589,7]
```

All activations, accumulators, cell states, and hidden states are `float32`.
The selected production package stores weights and biases as float16, widens
each loaded scalar/vector to float32 before accumulation, and otherwise
executes the same equations. A byte-exact FP32 package remains available as the
diagnostic baseline. The boundary layout is batch, frame, feature (`BTF`), so
the complete sequence for
one batch item is contiguous. Biases remain row-major; recurrent matrices use
the GPU-coalesced layout below. No transpose or CPU repacking is required
between the three stages. Static
frontend and tail graphs are exported for B1, B8, B16, and B32. The frontend
contains only convolution, instance normalization, absolute value, max pool,
leaky ReLU, and one final transpose. The tail contains only MatMul, Add, and
leaky ReLU.

## Weight buffer

`pyannote-segmentation-3.0-lstm-f32.bin` is a headerless, little-endian f32
storage buffer. Its size is 5,521,408 bytes and every tensor begins at a
256-byte-aligned offset. The authoritative offsets, shapes, lengths, hashes,
and source parameter names are in
`pyannote-segmentation-3.0-lstm.json`.

Blocks are ordered by layer, then forward direction, then reverse direction.
Each direction contains:

1. `matrix`: logical `[512, input_size + 128]`, physically
   `[4 gates, columns/4, 128 hidden lanes, 4 input lanes]`. For a fixed input
   vec4, adjacent GPU lanes fetch adjacent 16-byte weight vectors instead of
   striding by a complete source row. Reversing this transpose exactly recovers
   `concat(weight_ih[row], weight_hh[row])`.
2. `bias_ih`: `[512]`, copied exactly from PyTorch.
3. `bias_hh`: `[512]`, copied exactly from PyTorch.

Layer 0 has `input_size=60`, a `[512,188]` matrix, and a 389,120-byte block per
direction. Layers 1–3 have `input_size=256`, a `[512,384]` matrix, and a
790,528-byte block per direction. Keeping the biases separate preserves the
checkpoint exactly; a shader may add both while accumulating the affine term.

The production `pyannote-segmentation-3.0-lstm-f16.bin` uses the identical block
order and logical layout with little-endian IEEE binary16 elements. Its strict
version-3 metadata is `pyannote-segmentation-3.0-lstm-f16.json`; it declares
`shader-f16` as required and float32 accumulation explicitly. The package is
2,760,704 bytes versus 5,521,408 bytes for FP32, so it halves every recurrent
matrix read as well as the resident weight buffer. Chrome/M3 parity and
throughput passed before this package was promoted to the production manifest.

PyTorch gate rows use IFGO order:

| Rows | Gate | Activation |
|---:|---|---|
| `0..127` | input `i` | sigmoid |
| `128..255` | forget `f` | sigmoid |
| `256..383` | candidate `g` | tanh |
| `384..511` | output `o` | sigmoid |

For each layer and direction, with zero initial hidden and cell states:

```text
z = matrix @ concat(x_t, h_prev) + bias_ih + bias_hh
i = sigmoid(z[0:128])
f = sigmoid(z[128:256])
g = tanh(z[256:384])
o = sigmoid(z[384:512])
c_t = f * c_prev + i * g
h_t = o * tanh(c_t)
```

The forward direction visits frames `0..588`; reverse visits `588..0` but
writes each result back to its original frame index. The layer output feature
axis is `concat(forward_h, reverse_h)`, in that order. Layers 1–3 therefore
consume 256 features. There is no inference dropout because the model is in
evaluation mode.

## GPU buffer sizes

The manifest records exact byte counts under
`models.segmentation.split.buffer_bytes_by_batch`. Useful values are:

| Batch | Waveform | First-conv activation | Frontend output | Recurrent output | Two recurrent ping-pong buffers | Input-affine scratch | Logits |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 640,000 | 5,112,000 | 141,360 | 603,136 | 1,206,272 | 2,412,544 | 16,492 |
| 8 | 5,120,000 | 40,896,000 | 1,130,880 | 4,825,088 | 9,650,176 | 19,300,352 | 131,936 |
| 16 | 10,240,000 | 81,792,000 | 2,261,760 | 9,650,176 | 19,300,352 | 38,600,704 | 263,872 |
| 32 | 20,480,000 | 163,584,000 | 4,523,520 | 19,300,352 | 38,600,704 | 77,201,408 | 527,744 |

B32 fits the target M3 after requesting its maximum storage-buffer limit, but
measured throughput favors B8: B8 settles at 106.8 ms per batch while B16 takes
244.7 ms. Production therefore uses B8, projecting about 5.0 seconds for all
370 VAD chunks in the one-hour benchmark. Adapters limited to WebGPU's portable
128 MiB storage binding must select B16 or below because the B32 Sinc activation
is 163.6 MB. The custom LSTM itself does not have that large temporary
allocation.

## Browser runtime implementation

Production uses two dispatches per layer. The first evaluates the independent
`bias_ih + bias_hh + W_ih*x_t` prefix across every frame. A 256-invocation
workgroup tiles eight frames, loads each packed FP16 weight vector once, and
retains eight independent FP32 accumulation chains. It writes a reusable FP32
preactivation arena. The recurrent dispatch then uses one 256-invocation
workgroup for each `(batch,direction)` pair; every invocation owns two gate
rows and evaluates only `W_hh*h_(t-1)` before the state update. Both directions
write disjoint halves of the output feature axis. Compute-pass boundaries make
the preactivation arena and each layer's ping-pong output visible to its
consumer.

The split point is exactly between the original input and recurrent loops.
Storing the already-FP32 input prefix does not reassociate either dot product,
so the production output is byte-identical to the retained single-dispatch
`persistent` diagnostic baseline.

Metal fast-math can return a non-finite value for `tanh` on the trained cell
states, which reach roughly 490 even though the result is fully saturated.
The WGSL implementation clamps sigmoid inputs to +/-30 and tanh inputs to
+/-15. Both bounds are beyond float32 activation saturation and preserve the
model result while avoiding that backend-specific non-finite path.

The older ORT split diagnostic keeps frontend input/output and recurrent tail input in external GPU
tensors on ORT's own JSEP device. Graph capture is disabled: it regressed model
throughput on Chrome/M3, and JSEP 1.27 did not reliably populate the tail's
preallocated GPU output. The tail instead owns and downloads its final output,
so the only host transfers are one waveform upload and one final logits
readback. With production FP16 weights, explicitly owned buffers total
4,748,464 bytes for B1 and 18,661,888 bytes for B8, excluding ORT's opaque
internal arena and final output allocation.

The packed matrix deliberately keeps input and recurrent columns in the same
coalesced gate/column/hidden layout. The two production shaders consume their
respective contiguous column ranges without changing the artifact format.

## Input-affine split A/B

The original balanced isolated-Chrome B8 split A/B measured 14 settled samples
per variant. The retained single-dispatch `persistent` baseline had pooled
medians of 48.7475 ms wall and 46.891008 ms GPU for the full raw VAD call;
`input-affine-tile4` measured 35.2525 ms wall and 33.619968 ms GPU. Tile-4
therefore remains the explicit split-kernel diagnostic baseline.

A later production-build B8 A/B ran tile-4/tile-8/tile-4/tile-8, with seven
settled whole-call samples inside each run. Pooled medians were 36.9775 ms wall
and 35.389440 ms GPU for tile-4 versus 35.9500 ms wall and 33.914880 ms GPU for
production tile-8. Tile-8 saved 1.0275 ms wall (2.78%) and 1.474560 ms GPU
(4.17%) per call. Its four-layer input-affine profile fell from 9.0075 to
7.7225 ms (14.27%); across 47 B8 calls that projects to 48.29 ms less wall time
before dual-device overlap. Absolute times varied between the paired runs, so
the conclusion uses the pooled settled samples and the isolated subgroup, not
one favorable invocation.

All persistent, tile-4, and tile-8 checks produced SHA-256
`20c74873f618bd7f6b846f289a898a7bb8fdc44964eb8a80aa580944473b2323`,
the same ORT error metrics, no non-finite values, and 4712/4712 matching frame
argmax decisions. Tile-8 halves input-affine workgroups at B8 from 9472 to 4736
across four layers. Its local workgroup input tile is 8192 bytes rather than
tile-4's 4096 bytes, but both use the same 19,300,352-byte persistent
preactivation arena. The LSTM owns exactly 31,711,360 GPU-buffer bytes and the
complete direct VAD owns 44,145,664 bytes on either split geometry. Reported
production logical CPU residency also remains 5,571,936 bytes, with 12,058,624
bytes in the fixed WASM heaps.

## FP16-weight A/B diagnostic

Generate or byte-verify both packages through the regular pyannote exporter.
The FP16 package can also be composed against the PyTorch and ONNX references:

```bash
uv run --project web/scripts/models --locked \
  python3 web/scripts/models/verify_pyannote_split.py \
  --batch 1 --lstm-precision float16
```

With the development server running, compare identical deterministic input in
Chrome. Change only `lstm=f32` to `lstm=f16` between runs:

```text
http://127.0.0.1:5173/ort-diagnostic.html?model=vad-compare&batch=8&verify=1&runs=8&profile=1&lstm=f32
http://127.0.0.1:5173/ort-diagnostic.html?model=vad-compare&batch=8&verify=1&runs=8&profile=1&lstm=f16
```

On the retained pre-input-affine kernel, B8 FP16 reduces Senko-owned buffers from 21,422,592 to 18,661,888
bytes (20.43 to 17.80 MiB). ORT's opaque frontend/tail allocations are
unchanged. On Chrome/M3, FP32 settled at 95.9 ms per complete split run and
59.2 ms for its profiled LSTM stage; FP16 settled at 82.8 ms and 45.2 ms,
respectively. Profiled end-to-end split time fell from 103.1 to 88.6 ms. FP32
therefore remains an explicit diagnostic baseline, while FP16 is the production
default.

## Reference validation

The reference runner reads the deployed binary rather than the source state
dict and evaluates the equations above one frame at a time:

```bash
uv run --project web/scripts/models --locked \
  python3 web/scripts/models/verify_pyannote_split.py --batch 1
```

For the pinned checkpoint, B1 composition produced these maximum absolute
errors:

- frontend ONNX versus PyTorch: `5.25e-6`;
- explicit packaged LSTM versus PyTorch LSTM: `2.63e-6`;
- frontend -> explicit LSTM -> tail versus full frozen PyTorch logits:
  `7.87e-6` with 100% argmax agreement;
- split composition versus the monolithic ONNX graph: `3.87e-5`.

These are float32 accumulation-order differences, not semantic differences.

Direct Chrome/M3 comparison against the monolithic fallback reference produced:

- B1: maximum absolute error `2.294e-4`, RMS error `9.99e-5`, and 589/589
  matching frame argmax decisions;
- B8: maximum absolute error `2.363e-4`, RMS error `8.07e-5`, and 4712/4712
  matching frame argmax decisions;
- B1 recurrent output fingerprint: sum `-277.77443`, L2 `70.59137`, with all
  150,784 values finite.

For the pinned checkpoint and deterministic B1 verifier input, FP16 weights with
FP32 accumulation produced maximum recurrent error `7.683e-4`, maximum final
logit error `1.511e-3`, mean final logit error `6.101e-4`, and 589/589 matching
frame argmax decisions versus the full PyTorch model. Browser validation also
passed at B8: maximum logit error `1.835e-3`, RMS error `7.371e-4`, and
4712/4712 matching frame argmax decisions versus the monolithic reference.
