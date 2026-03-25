# Senko Documentation

### `Diarizer`
```python
import senko
diarizer = senko.Diarizer(device='auto', vad='auto', clustering='auto', warmup=True, quiet=True, mer_cos=None)
```
- `device`: Device to use for VAD & embeddings stage (`auto`, `cuda`, `coreml`, `cpu`)
    - `auto` automatically selects `coreml` if on macOS, if not, then `cuda`, if not, then `cpu`
- `vad`: Voice Activity Detection model to use (`auto`, `pyannote`, `silero`)
    - `auto` automatically selects `pyannote` for `cuda` & `coreml`, `silero` for `cpu`
    - `pyannote` uses Pyannote VAD (requires `cuda` for optimal performance)
    - `silero` uses Silero VAD (runs on CPU; not available on macOS)
- `clustering`: Clustering location when `device` == `cuda` (`auto`, `gpu`, `cpu`)
    - Only applies to CUDA devices; non-CUDA devices always use CPU clustering
    - `auto` uses GPU clustering for CUDA devices with compute capability >= 7.0, CPU clustering otherwise
    - `gpu` uses GPU clustering on CUDA devices with compute capability >= 7.0, falls back to CPU clustering with warning otherwise
    - `cpu` forces CPU clustering (most accurate; see [evals](./evaluation/README.md#rtx-5090--ryzen-9-9950x-gpu-clustering))
- `warmup`: Warm up CAM++ embedding model and clustering objects during initialization
    - If warmup is not done, the first few runs of the pipeline will be a bit slower
- `quiet`: Suppress progress updates and all other output to stdout
- `mer_cos`: Override the cosine-similarity merge threshold for both spectral and UMAP+HDBSCAN clustering
    - Must be > 0 and <= 1
    - `None` keeps the default value from `senko/cluster/conf/*.yaml` (`0.875`)
    - After initial clustering, clusters whose centroid cosine similarity is >= `mer_cos` are merged
    - If you see too many speakers (over‑splitting), try lowering `mer_cos`
    - If you see too few speakers (over‑merging), try raising `mer_cos`

### `diarize()`
```python
result_data = diarizer.diarize(wav_path='audio.wav', accurate=None, generate_colors=False)
```
#### Parameters
- `wav_path`: Path to the audio file (16kHz mono 16-bit WAV format)
- `accurate`: Use shorter subsegments & smaller shift for (very slightly) better accuracy (`None`, `True`, `False`)
    - `None` (default): Auto-enables if `device == 'cuda'` and `vad == 'pyannote'`
    - Accuracy difference not stark enough in my testing to warrant turning this on for when not `device == 'cuda'` and `vad == 'pyannote'`.
    - Only reason to turn on would be to get better output parity with `cuda` if on `coreml` or `cpu`.
- `generate_colors`: Whether to generate speaker color sets for visualization

#### Returns
Dictionary (`result_data`) containing keys:
- `raw_segments`: Raw diarization output
    - A list of speaking segments (dictionaries) with keys `start`, `end`, `speaker`
- `raw_speakers_detected`: Number of unique speakers found in `raw_segments`
- `merged_segments`: Cleaned diarization output
    - Same format as `raw_segments`
    - Segments <= 0.78 seconds in length are removed
    - Adjacent segments of the same speaker that have a silence in between them of <= 4 seconds are merged into one segment
- `merged_speakers_detected`: Number of unique speakers found in `merged_segments`
- `speaker_centroids`: Voice fingerprints for each detected speaker
    - Dictionary mapping speaker IDs to 192-dimensional numpy arrays
    - Each centroid is the mean of all audio embeddings for that speaker
    - Can be used for speaker comparison/identification across different audio files
- `timing_stats`: Dictionary of how long each stage of the pipeline took in seconds, as well as the total time
    - Keys: `total_time`, `vad_time`, `fbank_time`, `embeddings_time`, `clustering_time`
- `speaker_color_sets`: 10 sets of speaker colors (if requested)
- `vad`: Voice activity detection segments
    - List of `(start, end)` tuples in seconds produced by the VAD stage, marking every region of the audio that contains speech

#### Raises
- `senko.AudioFormatError` if audio file is not in the required 16kHz mono 16-bit WAV format

### `diarize_samples()`
```python
result_data = diarizer.diarize_samples(samples, sample_rate=16000, accurate=None, generate_colors=False, source_name="<memory>")
```
#### Parameters
- `samples`: In-memory mono 16kHz audio as a `numpy.ndarray` or `torch.Tensor`
  - Floating-point inputs must already be normalized to `[-1, 1]`
  - Integer PCM inputs must use `uint8`, `int16`, or `int32`
- `sample_rate`: Must be `16000`
- `accurate`: Same behavior as `diarize()`
- `generate_colors`: Same behavior as `diarize()`
- `source_name`: Label used in progress output

#### Returns
- Same dictionary shape as `diarize()`

#### Raises
- `senko.AudioFormatError` if the input is not 16kHz mono audio or uses an unsupported/ambiguous in-memory representation

### `speaker_similarity()`
```python
if senko.speaker_similarity(centroid1, centroid2) >= 0.875:
    print('Speakers are the same')
```
Calculate cosine similarity between two speaker centroids (voice fingerprints).
#### Parameters
- `centroid1`: First speaker centroid (192-dimensional numpy array)
- `centroid2`: Second speaker centroid (192-dimensional numpy array)

#### Returns
- `float`: Cosine similarity score between -1 and 1 (<1 rarely if ever happens with speaker embeddings)

### `save_json()`
```python
senko.save_json(segments, output_path)
```
Save diarization segments to a JSON file.
#### Parameters
- `segments`: List of segment dictionaries with keys `start`, `end`, `speaker`
  - Typically `result["raw_segments"]` or `result["merged_segments"]` from `diarize()`
- `output_path`: Path where the JSON file will be saved

### `save_rttm()`
```python
senko.save_rttm(segments, wav_path, output_path)
```
Save diarization segments in RTTM (Rich Transcription Time Marked) format, compatible with standard diarization evaluation tools.
#### Parameters
- `segments`: List of segment dictionaries with keys `start`, `end`, `speaker`
  - Typically `result["raw_segments"]` or `result["merged_segments"]` from `diarize()`
- `wav_path`: Path to the original audio file (used to extract file ID for RTTM format)
- `output_path`: Path where the RTTM file will be saved

### Output Format
Speaker segments (`raw_segments`/`merged_segments`):
```
[
  {
    "start": 0.0,
    "end": 5.2,
    "speaker": "SPEAKER_01"
  },
  {
    "start": 5.2,
    "end": 10.8,
    "speaker": "SPEAKER_02"
  },
  ...
]
```
Speaker centroids (`speaker_centroids`):
```
{
  "SPEAKER_01": array([0.123, -0.456, 0.789, ...]),  # 192-dimensional numpy array
  "SPEAKER_02": array([-0.234, 0.567, -0.890, ...]), # 192-dimensional numpy array
  ...
}
```
Color sets (`speaker_color_sets`):
```
{
    "0": {
      "SPEAKER_01": "#ea759c",
      "SPEAKER_02": "#579c3a",
      "SPEAKER_03": "#100058",
    },
    "1": {
      "SPEAKER_01": "#97de7b",
      "SPEAKER_02": "#4c56b6",
      "SPEAKER_03": "#480000",
    },
    "2": {
      "SPEAKER_01": "#8393f9",
      "SPEAKER_02": "#bf5d01",
      "SPEAKER_03": "#003a38",
    },
    ...
}
```
VAD segments (`vad`):
```
[
  (0.0, 2.1),
  (2.4, 6.7),
  (7.1, 10.2),
  ...
]
```
