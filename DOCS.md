# Senko Documentation

### `Diarizer`
```python
import senko
diarizer = senko.Diarizer(torch_device='auto', warmup=True, quiet=True)
```
- `torch_device`: Device to use for PyTorch operations (`auto`, `cuda`, `mps`, `cpu`)
    - `auto` automatically selects `cuda` if available, if not, then `mps` (Apple Silicon), if not, then `cpu`
- `warmup`: Warm up CAM++ embeddings model and clustering objects during initialization
    - If warmup is not done, the first few runs of the pipeline will be a bit slower
- `quiet`: Suppress progress updates and all other output to stdout

### `diarize()`
```python
result_data = diarizer.diarize(wav_path='audio.wav', generate_colors=False)
```
#### Parameters
- `wav_path`: Path to the audio file (16kHz mono 16-bit WAV format)
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

#### Raises
- `senko.AudioFormatError` if audio file is not in the required 16kHz mono 16-bit WAV format

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
