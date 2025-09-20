# Senko
> 閃光 (senkō) - a flash of light

A very fast and accurate speaker diarization pipeline.

1 hour of audio processed in 5 seconds (RTX 4090, Ryzen 9 7950X). ~17x faster than [Pyannote 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

On Apple M3, 1 hour in 7.7 seconds (~42x faster).

The pipeline achieves a best score of 10.5% DER on VoxConverse, 9.3% on AISHELL-4, and 24.9% on AMI (IHM/SDM).

Senko powers the [Zanshin](https://github.com/narcotic-sh/zanshin) media player.

## Usage
```python
import senko, json

diarizer = senko.Diarizer(device='auto', warmup=True, quiet=False)

wav_path = 'audio.wav' # 16kHz mono 16-bit wav
result = diarizer.diarize(wav_path, generate_colors=False)

senko.save_json(result["merged_segments"], 'audio_diarized.json')
senko.save_rttm(result["merged_segments"], wav_path, 'audio_diarized.rttm')
```
See `examples/diarize.py` for an interactive script, and also read `DOCS.md`

## Installation
Senko has been tested to work on Linux, macOS, and WSL, with Python version `3.11.13`.

Pre-requisites:
- `clang` - on Linux/WSL, a seperate install; on macOS, have the Xcode Developer Tools installed
- [`uv`](https://docs.astral.sh/uv/#installation)

Create a Python virtual environment and activate it
```
uv venv --python 3.11.13 .venv
source .venv/bin/activate
```
Then install Senko
```bash
# For NVIDIA GPUs with CUDA compute capability >= 7.5 (~GTX 16 series and newer)
uv pip install "git+https://github.com/narcotic-sh/senko.git[nvidia]"

# For NVIDIA GPUs with CUDA compute capability < 7.5 (~GTX 10 series and older)
uv pip install "git+https://github.com/narcotic-sh/senko.git[nvidia-old]"

# For Mac (macOS 14+) and CPU execution on all other platforms
uv pip install "git+https://github.com/narcotic-sh/senko.git"
```
For NVIDIA, make sure the installed driver is CUDA 12 capable (should see `CUDA Version: 12.x` in `nvidia-smi`).

For setting up Senko for development, see `DEV_SETUP.md`.

## Accuracy
Settings `collar=0.25` and `skip_overlap=False` were used for all evaluations.

<center>

| Benchmark | Best DER |
|:--------:|:-----:|
| VoxConverse | 10.5% |
| AISHELL-4 | 9.3% |
| AMI (IHM) | 24.9% |
| AMI (SDM) | 24.9% |

</center>

See the [evaluation](/evaluation) folder for more details.

## Technical Details
Senko is a modified version of the speaker diarization pipeline found in the excellent [3D-Speaker](https://github.com/modelscope/3D-Speaker/tree/main/egs/3dspeaker/speaker-diarization) project.
It consists of four stages: VAD (voice activity detection), Fbank feature extraction, speaker embeddings generation, and clustering (spectral or UMAP+HDBSCAN).

The following modifications have been made:
- VAD model has been swapped from FSMN-VAD to either Pyannote [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) or [Silero VAD](https://github.com/snakers4/silero-vad)
- Fbank feature extraction is done fully upfront, in C++, using all available CPU cores
- Batched inference of CAM++ embeddings model
- Clustering when on NVIDIA (with a GPU of CUDA compute capability 7.0+) can be done on the GPU through [RAPIDS](https://docs.rapids.ai/api/cuml/stable/zero-code-change/)

## FAQ
<details>
<summary>Is there any way to visualize the output diarization data?</summary>
<br>
Absolutely. The <a href="https://github.com/narcotic-sh/zanshin">Zanshin</a> media player is entirely made for this purpose. Zanshin is powered by Senko, so the easiest way to visualize the diarization data is by simply using it. It's currently available for Mac (Apple Silicon) with packaging. It also works on Windows and Linux, but without packaging (coming soon); you'll need to clone the repo and launch it through the terminal. See <a href="https://github.com/narcotic-sh/zanshin/blob/main/DEV_SETUP.md">here</a> for instructions.
<br>
<br>
You can also load in the diarization data that Senko generates manually into Zanshin if you want. First, turn off diarization in Zanshin by going into Settings and turning off <code>Identify Speakers</code>. Then, after you add a media item, click on it and on the player page press the <code>H</code> key. In the textbox that appears, paste the contents of the output JSON file that <code>examples/diarize.py</code> generates.
</details>
<details>
<summary>What languages does Senko support?</summary>
<br>
Generally the pipeline should work for any language, as it relies on acoustic patterns as opposed to words or speech patterns. That being said, the embeddings model used in this pipeline was trained on a mix of English and Mandarin Chinese. So the pipeline will likely work best on English and Mandarin Chinese.
</details>
<details>
<summary>Are overlapping speaker segments detected correctly?</summary>
<br>
The current output will not have any overlapping speaker segments; i.e. only one speaker max is reported to be speaking at any given time. However, despite this, the current pipeline still performs great in determining who the dominant speaker is at any given time in chaotic audio with speakers talking over each other (example: casual podcasts). That said, detecting overlapping speaker segments is a planned feature thanks to the Pyannote segmentation-3.0 model (which we currently only use for VAD) supporting it.
</details>
<details>
<summary>How fast is the pipeline on CPU (<code>cpu</code>)?</summary>
<br>
On a Ryzen 9 9950X it takes 42 seconds to process 1 hour of audio.
</details>
<details>
<summary>Does the entire pipeline run fully on the GPU, if available?</summary>
<br>
With <code>cuda</code>, all parts of the pipeline except Fbank feature extraction (which always runs on the CPU) do, by default, run on the GPU (though you can override this behaviour using the <code>vad</code> and <code>clustering</code> arguments of the <code>Diarizer</code> object). However, CPU performance still significantly impacts overall speed even for the GPU-accelerated stages.
<br><br>
During the embeddings generation phase, for example, while the actual model inference happens on the GPU with minimal CPU-GPU memory transfers (just input/output), the CPU handles all the orchestration work: Python loops for batching, tensor preparation, padding operations dispatch, and managing the inference pipeline. All this orchestration runs single-threaded on the CPU. This means a faster CPU will improve performance even when using a powerful GPU, as the CPU coordinates all the GPU operations.
<br><br>
Therefore, for optimal performance, pair a fast GPU with a fast CPU. The CPU bottleneck becomes more noticeable with very fast GPUs (ex. RTX 4090) where the GPU can execute the batch preparation and inference faster than the CPU can orchestrate/dispatch these operations.
<br><br>
As for Mac, both the VAD and embeddings gen phases run on the ANE (Apple Neural Engine) & CPU through CoreML. The fbank stage and clustering run purely on the CPU.
</details>
<details>
<summary>Known limitations?</summary>
<br>
- The pipeline works best when the audio recording quality is good. Ideal setting: professional podcast studio. Heavy background noise, background music, or a generally low fidelity recording will degrade the diarization performance significantly. Note that it's also possible to have generally good recording quality but still low fidelity recorded voice quality; an example is <a href="https://www.youtube.com/watch?v=89K8-4tHhgc">this</a>.
<br><br>
- It is rare but possible that voices that sound very similar get clustered as one voice. This can happen if the voices are genuinely extremely similar, or, more commonly, if the audio recording fidelity is low.
<br><br>
- The same voice recorded with >1 microphones or in >1 recording settings within the same audio file will often get detected as >1 speakers.
<br><br>
- If a single person makes >1 voices in the same recording (as in change the auditory texture/tone of their voice; like if they do an impression of someone else, for example), their speech will almost certainly get detected as >1 speakers.
</details>

## Community & Support
Join the [Discord](https://discord.gg/Nf7m5Ftk3c) server to ask questions, suggest features, talk about Senko and Zanshin development etc.

## Future Improvements & Directions
- Overlapping speaker segments support
- Improve speaker colors generation algorithm
- Support for Intel and AMD GPUs
- Experiment with `torch.compile()`
- Experiment with Modular [MAX](https://www.modular.com/blog/bring-your-own-pytorch-model) engine (faster CPU inference speed?)
- Experiment with [Resonate](https://alexandrefrancois.org/Resonate/) for superior audio feature extraction
- Background noise removal ([DeepFilterNet](https://github.com/Rikorose/DeepFilterNet))
- Live progress reporting
