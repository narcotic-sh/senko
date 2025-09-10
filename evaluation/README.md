# Senko Evaluations
Settings `collar=0.25` and `skip_overlap=False` were used for all evaluations.

Evaluations were performed using `eval.py`.

### VoxConverse
A dataset of conversational speech from YouTube videos. Primarily English, with a bit of other European languages.

<center>

| Device | VAD | Clustering Location | Global DER | Global RTF | System |
|:--------:|:-----:|:-------------------:|:------------:|:------------:|:------------:|
| `cuda` | pyannote | CPU | 10.5% | 0.0021401 | RTX 5090 + Ryzen 9 9950X |
| `mps` | silero | CPU | 11.1% | 0.0078063 | Apple M3 |
| `cuda` | pyannote | GPU | 14.5% | 0.0015595 | RTX 5090 + Ryzen 9 9950X |

</center>

### AISHELL-4
A dataset of meeting recordings in Mandarin Chinese.

<center>

| Device | VAD | Clustering Location | Global DER | Global RTF | System |
|:--------:|:-----:|:-------------------:|:------------:|:------------:|:------------:|
| `cuda` | pyannote | GPU | 9.3% | 0.0015444 | RTX 5090 + Ryzen 9 9950X |
| `cuda` | pyannote | CPU | 9.4% | 0.0034435 | RTX 5090 + Ryzen 9 9950X |
| `mps` | silero | CPU | 10.7% | 0.0060743 | Apple M3 |

</center>

### AMI
A dataset of meeting recordings in English, with participants recorded using headset and distant microphones.

### IHM (Individual Headset Microphone)

<center>

| Device | VAD | Clustering Location | Global DER | Global RTF | System |
|:--------:|:-----:|:-------------------:|:------------:|:------------:|:------------:|
| `cuda` | pyannote | GPU | 24.9% | 0.0014214 | RTX 5090 + Ryzen 9 9950X |
| `cuda` | pyannote | CPU | 24.9% | 0.0028280 | RTX 5090 + Ryzen 9 9950X |
| `mps` | silero | CPU | 26.0% | 0.0060978 | Apple M3 |

</center>

### SDM (Single Distant Microphone)

<center>

| Device | VAD | Clustering Location | Global DER | Global RTF | System |
|:--------:|:-----:|:-------------------:|:------------:|:------------:|:------------:|
| `cuda` | pyannote | GPU | 24.9% | 0.0014103 | RTX 5090 + Ryzen 9 9950X |
| `cuda` | pyannote | CPU | 24.9% | 0.0028629 | RTX 5090 + Ryzen 9 9950X |
| `mps` | silero | CPU | 32.8% | 0.0055471 | Apple M3 |

</center>