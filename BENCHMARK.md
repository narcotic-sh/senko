# Benchmarking
To get the figures listed in the README for time taken to diarize 1 hour of audio on an RTX 4090 + Ryzen 9 7950X machine and an M3 MacBook Air, as well as the CPU performance figure in the FAQ section using a Ryzen 9 9950X, the audio from [this](https://www.youtube.com/watch?v=GT_sXIUJPUo) video was used. It was downloaded and decompressed into a 16 KHz mono wav file in the following manner.
```
yt-dlp -f bestaudio GT_sXIUJPUo -o "temp_GT_sXIUJPUo.%(ext)s"
ffmpeg -i "temp_GT_sXIUJPUo."* -acodec pcm_s16le -ac 1 -ar 16000 cowen.wav
rm "temp_GT_sXIUJPUo."*
```
Then the `examples/diarize.py` script was run, and `cowen.wav` was entered after the warmup.

For the sake of reproducibility, the exact `cowen.wav` file used can be downloaded from [here](https://www.dropbox.com/scl/fi/77kgl6luhmsm6k30x1muf/cowen.wav?rlkey=n6goatgi3pjpgn7glna654f2a&dl=1).