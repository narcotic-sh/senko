# Senko on Modal
```sh
# Deploy Modal app
uv run modal deploy main.py

# Diarize a wav file
uv run client.py path/to/audio.wav

# Stop Modal app
uv run modal app stop senko-diarization
```
Using (CPU) memory snapshotting, we're able to shave off ~20 seconds from the cold-boot time.

<img width="1128" height="202" alt="Screenshot 2025-12-07 at 8 12 28 PM" src="https://github.com/user-attachments/assets/7df0a866-21e8-49a0-acd5-100c832cc695" />

This is done through the recommended [two-stage initialization process](https://modal.com/docs/guide/memory-snapshot#:~:text=two%2Dstage%20initialization%20process) where we first load model weights on CPU before the snapshot, and, after snapshot, transfer them to the GPU. Snapshotting also, of course, saves us from redoing all the heavy Python library/module imports, like `torch` etc.

RAPIDS clustering is the only component that must be initialized after snapshot restoration, as it requires a GPU attached. Trying to initialize it before the snapshot results in crashes. It is also responsible for the experimental [GPU snapshotting](https://modal.com/blog/gpu-mem-snapshots) feature not working, due to it not playing nicely with RAPIDS.