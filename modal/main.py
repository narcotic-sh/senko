import modal
import tempfile
import os

app = modal.App("senko-diarization")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "ffmpeg")
    #.uv_pip_install("senko[nvidia] @ git+https://github.com/narcotic-sh/senko.git@modal", force_build=True)
    .uv_pip_install("senko[nvidia] @ git+https://github.com/narcotic-sh/senko.git@modal")
)

@app.cls(
    gpu="A10G",
    image=image,
    timeout=600,
    enable_memory_snapshot=True,  # CPU snapshot only
)
class SenkoDiarizer:
    @modal.enter(snap=True)
    def preload_modules(self):
        """CPU snapshot stage: build CPU-initialized diarizer without touching CUDA."""
        import senko
        self.diarizer = senko.Diarizer(device="cuda", warmup=True, quiet=False, defer_cuda_init=True)
        print("CPU snapshot stage: Senko diarizer (CPU) constructed.")

    @modal.enter(snap=False)
    def load_model(self):
        """Post-snapshot stage: initialize CUDA-backed diarizer and warm up."""
        import torch
        print("Loading Senko model with warmup (GPU)...")
        self.diarizer.activate_cuda()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("Model loaded and ready on GPU!")

    @modal.method()
    def diarize_from_bytes(self, audio_bytes: bytes) -> dict:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = self.diarizer.diarize(tmp_path, generate_colors=True)
            if result is None:
                return {"error": "No speakers detected in the audio"}

            return {
                "merged_segments": result["merged_segments"],
                "merged_speakers_detected": result["merged_speakers_detected"],
                "timing_stats": result["timing_stats"],
            }
        finally:
            os.unlink(tmp_path)


@app.local_entrypoint()
def main(wav_path: str):
    if not os.path.exists(wav_path):
        print(f"Error: File '{wav_path}' not found!")
        return

    print(f"Processing {wav_path}...")

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    diarizer = SenkoDiarizer()
    result = diarizer.diarize_from_bytes.remote(audio_bytes)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\nDetected {result['merged_speakers_detected']} speakers")
    print(f"Timing: {result['timing_stats']}")
    print("\nSegments:")
    for seg in result["merged_segments"]:
        print(f"  {seg['speaker']}: {seg['start']:.2f}s - {seg['end']:.2f}s")
