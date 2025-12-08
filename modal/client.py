import modal
import sys
import os

def diarize(wav_path: str) -> dict:
    SenkoDiarizer = modal.Cls.from_name("senko-diarization", "SenkoDiarizer")
    diarizer = SenkoDiarizer()

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    return diarizer.diarize_from_bytes.remote(audio_bytes)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python client.py <wav_path>")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"Error: File '{wav_path}' not found!")
        sys.exit(1)

    print(f"Processing {wav_path}...")
    result = diarize(wav_path)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"\nDetected {result['merged_speakers_detected']} speakers")
    print(f"Timing: {result['timing_stats']}")
    print("\nSegments:")
    for seg in result["merged_segments"]:
        print(f"  {seg['speaker']}: {seg['start']:.2f}s - {seg['end']:.2f}s")
