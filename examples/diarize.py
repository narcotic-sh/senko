# Example Senko usage script. Run like so:
# python diarize.py --device cuda|mps|cpu
# Diarization output (cleaned/merged, not raw) along with generated speaker color sets will be saved in ./results

import os
import json
import senko
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Senko example diarization script')
    parser.add_argument('--device',
                       choices=['cuda', 'mps', 'cpu', 'auto'],
                       default='auto',
                       help='Torch device to use for processing (default: auto)')
    args = parser.parse_args()

    diarizer = senko.Diarizer(torch_device=args.device, warmup=True, quiet=False)
    print("Diarizer warmed up and ready!\n")

    while True:
        wav_path = input("Enter path to WAV file (or 'quit' to exit): ").strip()
        if wav_path.lower() in ['quit', 'exit', 'q']:
            break

        # Check if file exists
        if not os.path.exists(wav_path):
            print(f"Error: File '{wav_path}' not found!")
            continue

        # Check if it's a WAV file
        if not wav_path.lower().endswith('.wav'):
            print("Error: File doesn't have .wav extension.")
            continue

        try:
            # Run diarization
            result_data = diarizer.diarize(wav_path, generate_colors=True)

            if result_data is None: # No speakers detected
                continue

            # Extract everything diarize() returns (for demonstration)
            raw_segments = result_data["raw_segments"]
            raw_speakers_detected = result_data["raw_speakers_detected"]
            merged_segments = result_data["merged_segments"]
            merged_speakers_detected = result_data["merged_speakers_detected"]
            centroids = result_data["speaker_centroids"]
            timing_stats = result_data["timing_stats"]
            speaker_color_sets = result_data["speaker_color_sets"]

            # Compute cosine similarity between two speakers like this
            # similarity = senko.speaker_similarity(centroids['SPEAKER_01'], centroids['SPEAKER_02'])

            # Create result directory if it doesn't exist
            result_dir = Path("./result")
            result_dir.mkdir(exist_ok=True)

            # Generate output filename
            input_filename = Path(wav_path).stem
            output_path = result_dir / f"{input_filename}.json"

            # Save merged segments and generated color sets
            with open(output_path, 'w') as f:
                json.dump({
                    "merged_segments": merged_segments,
                    "speaker_color_sets": speaker_color_sets
                }, f, indent=2)


        except senko.AudioFormatError:
            # Error message already printed by diarizer
            continue

        except Exception as e:
            print(f"Error processing file: {str(e)}")
            continue