# Have senko installed, and also install the following:
# uv pip install pyannote.metrics pyannote.core librosa soundfile

# Then run this script like so:
# python voxconverse.py --subset test

"""
Senko Evaluation Script for VoxConverse Dataset
This script evaluates Senko speaker diarization performance on the VoxConverse dataset
using pyannote.metrics for computing the Diarization Error Rate (DER).

Expected directory structure:
./dev/
  audio/*.wav
  *.rttm
./test/
  audio/*.wav
  *.rttm
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

import senko
import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


def load_rttm_file(rttm_path: Path, uri: str = None) -> Annotation:
    """
    Load an RTTM file and convert it to a pyannote Annotation object.

    Args:
        rttm_path: Path to the RTTM file
        uri: URI for the annotation (defaults to filename without extension)

    Returns:
        pyannote.core.Annotation object
    """
    if uri is None:
        uri = rttm_path.stem

    annotation = Annotation(uri=uri)

    with open(rttm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            # RTTM format: SPEAKER file 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
            record_type = parts[0]
            if record_type != 'SPEAKER':
                continue

            file_id = parts[1]
            start_time = float(parts[3])
            duration = float(parts[4])
            speaker_id = parts[7]

            segment = Segment(start_time, start_time + duration)
            annotation[segment] = speaker_id

    return annotation


def segments_to_annotation(segments: List[Dict], uri: str) -> Annotation:
    """
    Convert Senko segments format to pyannote Annotation.

    Args:
        segments: List of segment dictionaries from Senko
        uri: URI for the annotation

    Returns:
        pyannote.core.Annotation object
    """
    annotation = Annotation(uri=uri)

    for segment in segments:
        start = segment['start']
        end = segment['end']
        speaker = segment['speaker']

        seg = Segment(start, end)
        annotation[seg] = speaker

    return annotation


def find_audio_and_rttm_files(subset_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find corresponding audio and RTTM files in a subset directory (dev or test).

    Args:
        subset_dir: Path to subset directory (dev or test)

    Returns:
        List of (audio_file, rttm_file) tuples
    """
    file_pairs = []

    # Look for audio files in audio/ subdirectory
    audio_dir = subset_dir / "audio"
    if not audio_dir.exists():
        print(f"Warning: {audio_dir} does not exist")
        return file_pairs

    audio_files = list(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    # Find corresponding RTTM files
    for audio_file in audio_files:
        # Try different possible locations for RTTM files
        rttm_candidates = [
            subset_dir / f"{audio_file.stem}.rttm",  # In subset root
            audio_dir / f"{audio_file.stem}.rttm",   # In audio dir
            subset_dir / "rttm" / f"{audio_file.stem}.rttm",  # In rttm subdir
        ]

        for rttm_file in rttm_candidates:
            if rttm_file.exists():
                file_pairs.append((audio_file, rttm_file))
                break
        else:
            print(f"Warning: No RTTM file found for {audio_file.name}")

    return file_pairs


def setup_voxconverse_dataset(base_dir: Path = None) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Setup VoxConverse dataset from dev and test directories.

    Args:
        base_dir: Base directory containing dev and test folders (defaults to current directory)

    Returns:
        Dictionary with 'dev' and 'test' keys containing lists of (audio_file, rttm_file) tuples
    """
    if base_dir is None:
        base_dir = Path.cwd()

    datasets = {}

    for subset in ['dev', 'test']:
        subset_dir = base_dir / subset
        if subset_dir.exists():
            file_pairs = find_audio_and_rttm_files(subset_dir)
            datasets[subset] = file_pairs
            print(f"Found {len(file_pairs)} {subset} file pairs")
        else:
            print(f"Warning: {subset_dir} does not exist")
            datasets[subset] = []

    return datasets


def preprocess_audio_for_senko(audio_file: Path, output_dir: Path) -> Path:
    """
    Preprocess audio file to meet Senko requirements (16kHz mono 16-bit WAV).

    Args:
        audio_file: Input audio file
        output_dir: Directory to save preprocessed file

    Returns:
        Path to preprocessed WAV file
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        print("Warning: librosa and soundfile not available for audio preprocessing")
        print("Assuming audio is already in correct format...")
        return audio_file

    output_file = output_dir / f"{audio_file.stem}_processed.wav"

    if output_file.exists():
        return output_file

    # Load audio and convert to 16kHz mono
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)

    # Save as 16-bit WAV
    sf.write(output_file, audio, 16000, subtype='PCM_16')

    return output_file


def evaluate_file(audio_file: Path, rttm_file: Path, diarizer,
                 results_dir: Path, preprocess_audio: bool = True) -> Dict:
    """
    Evaluate a single audio file.

    Args:
        audio_file: Path to audio file
        rttm_file: Path to reference RTTM file
        diarizer: Senko diarizer instance
        results_dir: Directory to save results
        preprocess_audio: Whether to preprocess audio for Senko

    Returns:
        Dictionary with evaluation results
    """
    file_id = audio_file.stem
    print(f"\nProcessing {file_id}...")

    try:
        # Preprocess audio if needed
        if preprocess_audio and not audio_file.name.endswith('_processed.wav'):
            processed_audio_dir = results_dir / "processed_audio"
            processed_audio_dir.mkdir(exist_ok=True)
            processed_audio = preprocess_audio_for_senko(audio_file, processed_audio_dir)
        else:
            processed_audio = audio_file

        # Run Senko diarization
        print(f"Running Senko on {processed_audio.name}")
        result_data = diarizer.diarize(str(processed_audio), generate_colors=False)

        if result_data is None:
            print(f"No speakers detected in {file_id}")
            return {
                'file_id': file_id,
                'error': 'No speakers detected',
                'der': None,
                'senko_speakers': 0,
                'reference_speakers': 0
            }

        # Get merged segments (cleaned output)
        merged_segments = result_data["merged_segments"]
        senko_speakers = result_data["merged_speakers_detected"]
        timing_stats = result_data["timing_stats"]

        # Convert to pyannote Annotation
        hypothesis = segments_to_annotation(merged_segments, file_id)

        # Load reference annotation
        reference = load_rttm_file(rttm_file, file_id)
        reference_speakers = len(reference.labels())

        # Save Senko output as RTTM for inspection
        output_rttm = results_dir / f"{file_id}_senko.rttm"
        senko.save_rttm(merged_segments, str(processed_audio), str(output_rttm))

        # Calculate DER with 0.25s collar (standard for VoxConverse)
        metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
        components = metric(reference, hypothesis, detailed=True)
        der = components['diarization error rate']

        print(f"  Senko speakers: {senko_speakers}, Reference speakers: {reference_speakers}")
        print(f"  Processing time: {timing_stats.get('total_time', 0):.2f}s")
        print(f"  DER: {der:.3f}")
        print(f"  Components: FA={components.get('false alarm', 0):.1f}, "
              f"Miss={components.get('missed detection', 0):.1f}, "
              f"Conf={components.get('confusion', 0):.1f}")

        return {
            'file_id': file_id,
            'der': der,
            'senko_speakers': senko_speakers,
            'reference_speakers': reference_speakers,
            'processing_time': timing_stats.get('total_time', 0),
            'components': components,
            'error': None
        }

    except Exception as e:
        print(f"Error processing {file_id}: {str(e)}")
        return {
            'file_id': file_id,
            'error': str(e),
            'der': None,
            'senko_speakers': None,
            'reference_speakers': None
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Senko on VoxConverse dataset')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu', 'auto'], default='auto',
                       help='Device for Senko processing')
    parser.add_argument('--results_dir', type=Path, default='./senko_evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--subset', choices=['dev', 'test', 'both'], default='both',
                       help='Which subset to evaluate')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--no_preprocess', action='store_true',
                       help='Skip audio preprocessing (assume correct format)')

    args = parser.parse_args()

    # Create results directory
    args.results_dir.mkdir(exist_ok=True)

    # Check that dev and test directories exist
    if not Path("dev").exists() and not Path("test").exists():
        print("Error: Neither 'dev' nor 'test' directories found in current directory!")
        print("Please ensure you have extracted the VoxConverse dataset:")
        print("  ./dev/audio/*.wav and ./dev/*.rttm")
        print("  ./test/audio/*.wav and ./test/*.rttm")
        sys.exit(1)

    # Initialize Senko diarizer
    print(f"Initializing Senko diarizer (device: {args.device})...")
    diarizer = senko.Diarizer(torch_device=args.device, warmup=True, quiet=False)
    print("Diarizer ready!")

    # Setup dataset
    datasets = setup_voxconverse_dataset()

    # Collect all file pairs based on subset selection
    all_file_pairs = []
    if args.subset in ['dev', 'both'] and 'dev' in datasets:
        all_file_pairs.extend([(f, r, 'dev') for f, r in datasets['dev']])
    if args.subset in ['test', 'both'] and 'test' in datasets:
        all_file_pairs.extend([(f, r, 'test') for f, r in datasets['test']])

    if not all_file_pairs:
        print("Error: No audio/RTTM file pairs found!")
        sys.exit(1)

    # Limit number of files if specified
    if args.max_files:
        all_file_pairs = all_file_pairs[:args.max_files]

    print(f"\nEvaluating {len(all_file_pairs)} files...")

    # Evaluate each file
    results = {'dev': [], 'test': []}
    global_metrics = {'dev': DiarizationErrorRate(collar=0.25, skip_overlap=False),
                     'test': DiarizationErrorRate(collar=0.25, skip_overlap=False)}

    for i, (audio_file, rttm_file, subset) in enumerate(all_file_pairs):
        print(f"\n=== {subset.upper()} File {i+1}/{len(all_file_pairs)} ===")

        result = evaluate_file(
            audio_file, rttm_file, diarizer, args.results_dir,
            preprocess_audio=not args.no_preprocess
        )
        result['subset'] = subset
        results[subset].append(result)

        # Add to global metric if successful
        if result['der'] is not None:
            try:
                reference = load_rttm_file(rttm_file, result['file_id'])
                # Load Senko output
                senko_rttm = args.results_dir / f"{result['file_id']}_senko.rttm"
                if senko_rttm.exists():
                    hypothesis = load_rttm_file(senko_rttm, result['file_id'])
                    global_metrics[subset](reference, hypothesis)
            except Exception as e:
                print(f"Warning: Could not add {result['file_id']} to global metric: {e}")

    # Calculate overall statistics
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")

    summary = {}

    for subset in ['dev', 'test']:
        if not results[subset]:
            continue

        successful_results = [r for r in results[subset] if r['der'] is not None]

        if successful_results:
            mean_der = np.mean([r['der'] for r in successful_results])
            median_der = np.median([r['der'] for r in successful_results])
            global_der = abs(global_metrics[subset])

            # Processing time stats
            times = [r.get('processing_time', 0) for r in successful_results if r.get('processing_time')]
            mean_time = np.mean(times) if times else 0

            print(f"\n{subset.upper()} SET:")
            print(f"  Files processed successfully: {len(successful_results)}/{len(results[subset])}")
            print(f"  Mean DER (per-file average): {mean_der:.3f}")
            print(f"  Median DER: {median_der:.3f}")
            print(f"  Global DER (accumulated): {global_der:.3f}")
            print(f"  Average processing time: {mean_time:.2f}s")

            # Speaker count analysis
            senko_counts = [r['senko_speakers'] for r in successful_results if r['senko_speakers'] is not None]
            ref_counts = [r['reference_speakers'] for r in successful_results if r['reference_speakers'] is not None]
            if senko_counts and ref_counts:
                print(f"  Average Senko speakers: {np.mean(senko_counts):.1f}")
                print(f"  Average reference speakers: {np.mean(ref_counts):.1f}")

            summary[subset] = {
                'global_der': global_der,
                'mean_der': mean_der,
                'median_der': median_der,
                'successful_files': len(successful_results),
                'total_files': len(results[subset]),
                'mean_processing_time': mean_time
            }

    # Save detailed results
    results_file = args.results_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary,
            'device': args.device,
            'subset_evaluated': args.subset,
            'file_results': {
                'dev': results['dev'],
                'test': results['test']
            }
        }, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")
    print(f"Individual RTTM outputs saved in: {args.results_dir}")


if __name__ == "__main__":
    main()