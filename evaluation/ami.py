# Credit: Script based on https://github.com/FluidInference/FluidAudio/tree/main/Sources/FluidAudioCLI/DatasetParsers

# uv pip install requests
# python ami.py --split test --variant sdm|ihm --output ./test/

# Downloads AMI test set

import os
import sys
import argparse
import requests
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import shutil

class AMIDatasetDownloader:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; AMI Dataset Downloader)'
        })

    OFFICIAL_SPLITS = {
        # https://github.com/pyannote/AMI-diarization-setup/blob/main/lists/test.meetings.txt
        # Full-corpus-ASR partition: https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml
        'test': ['IS1009', 'ES2004', 'TS3003', 'EN2002']
    }

    def get_meeting_sessions(self, meeting_id: str, split: str) -> List[str]:
        # sessions a, b, c, d
        return [f"{meeting_id}{session}" for session in ['a', 'b', 'c', 'd']]

    def download_ami_annotations(self, output_dir: Path, force: bool = False) -> bool:
        """Download AMI annotations archive and extract."""
        annotations_dir = output_dir / "ami_public_1.6.2"
        segments_dir = annotations_dir / "segments"
        meetings_file = annotations_dir / "corpusResources" / "meetings.xml"

        if not force and segments_dir.exists() and meetings_file.exists():
            print(f"AMI annotations already exist in {annotations_dir}")
            return True

        print("Downloading AMI annotations from Edinburgh University...")

        # Create directories
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # Download annotations ZIP
        zip_url = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
        zip_file = annotations_dir / "ami_public_manual_1.6.2.zip"

        print(f"Downloading AMI manual annotations archive (22MB)...")

        try:
            response = self.session.get(zip_url, stream=True)
            response.raise_for_status()

            with open(zip_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print("Extracting AMI annotations archive...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(annotations_dir)

            # Clean up ZIP file
            zip_file.unlink()

            # Verify extraction
            if segments_dir.exists() and meetings_file.exists():
                print("AMI annotations download and extraction completed")
                return True
            else:
                print("Warning: Extraction completed but expected files not found")
                return False

        except Exception as e:
            print(f"Failed to download AMI annotations: {e}")
            return False

    def download_ami_audio_file(self, meeting_id: str, variant: str) -> Optional[bytes]:
        """Download a single AMI audio file."""
        base_urls = [
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus",
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus",
        ]

        # SDM => Array1-01.wav
        # IHM => Mix-Headset.wav
        file_pattern = 'Array1-01.wav' if variant.lower() == 'sdm' else 'Mix-Headset.wav'

        for base_url in base_urls:
            url = f"{base_url}/{meeting_id}/audio/{meeting_id}.{file_pattern}"

            try:
                print(f"     Downloading from: {url}")
                response = self.session.get(url, stream=True)

                if response.status_code == 200:
                    content = response.content
                    if len(content) > 1000 and (b'RIFF' in content[:12] or b'fLaC' in content[:4]):
                        size_mb = len(content) / (1024 * 1024)
                        print(f"     Downloaded {size_mb:.1f} MB")
                        return content
                    else:
                        print(f"     Downloaded file is not valid audio")
                        continue
                elif response.status_code == 404:
                    print(f"     File not found (HTTP 404) - trying next URL...")
                    continue
                else:
                    print(f"     HTTP error: {response.status_code} - trying next URL...")
                    continue

            except Exception as e:
                print(f"     Download error: {e} - trying next URL...")
                continue

        print("     Failed to download from all available URLs")
        return None

    def parse_speaker_mapping(self, meetings_file: Path, meeting_id: str) -> Optional[Dict[str, str]]:
        """Parse speaker mapping from meetings.xml file."""
        try:
            tree = ET.parse(meetings_file)
            root = tree.getroot()

            for meeting in root.findall('.//meeting'):
                if meeting.get('observation') == meeting_id:
                    mapping = {}
                    for speaker in meeting.findall('.//speaker'):
                        nxt_agent = speaker.get('nxt_agent')
                        global_name = speaker.get('global_name')
                        if nxt_agent and global_name:
                            mapping[nxt_agent] = global_name

                    return {
                        'A': mapping.get('A', 'UNKNOWN'),
                        'B': mapping.get('B', 'UNKNOWN'),
                        'C': mapping.get('C', 'UNKNOWN'),
                        'D': mapping.get('D', 'UNKNOWN')
                    }

            return None

        except Exception as e:
            print(f"Error parsing speaker mapping: {e}")
            return None

    def parse_segments_file(self, segments_file: Path, speaker_code: str) -> List[Dict]:
        """Parse AMI segments XML file."""
        try:
            tree = ET.parse(segments_file)
            root = tree.getroot()

            segments = []
            for segment in root.findall('.//segment'):
                segment_id = segment.get('{http://nite.sourceforge.net/}id')
                start_time = segment.get('transcriber_start')
                end_time = segment.get('transcriber_end')

                if segment_id and start_time and end_time:
                    try:
                        start = float(start_time)
                        end = float(end_time)
                        duration = end - start

                        segments.append({
                            'segment_id': segment_id,
                            'speaker_code': speaker_code,
                            'start_time': start,
                            'end_time': end,
                            'duration': duration
                        })
                    except ValueError:
                        continue

            return segments

        except Exception as e:
            print(f"Error parsing segments file: {e}")
            return []

    def convert_to_rttm(self, segments: List[Dict], speaker_mapping: Dict[str, str],
                       meeting_id: str, output_file: Path):
        """Convert segments to RTTM format."""
        with open(output_file, 'w') as f:
            for segment in segments:
                speaker_code = segment['speaker_code']
                participant_id = speaker_mapping.get(speaker_code, f'SPEAKER_{speaker_code}')

                # RTTM format: SPEAKER file 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
                f.write(f"SPEAKER {meeting_id} 1 {segment['start_time']:.3f} "
                       f"{segment['duration']:.3f} <NA> <NA> {participant_id} <NA> <NA>\n")

    def process_meeting(self, meeting_id: str, variant: str, annotations_dir: Path,
                       output_dir: Path, force: bool = False) -> bool:
        """Process a single AMI meeting."""
        print(f"\nProcessing {meeting_id}...")

        # Output paths
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_file = audio_dir / f"{meeting_id}.wav"
        rttm_file = output_dir / f"{meeting_id}.rttm"

        # Skip if files exist and not forcing
        if not force and audio_file.exists() and rttm_file.exists():
            print(f"   Skipping {meeting_id} (already exists)")
            return True

        # Download audio file
        if force or not audio_file.exists():
            print(f"   Downloading audio file...")
            audio_data = self.download_ami_audio_file(meeting_id, variant)
            if audio_data is None:
                print(f"   Failed to download audio for {meeting_id}")
                return False

            with open(audio_file, 'wb') as f:
                f.write(audio_data)
            print(f"   Downloaded {meeting_id}.wav")

        # Process annotations
        if force or not rttm_file.exists():
            print(f"   Processing annotations...")

            # Get speaker mapping
            meetings_file = annotations_dir / "corpusResources" / "meetings.xml"
            speaker_mapping = self.parse_speaker_mapping(meetings_file, meeting_id)

            if speaker_mapping is None:
                print(f"   No speaker mapping found for {meeting_id}")
                return False

            print(f"   Speaker mapping: A={speaker_mapping['A']}, B={speaker_mapping['B']}, "
                  f"C={speaker_mapping['C']}, D={speaker_mapping['D']}")

            # Parse all speaker segments
            all_segments = []
            segments_dir = annotations_dir / "segments"

            for speaker_code in ['A', 'B', 'C', 'D']:
                segment_file = segments_dir / f"{meeting_id}.{speaker_code}.segments.xml"

                if segment_file.exists():
                    segments = self.parse_segments_file(segment_file, speaker_code)
                    all_segments.extend(segments)
                    print(f"   Loaded {len(segments)} segments for speaker {speaker_code}")

            # Sort by start time
            all_segments.sort(key=lambda x: x['start_time'])

            # Convert to RTTM
            self.convert_to_rttm(all_segments, speaker_mapping, meeting_id, rttm_file)
            print(f"   Created {meeting_id}.rttm with {len(all_segments)} segments")

        return True

    def download_official_split(self, split: str, variant: str = "sdm",
                              output_dir: Path = None, force: bool = False) -> bool:

        if split not in self.OFFICIAL_SPLITS:
            raise ValueError(f"Invalid split. Must be one of: {list(self.OFFICIAL_SPLITS.keys())}")

        if output_dir is None:
            output_dir = Path(f"./{split}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        variant_name = "Single Distant Microphone" if variant == "sdm" else "Individual Headset Microphones"
        print(f"Downloading AMI {variant_name} - {split.upper()} set")
        print(f"Target directory: {output_dir}")

        # Get all meeting sessions for this split
        all_meetings = []
        for meeting_base in self.OFFICIAL_SPLITS[split]:
            sessions = self.get_meeting_sessions(meeting_base, split)
            all_meetings.extend(sessions)

        print(f"Official {split} set contains {len(all_meetings)} meetings:")
        print(f"  Base meetings: {', '.join(self.OFFICIAL_SPLITS[split])}")
        print(f"  All sessions: {', '.join(all_meetings)}")

        # Download annotations first
        annotations_dir = output_dir / "ami_public_1.6.2"
        if not self.download_ami_annotations(output_dir, force):
            return False

        # Process each meeting
        successful = 0
        failed = 0

        for meeting_id in all_meetings:
            try:
                if self.process_meeting(meeting_id, variant, annotations_dir, output_dir, force):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing {meeting_id}: {e}")
                failed += 1

        print(f"\nAMI {split.upper()} set download completed")
        print(f"Successfully processed: {successful} files")
        print(f"Failed: {failed} files")
        print(f"Total files: {successful + failed}/{len(all_meetings)}")

        if successful > 0:
            print(f"\nDataset ready! Structure:")
            print(f"  {output_dir}/")
            print(f"    audio/*.wav")
            print(f"    *.rttm")
            print(f"\nTo evaluate: python voxconverse.py --subset {split}")

        return successful > 0


def main():
    parser = argparse.ArgumentParser(description='Download AMI dataset using official splits')
    parser.add_argument('--split', choices=['test'], required=True,
                       help='Official AMI split to download')
    parser.add_argument('--variant', choices=['sdm', 'ihm'], default='sdm',
                       help='AMI variant: sdm (Single Distant Mic) or ihm (Individual Headset Mics)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output directory (default: ./{split})')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if files exist')

    args = parser.parse_args()

    # Set default output directory
    if args.output is None:
        args.output = Path(f"./{args.split}")

    # Download dataset
    downloader = AMIDatasetDownloader()
    success = downloader.download_official_split(
        split=args.split,
        variant=args.variant,
        output_dir=args.output,
        force=args.force
    )

    if not success:
        print("Download failed!")
        sys.exit(1)

    print("\nDownload completed successfully!")

    # Show official split info
    print(f"\n=== Official AMI {args.split.upper()} Set ===")
    if args.split == 'test':
        print("Test set: 4 base meetings × 4 sessions = 16 files")
        print("This is the standard evaluation set used in papers")


if __name__ == "__main__":
    main()
