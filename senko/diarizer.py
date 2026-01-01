from .warning_filters import apply_warning_filters
apply_warning_filters()

import os
import yaml
import time
import wave
import ctypes
import psutil
import warnings
import numpy as np
from termcolor import colored

from . import config
from .colors import generate_speaker_colors
from .utils import time_method, suppress_stdout_stderr, timed_operation
if not config.DARWIN:
    import torch
    torch_version_str = torch.__version__.split('+')[0]  # Remove any suffix like '+cu126'
    torch_version = tuple(map(int, torch_version_str.split('.')[:2]))
    use_legacy_tf32 = torch_version < (2, 9)
    
    def set_fp32_precision(mode='ieee'):
        if use_legacy_tf32:
            allow_tf32 = (mode == 'tf32')
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        else:
            torch.backends.cuda.matmul.fp32_precision = mode
            torch.backends.cudnn.fp32_precision = mode

class AudioFormatError(Exception):
    """Raised when audio file is not in the required 16kHz mono 16-bit WAV format"""
    pass

class Diarizer:
    def __init__(self, device='auto', vad='auto', clustering='auto', warmup=True, quiet=True, mer_cos=None):

        self.quiet = quiet

        ############
        ## Device ##
        ############

        if config.DARWIN:
            self.device = 'coreml'
            self.torch_device = None
        else:
            self.torch_device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) if device == 'auto' else torch.device(device)
            self.device = self.torch_device.type
            self.logical_cores = psutil.cpu_count(logical=True)

        self._print(f"Using device: {self.device}")

        #########
        ## VAD ##
        #########

        if vad not in ['auto', 'pyannote', 'silero']:
            raise ValueError(f"Invalid VAD type: {vad}. Must be ['auto', 'pyannote'" + (", 'silero']" if not self.device == 'coreml' else "]"))

        if self.device == 'coreml' and vad == 'silero':
            raise ValueError("Only pyannote VAD (CoreML) available on macOS")

        # Determine VAD model type based on parameter or auto-selection
        self.vad_model_type = ('pyannote' if self.device in ['cuda', 'coreml'] else 'silero') if vad == 'auto' else vad.lower()

        # If pyannote, check if it's actually available
        if self.vad_model_type == 'pyannote' and self.device != 'coreml':
            try:
                import pyannote.audio
            except ModuleNotFoundError:
                self.vad_model_type = 'silero'

        # Pyannote VAD
        if self.vad_model_type == 'pyannote':
            # CUDA
            if self.device == 'cuda':
                try:
                    from pyannote.audio.utils.reproducibility import ReproducibilityWarning
                    warnings.filterwarnings("ignore", category=ReproducibilityWarning)
                except ImportError:
                    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*", category=UserWarning)

                from pyannote.audio.pipelines import VoiceActivityDetection
                from pyannote.audio import Model

                # Allow-list needed classes for safe weights-only loading on PyTorch 2.6+
                from pyannote.audio.core.task import Specifications, Problem, Resolution
                from pyannote.audio.models.segmentation import PyanNet
                with torch.serialization.safe_globals([torch.torch_version.TorchVersion, Specifications, Problem, Resolution, PyanNet]):
                    model = Model.from_pretrained(config.PYANNOTE_SEGMENTATION_PT_MODEL_PATH, map_location=self.torch_device)
                self.vad_pipeline_pyannote_cuda = VoiceActivityDetection(segmentation=model)
                self.vad_pipeline_pyannote_cuda.instantiate({
                    "min_duration_on": 0.25,  # Remove speech regions shorter than 250ms
                    "min_duration_off": 0.1   # Fill non-speech regions shorter than 100ms
                })
                self.vad_pipeline_pyannote_cuda.to(self.torch_device)
            # CoreML
            else:
                from .vad_coreml import VADProcessorCoreML
                self.vad_processor_pyannote_coreml = VADProcessorCoreML(
                    lib_path=config.VAD_COREML_LIB_PATH,
                    model_path=config.PYANNOTE_SEGMENTATION_COREML_MODEL_PATH
                )

        # Silero VAD
        else:
            from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
            self.vad_model_silero = load_silero_vad()
            self.read_audio_silero = read_audio
            self.get_speech_timestamps_silero = get_speech_timestamps

        self._print(f'Using {self.vad_model_type} VAD' + (' (CoreML)' if self.device == 'coreml' else ''))

        # silero sets torch threads to 1; set it back to make full use of all cores
        if self.vad_model_type == 'silero':
            self._set_torch_num_threads()

        ##############################
        ## Fbank feature extraction ##
        ##############################

        # Extract features on GPU through kaldifeat
        if self.device == 'cuda':
            try:
                import kaldifeat
                self.kaldifeat = kaldifeat
                # Configure kaldifeat options to match C++ extractor
                opts = self.kaldifeat.FbankOptions()
                opts.frame_opts.samp_freq = 16000
                opts.frame_opts.frame_shift_ms = 10.0
                opts.frame_opts.frame_length_ms = 25.0
                opts.frame_opts.dither = 0.0
                opts.frame_opts.preemph_coeff = 0.97
                opts.frame_opts.remove_dc_offset = True
                opts.frame_opts.window_type = "povey"
                opts.frame_opts.round_to_power_of_two = True
                opts.frame_opts.blackman_coeff = 0.42
                opts.frame_opts.snip_edges = True
                opts.mel_opts.num_bins = 80
                opts.mel_opts.low_freq = 20
                opts.mel_opts.high_freq = 0
                opts.mel_opts.vtln_low = 100
                opts.mel_opts.vtln_high = -500
                opts.use_energy = False
                opts.energy_floor = 1.0
                opts.raw_energy = True
                opts.use_log_fbank = True
                opts.use_power = True
                opts.device = self.torch_device
                self.kaldifeat_fbank = self.kaldifeat.Fbank(opts)
                self.use_gpu_fbank = True
            except ImportError:
                self.use_gpu_fbank = False
        
        # Extract features on CPU using C++ lib
        if not getattr(self, 'use_gpu_fbank', False):
            self.lib = ctypes.CDLL(config.FBANK_LIB_PATH)

            class FbankFeatures(ctypes.Structure):
                _fields_ = [
                    ("data", ctypes.POINTER(ctypes.c_float)),
                    ("frames_per_subsegment", ctypes.POINTER(ctypes.c_size_t)),
                    ("subsegment_offsets", ctypes.POINTER(ctypes.c_size_t)),
                    ("num_subsegments", ctypes.c_size_t),
                    ("total_frames", ctypes.c_size_t),
                    ("feature_dim", ctypes.c_size_t)
                ]

            self.FbankFeatures = FbankFeatures
            self.lib.create_fbank_extractor.restype = ctypes.c_void_p
            self.lib.destroy_fbank_extractor.argtypes = [ctypes.c_void_p]
            self.lib.destroy_fbank_extractor.restype = None
            self.lib.extract_fbank_features.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
            ]
            self.lib.extract_fbank_features.restype = FbankFeatures
            self.lib.free_fbank_features.argtypes = [ctypes.POINTER(FbankFeatures)]
            self.lib.free_fbank_features.restype = None
            self.fbank_extractor = self.lib.create_fbank_extractor()

        ################
        ## Embeddings ##
        ################

        # Load embedding model
        with timed_operation("Loading embedding model ........", self.quiet):
            # CoreML
            if self.device == 'coreml':
                with suppress_stdout_stderr():
                    import coremltools as ct
                    self.embeddings_model = ct.models.MLModel(config.EMBEDDINGS_COREML_PATH)
                    self.coreml_fixed_frames = 150
                    self.coreml_batch_size = 16
            # CUDA
            elif self.device == 'cuda':
                self.embeddings_model = torch.jit.load(config.EMBEDDINGS_JIT_CUDA_MODEL_PATH, map_location=self.torch_device)
                self.embeddings_model.eval()
            # CPU
            else:
                from .camplusplus import CAMPPlus
                self.embeddings_model = CAMPPlus(feat_dim=80, embedding_size=192)
                self.embeddings_model.load_state_dict(torch.load(config.EMBEDDINGS_PT_MODEL_PATH, map_location=self.torch_device, weights_only=True))
                self.embeddings_model.eval()
                self.embeddings_model.to(self.torch_device)

        # Warm up embedding model
        if warmup:
            with timed_operation("Warming up embedding model .....", self.quiet):
                if self.device == 'coreml':
                    for i in range(64):
                        dummy_input = np.random.randn(self.coreml_batch_size, self.coreml_fixed_frames, 80).astype(np.float32)
                        _ = self.embeddings_model.predict({'input_features': dummy_input})
                else:
                    with torch.no_grad():
                        for i in range(4):
                            dummy = torch.randn(80, 148, 80, device=self.torch_device)
                            _ = self.embeddings_model(dummy)

        ################
        ## Clustering ##
        ################

        with open(config.SPECTRAL_YAML, 'r') as spectral_yaml, open(config.UMAP_HDBSCAN_YAML, 'r') as umap_hdbscan_yaml:
            self.spectral_config = yaml.safe_load(spectral_yaml)
            self.umap_hdbscan_config = yaml.safe_load(umap_hdbscan_yaml)

            if mer_cos is not None:
                if not (0 < mer_cos <= 1):
                    raise ValueError("mer_cos must be > 0 and <= 1")
                self.spectral_config['cluster']['args']['mer_cos'] = mer_cos
                self.umap_hdbscan_config['cluster']['args']['mer_cos'] = mer_cos

            # Determine clustering location based on parameter or auto-selection
            if self.device != 'cuda':
                # Non-CUDA devices always use CPU clustering
                use_gpu_clustering = False
            else:
                # Windows builds don't ship RAPIDS; always fall back to CPU clustering.
                if config.WINDOWS:
                    if clustering.lower() == 'gpu':
                        self._print("Warning: GPU clustering requested on Windows, but RAPIDS is not installed. Falling back to CPU clustering.")
                    use_gpu_clustering = False
                else:
                    # CUDA devices can choose between GPU and CPU clustering
                    cuda_compute_capable = torch.cuda.get_device_capability()[0] >= 7
                    if clustering == 'auto':
                        use_gpu_clustering = cuda_compute_capable
                    elif clustering.lower() == 'gpu':
                        if cuda_compute_capable:
                            use_gpu_clustering = True
                        else:
                            self._print("Warning: GPU clustering requested but CUDA compute capability < 7.0. Falling back to CPU clustering.")
                            use_gpu_clustering = False
                    elif clustering.lower() == 'cpu':
                        use_gpu_clustering = False
                    else:
                        raise ValueError(f"Invalid clustering type: {clustering}. Must be 'auto', 'gpu', or 'cpu'")

            if use_gpu_clustering:
                try:
                    from .cluster.cluster_gpu import CommonClustering as ClusteringClass
                    self.clustering_location = 'gpu'
                except ImportError:
                    self._print("Warning: RAPIDS is unavailable. Falling back to CPU clustering.")
                    from .cluster.cluster_cpu import CommonClustering as ClusteringClass
                    self.clustering_location = 'cpu'
            else:
                from .cluster.cluster_cpu import CommonClustering as ClusteringClass
                self.clustering_location = 'cpu'

            self.spectral_cluster = ClusteringClass(**self.spectral_config['cluster']['args'])
            self.umap_hdbscan_cluster = ClusteringClass(**self.umap_hdbscan_config['cluster']['args'])

        self._print(f'Using {self.clustering_location.upper()} clustering')

        # Warmup clustering objects
        if warmup:
            with timed_operation("Warming up clustering objects ...", self.quiet):
                dummy_embeddings = np.random.randn(4250, 192).astype(np.float32)
                with suppress_stdout_stderr():
                    _ = self.umap_hdbscan_cluster(dummy_embeddings)

    def __del__(self):
        if hasattr(self, 'fbank_extractor') and self.fbank_extractor:
            self.lib.destroy_fbank_extractor(self.fbank_extractor)

    def diarize(self, wav_path, accurate=None, generate_colors=False):
        self._timing_stats = {}
        total_start = time.time()

        # Print filename
        self._print(f"\n    \033[38;2;120;167;214m{os.path.basename(wav_path)}\033[0m")

        # Verify correct format (16kHz mono 16-bit WAV)
        with wave.open(wav_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            bit_depth = wav_file.getsampwidth() * 8  # getsampwidth returns bytes, multiply by 8 for bits

            if channels != 1 or sample_rate != 16000 or bit_depth != 16:
                error_msg = f"\tError: Audio file must be 16kHz mono 16-bit WAV format.\n"
                error_msg += f"\tCurrent format: {sample_rate}Hz, {channels} channel(s), {bit_depth}-bit\n\n"
                error_msg += "\tTo convert your file to the correct format, run:\n"
                error_msg += f"\tffmpeg -i {wav_path} -acodec pcm_s16le -ac 1 -ar 16000 {os.path.splitext(wav_path)[0]}_mono.wav\n"
                self._print(colored(error_msg, 'red'))
                raise AudioFormatError(f"Audio file must be 16kHz mono 16-bit WAV format. Current: {sample_rate}Hz, {channels} channel(s), {bit_depth}-bit")

        # VAD (voice activity detection)
        vad_segments = self._perform_vad(wav_path)

        if os.getenv("DUMP_VAD"):
            print(f"\n    VAD segments ({len(vad_segments)}):")
            if vad_segments:
                for start, end in vad_segments:
                    print(f"    - {start:.3f} -> {end:.3f}")
            else:
                print("    (none)")

        if not vad_segments:
            self._print(colored("\n    No speakers detected in the audio!\n", 'yellow'))
            return None

        # Generate subsegments
        subsegments = self._generate_subsegments(vad_segments, accurate)

        # Extract Fbank feature for each subsegment
        features_flat, frames_per_subsegment, subsegment_offsets, feature_dim = self._extract_fbank_features(wav_path, subsegments)

        # Convert subsegment_offsets to integers for slicing
        subsegment_offsets = [int(offset) for offset in subsegment_offsets]

        # Generate CAM++ speaker embedding for each subsegment feature
        embeddings = self._generate_embeddings(features_flat, frames_per_subsegment, subsegment_offsets, feature_dim)

        # Cluster speaker embeddings
        raw_segments, merged_segments, centroids = self._perform_clustering(embeddings, subsegments)

        # Print diarization time
        total_time = round(time.time() - total_start, 2)
        self._timing_stats["total_time"] = total_time
        self._print(colored(f"\n    Total diarization time: ", 'light_cyan'), end="")
        self._print(f"{total_time:.2f}s\n")

        # Calculate unique speaker counts
        raw_speakers_detected = len(set(segment['speaker'] for segment in raw_segments))
        merged_speakers_detected = len(set(segment['speaker'] for segment in merged_segments))

        # Output dictionary
        result = {
            "raw_segments": raw_segments,
            "raw_speakers_detected": raw_speakers_detected,
            "merged_speakers_detected": merged_speakers_detected,
            "merged_segments": merged_segments,
            "speaker_centroids": centroids,
            "timing_stats": self._timing_stats,
            "vad": vad_segments,
        }

        # Generate 10 speaker color sets (if requested)
        if generate_colors:
            speaker_color_sets = {}
            for i in range(10):
                speaker_color_sets[str(i)] = generate_speaker_colors(merged_segments, i)
            result["speaker_color_sets"] = speaker_color_sets

        return result

    @time_method('vad_time', 'Voice activity detection')
    def _perform_vad(self, wav_path):
        if self.vad_model_type == 'pyannote':
            # CUDA
            if self.device == 'cuda':
                set_fp32_precision('ieee')
                vad_result = self.vad_pipeline_pyannote_cuda(wav_path)
                segments = [(segment.start, segment.end) for segment in vad_result.get_timeline()]
            # CoreML
            else:
                segments = self.vad_processor_pyannote_coreml.process_audio(wav_path)
        # Silero
        else:
            self._set_torch_num_threads(1)  # silero vad is single threaded

            wav = self.read_audio_silero(wav_path)
            speech_timestamps = self.get_speech_timestamps_silero(wav, self.vad_model_silero, threshold=0.55, min_speech_duration_ms=250, min_silence_duration_ms=100, return_seconds=False)

            # Convert from samples to seconds manually for full precision
            sample_rate = 16000
            segments = []
            for ts in speech_timestamps:
                start_sec = float(ts['start']) / sample_rate
                end_sec = float(ts['end']) / sample_rate
                segments.append((start_sec, end_sec))

            # Restore num threads to make full use of all CPU cores
            self._set_torch_num_threads()

        return segments

    def _generate_subsegments(self, vad_segments, accurate):
        # If accurate mode unspecified, turn it on only if on NVIDIA & pyannote VAD selected
        accurate = (self.vad_model_type == 'pyannote' and self.device == 'cuda') if accurate == None else accurate

        # Slightly shorter subsegments & smaller shift with accurate mode on
        segment_duration = 1.45 if accurate else 1.5
        shift = segment_duration / (3.0 if accurate else 2.5)

        subsegments = []

        for start, end in vad_segments:
            sub_start = start

            while sub_start + segment_duration < end:
                subsegments.append((sub_start, sub_start + segment_duration))
                sub_start += shift

            if sub_start < end:
                sub_start = min(end - segment_duration, sub_start)
                subsegments.append((sub_start, end))

        return subsegments

    @time_method('fbank_time', 'Fbank feature extraction')
    def _extract_fbank_features(self, wav_path, subsegments):
        # GPU path: use kaldifeat when available on CUDA
        if getattr(self, 'use_gpu_fbank', False):
            return self._extract_fbank_features_gpu(wav_path, subsegments)

        # Convert subsegments to flat array
        subseg_array = np.array(subsegments, dtype=np.float32).flatten()
        subseg_ptr = subseg_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Call C++ lib function
        wav_path_bytes = wav_path.encode('utf-8')
        features = self.lib.extract_fbank_features(self.fbank_extractor, wav_path_bytes, subseg_ptr, len(subsegments))

        # Convert results to numpy arrays
        total_floats = features.total_frames * features.feature_dim
        features_np = np.ctypeslib.as_array(features.data, shape=(total_floats,))
        features_copy = features_np.copy()

        frames_per_seg_np = np.ctypeslib.as_array(features.frames_per_subsegment, shape=(features.num_subsegments,))
        frames_per_seg_copy = frames_per_seg_np.copy()

        # Get the offsets
        subsegment_offsets_np = np.ctypeslib.as_array(features.subsegment_offsets, shape=(features.num_subsegments,))
        subsegment_offsets_copy = subsegment_offsets_np.copy()

        # Free the C++ allocated memory
        self.lib.free_fbank_features(ctypes.byref(features))

        return features_copy, frames_per_seg_copy, subsegment_offsets_copy, features.feature_dim

    def _extract_fbank_features_gpu(self, wav_path, subsegments):
        import torchaudio
        import torch.nn.functional as F
        
        sample_rate = 16000
        min_len = 400  # match C++ padding
        wav, sr = torchaudio.load(wav_path)  # shape: (1, num_samples) on CPU
        wav = wav.squeeze(0)  # drop channel dim to 1-D samples

        BATCH_SEGMENTS = 256

        features_list = []
        frames_per_subsegment = []
        subsegment_offsets = []
        offset = 0
        mel_bins = 80

        for b in range(0, len(subsegments), BATCH_SEGMENTS):
            batch_segments = subsegments[b:b + BATCH_SEGMENTS]
            segment_tensors = []
            for start_sec, end_sec in batch_segments:
                start = int(start_sec * sample_rate)
                length = int((end_sec - start_sec) * sample_rate)
                if length <= 0:
                    length = 1
                if start + length > wav.numel():
                    length = wav.numel() - start
                seg = wav[start:start + length]
                if seg.numel() < min_len:
                    pad = min_len - seg.numel()
                    seg = F.pad(seg, (0, pad))
                # move just this segment to GPU for fbank computation
                segment_tensors.append(seg.to(self.torch_device))

            # Compute features for this batch
            with torch.no_grad():
                feats_batch = self.kaldifeat_fbank(segment_tensors)

            if feats_batch:
                mel_bins = feats_batch[0].shape[1]

            for curr in feats_batch:
                frames = curr.size(0)
                if frames > 0:
                    curr = curr - curr.mean(dim=0, keepdim=True)
                curr_cpu = curr.cpu()
                curr_flat = curr_cpu.contiguous().view(-1)
                features_list.append(curr_flat)
                frames_per_subsegment.append(frames)
                subsegment_offsets.append(offset)
                offset += curr_flat.numel()

            # free GPU memory between batches
            torch.cuda.empty_cache()

        if features_list:
            features_tensor = torch.cat(features_list, dim=0)  # CPU tensor
        else:
            features_tensor = torch.empty(0, device='cpu', dtype=torch.float32)

        frames_np = np.array(frames_per_subsegment, dtype=np.int64)
        offsets_np = np.array(subsegment_offsets, dtype=np.int64)

        return features_tensor, frames_np, offsets_np, mel_bins

    @time_method('embeddings_time', 'Embeddings generation')
    def _generate_embeddings(self, features_flat, frames_per_subsegment, subsegment_offsets, feature_dim):
        if self.device == 'coreml':
            return self._generate_embeddings_coreml(features_flat, frames_per_subsegment, subsegment_offsets, feature_dim)

        if self.vad_model_type == 'pyannote' and self.device == 'cuda':
            set_fp32_precision('tf32')

        # Move features to torch device (avoid extra copy if already a tensor)
        if isinstance(features_flat, torch.Tensor):
            big_tensor = features_flat.to(self.torch_device)
        else:
            big_tensor = torch.from_numpy(features_flat).to(self.torch_device)

        feature_tensors = []
        for i, (frames, offset) in enumerate(zip(frames_per_subsegment, subsegment_offsets)):
            if frames == 0:
                feature_tensors.append(torch.zeros(1, feature_dim, device=self.torch_device))
                continue
            num_floats = int(frames * feature_dim)
            t = big_tensor[offset:offset + num_floats].view(int(frames), int(feature_dim))
            feature_tensors.append(t)

        BATCH_SIZE = 64
        embeddings = []
        for b in range(0, len(feature_tensors), BATCH_SIZE):
            curr_batch_size = min(BATCH_SIZE, len(feature_tensors) - b)
            curr_batch = feature_tensors[b:b + curr_batch_size]

            # Find max length in current batch
            max_len = max(t.size(0) for t in curr_batch)

            # Pad tensors in the batch to max_len
            padded_tensors = []
            for t in curr_batch:
                pad_amt = max_len - t.size(0)
                if pad_amt > 0:
                    # Pad along the frames dimension (dim=0 after unsqueeze)
                    t = torch.nn.functional.pad(t.unsqueeze(0), (0, 0, 0, pad_amt), "constant", 0.0)
                else:
                    t = t.unsqueeze(0)
                padded_tensors.append(t)

            # Stack into batch tensor
            batch = torch.cat(padded_tensors, 0)

            # Run CAM++ inference for this batch
            with torch.no_grad():
                batch_embeddings = self.embeddings_model(batch)
                embeddings.append(batch_embeddings)

        # Concatenate all embeddings and move to CPU
        final_embeddings = torch.cat(embeddings, 0).cpu().numpy()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return final_embeddings

    def _generate_embeddings_coreml(self, features_flat, frames_per_subsegment, subsegment_offsets, feature_dim):
        FIXED_FRAMES = self.coreml_fixed_frames  # 150

        # Prepare features
        processed_features = []
        for i, (frames, offset) in enumerate(zip(frames_per_subsegment, subsegment_offsets)):
            if frames == 0:
                # Zero features for empty segments
                features = np.zeros((FIXED_FRAMES, feature_dim), dtype=np.float32)
            else:
                # Extract features for this subsegment
                num_floats = int(frames * feature_dim)
                features = features_flat[offset:offset + num_floats].reshape(int(frames), int(feature_dim))

                # Prepare for fixed-size model
                if frames <= FIXED_FRAMES:
                    # Pad with zeros
                    padded = np.zeros((FIXED_FRAMES, feature_dim), dtype=np.float32)
                    padded[:frames, :] = features
                    features = padded
                else:
                    # If longer than FIXED_FRAMES (should never happen, but in case): center crop
                    start = (frames - FIXED_FRAMES) // 2
                    features = features[start:start + FIXED_FRAMES, :]

            processed_features.append(features)

        # Convert to numpy array
        all_features = np.array(processed_features)  # Shape: (num_segments, FIXED_FRAMES, 80)

        embeddings = []

        # Process in batches
        batch_size = self.coreml_batch_size
        num_segments = len(processed_features)
        for i in range(0, num_segments, batch_size):
            end_idx = min(i + batch_size, num_segments)
            actual_batch_size = end_idx - i

            # Create batch tensor (padded if necessary)
            batch_features = np.zeros((batch_size, FIXED_FRAMES, feature_dim), dtype=np.float32)
            batch_features[:actual_batch_size] = all_features[i:end_idx]

            # Run batched CoreML inference
            output = self.embeddings_model.predict({'input_features': batch_features})
            batch_embeddings = output['embeddings']

            # Only keep the embeddings we actually need (in case of padding)
            embeddings.extend(batch_embeddings[:actual_batch_size])

        # Stack all embeddings
        final_embeddings = np.vstack(embeddings)

        return final_embeddings

    @time_method('clustering_time', 'Clustering', last=True)
    def _perform_clustering(self, embeddings, subsegments):
        wav_length = subsegments[-1][1]  # Using last segment's end time

        with suppress_stdout_stderr():
            if wav_length < 1200.0:                          # Use spectral clustering for short audio (< 20 min)
                labels = self.spectral_cluster(embeddings)
            else:                                            # Use UMAP+HDBSCAN for longer audio
                labels = self.umap_hdbscan_cluster(embeddings)

        # Normalize labels to consecutive integers starting from 0 (these will be incremented by 1 when creating final speaker IDs)
        new_labels = np.zeros(len(labels), dtype=int)
        uniq = np.unique(labels)
        for i in range(len(uniq)):
            new_labels[labels==uniq[i]] = i

        # Calculate speaker centroids (voice fingerprints)
        centroids = {}
        for speaker_idx in range(len(uniq)):
            speaker_mask = new_labels == speaker_idx
            speaker_embeddings = embeddings[speaker_mask]
            centroid = np.mean(speaker_embeddings, axis=0)
            speaker_id = f"SPEAKER_{speaker_idx + 1:02d}"
            centroids[speaker_id] = centroid

        # Combine subsegments with their speaker labels
        seg_list = [(i, j) for i, j in zip(subsegments, new_labels)]

        # Create segments list with speaker information
        new_seg_list = []
        for i, seg in enumerate(seg_list):
            seg_st, seg_ed = seg[0]
            seg_st = max(0.0, float(seg_st))  # Ensure start is non-negative
            seg_ed = max(0.0, float(seg_ed))  # Ensure end is non-negative
            cluster_id = seg[1] + 1
            speaker_id = f"SPEAKER_{cluster_id:02d}"

            if i == 0:
                new_seg_list.append({"speaker": speaker_id, "start": seg_st, "end": seg_ed})
            elif speaker_id == new_seg_list[-1]["speaker"]:
                if seg_st > new_seg_list[-1]["end"]:
                    new_seg_list.append({"speaker": speaker_id, "start": seg_st, "end": seg_ed})
                else:
                    new_seg_list[-1]["end"] = seg_ed
            else:
                if seg_st < new_seg_list[-1]["end"]:
                    p = max(0.0, (new_seg_list[-1]["end"] + seg_st) / 2)  # Ensure midpoint is non-negative
                    new_seg_list[-1]["end"] = p
                    seg_st = p
                new_seg_list.append({"speaker": speaker_id, "start": seg_st, "end": seg_ed})

        # Raw (un-merged) segments
        raw_segments = new_seg_list

        # Merge segments
        merged_segments = self._merge_segments(new_seg_list)

        # Calculate total speaking time per speaker
        # Use raw_segments to ensure we account for all speakers, even those filtered out during merging
        speaker_times = {}
        for segment in raw_segments:
            speaker = segment['speaker']
            duration = segment['end'] - segment['start']
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration

        # Sort speakers by total speaking time (descending)
        sorted_speakers = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)

        # Create mapping from old to new speaker IDs
        speaker_mapping = {}
        for new_idx, (old_speaker, _) in enumerate(sorted_speakers, 1):
            speaker_mapping[old_speaker] = f"SPEAKER_{new_idx:02d}"

        # Update all segments with new speaker IDs (avoid double-mapping shared dicts)
        updated = set()
        for segment in raw_segments + merged_segments:
            seg_id = id(segment)
            if seg_id in updated:
                continue
            segment['speaker'] = speaker_mapping[segment['speaker']]
            updated.add(seg_id)

        # Update centroids with new speaker IDs
        new_centroids = {}
        for old_speaker, new_speaker in speaker_mapping.items():
            if old_speaker in centroids:
                new_centroids[new_speaker] = centroids[old_speaker]
        centroids = new_centroids

        return raw_segments, merged_segments, centroids

    def _merge_segments(self, segments):
        merged_segments = []
        current_segment = None

        # Step 1: Merge segments with gaps <= 4 seconds
        for segment in segments:
            if current_segment is None:
                current_segment = segment
            else:
                if current_segment['speaker'] == segment['speaker'] and segment['start'] - current_segment['end'] <= 4:
                    current_segment['end'] = segment['end']
                else:
                    merged_segments.append(current_segment)
                    current_segment = segment

        if current_segment is not None:
            merged_segments.append(current_segment)

        # Step 2: Remove segments shorter than or equal to 0.78 seconds
        i = 0
        while i < len(merged_segments):
            segment = merged_segments[i]
            if segment['end'] - segment['start'] <= 0.78:
                if i > 0 and i < len(merged_segments) - 1:
                    prev_segment = merged_segments[i - 1]
                    next_segment = merged_segments[i + 1]
                    if prev_segment['speaker'] == next_segment['speaker']:
                        prev_segment['end'] = next_segment['end']
                        merged_segments.pop(i + 1)
                merged_segments.pop(i)
            else:
                i += 1

        return merged_segments

    def _print(self, str="", end=None):
        if not self.quiet:
            if end == None:
                print(str)
            else:
                print(str, end=end)

    def _set_torch_num_threads(self, num=None):
        torch.set_num_threads(num if num is not None else self.logical_cores)
