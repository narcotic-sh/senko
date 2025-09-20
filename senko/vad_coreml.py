import ctypes
from ctypes import c_double, c_void_p, POINTER, c_int32, c_char_p

class VADProcessorCoreML:
    def __init__(self, lib_path, model_path, min_duration_on=0.25, min_duration_off=0.1):
        # Load library
        self.lib = ctypes.CDLL(lib_path)
        self.lib.vad_create.restype = c_void_p
        self.lib.vad_load_model.argtypes = [c_void_p, c_char_p]
        self.lib.vad_load_model.restype = c_int32
        self.lib.vad_set_min_duration_on.argtypes = [c_void_p, c_double]
        self.lib.vad_set_min_duration_off.argtypes = [c_void_p, c_double]
        self.lib.vad_process_file.argtypes = [c_void_p, c_char_p, POINTER(c_int32)]
        self.lib.vad_process_file.restype = c_void_p
        self.lib.vad_free_segments.argtypes = [c_void_p]
        self.lib.vad_destroy.argtypes = [c_void_p]

        # Create processor
        self.processor = self.lib.vad_create()

        # Set parameters
        self.lib.vad_set_min_duration_on(self.processor, min_duration_on)
        self.lib.vad_set_min_duration_off(self.processor, min_duration_off)

        # Load model
        if not self.lib.vad_load_model(self.processor, model_path.encode('utf-8')):
            raise RuntimeError(f"Failed to load model from {model_path}")

    def process_audio(self, audio_path):
        """Process audio file and return VAD segments"""
        count = c_int32()
        segments_ptr = self.lib.vad_process_file(
            self.processor,
            str(audio_path).encode('utf-8'),
            ctypes.byref(count)
        )

        if not segments_ptr or count.value == 0:
            return []

        # Check for special invalid pointer
        if segments_ptr == 1:
            return []

        # Extract segments
        double_ptr = ctypes.cast(segments_ptr, POINTER(c_double))
        segments = []

        for i in range(count.value):
            start = double_ptr[i * 2]
            end = double_ptr[i * 2 + 1]
            segments.append((start, end))

        # Free memory
        self.lib.vad_free_segments(segments_ptr)

        return segments

    def __del__(self):
        if hasattr(self, 'processor') and self.processor:
            self.lib.vad_destroy(self.processor)