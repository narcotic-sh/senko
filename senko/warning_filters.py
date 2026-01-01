import warnings

def apply_warning_filters():
    warnings.filterwarnings("ignore", message=".*Matplotlib.*")
    warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*invalid escape sequence.*")
    warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
    warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")
    warnings.filterwarnings("ignore", message=".*torchaudio.load_with_torchcodec.*")
    warnings.filterwarnings("ignore", message=".*torchaudio.sox_effects.sox_effects.apply_effects_file.*")
    warnings.filterwarnings("ignore", message=".*torio.io._streaming_media_decoder.StreamingMediaDecoder.*")
    warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*")