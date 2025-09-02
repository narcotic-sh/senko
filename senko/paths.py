import platform
from pathlib import Path
import importlib.resources

IS_DARWIN = platform.system() == 'Darwin'
LIB_EXTENSION = '.dylib' if IS_DARWIN else '.so'
LIB_FILENAME = f'libfbank_extractor{LIB_EXTENSION}'

# By default, we assume a standard installation where all data is located relative to the package's installed location
PACKAGE_DIR = importlib.resources.files('senko')
IS_DEV_MODE = not (PACKAGE_DIR / LIB_FILENAME).exists()

if IS_DEV_MODE:
    # We are in an editable 'redirect' install. Find the project root by going up from this file's location.
    # senko/paths.py -> senko -> project_root
    project_root = Path(__file__).parent.parent

    # In development, compiled artifacts are in the build directory.
    BUILD_DIR = project_root / 'build'
    if not (BUILD_DIR / LIB_FILENAME).exists():
        raise FileNotFoundError(
            f"Could not find '{LIB_FILENAME}' in development build directory "
            f"('{BUILD_DIR}'). Please run 'uv pip install -e .' successfully."
        )

    # In development, the 'models' directory is at the project root.
    MODELS = project_root / 'models'
    # The compiled library is in the build directory.
    FBANK_LIB_PATH = str(BUILD_DIR / LIB_FILENAME)
else:
    # In a standard installation, 'models' and the library are installed alongside the Python files
    MODELS = PACKAGE_DIR / 'models'
    FBANK_LIB_PATH = str(PACKAGE_DIR / LIB_FILENAME)

# The YAML configuration files are always part of the source tree, so we can safely locate them relative to the package source path
CLUSTER = PACKAGE_DIR / 'cluster'
SPECTRAL_YAML = str(CLUSTER / 'conf' / 'spectral.yaml')
UMAP_HDBSCAN_YAML = str(CLUSTER / 'conf' / 'umap_hdbscan.yaml')

# Check if the primary models directory exists where we expect it
if not MODELS.exists():
    raise FileNotFoundError(
        f"The 'models' directory was not found at the expected location: {MODELS}"
    )

# Define final model paths based on the resolved 'MODELS' directory
PYANNOTE_SEGMENTATION_MODEL_PATH = str(MODELS / 'pyannote_segmentation_3.0/pytorch_model.bin')
TORCH_DEVICE = 'mps' if IS_DARWIN else 'cuda'
EMBEDDINGS_JIT_MODEL_PATH = str(MODELS / f'camplusplus_traced_{TORCH_DEVICE}_optimized.pt')
EMBEDDINGS_PT_MODEL_PATH = str(MODELS / 'speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt')