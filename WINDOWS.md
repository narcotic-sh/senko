# Senko Windows Install Instructions
Tested on Windows 11 Pro (25H2).

Prerequisites:
- `MSVC` - Microsoft Visual C++ Compiler

  ```cmd
  winget install Microsoft.VisualStudio.2022.BuildTools --source winget --force --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.26100"
  ```
  
- [`uv`](https://docs.astral.sh/uv/#installation)

Create a Python virtual environment and activate it
```sh
uv venv --python 3.13 .venv

# PowerShell
.venv\Scripts\Activate

# Command Prompt
.venv\Scripts\activate.bat
```
Then install Senko
```bash
# For NVIDIA GPUs with CUDA compute capability >= 7.5 (~GTX 16 series and newer)
uv pip install "git+https://github.com/narcotic-sh/senko.git[nvidia-windows]"

# For NVIDIA GPUs with CUDA compute capability < 7.5 (~GTX 10 series and older)
uv pip install "git+https://github.com/narcotic-sh/senko.git[nvidia-old-windows]"

# For CPU execution
uv pip install "git+https://github.com/narcotic-sh/senko.git"
```
For NVIDIA, make sure the installed driver is CUDA 12 capable (should see `CUDA Version: 12+` in `nvidia-smi`).

For setting up Senko for development, see [`DEV_SETUP.md`](DEV_SETUP.md).

## Native Windows vs. WSL
There are a couple downsides to running Senko natively on Windows vs. on WSL:

- Clustering cannot run on the GPU ([RAPIDS](https://rapids.ai/) doesn't support native Windows)
- Fbank feature extraction cannot run on the GPU ([`kaldifeat`](https://github.com/csukuangfj/kaldifeat) only has prebuilt wheels for Linux)

For CPU-only diarization though, native Windows yields a significant speed boost vs. WSL.