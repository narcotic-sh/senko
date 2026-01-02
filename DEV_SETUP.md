# Senko Development Setup Guide

Pre-requisites
- `gcc/clang` if on Linux/macOS/WSL
- `MSVC` if on Windows

  ```cmd
  winget install Microsoft.VisualStudio.2022.BuildTools --source winget --force --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.26100"
  ```

- `git`
- [`uv`](https://docs.astral.sh/uv/#installation)

## Development Setup Steps
First create and activate a Python virtual environment
```sh
uv venv --python 3.13 .venv

# bash/zsh etc.
source .venv/bin/activate

# PowerShell
.venv\Scripts\Activate

# Command Prompt
.venv\Scripts\activate.bat
```
Clone the Senko repository
```sh
git clone https://github.com/narcotic-sh/senko.git
```
Then install using editable mode (`-e`)
```sh
# Can add [nvidia], [nvidia-old], [nvidia-windows] etc. 
uv pip install -e "/path/to/cloned/senko"
```
Now you can modify the Senko code in the cloned repository folder. Changes will be reflected immediately if only the Python code was changed; if the C++/Swift code was changed, you'll have to run the `uv pip install -e` command again to rebuild the C++/Swift code.

Then use Senko like normal in scripts:
```python
from senko import Diarizer
import json

diarizer = Diarizer(device='auto', warmup=True, quiet=False)
result = diarizer.diarize('audio.wav', generate_colors=False) # 16kHz mono 16-bit wav

with open('./audio_diarized.json', 'w') as f:
    json.dump(result["merged_segments"], f, indent=2)
```
Also see [`examples/diarize.py`](examples/diarize.py).