import argparse
import torch
import requests
from pathlib import Path
from senko import CAMPPlus

#############
## Helpers ##
#############

def download_model():
    """Download the CAMPlus model if it doesn't exist and return the local path."""
    model_dir = Path("./models/speech_campplus_sv_zh_en_16k-common_advanced")
    model_path = model_dir / "campplus_cn_en_common.pt"

    if model_path.exists():
        print(f"Model already exists at: {model_path}")
        return str(model_path)

    print(f"Model not found. Downloading to: {model_path}")
    model_dir.mkdir(parents=True, exist_ok=True)

    url = "https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/resolve/master/campplus_cn_en_common.pt"

    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = downloaded / total_size * 100
                        print(f"\rDownloading: {pct:.1f}%", end="", flush=True)
    except requests.RequestException as e:
        raise RuntimeError(f"Error downloading model: {e}") from e

    print(f"\nModel downloaded successfully to: {model_path}")
    return str(model_path)


def select_device(device_str):
    """Return an appropriate torch.device for the requested backend."""
    dev = device_str.lower()
    if dev.startswith("cuda") and torch.cuda.is_available():
        return torch.device(dev)
    if dev == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if dev not in {"cpu", "mps", "cuda"}:
        print(f"Unrecognised device '{device_str}', falling back to CPU.")
    return torch.device("cpu")

##########################
## Main tracing routine ##
##########################

def trace_model(device_str):
    device = select_device(device_str)
    print(f"Using device: {device}")

    model_path = download_model()
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval().to(device)

    dummy = torch.randn(40, 150, 80, device=device)
    print(f"Dummy input shape: {dummy.shape}")

    with torch.no_grad():
        original_out = model(dummy)

    trace_file = f"./tracing/camplusplus_traced_{device.type}.pt"
    traced = torch.jit.trace(model, dummy)
    traced.save(trace_file)
    print(f"Traced model saved to {trace_file}")

    loaded = torch.jit.load(trace_file).to(device)
    with torch.no_grad():
        traced_out = loaded(dummy)

    max_diff = torch.max(torch.abs(original_out - traced_out)).item()
    match = torch.allclose(original_out, traced_out, atol=1e-5)
    print(f"\nMaximum difference: {max_diff}")
    print(f"Outputs match:      {match}")

    return original_out, traced_out

############
## main() ##
############

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trace CAMPPlus and save a TorchScript model."
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        default=None,
        help="Device to run on (default: auto-select)",
    )
    args = parser.parse_args()

    # Auto-select if not provided
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    trace_model(args.device)
