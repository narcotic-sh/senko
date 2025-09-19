import argparse
import statistics
import time
import torch

#############
## Helpers ##
#############

def select_device(device_str):
    dev = device_str.lower()
    if dev.startswith("cuda") and torch.cuda.is_available():
        return torch.device(dev)
    if dev == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if dev not in {"cpu", "mps", "cuda"}:
        print(f"Unrecognised device '{device_str}', falling back to CPU.")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

##################
## Benchmarking ##
##################

def benchmark(model, device, input_shape=(40, 148, 80), runs=100, warmup=10):
    model = model.to(device)
    sample = torch.randn(input_shape, device=device)
    times = []

    # warm-up
    for _ in range(warmup):
        model(sample)
        synchronize(device)

    # timed
    for _ in range(runs):
        start = time.perf_counter()
        model(sample)
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "avg_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "med_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }

###########################
## Optimization pipeline ##
###########################

def optimise(device_str="mps"):
    device = select_device(device_str)
    print(f"Using device: {device}")

    traced = f"./tracing/camplusplus_traced_{device.type}.pt"
    optimised = f"./tracing/camplusplus_traced_{device.type}_optimized.pt"

    model = torch.jit.load(traced).to(device)
    opt_model = torch.jit.optimize_for_inference(model).to(device)
    opt_model.save(optimised)
    print(f"Optimised model saved to {optimised}")

    # Benchmark both
    print("\nBenchmarking original model …")
    stats_orig = benchmark(model, device)

    empty_cache(device)

    print("\nBenchmarking optimised model …")
    stats_opt = benchmark(opt_model, device)

    speedup = (stats_orig["avg_ms"] - stats_opt["avg_ms"]) / stats_orig["avg_ms"] * 100
    print("\nResults:")
    for tag, s in [("Original", stats_orig), ("Optimised", stats_opt)]:
        print(
            f"{tag:9} | avg {s['avg_ms']:.2f} ms | std {s['std_ms']:.2f} ms | "
            f"med {s['med_ms']:.2f} ms | min {s['min_ms']:.2f} ms | "
            f"max {s['max_ms']:.2f} ms"
        )
    print(f"\nSpeed-up: {speedup:.1f}%")

    return opt_model

############
## main() ##
############

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimise a traced CAMPPlus TorchScript model."
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        default=None,
        help="Device to run on (default: auto-select)",
    )
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    optimise(args.device)
