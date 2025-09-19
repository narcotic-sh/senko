# CAM++ CoreML, TorchScript Tracing
This directory contains scripts for creating either a TorchScript CUDA version or CoreML version of the CAM++ speaker embedding model.

Run all command below from the root of the repo.

## CoreML
```
python -m tracing.coreml.convert
```

## TorchScript CUDA
```bash
# First trace
python -m tracing.torch.trace --device cuda

# Then optimize
python -m tracing.torch.optimize --device cuda
```