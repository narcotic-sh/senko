# CAM++ JIT Tracing

This directory contains scripts for creating optimized TorchScript versions of the CAM++ speaker verification/embeddings gen model.

## Overview

The tracing process consists of two sequential steps:
1. **Tracing** (`trace.py`): Downloads model weights (if not already present) from ModelScope and creates a JIT-traced TorchScript version
2. **Optimization** (`optimize.py`): Applies PyTorch inference optimizations and benchmarks performance

## Usage

Both scripts must be run from the project root directory using Python module syntax:

### Step 1: Tracing

```bash
python -m tracing.trace --device cuda|mps|cpu
```

### Step 2: Optimization

```bash
python -m tracing.optimize --device cuda|mps|cpu
```
