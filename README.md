# Lattice Boltzmann Method for Lid-Driven Cavity Flow

A high-performance GPU-accelerated implementation of the Lattice Boltzmann Method (LBM) for simulating lid-driven cavity flow using PyTorch.

## 🎯 Overview

This project implements the D2Q9 Lattice Boltzmann Method with:
- **GPU acceleration** using PyTorch/CUDA
- **Memory-optimized** algorithms with double buffering
- **Real-time visualization** of flow evolution
- **Benchmark validation** against classical results

### Key Features

- ✅ **Efficient Implementation**: 32MB memory usage for 512² grid (89% of theoretical minimum)
- ✅ **Fast Performance**: <20 minutes runtime for 512² resolution on GPU
- ✅ **Physical Accuracy**: Validates against Ghia et al. (1982) benchmark
- ✅ **Real-time Monitoring**: Live velocity magnitude visualization
- ✅ **Memory Optimized**: Double buffering and in-place operations

## 🚀 Quick Start

### Prerequisites

- CUDA-capable GPU (recommended) or CPU
- Python 3.8+
- Conda package manager

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Create conda environment
conda create -n lbm_pytorch python=3.10
conda activate lbm_pytorch

pip install -r requirements.txt
```


## 🏃‍♂️ Running the Simulation

### Basic Usage

```bash
# Activate environment
conda activate lbm_pytorch

# Run the simulation
python Pytorch_LBM_accelerated.py
```

### Configuration Options

Edit the configuration section in `Pytorch_LBM_accelerated.py`:

```python
# Grid resolution
Nx, Ny = 512, 512               # 128x128, 256x256, 512x512, 1024x1024

# Physical parameters
tau = 0.6                       # Relaxation time (controls viscosity)
u_lid = 0.05                    # Lid velocity (affects Reynolds number)

# Simulation control
max_iter = 130_000              # Maximum iterations
conv_tol = 1e-7                 # Convergence tolerance
viz_every = 50                  # Visualization update frequency
```

