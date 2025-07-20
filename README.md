# Bethe-Functions

## What is this?

This is an open source GitHub repository containing code for the functions introduced in (link arxiv paper here) related to the Bethe Ansatz for the Heisenberg-Ising (XXZ) Spin-1/2 chain. You will find code which contains definitions for the functions themselves along with code to generate plots for observables of the model (only the one-point function for now). Additionally, there is code written to run on GPUs in high performance computing clusters.

**NEW: Multi-GPU Support** - The repository now includes a comprehensive multi-GPU implementation that enables calculations on problems exceeding single-GPU memory limits with near-linear scaling across multiple GPUs.

## Why?

* Provide source code for the plots and numerical evidence for the conjectures found in (link arxiv paper here)
* Advance research and simulation in the fields relating to the Bethe Ansatz and XXZ spin chain by providing open source access to relevant code
* Enable large-scale quantum many-body calculations using multi-GPU parallelization

## Quick Start

### Basic Python Implementation
```bash
cd "One-point and Bethe Functions"
python3 OnePointFuncConsole.py 2 5 0.03 1 2
```

### Single-GPU Implementation
```bash
cd spinChain-main
python setup.py build_ext --inplace
pip install -e .
```

### Multi-GPU Implementation (Recommended)
```bash
cd spinChain-main
python setup_multigpu.py build_ext --inplace
pip install -e .
```

## Installation Guide

### Prerequisites

#### System Requirements
- **Operating System**: Linux (tested), Windows (partial support), macOS (partial support)
- **Python**: 3.7 or higher
- **Memory**: 8GB RAM minimum (16GB+ recommended for large problems)
- **Storage**: 2GB free space

#### Required Dependencies
- **CUDA Toolkit**: 10.0 or higher (11.0+ recommended)
  ```bash
  # Check CUDA installation
  nvcc --version
  nvidia-smi
  ```
- **CMake**: 3.18 or higher
  ```bash
  # Install on Ubuntu/Debian
  sudo apt update && sudo apt install cmake
  
  # Install on CentOS/RHEL
  sudo yum install cmake3
  
  # Check version
  cmake --version
  ```
- **C++ Compiler**: GCC 7+ or Clang 8+ with C++17 support

#### Python Dependencies
```bash
# Install core dependencies
pip install numpy>=1.18.0 scipy>=1.5.0 matplotlib>=3.0.0

# Install pybind11 for C++ bindings
pip install pybind11>=2.6.0

# Optional: for enhanced testing and benchmarking
pip install pytest pandas seaborn
```

### Installation Methods

#### Method 1: Multi-GPU Installation (Recommended)

**Step 1: Clone and Navigate**
```bash
git clone https://github.com/your-repo/Bethe-Functions.git
cd Bethe-Functions/spinChain-main
```

**Step 2: Check GPU Compatibility**
```bash
# Verify GPUs meet minimum requirements (compute capability ≥ 6.1)
python -c "
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv'], 
                       capture_output=True, text=True)
print(result.stdout)
"
```

**Step 3: Build Multi-GPU Version**
```bash
# Method 3a: Using enhanced setup.py (recommended)
python setup_multigpu.py build_ext --inplace
pip install -e .

# Method 3b: Using CMake directly
mkdir build && cd build
cmake -DENABLE_MULTIGPU=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..

# Method 3c: Using original setup with enhanced CMake
cp CMakeLists_MultiGPU.txt CMakeLists.txt
python setup.py build_ext --inplace
pip install -e .
```

**Step 4: Verify Installation**
```bash
python -c "
import libspinChainMultiGPU as mgpu
chain = mgpu.MultiGPUSpinChain()
print(f'Available GPUs: {chain.get_available_gpus()}')
print(f'GPU Info: {chain.get_gpu_info()}')
"
```

#### Method 2: Single-GPU Installation

**For systems with single GPU or legacy compatibility:**
```bash
cd spinChain-main
python setup.py build_ext --inplace
pip install -e .

# Verify installation
python -c "import libspinChain; print('Single GPU installation successful')"
```

#### Method 3: CPU-Only Installation

**For systems without CUDA:**
```bash
cd "One-point and Bethe Functions"

# Install Python dependencies
pip install numpy scipy matplotlib

# Test basic functionality
python OnePointFuncConsole.py 2 5 0.03 1 2
```

### Installation Verification

#### Quick Test
```bash
# Navigate to test directory
cd spinChain-main

# Run quick multi-GPU test
python test_multigpu.py --quick

# Expected output:
# Available GPUs: [0, 1, 2, ...]
# Running correctness tests...
# Running performance tests...
# TEST SUMMARY: All tests passed
```

#### Comprehensive Test
```bash
# Run full test suite (may take 10-30 minutes)
python test_multigpu.py

# Run specific test categories
python test_multigpu.py --performance --scalability --output results.json
```

### Usage Examples

#### Basic Multi-GPU Usage
```python
import libspinChainMultiGPU as multigpu
import numpy as np

# Initialize multi-GPU system
chain = multigpu.MultiGPUSpinChain()

# Configure for your system
available_gpus = chain.get_available_gpus()
print(f"Found {len(available_gpus)} GPUs: {available_gpus}")

# Use first 2 GPUs
chain.set_active_gpus(available_gpus[:2])

# Configure performance parameters
chain.set_block_size(256)           # CUDA block size
chain.set_shared_memory_size(1024)  # KB per GPU

# Prepare your data (example)
nBasis, nUp = 100, 6
allBetheRoots = [complex(0.5 + 0.1*i, 0.1*j) 
                for i in range(nBasis) for j in range(nUp)]
allConfigs = list(range(nBasis * nUp))
allGaudinDets = [complex(1.0 + 0.01*i) for i in range(nBasis)]
sigma = list(range(720 * nUp))  # Permutations
delta = complex(0.5, 0.0)

# Run computation
result = chain.compute_basis_transform(
    allBetheRoots, allConfigs, allGaudinDets, sigma, delta
)

# Monitor performance
metrics = chain.get_performance_metrics()
print(f"Execution time: {metrics['last_execution_time']:.2f}s")
print(f"GPU utilization: {metrics['gpu_utilization']}")
print(f"Memory usage: {metrics['gpu_memory_usage_mb']} MB")
```

### Troubleshooting

#### Common Issues

**Issue**: `ImportError: No module named 'libspinChainMultiGPU'`
```bash
# Solution: Rebuild and reinstall
cd spinChain-main
python setup_multigpu.py clean --all
python setup_multigpu.py build_ext --inplace
pip install -e .
```

**Issue**: `CUDA out of memory`
```bash
# Solution: Reduce problem size or use more GPUs
python -c "
import libspinChainMultiGPU as mgpu
chain = mgpu.MultiGPUSpinChain()
memory_info = chain.get_memory_info()
print(f'Available memory: {memory_info[\"available_memory_mb\"]} MB')
print('Reduce nBasis or increase GPU count')
"
```

**Issue**: `No CUDA-capable device is detected`
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify CUDA toolkit path
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Issue**: CMake cannot find pybind11
```bash
# Install pybind11 development files
pip install "pybind11[global]"

# Or specify path manually
export PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
```

#### Performance Optimization

**For Maximum Performance:**
```python
# Use all available GPUs
chain.set_active_gpus(chain.get_available_gpus())

# Optimize block size for your GPUs
chain.set_block_size(512)  # Try 256, 512, or 1024

# Enable shared memory for smaller problems
chain.set_shared_memory_size(2 * 1024 * 1024)  # 2MB per GPU

# Monitor and tune
metrics = chain.get_performance_metrics()
if max(metrics['gpu_utilization']) < 0.8:
    # Increase block size or problem size
    pass
```

#### Getting Help

1. **Check Documentation**: See `MULTI_GPU_IMPLEMENTATION.md` for detailed technical information
2. **Run Diagnostics**: Use `test_multigpu.py` to identify issues
3. **Check System Requirements**: Ensure CUDA, CMake, and Python versions meet minimums
4. **Review Build Logs**: Look for compilation errors in build output
5. **GPU Compatibility**: Verify compute capability ≥ 6.1 with `nvidia-smi -q`

## Project Structure

* **One-point and Bethe Functions/** - Basic Python implementations and Jupyter notebooks
* **spinChain-main/** - GPU-accelerated implementations
  * `src/` - C++/CUDA source code
  * `multiGPU*` - Multi-GPU implementation files
  * `setup_multigpu.py` - Enhanced build system
  * `test_multigpu.py` - Comprehensive test suite
* **Plots and output files/** - Generated visualization outputs
* **CLAUDE.md** - Development guide for this repository
* **MULTI_GPU_IMPLEMENTATION.md** - Detailed technical documentation
