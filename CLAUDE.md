# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains implementations of Bethe Functions for the Heisenberg-Ising (XXZ) Spin-1/2 chain using the Bethe Ansatz method. The code is structured into two main components:

1. **Basic Python implementations** with Jupyter notebook analysis
2. **High-performance GPU-accelerated implementations** using CUDA and C++

## Directory Structure

- `One-point and Bethe Functions/` - Basic Python implementations and analysis
  - `DetFormula.ipynb` - Jupyter notebook with theoretical background and function implementations
  - `OnePointFuncConsole.py` - Command-line script for generating one-point function plots  
  - `GapProb.py` - Gap probability calculations
  - `onePoint.sh` - SLURM batch script for HPC cluster execution

- `spinChain-main/` - GPU-accelerated implementations
  - `src/` - C++/CUDA source code for high-performance computations
  - `spinChain/` - Python wrapper modules
  - `CMakeLists.txt` - CMake build configuration
  - `setup.py` - Python package setup with custom CMake build

- `Plots and output files/` - Generated visualization outputs and computation results

## Common Development Commands

### Basic Python Implementation
```bash
# Run one-point function calculation
cd "One-point and Bethe Functions"
python3 OnePointFuncConsole.py [N] [L] [delta] [initial_conditions...]

# Example: N=2 particles, L=5 sites, delta=0.03, initial positions at 1,2
python3 OnePointFuncConsole.py 2 5 0.03 1 2

# Gap probability calculations
python3 GapProb.py
```

### GPU Implementation Build Process
```bash
cd spinChain-main

# Build the C++/CUDA extension
python setup.py build_ext --inplace

# Install as Python package (recommended for development)
pip install -e .

# Clean build (if needed)
rm -rf build/ *.so
```

### HPC Cluster Usage
```bash
# Submit job to SLURM scheduler
sbatch onePoint.sh [N] [L] [delta] [initial_conditions...]

# Example: N=2 particles, L=5 sites, delta=0.03, positions 2,3
sbatch onePoint.sh 2 5 0.03 2 3

# Check job status
squeue -u $USER

# View output logs
ls "Plots and output files/"plotStd*.txt
```

## Key Components

### Mathematical Framework
The codebase implements:
- Bethe Ansatz equations for XXZ spin chain
- One-point correlation functions with time evolution
- Eigenfunction calculations using permutation sums over particle configurations
- Determinant formulas for coefficient calculations via matrix operations
- Fixed-point iteration methods for solving transcendental Bethe equations
- Complex arithmetic throughout for handling Bethe roots

### Build Architecture
- **Hybrid Build System**: CMake for CUDA compilation, setuptools for Python packaging
- **CMake Configuration**: Targets CUDA compute capabilities 6.1 and 7.5
- **Python Bindings**: pybind11 for seamless C++/CUDA integration
- **Custom Build Class**: `CMakeBuild` in setup.py automatically invokes CMake during pip install

### Parameter Space
- `N`: Number of up-spin particles (determines system complexity)
- `L`: Number of lattice sites (system size)
- `delta`: Anisotropy parameter (Δ in equations, controls interaction strength)
- Initial conditions: Specify particle positions at t=0 (values between 0 and L-1)

## Dependencies

### Python Requirements
- numpy (matrix operations, complex arithmetic)
- scipy (special functions, linear algebra)
- matplotlib (plotting and visualization)
- numba (JIT compilation for performance)
- itertools (built-in, for permutation generation)

### GPU Build Requirements
- CUDA toolkit (≥10.0 recommended)
- CMake (≥3.10)
- pybind11 (for Python-C++ bindings)
- C++14 compatible compiler
- Compatible GPU with compute capability ≥6.1

### HPC Environment Dependencies
- SLURM workload manager
- Xvfb (virtual display for headless plotting)
- module system with Python support

## File Naming Conventions

Generated outputs follow these patterns:
- **Plot files**: `n[N] l[L] d[delta] [initial_conditions] (job [SLURM_JOB_ID]).png`
- **SLURM logs**: `plotStdOut (job [SLURM_JOB_ID]).txt` and `plotStdError (job [SLURM_JOB_ID]).txt`
- All outputs saved to `Plots and output files/` directory

## Development Notes

### No Formal Testing Infrastructure
- Interactive testing via Jupyter notebook cells
- Mathematical verification through identity matrix checks and convergence validation
- Example calculations embedded in notebook for validation

### Performance Considerations
- **CPU Implementation**: Pure Python with NumPy for problem sizes N≤4, L≤10
- **GPU Implementation**: CUDA kernels for larger systems with parallel amplitude calculations
- **Memory Management**: Efficient handling of combinatorial spaces that grow as C(L,N)
- **Numerical Stability**: Complex arithmetic with careful handling of small denominators

### Mathematical Implementation Details
- Uses permutation group operations for eigenfunction construction
- Matrix determinants computed via NumPy's LAPACK interface
- Root-finding for transcendental Bethe equations using iterative methods
- Time evolution implemented via matrix exponentials

### Troubleshooting Common Issues
- **CUDA Build Failures**: Check that pybind11 paths in CMakeLists.txt match your environment
- **Memory Issues**: Large L and N values require substantial RAM; consider GPU implementation
- **Convergence Problems**: Bethe equation solutions may not converge for certain parameter ranges
- **Display Issues on HPC**: Ensure Xvfb is properly configured for headless plotting



### Multi-GPU plan 

● Update Todos
  ⎿  ☐ Analyze current GPU implementation architecture
     ☐ Research multi-GPU memory management strategies
     ☐ Design virtual memory pooling across GPUs
     ☐ Plan workload distribution mechanisms
     ☐ Design GPU synchronization strategy

● Update Todos
  ⎿  ☐ Analyze current GPU implementation architecture
     ☐ Research multi-GPU memory management strategies
     ☐ Design virtual memory pooling across GPUs
     ☐ Plan workload distribution mechanisms
     ☐ Design GPU synchronization strategy

● Task(Analyze current GPU code)
  ⎿  Done (15 tool uses · 36.1k tokens · 1m 29.1s)

● Update Todos
  ⎿  ☒ Analyze current GPU implementation architecture
     ☐ Research multi-GPU memory management strategies
     ☐ Design virtual memory pooling across GPUs
     ☐ Plan workload distribution mechanisms
     ☐ Design GPU synchronization strategy

● Multi-GPU Implementation Plan for Bethe Functions Application

  Phase 1: Architecture Foundation

  1.1 Multi-GPU Device Management

  - GPU Discovery & Enumeration: Implement automatic detection of available GPUs with capability checking
  (compute ≥6.1)
  - Device Context Management: Create per-GPU CUDA contexts with proper resource isolation
  - Memory Pool Initialization: Pre-allocate memory pools on each GPU for efficient runtime allocation
  - Device Affinity: CPU thread-to-GPU binding for optimal NUMA performance

  1.2 Virtual Memory Pooling Strategy

  - Unified Memory Architecture: Implement a virtual memory manager that abstracts GPU memory as a single
  address space
  - Memory Segmentation: Divide problem data into chunks that can be distributed across available GPU memory
  - Dynamic Allocation: Smart memory allocator that can span multiple GPUs when single-GPU memory is
  insufficient
  - Memory Migration: Ability to move data between GPUs based on computational needs

  Phase 2: Data Distribution Architecture

  2.1 Problem Decomposition Strategy

  - Energy-State Partitioning: Primary strategy - distribute nBasis energy states across GPUs
    - Each GPU handles energiesPerGPU = ceil(nBasis / nGPUs) states
    - Natural load balancing since each energy state has similar computational complexity
    - Minimal inter-GPU communication required

  2.2 Memory Distribution Patterns

  - Replicated Data: Broadcast to all GPUs
    - allConfigs (particle configurations)
    - sigma (permutation arrays)
    - Small lookup tables and constants
  - Partitioned Data: Distribute across GPUs
    - allBetheRoots[startEnergy:endEnergy] per GPU
    - allGaudinDets[startEnergy:endEnergy] per GPU
    - Output matrices d_e2c and d_c2e blocks

  2.3 Virtual Memory Implementation

  - GPU Memory Virtualization: Create abstraction layer that presents multiple GPU memories as single virtual
   space
  - Address Translation: Map virtual addresses to (GPU_ID, local_address) pairs
  - Automatic Paging: Move memory segments between GPUs based on access patterns
  - Overflow Handling: Utilize system memory when combined GPU memory is insufficient

  Phase 3: Computation Distribution

  3.1 Kernel Modification Strategy

  - Dynamic Memory Allocation: Replace hardcoded [1716] arrays with dynamic shared memory
  - Configurable Block Sizes: Adapt thread block dimensions based on available memory per GPU
  - Stream-Based Execution: Use CUDA streams to overlap computation with data transfers

  3.2 Workload Balancing

  - Static Partitioning: Initial even distribution of energy states
  - Dynamic Load Balancing: Monitor GPU utilization and redistribute work if imbalances detected
  - Work Stealing: Allow idle GPUs to take work from busy ones

  3.3 Memory Access Optimization

  - Data Locality: Ensure each GPU primarily accesses its local memory
  - Prefetching: Anticipate memory access patterns and pre-load data
  - Cache Management: Optimize GPU L2 cache usage across multiple devices

  Phase 4: Communication & Synchronization

  4.1 Inter-GPU Communication

  - Minimal Communication Design: Leverage embarrassingly parallel nature
  - Result Aggregation: Collect partial results from each GPU
  - P2P Memory Transfers: Use GPU-to-GPU direct memory access when available
  - Host-Mediated Transfers: Fallback through system memory when direct P2P unavailable

  4.2 Synchronization Strategy

  - Barrier Synchronization: Coordinate GPUs at key computation phases
  - Event-Based Coordination: Use CUDA events for fine-grained synchronization
  - Stream Dependencies: Chain operations across GPUs using stream dependencies

  4.3 Error Handling & Recovery

  - GPU Failure Detection: Monitor for GPU errors and handle graceful degradation
  - Work Redistribution: Reassign failed GPU's work to remaining devices
  - Checkpointing: Save intermediate results for long-running computations

  Phase 5: Memory Management Implementation

  5.1 Virtual Memory Pool Architecture

  class MultiGPUMemoryPool {
      - std::vector<GPUMemoryPool> gpu_pools;
      - VirtualAddressSpace virtual_space;
      - MemoryMigrationManager migration_manager;
      + allocate(size_t bytes) -> VirtualPointer
      + deallocate(VirtualPointer ptr)
      + migrate(VirtualPointer ptr, int target_gpu)
  }

  5.2 Smart Memory Allocation

  - First-Fit Strategy: Try to allocate on GPU with most free memory
  - Fragmentation Management: Implement memory compaction when needed
  - Allocation Tracking: Monitor memory usage patterns for optimization
  - Out-of-Memory Handling: Automatic spilling to system memory

  5.3 Data Movement Optimization

  - Asynchronous Transfers: Overlap data movement with computation
  - Bandwidth Optimization: Use optimal transfer patterns (2D/3D copies when beneficial)
  - Transfer Scheduling: Queue and prioritize memory operations

  Phase 6: API & Integration

  6.1 Python Interface Enhancement

  - Multi-GPU Configuration: Allow users to specify GPU count and selection
  - Automatic Scaling: Transparent multi-GPU usage without API changes
  - Performance Monitoring: Expose GPU utilization and memory usage metrics
  - Fallback Mechanisms: Graceful degradation to single-GPU or CPU-only execution

  6.2 Build System Updates

  - Multi-GPU Detection: CMake modifications to detect and configure multiple GPUs
  - Conditional Compilation: Enable/disable multi-GPU features based on hardware
  - Library Dependencies: Add any additional CUDA libraries (e.g., NCCL for communication)

  Phase 7: Testing & Optimization

  7.1 Performance Validation

  - Scaling Tests: Measure performance with 1, 2, 4, 8+ GPUs
  - Memory Efficiency: Monitor memory utilization across all devices
  - Load Balance Analysis: Ensure even GPU utilization
  - Bandwidth Utilization: Measure inter-GPU and GPU-CPU transfer rates

  7.2 Correctness Verification

  - Result Consistency: Ensure multi-GPU results match single-GPU output
  - Numerical Stability: Verify no precision loss from data movement
  - Edge Case Testing: Handle scenarios with insufficient memory, GPU failures

  Expected Performance Improvements

  Theoretical Speedups

  - Computation: Near-linear scaling (0.8-0.95x per additional GPU)
  - Memory Capacity: Multiplicative increase in problem size capability
  - Memory Bandwidth: Combined bandwidth of all GPUs for large data sets

  Target Metrics

  - 4 GPUs: Handle problems 3-4x larger than single-GPU limits
  - 8 GPUs: Enable calculations previously impossible due to memory constraints
  - Efficiency: Maintain >80% single-GPU performance per device

  This plan leverages the embarrassingly parallel nature of the Bethe function calculations while
  implementing sophisticated memory management to enable calculations that exceed single-GPU memory limits.
