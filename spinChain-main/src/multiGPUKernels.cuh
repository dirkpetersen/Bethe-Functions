#ifndef MULTIGPU_KERNELS_H_
#define MULTIGPU_KERNELS_H_

#include "defs.h"
#include "multiGPUMemory.h"
#include "multiGPUControl.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/**
 * @brief Multi-GPU kernel execution parameters
 */
struct MultiGPUKernelParams {
    int nBasis;
    int nUp;
    int nPerm;
    size_t shared_memory_size;
    int block_size;
    int grid_size;
    
    MultiGPUKernelParams() : nBasis(0), nUp(0), nPerm(0), shared_memory_size(0), 
                            block_size(256), grid_size(0) {}
};

/**
 * @brief Multi-GPU workload distribution for basis transformation
 */
struct GPUWorkload {
    int gpu_id;
    int start_energy;
    int end_energy;
    int energy_count;
    VirtualPointer d_allBetheRoots;
    VirtualPointer d_allConfigs;
    VirtualPointer d_allGaudinDets;
    VirtualPointer d_sigma;
    VirtualPointer d_e2c;
    VirtualPointer d_c2e;
    cudaStream_t stream;
    
    GPUWorkload() : gpu_id(-1), start_energy(0), end_energy(0), energy_count(0), stream(nullptr) {}
};

// Enhanced CUDA kernels with dynamic memory allocation
__global__ void multiGPUComputeBasisTransform(
    const thrust::complex<double>* allBetheRoots,
    const unsigned long* allConfigs,
    const thrust::complex<double>* allGaudinDets,
    const unsigned long* sigma,
    thrust::complex<double>* e2c_output,
    thrust::complex<double>* c2e_output,
    thrust::complex<double> delta,
    int nBasis,
    int nUp,
    int nPerm,
    int start_energy,
    int end_energy
);

__global__ void multiGPUComputeBasisTransformShared(
    const thrust::complex<double>* allBetheRoots,
    const unsigned long* allConfigs,
    const thrust::complex<double>* allGaudinDets,
    const unsigned long* sigma,
    thrust::complex<double>* e2c_output,
    thrust::complex<double>* c2e_output,
    thrust::complex<double> delta,
    int nBasis,
    int nUp,
    int nPerm,
    int start_energy,
    int end_energy
);

// Optimized device functions (same as original but with improved memory access)
__device__ thrust::complex<double> gpuProductOptimized(
    const thrust::complex<double>* vector, 
    unsigned long size
);

__device__ thrust::complex<double> intPowerOptimized(
    thrust::complex<double> base, 
    unsigned long power
);

__device__ thrust::complex<double> gpuAmplitudeOptimized(
    const thrust::complex<double>* betheRoots,
    const unsigned long* sigma,
    unsigned long nUp,
    thrust::complex<double> delta
);

// Multi-GPU coordination functions
class MultiGPUBasisTransform {
public:
    MultiGPUBasisTransform();
    ~MultiGPUBasisTransform();
    
    // Main computation interface
    std::pair<MyComplexVector, MyComplexVector> compute(
        const MyComplexVector& allBetheRoots,
        const MyLongVector& allConfigs,
        const MyComplexVector& allGaudinDets,
        const MyLongVector& sigma,
        const MyComplexType& delta
    );
    
    // Configuration
    void setGpuList(const std::vector<int>& gpu_ids);
    void setSharedMemorySize(size_t bytes_per_gpu);
    void setBlockSize(int block_size);
    
    // Performance monitoring
    double getLastExecutionTime() const { return last_execution_time_; }
    std::vector<double> getGpuUtilization() const;
    std::vector<size_t> getGpuMemoryUsage() const;
    
private:
    std::vector<int> gpu_ids_;
    std::unique_ptr<MultiGPUStreamManager> stream_manager_;
    size_t shared_memory_per_gpu_;
    int block_size_;
    double last_execution_time_;
    
    bool distributeWorkload(
        const MyComplexVector& allBetheRoots,
        const MyLongVector& allConfigs,
        const MyComplexVector& allGaudinDets,
        const MyLongVector& sigma,
        const MyComplexType& delta,
        std::vector<GPUWorkload>& workloads
    );
    
    bool executeWorkload(const std::vector<GPUWorkload>& workloads);
    
    bool collectResults(
        const std::vector<GPUWorkload>& workloads,
        MyComplexVector& e2c,
        MyComplexVector& c2e
    );
    
    void cleanupWorkload(const std::vector<GPUWorkload>& workloads);
    
    MultiGPUKernelParams calculateKernelParams(
        int nBasis, int nUp, int nPerm, int gpu_count
    );
    
    size_t calculateSharedMemoryRequirement(int nBasis, int nUp);
};

// Memory management utilities for multi-GPU kernels
class MultiGPUMemoryManager {
public:
    static VirtualPointer allocateAndCopy(
        const void* host_data,
        size_t bytes,
        int preferred_gpu = -1
    );
    
    static bool copyToHost(
        const VirtualPointer& device_ptr,
        void* host_data,
        size_t bytes
    );
    
    static bool copyBetweenGPUs(
        const VirtualPointer& src,
        VirtualPointer& dst,
        size_t bytes
    );
    
    static void deallocateAll(const std::vector<VirtualPointer>& ptrs);
};

// Legacy interface with multi-GPU support
std::pair<MyComplexVector, MyComplexVector> gpuComputeBasisTransformMultiGPU(
    const MyComplexVector allBetheRoots,
    const MyLongVector allConfigs,
    const MyComplexVector allGaudinDets,
    const MyLongVector sigma,
    const MyComplexType delta
);

// Performance utilities
void benchmarkMultiGPU(
    int nBasis_min, int nBasis_max,
    int nUp_min, int nUp_max,
    const std::vector<int>& gpu_counts
);

void profileMemoryUsage(
    int nBasis, int nUp, int nPerm,
    const std::vector<int>& gpu_ids
);

#endif // MULTIGPU_KERNELS_H_