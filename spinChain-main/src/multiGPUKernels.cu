#include "multiGPUKernels.cuh"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cooperative_groups.h>

typedef thrust::complex<double> GPUComplexType;

// Enhanced device functions with better memory access patterns
__device__ GPUComplexType gpuProductOptimized(const GPUComplexType* vector, unsigned long size) {
    GPUComplexType result(1);
    for (unsigned long i = 0; i < size; i++) {
        result *= vector[i];
    }
    return result;
}

__device__ GPUComplexType intPowerOptimized(GPUComplexType base, unsigned long power) {
    GPUComplexType result(1);
    while (power > 0) {
        if (power % 2 == 1) {
            result *= base;
        }
        base *= base;
        power /= 2;
    }
    return result;
}

__device__ GPUComplexType gpuAmplitudeOptimized(
    const GPUComplexType* betheRoots,
    const unsigned long* sigma,
    unsigned long nUp,
    GPUComplexType delta) {
    
    GPUComplexType amplitude(1);
    
    for (unsigned long i = 0; i < nUp; i++) {
        for (unsigned long j = i+1; j < nUp; j++) {
            if (sigma[i] > sigma[j]) {
                auto si = sigma[i];
                auto sj = sigma[j];
                amplitude *= -(GPUComplexType(1) + betheRoots[si]*betheRoots[sj] - delta*betheRoots[si])
                            / (GPUComplexType(1) + betheRoots[si]*betheRoots[sj] - delta*betheRoots[sj]);
            }
        }
    }
    return amplitude;
}

// Multi-GPU kernel with dynamic memory allocation
__global__ void multiGPUComputeBasisTransform(
    const GPUComplexType* allBetheRoots,
    const unsigned long* allConfigs,
    const GPUComplexType* allGaudinDets,
    const unsigned long* sigma,
    GPUComplexType* e2c_output,
    GPUComplexType* c2e_output,
    GPUComplexType delta,
    int nBasis,
    int nUp,
    int nPerm,
    int start_energy,
    int end_energy) {
    
    int energyIndex = blockIdx.x * blockDim.x + threadIdx.x + start_energy;
    
    if (energyIndex >= end_energy) return;
    
    // Use global memory arrays instead of fixed-size local arrays
    // This allows for larger problem sizes but may be slower
    auto gaudinDet = allGaudinDets[energyIndex];
    auto betheProduct = gpuProductOptimized(&allBetheRoots[energyIndex * nUp], nUp);
    
    for (unsigned long permIndex = 0; permIndex < nPerm; permIndex++) {
        auto amplitude = gpuAmplitudeOptimized(
            &allBetheRoots[energyIndex * nUp], 
            &sigma[permIndex * nUp], 
            nUp, 
            delta
        );
        
        for (unsigned long positionIndex = 0; positionIndex < nBasis; positionIndex++) {
            GPUComplexType accum(1);
            for (unsigned long i = 0; i < nUp; i++) {
                accum *= intPowerOptimized(
                    allBetheRoots[energyIndex * nUp + sigma[permIndex * nUp + i]], 
                    allConfigs[positionIndex * nUp + i]
                );
            }
            
            // Atomic operations for thread-safe accumulation
            // Note: This is a simplified version - proper atomic complex addition would be needed
            GPUComplexType e2c_term = amplitude * accum;
            GPUComplexType c2e_term = GPUComplexType(1) / (gaudinDet * betheProduct * amplitude * accum);
            
            // For now, assuming each thread handles a complete energy state
            // In a more sophisticated version, we'd need atomic operations
            e2c_output[energyIndex * nBasis + positionIndex] += e2c_term;
            c2e_output[positionIndex * nBasis + energyIndex] += c2e_term;
        }
    }
}

// Optimized kernel using shared memory
__global__ void multiGPUComputeBasisTransformShared(
    const GPUComplexType* allBetheRoots,
    const unsigned long* allConfigs,
    const GPUComplexType* allGaudinDets,
    const unsigned long* sigma,
    GPUComplexType* e2c_output,
    GPUComplexType* c2e_output,
    GPUComplexType delta,
    int nBasis,
    int nUp,
    int nPerm,
    int start_energy,
    int end_energy) {
    
    extern __shared__ GPUComplexType shared_memory[];
    
    int energyIndex = blockIdx.x * blockDim.x + threadIdx.x + start_energy;
    
    if (energyIndex >= end_energy) return;
    
    // Calculate shared memory layout
    GPUComplexType* e2cVector = &shared_memory[threadIdx.x * nBasis * 2];
    GPUComplexType* c2eVector = &shared_memory[threadIdx.x * nBasis * 2 + nBasis];
    
    // Initialize shared memory arrays
    for (int i = 0; i < nBasis; i++) {
        e2cVector[i] = GPUComplexType(0);
        c2eVector[i] = GPUComplexType(0);
    }
    
    auto gaudinDet = allGaudinDets[energyIndex];
    auto betheProduct = gpuProductOptimized(&allBetheRoots[energyIndex * nUp], nUp);
    
    for (unsigned long permIndex = 0; permIndex < nPerm; permIndex++) {
        auto amplitude = gpuAmplitudeOptimized(
            &allBetheRoots[energyIndex * nUp], 
            &sigma[permIndex * nUp], 
            nUp, 
            delta
        );
        
        for (unsigned long positionIndex = 0; positionIndex < nBasis; positionIndex++) {
            GPUComplexType accum(1);
            for (unsigned long i = 0; i < nUp; i++) {
                accum *= intPowerOptimized(
                    allBetheRoots[energyIndex * nUp + sigma[permIndex * nUp + i]], 
                    allConfigs[positionIndex * nUp + i]
                );
            }
            
            e2cVector[positionIndex] += amplitude * accum;
            c2eVector[positionIndex] += GPUComplexType(1) / (gaudinDet * betheProduct * amplitude * accum);
        }
    }
    
    // Copy results to global memory
    for (int i = 0; i < nBasis; i++) {
        e2c_output[energyIndex * nBasis + i] = e2cVector[i];
        c2e_output[i * nBasis + energyIndex] = c2eVector[i];
    }
}

// MultiGPUBasisTransform implementation
MultiGPUBasisTransform::MultiGPUBasisTransform() 
    : shared_memory_per_gpu_(0), block_size_(256), last_execution_time_(0.0) {
    
    // Initialize with all available GPUs by default
    if (g_multiGpuController) {
        gpu_ids_ = g_multiGpuController->getActiveDevices();
    } else {
        initializeMultiGPU();
        if (g_multiGpuController) {
            gpu_ids_ = g_multiGpuController->getActiveDevices();
        }
    }
    
    if (!gpu_ids_.empty()) {
        stream_manager_ = std::make_unique<MultiGPUStreamManager>(gpu_ids_);
        stream_manager_->createStreams(2); // 2 streams per GPU
    }
}

MultiGPUBasisTransform::~MultiGPUBasisTransform() {
    if (stream_manager_) {
        stream_manager_->destroyStreams();
    }
}

std::pair<MyComplexVector, MyComplexVector> MultiGPUBasisTransform::compute(
    const MyComplexVector& allBetheRoots,
    const MyLongVector& allConfigs,
    const MyComplexVector& allGaudinDets,
    const MyLongVector& sigma,
    const MyComplexType& delta) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<GPUWorkload> workloads;
    
    // Distribute workload across GPUs
    if (!distributeWorkload(allBetheRoots, allConfigs, allGaudinDets, sigma, delta, workloads)) {
        std::cerr << "Failed to distribute workload across GPUs" << std::endl;
        return {};
    }
    
    // Execute computation on all GPUs
    if (!executeWorkload(workloads)) {
        std::cerr << "Failed to execute workload on GPUs" << std::endl;
        cleanupWorkload(workloads);
        return {};
    }
    
    // Collect results from all GPUs
    MyComplexVector e2c, c2e;
    if (!collectResults(workloads, e2c, c2e)) {
        std::cerr << "Failed to collect results from GPUs" << std::endl;
        cleanupWorkload(workloads);
        return {};
    }
    
    // Cleanup
    cleanupWorkload(workloads);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_execution_time_ = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "Multi-GPU computation completed in " << last_execution_time_ << " seconds" << std::endl;
    
    return {e2c, c2e};
}

void MultiGPUBasisTransform::setGpuList(const std::vector<int>& gpu_ids) {
    gpu_ids_ = gpu_ids;
    
    // Recreate stream manager with new GPU list
    if (stream_manager_) {
        stream_manager_->destroyStreams();
    }
    stream_manager_ = std::make_unique<MultiGPUStreamManager>(gpu_ids_);
    stream_manager_->createStreams(2);
}

void MultiGPUBasisTransform::setSharedMemorySize(size_t bytes_per_gpu) {
    shared_memory_per_gpu_ = bytes_per_gpu;
}

void MultiGPUBasisTransform::setBlockSize(int block_size) {
    block_size_ = block_size;
}

bool MultiGPUBasisTransform::distributeWorkload(
    const MyComplexVector& allBetheRoots,
    const MyLongVector& allConfigs,
    const MyComplexVector& allGaudinDets,
    const MyLongVector& sigma,
    const MyComplexType& delta,
    std::vector<GPUWorkload>& workloads) {
    
    int nBasis = allGaudinDets.size();
    int nUp = allBetheRoots.size() / nBasis;
    int nPerm = sigma.size() / nUp;
    
    // Distribute energy states across GPUs
    WorkDistributor distributor(gpu_ids_);
    auto chunks = distributor.distributeEvenly(nBasis);
    
    workloads.resize(chunks.size());
    
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        auto& workload = workloads[i];
        
        workload.gpu_id = chunk.gpu_id;
        workload.start_energy = chunk.start_index;
        workload.end_energy = chunk.end_index;
        workload.energy_count = chunk.work_size;
        workload.stream = stream_manager_->getStream(chunk.gpu_id, 0);
        
        // Set active GPU
        g_multiGpuController->setActiveDevice(chunk.gpu_id);
        
        // Allocate and copy input data
        size_t bethe_roots_size = workload.energy_count * nUp * sizeof(MyComplexType);
        size_t configs_size = nBasis * nUp * sizeof(MyLongType);
        size_t gaudin_dets_size = workload.energy_count * sizeof(MyComplexType);
        size_t sigma_size = nPerm * nUp * sizeof(MyLongType);
        
        // Allocate GPU memory using multi-GPU pool
        workload.d_allBetheRoots = allocateMultiGPU(bethe_roots_size, chunk.gpu_id);
        workload.d_allConfigs = allocateMultiGPU(configs_size, chunk.gpu_id);
        workload.d_allGaudinDets = allocateMultiGPU(gaudin_dets_size, chunk.gpu_id);
        workload.d_sigma = allocateMultiGPU(sigma_size, chunk.gpu_id);
        
        // Allocate output memory
        size_t output_size = workload.energy_count * nBasis * sizeof(MyComplexType);
        workload.d_e2c = allocateMultiGPU(output_size, chunk.gpu_id);
        workload.d_c2e = allocateMultiGPU(output_size, chunk.gpu_id);
        
        if (!workload.d_allBetheRoots.isValid() || !workload.d_allConfigs.isValid() ||
            !workload.d_allGaudinDets.isValid() || !workload.d_sigma.isValid() ||
            !workload.d_e2c.isValid() || !workload.d_c2e.isValid()) {
            std::cerr << "Failed to allocate GPU memory for GPU " << chunk.gpu_id << std::endl;
            return false;
        }
        
        // Copy input data to GPU
        cudaMemcpyAsync(workload.d_allBetheRoots.getDevicePtr(),
                       &allBetheRoots[workload.start_energy * nUp],
                       bethe_roots_size, cudaMemcpyHostToDevice, workload.stream);
        
        cudaMemcpyAsync(workload.d_allConfigs.getDevicePtr(),
                       allConfigs.data(), configs_size, 
                       cudaMemcpyHostToDevice, workload.stream);
        
        cudaMemcpyAsync(workload.d_allGaudinDets.getDevicePtr(),
                       &allGaudinDets[workload.start_energy],
                       gaudin_dets_size, cudaMemcpyHostToDevice, workload.stream);
        
        cudaMemcpyAsync(workload.d_sigma.getDevicePtr(),
                       sigma.data(), sigma_size,
                       cudaMemcpyHostToDevice, workload.stream);
        
        // Initialize output arrays to zero
        cudaMemsetAsync(workload.d_e2c.getDevicePtr(), 0, output_size, workload.stream);
        cudaMemsetAsync(workload.d_c2e.getDevicePtr(), 0, output_size, workload.stream);
    }
    
    return true;
}

bool MultiGPUBasisTransform::executeWorkload(const std::vector<GPUWorkload>& workloads) {
    for (const auto& workload : workloads) {
        g_multiGpuController->setActiveDevice(workload.gpu_id);
        
        // Calculate kernel launch parameters
        int grid_size = (workload.energy_count + block_size_ - 1) / block_size_;
        
        // Determine which kernel to use based on available shared memory
        size_t shared_mem_required = calculateSharedMemoryRequirement(
            workload.energy_count, // This should be nBasis, but we're working per energy state
            workload.d_allBetheRoots.getSize() / (workload.energy_count * sizeof(MyComplexType)) // nUp
        );
        
        GPUComplexType delta_gpu(std::real(MyComplexType(1.0)), std::imag(MyComplexType(1.0)));
        
        if (shared_mem_required <= shared_memory_per_gpu_ && shared_memory_per_gpu_ > 0) {
            // Use shared memory kernel
            multiGPUComputeBasisTransformShared<<<grid_size, block_size_, 
                                                 shared_mem_required, workload.stream>>>(
                static_cast<const GPUComplexType*>(workload.d_allBetheRoots.getDevicePtr()),
                static_cast<const unsigned long*>(workload.d_allConfigs.getDevicePtr()),
                static_cast<const GPUComplexType*>(workload.d_allGaudinDets.getDevicePtr()),
                static_cast<const unsigned long*>(workload.d_sigma.getDevicePtr()),
                static_cast<GPUComplexType*>(workload.d_e2c.getDevicePtr()),
                static_cast<GPUComplexType*>(workload.d_c2e.getDevicePtr()),
                delta_gpu,
                workload.energy_count, // nBasis for this GPU
                workload.d_allBetheRoots.getSize() / (workload.energy_count * sizeof(MyComplexType)), // nUp
                workload.d_sigma.getSize() / (workload.d_allBetheRoots.getSize() / (workload.energy_count * sizeof(MyComplexType)) * sizeof(MyLongType)), // nPerm
                0, // start_energy relative to this GPU
                workload.energy_count // end_energy relative to this GPU
            );
        } else {
            // Use global memory kernel
            multiGPUComputeBasisTransform<<<grid_size, block_size_, 0, workload.stream>>>(
                static_cast<const GPUComplexType*>(workload.d_allBetheRoots.getDevicePtr()),
                static_cast<const unsigned long*>(workload.d_allConfigs.getDevicePtr()),
                static_cast<const GPUComplexType*>(workload.d_allGaudinDets.getDevicePtr()),
                static_cast<const unsigned long*>(workload.d_sigma.getDevicePtr()),
                static_cast<GPUComplexType*>(workload.d_e2c.getDevicePtr()),
                static_cast<GPUComplexType*>(workload.d_c2e.getDevicePtr()),
                delta_gpu,
                workload.energy_count, // nBasis for this GPU
                workload.d_allBetheRoots.getSize() / (workload.energy_count * sizeof(MyComplexType)), // nUp
                workload.d_sigma.getSize() / (workload.d_allBetheRoots.getSize() / (workload.energy_count * sizeof(MyComplexType)) * sizeof(MyLongType)), // nPerm
                0, // start_energy relative to this GPU
                workload.energy_count // end_energy relative to this GPU
            );
        }
        
        // Check for kernel launch errors
        cudaError_t launch_error = cudaGetLastError();
        if (launch_error != cudaSuccess) {
            std::cerr << "Kernel launch failed on GPU " << workload.gpu_id 
                      << ": " << cudaGetErrorString(launch_error) << std::endl;
            return false;
        }
    }
    
    // Synchronize all streams
    return stream_manager_->synchronizeAllStreams();
}

bool MultiGPUBasisTransform::collectResults(
    const std::vector<GPUWorkload>& workloads,
    MyComplexVector& e2c,
    MyComplexVector& c2e) {
    
    // Calculate total result size
    int total_nBasis = 0;
    for (const auto& workload : workloads) {
        total_nBasis += workload.energy_count;
    }
    
    e2c.resize(total_nBasis * total_nBasis);
    c2e.resize(total_nBasis * total_nBasis);
    
    // Copy results from each GPU
    for (const auto& workload : workloads) {
        g_multiGpuController->setActiveDevice(workload.gpu_id);
        
        size_t result_size = workload.energy_count * total_nBasis * sizeof(MyComplexType);
        
        // Copy e2c results
        cudaMemcpyAsync(&e2c[workload.start_energy * total_nBasis],
                       workload.d_e2c.getDevicePtr(),
                       result_size, cudaMemcpyDeviceToHost, workload.stream);
        
        // Copy c2e results (note the transposed indexing)
        for (int i = 0; i < workload.energy_count; ++i) {
            cudaMemcpyAsync(&c2e[(workload.start_energy + i) * total_nBasis],
                           static_cast<const MyComplexType*>(workload.d_c2e.getDevicePtr()) + i * total_nBasis,
                           total_nBasis * sizeof(MyComplexType), 
                           cudaMemcpyDeviceToHost, workload.stream);
        }
    }
    
    // Synchronize all transfers
    return stream_manager_->synchronizeAllStreams();
}

void MultiGPUBasisTransform::cleanupWorkload(const std::vector<GPUWorkload>& workloads) {
    for (const auto& workload : workloads) {
        deallocateMultiGPU(workload.d_allBetheRoots);
        deallocateMultiGPU(workload.d_allConfigs);
        deallocateMultiGPU(workload.d_allGaudinDets);
        deallocateMultiGPU(workload.d_sigma);
        deallocateMultiGPU(workload.d_e2c);
        deallocateMultiGPU(workload.d_c2e);
    }
}

size_t MultiGPUBasisTransform::calculateSharedMemoryRequirement(int nBasis, int nUp) {
    // Each thread needs 2 * nBasis complex numbers in shared memory
    return block_size_ * 2 * nBasis * sizeof(GPUComplexType);
}

std::vector<double> MultiGPUBasisTransform::getGpuUtilization() const {
    std::vector<double> utilization;
    if (g_multiGpuController) {
        for (int gpu_id : gpu_ids_) {
            utilization.push_back(g_multiGpuController->getDeviceUtilization(gpu_id));
        }
    }
    return utilization;
}

std::vector<size_t> MultiGPUBasisTransform::getGpuMemoryUsage() const {
    std::vector<size_t> memory_usage;
    if (g_multiGpuController) {
        for (int gpu_id : gpu_ids_) {
            memory_usage.push_back(g_multiGpuController->getDeviceMemoryUsage(gpu_id));
        }
    }
    return memory_usage;
}

// Legacy interface implementation
std::pair<MyComplexVector, MyComplexVector> gpuComputeBasisTransformMultiGPU(
    const MyComplexVector allBetheRoots,
    const MyLongVector allConfigs,
    const MyComplexVector allGaudinDets,
    const MyLongVector sigma,
    const MyComplexType delta) {
    
    static MultiGPUBasisTransform transformer;
    return transformer.compute(allBetheRoots, allConfigs, allGaudinDets, sigma, delta);
}