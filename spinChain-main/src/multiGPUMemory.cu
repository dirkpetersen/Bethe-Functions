#include "multiGPUMemory.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>

// Global multi-GPU memory pool instance
std::unique_ptr<MultiGPUMemoryPool> g_multiGpuPool = nullptr;

// GPUMemoryPool implementation
GPUMemoryPool::GPUMemoryPool(int gpu_id) : gpu_id_(gpu_id), total_memory_(0), available_memory_(0) {
    // Set active device
    cudaSetDevice(gpu_id_);
    
    // Get memory information
    updateMemoryInfo();
    
    std::cout << "Initialized GPU " << gpu_id_ << " with " 
              << (total_memory_ / (1024*1024)) << " MB total memory" << std::endl;
}

GPUMemoryPool::~GPUMemoryPool() {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    // Free all remaining allocated blocks
    cudaSetDevice(gpu_id_);
    for (auto& block : allocated_blocks_) {
        cudaFree(block.first);
    }
    allocated_blocks_.clear();
}

VirtualPointer GPUMemoryPool::allocate(size_t bytes) {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    if (bytes > available_memory_) {
        return VirtualPointer(); // Invalid pointer - not enough memory
    }
    
    cudaSetDevice(gpu_id_);
    
    void* device_ptr = nullptr;
    cudaError_t result = cudaMalloc(&device_ptr, bytes);
    
    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate " << bytes << " bytes on GPU " << gpu_id_ 
                  << ": " << cudaGetErrorString(result) << std::endl;
        return VirtualPointer();
    }
    
    allocated_blocks_[device_ptr] = bytes;
    available_memory_ -= bytes;
    
    return VirtualPointer(gpu_id_, device_ptr, bytes);
}

bool GPUMemoryPool::deallocate(const VirtualPointer& ptr) {
    if (!ptr.isValid() || ptr.getGpuId() != gpu_id_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    auto it = allocated_blocks_.find(ptr.getDevicePtr());
    if (it == allocated_blocks_.end()) {
        return false;
    }
    
    cudaSetDevice(gpu_id_);
    cudaFree(ptr.getDevicePtr());
    
    available_memory_ += it->second;
    allocated_blocks_.erase(it);
    
    return true;
}

size_t GPUMemoryPool::getAvailableMemory() const {
    return available_memory_;
}

size_t GPUMemoryPool::getTotalMemory() const {
    return total_memory_;
}

size_t GPUMemoryPool::getUsedMemory() const {
    return total_memory_ - available_memory_;
}

bool GPUMemoryPool::copyTo(const VirtualPointer& src, VirtualPointer& dst, int target_gpu) {
    if (!src.isValid() || src.getGpuId() != gpu_id_) {
        return false;
    }
    
    // For now, use host memory as intermediate buffer
    // TODO: Implement direct P2P transfers when available
    
    std::vector<char> host_buffer(src.getSize());
    
    // Copy from source GPU to host
    cudaSetDevice(gpu_id_);
    cudaError_t result = cudaMemcpy(host_buffer.data(), src.getDevicePtr(), 
                                   src.getSize(), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        return false;
    }
    
    // Copy from host to destination GPU
    cudaSetDevice(target_gpu);
    result = cudaMemcpy(dst.getDevicePtr(), host_buffer.data(), 
                       src.getSize(), cudaMemcpyHostToDevice);
    
    return result == cudaSuccess;
}

bool GPUMemoryPool::copyFrom(const VirtualPointer& src, VirtualPointer& dst) {
    if (!dst.isValid() || dst.getGpuId() != gpu_id_) {
        return false;
    }
    
    // Similar to copyTo but in reverse direction
    std::vector<char> host_buffer(src.getSize());
    
    // Copy from source GPU to host
    cudaSetDevice(src.getGpuId());
    cudaError_t result = cudaMemcpy(host_buffer.data(), src.getDevicePtr(), 
                                   src.getSize(), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        return false;
    }
    
    // Copy from host to this GPU
    cudaSetDevice(gpu_id_);
    result = cudaMemcpy(dst.getDevicePtr(), host_buffer.data(), 
                       src.getSize(), cudaMemcpyHostToDevice);
    
    return result == cudaSuccess;
}

void GPUMemoryPool::updateMemoryInfo() {
    cudaSetDevice(gpu_id_);
    
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    
    total_memory_ = total_bytes;
    // Keep some memory reserved for system operations
    available_memory_ = free_bytes - (free_bytes / 10); // Reserve 10%
}

// VirtualAddressSpace implementation
uint64_t VirtualAddressSpace::registerPointer(const VirtualPointer& ptr) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    
    uint64_t virtual_id = next_virtual_id_++;
    virtual_map_[virtual_id] = ptr;
    return virtual_id;
}

VirtualPointer VirtualAddressSpace::getPointer(uint64_t virtual_id) const {
    std::lock_guard<std::mutex> lock(map_mutex_);
    
    auto it = virtual_map_.find(virtual_id);
    if (it != virtual_map_.end()) {
        return it->second;
    }
    return VirtualPointer(); // Invalid pointer
}

bool VirtualAddressSpace::unregisterPointer(uint64_t virtual_id) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    
    return virtual_map_.erase(virtual_id) > 0;
}

// MemoryMigrationManager implementation
MemoryMigrationManager::MemoryMigrationManager(const std::vector<std::shared_ptr<GPUMemoryPool>>& pools) 
    : gpu_pools_(pools) {
    initializeP2PAccess();
}

bool MemoryMigrationManager::migrate(VirtualPointer& ptr, int target_gpu) {
    if (!ptr.isValid() || target_gpu < 0 || target_gpu >= gpu_pools_.size()) {
        return false;
    }
    
    if (ptr.getGpuId() == target_gpu) {
        return true; // Already on target GPU
    }
    
    // Allocate memory on target GPU
    VirtualPointer new_ptr = gpu_pools_[target_gpu]->allocate(ptr.getSize());
    if (!new_ptr.isValid()) {
        return false; // Failed to allocate on target GPU
    }
    
    // Copy data
    bool copy_success = false;
    if (canMigrate(ptr.getGpuId(), target_gpu)) {
        // Use direct P2P if available
        copy_success = gpu_pools_[ptr.getGpuId()]->copyTo(ptr, new_ptr, target_gpu);
    } else {
        // Use host memory as intermediate
        copy_success = gpu_pools_[target_gpu]->copyFrom(ptr, new_ptr);
    }
    
    if (copy_success) {
        // Free old memory and update pointer
        gpu_pools_[ptr.getGpuId()]->deallocate(ptr);
        ptr = new_ptr;
        return true;
    } else {
        // Failed to copy, clean up allocated memory
        gpu_pools_[target_gpu]->deallocate(new_ptr);
        return false;
    }
}

bool MemoryMigrationManager::canMigrate(int src_gpu, int dst_gpu) const {
    if (src_gpu < 0 || src_gpu >= p2p_access_matrix_.size() ||
        dst_gpu < 0 || dst_gpu >= p2p_access_matrix_[src_gpu].size()) {
        return false;
    }
    return p2p_access_matrix_[src_gpu][dst_gpu];
}

void MemoryMigrationManager::initializeP2PAccess() {
    int gpu_count = gpu_pools_.size();
    p2p_access_matrix_.resize(gpu_count);
    
    for (int i = 0; i < gpu_count; ++i) {
        p2p_access_matrix_[i].resize(gpu_count, false);
        p2p_access_matrix_[i][i] = true; // Self-access always available
        
        for (int j = 0; j < gpu_count; ++j) {
            if (i != j) {
                p2p_access_matrix_[i][j] = enableP2PAccess(i, j);
            }
        }
    }
}

bool MemoryMigrationManager::enableP2PAccess(int gpu1, int gpu2) {
    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, gpu1, gpu2);
    
    if (can_access) {
        cudaSetDevice(gpu1);
        cudaError_t result = cudaDeviceEnablePeerAccess(gpu2, 0);
        if (result == cudaSuccess || result == cudaErrorPeerAccessAlreadyEnabled) {
            std::cout << "Enabled P2P access from GPU " << gpu1 << " to GPU " << gpu2 << std::endl;
            return true;
        }
    }
    
    return false;
}

// MultiGPUMemoryPool implementation
MultiGPUMemoryPool::MultiGPUMemoryPool() : active_gpu_(0) {
    initializeGpuPools();
    
    virtual_space_ = std::make_unique<VirtualAddressSpace>();
    migration_manager_ = std::make_unique<MemoryMigrationManager>(gpu_pools_);
    
    std::cout << "Initialized multi-GPU memory pool with " << gpu_pools_.size() << " GPUs" << std::endl;
}

MultiGPUMemoryPool::~MultiGPUMemoryPool() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    gpu_pools_.clear();
    virtual_space_.reset();
    migration_manager_.reset();
}

VirtualPointer MultiGPUMemoryPool::allocate(size_t bytes, int preferred_gpu) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    int target_gpu = preferred_gpu;
    if (target_gpu < 0 || !isValidGpu(target_gpu)) {
        target_gpu = selectBestGpu(bytes);
    }
    
    if (target_gpu < 0) {
        std::cerr << "No suitable GPU found for allocation of " << bytes << " bytes" << std::endl;
        return VirtualPointer();
    }
    
    return gpu_pools_[target_gpu]->allocate(bytes);
}

bool MultiGPUMemoryPool::deallocate(const VirtualPointer& ptr) {
    if (!ptr.isValid()) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    int gpu_id = ptr.getGpuId();
    if (!isValidGpu(gpu_id)) {
        return false;
    }
    
    return gpu_pools_[gpu_id]->deallocate(ptr);
}

bool MultiGPUMemoryPool::migrate(VirtualPointer& ptr, int target_gpu) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (!isValidGpu(target_gpu)) {
        return false;
    }
    
    return migration_manager_->migrate(ptr, target_gpu);
}

size_t MultiGPUMemoryPool::getTotalMemory() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    size_t total = 0;
    for (const auto& pool : gpu_pools_) {
        total += pool->getTotalMemory();
    }
    return total;
}

size_t MultiGPUMemoryPool::getAvailableMemory() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    size_t available = 0;
    for (const auto& pool : gpu_pools_) {
        available += pool->getAvailableMemory();
    }
    return available;
}

size_t MultiGPUMemoryPool::getUsedMemory() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    size_t used = 0;
    for (const auto& pool : gpu_pools_) {
        used += pool->getUsedMemory();
    }
    return used;
}

int MultiGPUMemoryPool::selectBestGpu(size_t bytes) const {
    int best_gpu = -1;
    size_t max_available = 0;
    
    for (int i = 0; i < gpu_pools_.size(); ++i) {
        size_t available = gpu_pools_[i]->getAvailableMemory();
        if (available >= bytes && available > max_available) {
            max_available = available;
            best_gpu = i;
        }
    }
    
    return best_gpu;
}

std::vector<int> MultiGPUMemoryPool::getAvailableGpus() const {
    std::vector<int> available_gpus;
    for (int i = 0; i < gpu_pools_.size(); ++i) {
        available_gpus.push_back(i);
    }
    return available_gpus;
}

bool MultiGPUMemoryPool::setActiveGpu(int gpu_id) {
    if (!isValidGpu(gpu_id)) {
        return false;
    }
    
    active_gpu_ = gpu_id;
    cudaSetDevice(gpu_id);
    return true;
}

int MultiGPUMemoryPool::getActiveGpu() const {
    return active_gpu_;
}

void MultiGPUMemoryPool::initializeGpuPools() {
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    
    if (gpu_count == 0) {
        throw std::runtime_error("No CUDA devices found");
    }
    
    gpu_pools_.reserve(gpu_count);
    
    for (int i = 0; i < gpu_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        // Check if GPU has sufficient compute capability
        if (prop.major >= 6 || (prop.major == 6 && prop.minor >= 1)) {
            gpu_pools_.push_back(std::make_shared<GPUMemoryPool>(i));
        } else {
            std::cout << "Skipping GPU " << i << " (compute capability " 
                      << prop.major << "." << prop.minor << " < 6.1)" << std::endl;
        }
    }
    
    if (gpu_pools_.empty()) {
        throw std::runtime_error("No suitable GPUs found (compute capability >= 6.1 required)");
    }
}

bool MultiGPUMemoryPool::isValidGpu(int gpu_id) const {
    return gpu_id >= 0 && gpu_id < gpu_pools_.size();
}

// Global convenience functions
VirtualPointer allocateMultiGPU(size_t bytes, int preferred_gpu) {
    if (!g_multiGpuPool) {
        g_multiGpuPool = std::make_unique<MultiGPUMemoryPool>();
    }
    return g_multiGpuPool->allocate(bytes, preferred_gpu);
}

bool deallocateMultiGPU(const VirtualPointer& ptr) {
    if (!g_multiGpuPool) {
        return false;
    }
    return g_multiGpuPool->deallocate(ptr);
}

bool migrateMultiGPU(VirtualPointer& ptr, int target_gpu) {
    if (!g_multiGpuPool) {
        return false;
    }
    return g_multiGpuPool->migrate(ptr, target_gpu);
}