#ifndef MULTIGPU_MEMORY_H_
#define MULTIGPU_MEMORY_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "defs.h"

// Forward declarations
class MultiGPUMemoryPool;
class VirtualPointer;

/**
 * @brief Virtual pointer that abstracts GPU memory across multiple devices
 */
class VirtualPointer {
public:
    VirtualPointer() : gpu_id_(-1), device_ptr_(nullptr), size_(0) {}
    VirtualPointer(int gpu_id, void* device_ptr, size_t size) 
        : gpu_id_(gpu_id), device_ptr_(device_ptr), size_(size) {}
    
    int getGpuId() const { return gpu_id_; }
    void* getDevicePtr() const { return device_ptr_; }
    size_t getSize() const { return size_; }
    bool isValid() const { return gpu_id_ >= 0 && device_ptr_ != nullptr; }
    
private:
    int gpu_id_;
    void* device_ptr_;
    size_t size_;
};

/**
 * @brief Memory pool for a single GPU
 */
class GPUMemoryPool {
public:
    GPUMemoryPool(int gpu_id);
    ~GPUMemoryPool();
    
    VirtualPointer allocate(size_t bytes);
    bool deallocate(const VirtualPointer& ptr);
    size_t getAvailableMemory() const;
    size_t getTotalMemory() const;
    size_t getUsedMemory() const;
    
    // Memory migration support
    bool copyTo(const VirtualPointer& src, VirtualPointer& dst, int target_gpu);
    bool copyFrom(const VirtualPointer& src, VirtualPointer& dst);
    
private:
    int gpu_id_;
    size_t total_memory_;
    size_t available_memory_;
    std::mutex memory_mutex_;
    
    // Simple memory tracking - could be enhanced with more sophisticated allocator
    std::unordered_map<void*, size_t> allocated_blocks_;
    
    void updateMemoryInfo();
};

/**
 * @brief Virtual address space manager across multiple GPUs
 */
class VirtualAddressSpace {
public:
    VirtualAddressSpace() : next_virtual_id_(0) {}
    
    uint64_t registerPointer(const VirtualPointer& ptr);
    VirtualPointer getPointer(uint64_t virtual_id) const;
    bool unregisterPointer(uint64_t virtual_id);
    
private:
    uint64_t next_virtual_id_;
    std::unordered_map<uint64_t, VirtualPointer> virtual_map_;
    mutable std::mutex map_mutex_;
};

/**
 * @brief Memory migration manager for moving data between GPUs
 */
class MemoryMigrationManager {
public:
    MemoryMigrationManager(const std::vector<std::shared_ptr<GPUMemoryPool>>& pools);
    
    bool migrate(VirtualPointer& ptr, int target_gpu);
    bool canMigrate(int src_gpu, int dst_gpu) const;
    
private:
    std::vector<std::shared_ptr<GPUMemoryPool>> gpu_pools_;
    std::vector<std::vector<bool>> p2p_access_matrix_;
    
    void initializeP2PAccess();
    bool enableP2PAccess(int gpu1, int gpu2);
};

/**
 * @brief Main multi-GPU memory pool managing all GPU memory as unified virtual space
 */
class MultiGPUMemoryPool {
public:
    MultiGPUMemoryPool();
    ~MultiGPUMemoryPool();
    
    // Memory allocation/deallocation
    VirtualPointer allocate(size_t bytes, int preferred_gpu = -1);
    bool deallocate(const VirtualPointer& ptr);
    
    // Memory migration
    bool migrate(VirtualPointer& ptr, int target_gpu);
    
    // Memory information
    size_t getTotalMemory() const;
    size_t getAvailableMemory() const;
    size_t getUsedMemory() const;
    int getGpuCount() const { return gpu_pools_.size(); }
    
    // GPU selection strategies
    int selectBestGpu(size_t bytes) const;
    std::vector<int> getAvailableGpus() const;
    
    // Utility functions
    bool setActiveGpu(int gpu_id);
    int getActiveGpu() const;
    
private:
    std::vector<std::shared_ptr<GPUMemoryPool>> gpu_pools_;
    std::unique_ptr<VirtualAddressSpace> virtual_space_;
    std::unique_ptr<MemoryMigrationManager> migration_manager_;
    
    int active_gpu_;
    mutable std::mutex pool_mutex_;
    
    void initializeGpuPools();
    bool isValidGpu(int gpu_id) const;
};

// Global multi-GPU memory pool instance
extern std::unique_ptr<MultiGPUMemoryPool> g_multiGpuPool;

// Convenience functions for global pool access
VirtualPointer allocateMultiGPU(size_t bytes, int preferred_gpu = -1);
bool deallocateMultiGPU(const VirtualPointer& ptr);
bool migrateMultiGPU(VirtualPointer& ptr, int target_gpu);

#endif // MULTIGPU_MEMORY_H_