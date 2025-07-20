#ifndef MULTIGPU_CONTROL_H_
#define MULTIGPU_CONTROL_H_

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "multiGPUMemory.h"
#include "defs.h"

/**
 * @brief GPU device information structure
 */
struct GPUDeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t available_memory;
    int compute_major;
    int compute_minor;
    bool p2p_capable;
    bool is_active;
    
    GPUDeviceInfo() : device_id(-1), total_memory(0), available_memory(0),
                     compute_major(0), compute_minor(0), p2p_capable(false), is_active(false) {}
};

/**
 * @brief Multi-GPU device management and coordination
 */
class MultiGPUController {
public:
    MultiGPUController();
    ~MultiGPUController();
    
    // Device discovery and initialization
    bool initializeDevices();
    int getDeviceCount() const { return gpu_devices_.size(); }
    const std::vector<GPUDeviceInfo>& getDeviceInfo() const { return gpu_devices_; }
    
    // Device selection and management
    bool setActiveDevice(int device_id);
    int getActiveDevice() const;
    std::vector<int> getActiveDevices() const;
    bool setActiveDevices(const std::vector<int>& device_ids);
    
    // Memory management integration
    MultiGPUMemoryPool* getMemoryPool() { return memory_pool_.get(); }
    
    // Performance monitoring
    void updateDeviceUtilization();
    double getDeviceUtilization(int device_id) const;
    size_t getDeviceMemoryUsage(int device_id) const;
    
    // Load balancing
    int selectOptimalDevice(size_t memory_requirement) const;
    std::vector<int> distributeWorkload(int total_work_items) const;
    
    // Synchronization
    bool synchronizeDevice(int device_id);
    bool synchronizeAllDevices();
    
    // Error handling
    bool isDeviceValid(int device_id) const;
    std::string getLastError() const { return last_error_; }
    
private:
    std::vector<GPUDeviceInfo> gpu_devices_;
    std::vector<int> active_devices_;
    int primary_device_;
    std::unique_ptr<MultiGPUMemoryPool> memory_pool_;
    std::string last_error_;
    
    bool discoverDevices();
    bool validateDevice(int device_id, GPUDeviceInfo& info);
    void initializeP2PAccess();
    void setError(const std::string& error);
};

/**
 * @brief Work distribution strategy for multi-GPU execution
 */
class WorkDistributor {
public:
    struct WorkChunk {
        int gpu_id;
        int start_index;
        int end_index;
        int work_size;
    };
    
    WorkDistributor(const std::vector<int>& available_gpus);
    
    // Distribution strategies
    std::vector<WorkChunk> distributeEvenly(int total_work) const;
    std::vector<WorkChunk> distributeByCapability(int total_work, 
                                                 const std::vector<double>& gpu_capabilities) const;
    std::vector<WorkChunk> distributeByMemory(int total_work, 
                                            const std::vector<size_t>& memory_requirements) const;
    
    // Dynamic load balancing
    void adjustDistribution(std::vector<WorkChunk>& chunks, 
                           const std::vector<double>& completion_rates) const;
    
private:
    std::vector<int> available_gpus_;
    
    int calculateOptimalChunkSize(int total_work, int gpu_count) const;
};

/**
 * @brief CUDA stream manager for multi-GPU operations
 */
class MultiGPUStreamManager {
public:
    MultiGPUStreamManager(const std::vector<int>& gpu_ids);
    ~MultiGPUStreamManager();
    
    // Stream management
    cudaStream_t getStream(int gpu_id, int stream_index = 0);
    bool createStreams(int streams_per_gpu = 2);
    bool destroyStreams();
    
    // Synchronization
    bool synchronizeStream(int gpu_id, int stream_index = 0);
    bool synchronizeAllStreams();
    
    // Event management for cross-GPU synchronization
    bool createEvent(int gpu_id, const std::string& event_name);
    bool recordEvent(int gpu_id, const std::string& event_name, int stream_index = 0);
    bool waitForEvent(int gpu_id, const std::string& event_name, int stream_index = 0);
    
private:
    std::vector<int> gpu_ids_;
    std::map<int, std::vector<cudaStream_t>> streams_per_gpu_;
    std::map<int, std::map<std::string, cudaEvent_t>> events_per_gpu_;
    
    bool initialized_;
};

// Global multi-GPU controller instance
extern std::unique_ptr<MultiGPUController> g_multiGpuController;

// Convenience functions for global controller access
bool initializeMultiGPU();
void shutdownMultiGPU();
MultiGPUController* getMultiGPUController();

// Legacy compatibility functions (enhanced versions)
long setDeviceNumber(long deviceNumber);
long getDeviceNumber();
std::vector<long> getAvailableDevices();
bool setMultipleDevices(const std::vector<long>& deviceNumbers);

#endif // MULTIGPU_CONTROL_H_