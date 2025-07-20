#include "multiGPUControl.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

// Global multi-GPU controller instance
std::unique_ptr<MultiGPUController> g_multiGpuController = nullptr;

// MultiGPUController implementation
MultiGPUController::MultiGPUController() : primary_device_(0) {
    if (!discoverDevices()) {
        setError("Failed to discover GPU devices");
        return;
    }
    
    if (gpu_devices_.empty()) {
        setError("No suitable GPU devices found");
        return;
    }
    
    // Initialize memory pool
    try {
        memory_pool_ = std::make_unique<MultiGPUMemoryPool>();
    } catch (const std::exception& e) {
        setError(std::string("Failed to initialize memory pool: ") + e.what());
        return;
    }
    
    // Set up P2P access
    initializeP2PAccess();
    
    // Set primary device as active by default
    active_devices_.push_back(primary_device_);
    
    std::cout << "MultiGPUController initialized with " << gpu_devices_.size() << " GPUs" << std::endl;
}

MultiGPUController::~MultiGPUController() {
    // Synchronize all devices before cleanup
    synchronizeAllDevices();
    
    memory_pool_.reset();
    gpu_devices_.clear();
    active_devices_.clear();
}

bool MultiGPUController::initializeDevices() {
    return !gpu_devices_.empty();
}

bool MultiGPUController::setActiveDevice(int device_id) {
    if (!isDeviceValid(device_id)) {
        setError("Invalid device ID: " + std::to_string(device_id));
        return false;
    }
    
    cudaError_t result = cudaSetDevice(device_id);
    if (result != cudaSuccess) {
        setError("Failed to set active device: " + std::string(cudaGetErrorString(result)));
        return false;
    }
    
    primary_device_ = device_id;
    return true;
}

int MultiGPUController::getActiveDevice() const {
    int current_device;
    cudaGetDevice(&current_device);
    return current_device;
}

std::vector<int> MultiGPUController::getActiveDevices() const {
    return active_devices_;
}

bool MultiGPUController::setActiveDevices(const std::vector<int>& device_ids) {
    // Validate all device IDs first
    for (int device_id : device_ids) {
        if (!isDeviceValid(device_id)) {
            setError("Invalid device ID in list: " + std::to_string(device_id));
            return false;
        }
    }
    
    active_devices_ = device_ids;
    
    // Set the first device as primary
    if (!device_ids.empty()) {
        primary_device_ = device_ids[0];
        setActiveDevice(primary_device_);
    }
    
    return true;
}

void MultiGPUController::updateDeviceUtilization() {
    for (auto& device : gpu_devices_) {
        if (device.is_active) {
            cudaSetDevice(device.device_id);
            
            // Update memory information
            size_t free_bytes, total_bytes;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            device.available_memory = free_bytes;
        }
    }
}

double MultiGPUController::getDeviceUtilization(int device_id) const {
    if (!isDeviceValid(device_id)) {
        return 0.0;
    }
    
    const auto& device = gpu_devices_[device_id];
    if (device.total_memory == 0) {
        return 0.0;
    }
    
    double used_memory = device.total_memory - device.available_memory;
    return used_memory / device.total_memory;
}

size_t MultiGPUController::getDeviceMemoryUsage(int device_id) const {
    if (!isDeviceValid(device_id)) {
        return 0;
    }
    
    const auto& device = gpu_devices_[device_id];
    return device.total_memory - device.available_memory;
}

int MultiGPUController::selectOptimalDevice(size_t memory_requirement) const {
    int best_device = -1;
    size_t max_available = 0;
    double min_utilization = 1.0;
    
    for (const int device_id : active_devices_) {
        const auto& device = gpu_devices_[device_id];
        
        if (device.available_memory >= memory_requirement) {
            double utilization = getDeviceUtilization(device_id);
            
            // Prefer device with lower utilization and more available memory
            if (utilization < min_utilization || 
                (utilization == min_utilization && device.available_memory > max_available)) {
                best_device = device_id;
                max_available = device.available_memory;
                min_utilization = utilization;
            }
        }
    }
    
    return best_device;
}

std::vector<int> MultiGPUController::distributeWorkload(int total_work_items) const {
    if (active_devices_.empty()) {
        return {};
    }
    
    WorkDistributor distributor(active_devices_);
    auto chunks = distributor.distributeEvenly(total_work_items);
    
    std::vector<int> work_per_gpu(gpu_devices_.size(), 0);
    for (const auto& chunk : chunks) {
        work_per_gpu[chunk.gpu_id] = chunk.work_size;
    }
    
    return work_per_gpu;
}

bool MultiGPUController::synchronizeDevice(int device_id) {
    if (!isDeviceValid(device_id)) {
        return false;
    }
    
    cudaSetDevice(device_id);
    cudaError_t result = cudaDeviceSynchronize();
    
    if (result != cudaSuccess) {
        setError("Failed to synchronize device " + std::to_string(device_id) + 
                ": " + std::string(cudaGetErrorString(result)));
        return false;
    }
    
    return true;
}

bool MultiGPUController::synchronizeAllDevices() {
    bool success = true;
    
    for (int device_id : active_devices_) {
        if (!synchronizeDevice(device_id)) {
            success = false;
        }
    }
    
    return success;
}

bool MultiGPUController::isDeviceValid(int device_id) const {
    return device_id >= 0 && device_id < gpu_devices_.size();
}

bool MultiGPUController::discoverDevices() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        setError("No CUDA devices found");
        return false;
    }
    
    gpu_devices_.reserve(device_count);
    
    for (int i = 0; i < device_count; ++i) {
        GPUDeviceInfo device_info;
        if (validateDevice(i, device_info)) {
            gpu_devices_.push_back(device_info);
            std::cout << "Found GPU " << i << ": " << device_info.name 
                      << " (" << (device_info.total_memory / (1024*1024)) << " MB)" << std::endl;
        }
    }
    
    return !gpu_devices_.empty();
}

bool MultiGPUController::validateDevice(int device_id, GPUDeviceInfo& info) {
    cudaDeviceProp prop;
    cudaError_t result = cudaGetDeviceProperties(&prop, device_id);
    
    if (result != cudaSuccess) {
        return false;
    }
    
    // Check compute capability (require >= 6.1)
    if (prop.major < 6 || (prop.major == 6 && prop.minor < 1)) {
        std::cout << "Skipping GPU " << device_id << " (compute capability " 
                  << prop.major << "." << prop.minor << " < 6.1)" << std::endl;
        return false;
    }
    
    // Fill device info
    info.device_id = device_id;
    info.name = prop.name;
    info.total_memory = prop.totalGlobalMem;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.is_active = true;
    
    // Get current memory info
    cudaSetDevice(device_id);
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    info.available_memory = free_bytes;
    
    // Check P2P capability with other devices
    info.p2p_capable = false;
    for (int j = 0; j < device_id; ++j) {
        int can_access;
        cudaDeviceCanAccessPeer(&can_access, device_id, j);
        if (can_access) {
            info.p2p_capable = true;
            break;
        }
    }
    
    return true;
}

void MultiGPUController::initializeP2PAccess() {
    for (int i = 0; i < gpu_devices_.size(); ++i) {
        for (int j = i + 1; j < gpu_devices_.size(); ++j) {
            int can_access_ij, can_access_ji;
            cudaDeviceCanAccessPeer(&can_access_ij, i, j);
            cudaDeviceCanAccessPeer(&can_access_ji, j, i);
            
            if (can_access_ij) {
                cudaSetDevice(i);
                cudaError_t result = cudaDeviceEnablePeerAccess(j, 0);
                if (result == cudaSuccess || result == cudaErrorPeerAccessAlreadyEnabled) {
                    std::cout << "Enabled P2P access from GPU " << i << " to GPU " << j << std::endl;
                }
            }
            
            if (can_access_ji) {
                cudaSetDevice(j);
                cudaError_t result = cudaDeviceEnablePeerAccess(i, 0);
                if (result == cudaSuccess || result == cudaErrorPeerAccessAlreadyEnabled) {
                    std::cout << "Enabled P2P access from GPU " << j << " to GPU " << i << std::endl;
                }
            }
        }
    }
}

void MultiGPUController::setError(const std::string& error) {
    last_error_ = error;
    std::cerr << "MultiGPUController Error: " << error << std::endl;
}

// WorkDistributor implementation
WorkDistributor::WorkDistributor(const std::vector<int>& available_gpus) 
    : available_gpus_(available_gpus) {
}

std::vector<WorkDistributor::WorkChunk> WorkDistributor::distributeEvenly(int total_work) const {
    std::vector<WorkChunk> chunks;
    
    if (available_gpus_.empty() || total_work <= 0) {
        return chunks;
    }
    
    int work_per_gpu = total_work / available_gpus_.size();
    int remaining_work = total_work % available_gpus_.size();
    
    int current_start = 0;
    for (int i = 0; i < available_gpus_.size(); ++i) {
        int chunk_size = work_per_gpu + (i < remaining_work ? 1 : 0);
        
        WorkChunk chunk;
        chunk.gpu_id = available_gpus_[i];
        chunk.start_index = current_start;
        chunk.end_index = current_start + chunk_size;
        chunk.work_size = chunk_size;
        
        chunks.push_back(chunk);
        current_start += chunk_size;
    }
    
    return chunks;
}

std::vector<WorkDistributor::WorkChunk> WorkDistributor::distributeByCapability(
    int total_work, const std::vector<double>& gpu_capabilities) const {
    
    std::vector<WorkChunk> chunks;
    
    if (available_gpus_.size() != gpu_capabilities.size()) {
        return distributeEvenly(total_work); // Fallback
    }
    
    // Calculate total capability
    double total_capability = std::accumulate(gpu_capabilities.begin(), gpu_capabilities.end(), 0.0);
    
    if (total_capability <= 0.0) {
        return distributeEvenly(total_work); // Fallback
    }
    
    int current_start = 0;
    int remaining_work = total_work;
    
    for (int i = 0; i < available_gpus_.size(); ++i) {
        int chunk_size;
        if (i == available_gpus_.size() - 1) {
            // Last GPU gets all remaining work
            chunk_size = remaining_work;
        } else {
            // Calculate proportional work based on capability
            chunk_size = static_cast<int>(std::round(total_work * gpu_capabilities[i] / total_capability));
        }
        
        WorkChunk chunk;
        chunk.gpu_id = available_gpus_[i];
        chunk.start_index = current_start;
        chunk.end_index = current_start + chunk_size;
        chunk.work_size = chunk_size;
        
        chunks.push_back(chunk);
        current_start += chunk_size;
        remaining_work -= chunk_size;
    }
    
    return chunks;
}

std::vector<WorkDistributor::WorkChunk> WorkDistributor::distributeByMemory(
    int total_work, const std::vector<size_t>& memory_requirements) const {
    
    // For now, implement simple even distribution
    // TODO: Implement memory-aware distribution based on requirements
    return distributeEvenly(total_work);
}

void WorkDistributor::adjustDistribution(std::vector<WorkChunk>& chunks, 
                                       const std::vector<double>& completion_rates) const {
    // Simple load balancing: redistribute work from slow to fast GPUs
    // TODO: Implement more sophisticated dynamic load balancing
    
    if (chunks.size() != completion_rates.size()) {
        return;
    }
    
    // Find fastest and slowest GPUs
    auto min_rate_it = std::min_element(completion_rates.begin(), completion_rates.end());
    auto max_rate_it = std::max_element(completion_rates.begin(), completion_rates.end());
    
    if (min_rate_it != max_rate_it && *max_rate_it > *min_rate_it * 1.5) {
        // If there's significant imbalance, consider work stealing
        int slow_gpu_idx = std::distance(completion_rates.begin(), min_rate_it);
        int fast_gpu_idx = std::distance(completion_rates.begin(), max_rate_it);
        
        // Transfer some work from slow to fast GPU
        int work_to_transfer = chunks[slow_gpu_idx].work_size / 10; // Transfer 10%
        if (work_to_transfer > 0) {
            chunks[slow_gpu_idx].work_size -= work_to_transfer;
            chunks[slow_gpu_idx].end_index -= work_to_transfer;
            chunks[fast_gpu_idx].work_size += work_to_transfer;
            chunks[fast_gpu_idx].end_index += work_to_transfer;
        }
    }
}

int WorkDistributor::calculateOptimalChunkSize(int total_work, int gpu_count) const {
    // Simple heuristic: aim for chunk sizes that balance parallelism and overhead
    int min_chunk_size = 32;  // Minimum for good GPU utilization
    int max_chunk_size = 1024; // Maximum to ensure good load balancing
    
    int ideal_chunk_size = total_work / (gpu_count * 4); // 4 chunks per GPU
    return std::max(min_chunk_size, std::min(max_chunk_size, ideal_chunk_size));
}

// MultiGPUStreamManager implementation
MultiGPUStreamManager::MultiGPUStreamManager(const std::vector<int>& gpu_ids) 
    : gpu_ids_(gpu_ids), initialized_(false) {
}

MultiGPUStreamManager::~MultiGPUStreamManager() {
    destroyStreams();
}

cudaStream_t MultiGPUStreamManager::getStream(int gpu_id, int stream_index) {
    auto it = streams_per_gpu_.find(gpu_id);
    if (it != streams_per_gpu_.end() && stream_index < it->second.size()) {
        return it->second[stream_index];
    }
    return nullptr;
}

bool MultiGPUStreamManager::createStreams(int streams_per_gpu) {
    if (initialized_) {
        return true;
    }
    
    for (int gpu_id : gpu_ids_) {
        cudaSetDevice(gpu_id);
        
        std::vector<cudaStream_t> streams(streams_per_gpu);
        for (int i = 0; i < streams_per_gpu; ++i) {
            cudaError_t result = cudaStreamCreate(&streams[i]);
            if (result != cudaSuccess) {
                // Cleanup already created streams
                for (int j = 0; j < i; ++j) {
                    cudaStreamDestroy(streams[j]);
                }
                return false;
            }
        }
        
        streams_per_gpu_[gpu_id] = std::move(streams);
    }
    
    initialized_ = true;
    return true;
}

bool MultiGPUStreamManager::destroyStreams() {
    if (!initialized_) {
        return true;
    }
    
    bool success = true;
    
    for (auto& pair : streams_per_gpu_) {
        int gpu_id = pair.first;
        cudaSetDevice(gpu_id);
        
        for (cudaStream_t stream : pair.second) {
            cudaError_t result = cudaStreamDestroy(stream);
            if (result != cudaSuccess) {
                success = false;
            }
        }
    }
    
    // Destroy events
    for (auto& gpu_pair : events_per_gpu_) {
        int gpu_id = gpu_pair.first;
        cudaSetDevice(gpu_id);
        
        for (auto& event_pair : gpu_pair.second) {
            cudaEventDestroy(event_pair.second);
        }
    }
    
    streams_per_gpu_.clear();
    events_per_gpu_.clear();
    initialized_ = false;
    
    return success;
}

bool MultiGPUStreamManager::synchronizeStream(int gpu_id, int stream_index) {
    cudaStream_t stream = getStream(gpu_id, stream_index);
    if (stream == nullptr) {
        return false;
    }
    
    cudaSetDevice(gpu_id);
    cudaError_t result = cudaStreamSynchronize(stream);
    return result == cudaSuccess;
}

bool MultiGPUStreamManager::synchronizeAllStreams() {
    bool success = true;
    
    for (const auto& pair : streams_per_gpu_) {
        int gpu_id = pair.first;
        cudaSetDevice(gpu_id);
        
        for (cudaStream_t stream : pair.second) {
            cudaError_t result = cudaStreamSynchronize(stream);
            if (result != cudaSuccess) {
                success = false;
            }
        }
    }
    
    return success;
}

bool MultiGPUStreamManager::createEvent(int gpu_id, const std::string& event_name) {
    cudaSetDevice(gpu_id);
    
    cudaEvent_t event;
    cudaError_t result = cudaEventCreate(&event);
    if (result != cudaSuccess) {
        return false;
    }
    
    events_per_gpu_[gpu_id][event_name] = event;
    return true;
}

bool MultiGPUStreamManager::recordEvent(int gpu_id, const std::string& event_name, int stream_index) {
    auto gpu_it = events_per_gpu_.find(gpu_id);
    if (gpu_it == events_per_gpu_.end()) {
        return false;
    }
    
    auto event_it = gpu_it->second.find(event_name);
    if (event_it == gpu_it->second.end()) {
        return false;
    }
    
    cudaStream_t stream = getStream(gpu_id, stream_index);
    if (stream == nullptr) {
        return false;
    }
    
    cudaSetDevice(gpu_id);
    cudaError_t result = cudaEventRecord(event_it->second, stream);
    return result == cudaSuccess;
}

bool MultiGPUStreamManager::waitForEvent(int gpu_id, const std::string& event_name, int stream_index) {
    auto gpu_it = events_per_gpu_.find(gpu_id);
    if (gpu_it == events_per_gpu_.end()) {
        return false;
    }
    
    auto event_it = gpu_it->second.find(event_name);
    if (event_it == gpu_it->second.end()) {
        return false;
    }
    
    cudaStream_t stream = getStream(gpu_id, stream_index);
    if (stream == nullptr) {
        return false;
    }
    
    cudaSetDevice(gpu_id);
    cudaError_t result = cudaStreamWaitEvent(stream, event_it->second, 0);
    return result == cudaSuccess;
}

// Global convenience functions
bool initializeMultiGPU() {
    try {
        g_multiGpuController = std::make_unique<MultiGPUController>();
        return g_multiGpuController->initializeDevices();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize multi-GPU: " << e.what() << std::endl;
        return false;
    }
}

void shutdownMultiGPU() {
    g_multiGpuController.reset();
}

MultiGPUController* getMultiGPUController() {
    return g_multiGpuController.get();
}

// Legacy compatibility functions (enhanced versions)
long setDeviceNumber(long deviceNumber) {
    if (!g_multiGpuController) {
        initializeMultiGPU();
    }
    
    if (g_multiGpuController && g_multiGpuController->setActiveDevice(static_cast<int>(deviceNumber))) {
        return deviceNumber;
    }
    
    // Fallback to original implementation
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    
    if (deviceNumber < num_devices) {
        cudaSetDevice(deviceNumber);
    }
    
    int active_device;
    cudaGetDevice(&active_device);
    return active_device;
}

long getDeviceNumber() {
    if (g_multiGpuController) {
        return g_multiGpuController->getActiveDevice();
    }
    
    int active_device;
    cudaGetDevice(&active_device);
    return active_device;
}

std::vector<long> getAvailableDevices() {
    std::vector<long> devices;
    
    if (g_multiGpuController) {
        auto active_devices = g_multiGpuController->getActiveDevices();
        devices.assign(active_devices.begin(), active_devices.end());
    } else {
        int device_count;
        cudaGetDeviceCount(&device_count);
        for (int i = 0; i < device_count; ++i) {
            devices.push_back(i);
        }
    }
    
    return devices;
}

bool setMultipleDevices(const std::vector<long>& deviceNumbers) {
    if (!g_multiGpuController) {
        initializeMultiGPU();
    }
    
    if (g_multiGpuController) {
        std::vector<int> devices;
        for (long dev : deviceNumbers) {
            devices.push_back(static_cast<int>(dev));
        }
        return g_multiGpuController->setActiveDevices(devices);
    }
    
    return false;
}