/**
 * @Author: Multi-GPU Enhancement
 * @Filename: multiGPUWrapper.cpp
 * @Description: Enhanced Python bindings with multi-GPU support for Bethe Functions
 */

#include "goldCode.h"
#include "gpuCode.cuh"
#include "multiGPUKernels.cuh"
#include "multiGPUControl.h"
#include "multiGPUMemory.h"
#include <functional>
#include <utility>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

namespace py = pybind11;

// Multi-GPU Python interface class
class MultiGPUSpinChain {
public:
    MultiGPUSpinChain() {
        if (!initializeMultiGPU()) {
            throw std::runtime_error("Failed to initialize multi-GPU system");
        }
        transformer_ = std::make_unique<MultiGPUBasisTransform>();
    }
    
    ~MultiGPUSpinChain() {
        transformer_.reset();
        shutdownMultiGPU();
    }
    
    // Device management
    std::vector<int> getAvailableGPUs() const {
        if (g_multiGpuController) {
            return g_multiGpuController->getActiveDevices();
        }
        return {};
    }
    
    bool setActiveGPUs(const std::vector<int>& gpu_ids) {
        if (g_multiGpuController && transformer_) {
            bool success = g_multiGpuController->setActiveDevices(gpu_ids);
            if (success) {
                transformer_->setGpuList(gpu_ids);
            }
            return success;
        }
        return false;
    }
    
    py::dict getGPUInfo() const {
        py::dict info;
        if (g_multiGpuController) {
            const auto& devices = g_multiGpuController->getDeviceInfo();
            py::list gpu_list;
            
            for (const auto& device : devices) {
                py::dict gpu_info;
                gpu_info["device_id"] = device.device_id;
                gpu_info["name"] = device.name;
                gpu_info["total_memory_mb"] = device.total_memory / (1024 * 1024);
                gpu_info["available_memory_mb"] = device.available_memory / (1024 * 1024);
                gpu_info["compute_capability"] = py::make_tuple(device.compute_major, device.compute_minor);
                gpu_info["p2p_capable"] = device.p2p_capable;
                gpu_info["is_active"] = device.is_active;
                gpu_list.append(gpu_info);
            }
            
            info["gpus"] = gpu_list;
            info["gpu_count"] = devices.size();
        }
        return info;
    }
    
    // Performance monitoring
    py::dict getPerformanceMetrics() const {
        py::dict metrics;
        if (transformer_) {
            metrics["last_execution_time"] = transformer_->getLastExecutionTime();
            metrics["gpu_utilization"] = transformer_->getGpuUtilization();
            metrics["gpu_memory_usage_mb"] = py::cast(transformer_->getGpuMemoryUsage());
            
            // Convert memory usage to MB
            auto memory_usage = transformer_->getGpuMemoryUsage();
            std::vector<double> memory_usage_mb;
            for (size_t usage : memory_usage) {
                memory_usage_mb.push_back(static_cast<double>(usage) / (1024 * 1024));
            }
            metrics["gpu_memory_usage_mb"] = memory_usage_mb;
        }
        return metrics;
    }
    
    // Configuration
    void setBlockSize(int block_size) {
        if (transformer_) {
            transformer_->setBlockSize(block_size);
        }
    }
    
    void setSharedMemorySize(size_t bytes_per_gpu) {
        if (transformer_) {
            transformer_->setSharedMemorySize(bytes_per_gpu);
        }
    }
    
    // Main computation interface
    py::tuple computeBasisTransform(
        const MyComplexVector& allBetheRoots,
        const MyLongVector& allConfigs,
        const MyComplexVector& allGaudinDets,
        const MyLongVector& sigma,
        const MyComplexType& delta) {
        
        if (!transformer_) {
            throw std::runtime_error("Multi-GPU transformer not initialized");
        }
        
        auto result = transformer_->compute(allBetheRoots, allConfigs, allGaudinDets, sigma, delta);
        return py::make_tuple(result.first, result.second);
    }
    
    // Memory management utilities
    py::dict getMemoryInfo() const {
        py::dict memory_info;
        if (g_multiGpuPool) {
            memory_info["total_memory_mb"] = static_cast<double>(g_multiGpuPool->getTotalMemory()) / (1024 * 1024);
            memory_info["available_memory_mb"] = static_cast<double>(g_multiGpuPool->getAvailableMemory()) / (1024 * 1024);
            memory_info["used_memory_mb"] = static_cast<double>(g_multiGpuPool->getUsedMemory()) / (1024 * 1024);
            memory_info["gpu_count"] = g_multiGpuPool->getGpuCount();
        }
        return memory_info;
    }
    
    // Synchronization utilities
    bool synchronizeAllGPUs() const {
        if (g_multiGpuController) {
            return g_multiGpuController->synchronizeAllDevices();
        }
        return false;
    }
    
private:
    std::unique_ptr<MultiGPUBasisTransform> transformer_;
};

// Legacy wrapper functions for backwards compatibility
std::pair<MyComplexVector, MyComplexVector> computeBasisTransformLegacy(
    const MyComplexVector& allBetheRoots,
    const MyLongVector& allConfigs,
    const MyComplexVector& allGaudinDets,
    const MyLongVector& sigma,
    const MyComplexType& delta,
    bool use_multi_gpu = true) {
    
    if (use_multi_gpu) {
        return gpuComputeBasisTransformMultiGPU(allBetheRoots, allConfigs, allGaudinDets, sigma, delta);
    } else {
        return gpuComputeBasisTransform(allBetheRoots, allConfigs, allGaudinDets, sigma, delta);
    }
}

// Enhanced device management functions
long setDeviceNumberEnhanced(long deviceNumber) {
    if (!g_multiGpuController) {
        initializeMultiGPU();
    }
    return setDeviceNumber(deviceNumber);
}

std::vector<long> getAvailableDevicesEnhanced() {
    if (!g_multiGpuController) {
        initializeMultiGPU();
    }
    return getAvailableDevices();
}

bool setMultipleDevicesEnhanced(const std::vector<long>& deviceNumbers) {
    if (!g_multiGpuController) {
        initializeMultiGPU();
    }
    return setMultipleDevices(deviceNumbers);
}

// Benchmarking and profiling functions
py::dict benchmarkComputation(
    int nBasis_min, int nBasis_max,
    int nUp_min, int nUp_max,
    const std::vector<int>& gpu_counts) {
    
    py::dict results;
    py::list benchmark_results;
    
    for (int gpu_count : gpu_counts) {
        py::dict gpu_result;
        gpu_result["gpu_count"] = gpu_count;
        
        // Create test data
        int nBasis = nBasis_min;
        int nUp = nUp_min;
        
        MyComplexVector allBetheRoots(nBasis * nUp, MyComplexType(1.0, 0.1));
        MyLongVector allConfigs(nBasis * nUp, 1);
        MyComplexVector allGaudinDets(nBasis, MyComplexType(1.0));
        MyLongVector sigma(120 * nUp, 0); // Approximate factorial
        for (size_t i = 0; i < sigma.size(); ++i) {
            sigma[i] = i % nUp;
        }
        MyComplexType delta(0.5);
        
        // Time the computation
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            MultiGPUSpinChain chain;
            std::vector<int> gpu_ids;
            for (int i = 0; i < gpu_count; ++i) {
                gpu_ids.push_back(i);
            }
            
            if (chain.setActiveGPUs(gpu_ids)) {
                auto result = chain.computeBasisTransform(allBetheRoots, allConfigs, allGaudinDets, sigma, delta);
                
                auto end_time = std::chrono::high_resolution_clock::now();
                double execution_time = std::chrono::duration<double>(end_time - start_time).count();
                
                gpu_result["execution_time"] = execution_time;
                gpu_result["success"] = true;
                gpu_result["performance_metrics"] = chain.getPerformanceMetrics();
            } else {
                gpu_result["success"] = false;
                gpu_result["error"] = "Failed to set active GPUs";
            }
        } catch (const std::exception& e) {
            gpu_result["success"] = false;
            gpu_result["error"] = e.what();
        }
        
        benchmark_results.append(gpu_result);
    }
    
    results["benchmarks"] = benchmark_results;
    return results;
}

PYBIND11_MODULE(libspinChainMultiGPU, m) {
    m.doc() = "Enhanced libspinChain with multi-GPU support for Bethe Functions computation";
    
    // Multi-GPU SpinChain class
    py::class_<MultiGPUSpinChain>(m, "MultiGPUSpinChain")
        .def(py::init<>())
        .def("get_available_gpus", &MultiGPUSpinChain::getAvailableGPUs,
             "Get list of available GPU device IDs")
        .def("set_active_gpus", &MultiGPUSpinChain::setActiveGPUs,
             "Set which GPUs to use for computation")
        .def("get_gpu_info", &MultiGPUSpinChain::getGPUInfo,
             "Get detailed information about all GPUs")
        .def("get_performance_metrics", &MultiGPUSpinChain::getPerformanceMetrics,
             "Get performance metrics from last computation")
        .def("set_block_size", &MultiGPUSpinChain::setBlockSize,
             "Set CUDA block size for kernel execution")
        .def("set_shared_memory_size", &MultiGPUSpinChain::setSharedMemorySize,
             "Set shared memory size per GPU")
        .def("compute_basis_transform", &MultiGPUSpinChain::computeBasisTransform,
             "Compute basis transformation matrices using multi-GPU")
        .def("get_memory_info", &MultiGPUSpinChain::getMemoryInfo,
             "Get memory usage information across all GPUs")
        .def("synchronize_all_gpus", &MultiGPUSpinChain::synchronizeAllGPUs,
             "Synchronize all active GPUs");
    
    // Enhanced device management functions
    m.def("set_device_number", &setDeviceNumberEnhanced, 
          "Select the active GPU device (enhanced version)");
    m.def("get_device_number", &getDeviceNumber, 
          "Return the active GPU device");
    m.def("get_available_devices", &getAvailableDevicesEnhanced,
          "Get list of all available GPU devices");
    m.def("set_multiple_devices", &setMultipleDevicesEnhanced,
          "Set multiple active GPU devices");
    
    // Legacy compatibility functions
    m.def("gpu_compute_basis_transform", &computeBasisTransformLegacy,
          "Compute basis transformation (with multi-GPU option)",
          py::arg("allBetheRoots"), py::arg("allConfigs"), py::arg("allGaudinDets"),
          py::arg("sigma"), py::arg("delta"), py::arg("use_multi_gpu") = true);
    
    // Original single-GPU functions for comparison
    m.def("gpu_compute_basis_transform_single", &gpuComputeBasisTransform,
          "Original single-GPU basis transformation");
    
    // Utility and testing functions
    m.def("benchmark_computation", &benchmarkComputation,
          "Benchmark multi-GPU performance with different GPU counts");
    
    // Original functions for backwards compatibility
    m.def("sum", &sum);
    m.def("vector_sum", &vectorSum);
    m.def("inspect_array", &inspectArray);
    m.def("array_sum_sq", &arraySumSq);
    m.def("int_power", &intPower);
    m.def("product", &product);
    m.def("compute_amplitude", &computeAmplitude);
    m.def("compute_basis_transform", &computeBasisTransform);
    m.def("gpu_sum", &gpuSum);
    m.def("thrust_list_test", &thrustListTest);
    
    // Initialize multi-GPU system on module load
    m.def("initialize_multi_gpu", &initializeMultiGPU,
          "Initialize the multi-GPU system");
    m.def("shutdown_multi_gpu", &shutdownMultiGPU,
          "Shutdown the multi-GPU system");
    
    // Version and build information
    m.attr("__version__") = "2.0.0-multigpu";
    m.attr("multi_gpu_support") = true;
}