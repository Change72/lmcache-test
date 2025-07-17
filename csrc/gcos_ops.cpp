#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <vector>
#include <string>

#ifdef WITH_GCOS
#include "gcos.h"
#include "gcos_file.h"
#include "buffer.h"
#endif

namespace py = pybind11;

// GCOS function implementations
#ifdef WITH_GCOS

int initialize_gcos(const std::string& config_path) {
    try {
        return gcos_init(config_path.c_str());
    } catch (const std::exception& e) {
        throw std::runtime_error("GCOS initialization failed: " + std::string(e.what()));
    }
}

void cleanup_gcos() {
    try {
        gcos_close();
    } catch (const std::exception& e) {
        // Log but don't throw during cleanup
    }
}

py::object create_gcos_file(const std::string& filename, uint64_t size) {
    try {
        File* file = OpenFile(filename, size);
        if (!file) {
            throw std::runtime_error("Failed to create GCOS file: " + filename);
        }
        return py::cast(file, py::return_value_policy::take_ownership);
    } catch (const std::exception& e) {
        throw std::runtime_error("GCOS file creation failed: " + std::string(e.what()));
    }
}

py::object create_gcos_dma_buffer(size_t size, int device = 0) {
    try {
        auto dma_buf = CreateDma(size, device);
        if (!dma_buf) {
            throw std::runtime_error("Failed to create GCOS DMA buffer");
        }
        return py::cast(dma_buf.get(), py::return_value_policy::take_ownership);
    } catch (const std::exception& e) {
        throw std::runtime_error("GCOS DMA buffer creation failed: " + std::string(e.what()));
    }
}

uint64_t gpu_file_write(py::object file_obj, py::object dma_buf, uint64_t size, uint32_t device_id = 0) {
    try {
        File* file = file_obj.cast<File*>();
        DmaPtr dma_ptr = dma_buf.cast<DmaPtr>();
        if (!file || !dma_ptr) {
            throw std::runtime_error("Invalid file or DMA buffer objects");
        }
        return FileWrite(file, dma_ptr, size, device_id);
    } catch (const std::exception& e) {
        throw std::runtime_error("GCOS file write failed: " + std::string(e.what()));
    }
}

uint64_t gpu_file_read(py::object file_obj, py::object dma_buf, uint64_t size, uint32_t device_id = 0) {
    try {
        File* file = file_obj.cast<File*>();
        DmaPtr dma_ptr = dma_buf.cast<DmaPtr>();
        if (!file || !dma_ptr) {
            throw std::runtime_error("Invalid file or DMA buffer objects");
        }
        return FileRead(file, dma_ptr, size, device_id);
    } catch (const std::exception& e) {
        throw std::runtime_error("GCOS file read failed: " + std::string(e.what()));
    }
}

int init_allocator() {
    try {
        return InitAllocator();
    } catch (const std::exception& e) {
        throw std::runtime_error("GCOS allocator initialization failed: " + std::string(e.what()));
    }
}

void destroy_allocator() {
    try {
        DestroyAllocator();
    } catch (const std::exception& e) {
        // Log but don't throw during cleanup
    }
}

void gpu_store_kv_kernel(
    const std::vector<torch::Tensor>& kv_caches,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& storage_addresses,
    int num_tokens,
    int num_layers,
    int hidden_dim,
    py::object gpu_queues,
    int offset = 0
) {
    // Implementation for direct GPU-to-storage KV transfer
    // This would call the actual GCOS GPU kernels
    try {
        // TODO: Implement actual GCOS GPU kernel calls
        // For now, this is a placeholder that would need to be
        // implemented with the actual GCOS GPU queue operations
        throw std::runtime_error("GPU store KV kernel not yet implemented");
    } catch (const std::exception& e) {
        throw std::runtime_error("GCOS GPU store kernel failed: " + std::string(e.what()));
    }
}

void gpu_retrieve_kv_kernel(
    const std::vector<torch::Tensor>& kv_caches,
    const torch::Tensor& slot_mapping, 
    const torch::Tensor& storage_addresses,
    int num_tokens,
    int num_layers,
    int hidden_dim,
    py::object gpu_queues
) {
    // Implementation for direct storage-to-GPU KV transfer
    try {
        // TODO: Implement actual GCOS GPU kernel calls
        // For now, this is a placeholder that would need to be
        // implemented with the actual GCOS GPU queue operations  
        throw std::runtime_error("GPU retrieve KV kernel not yet implemented");
    } catch (const std::exception& e) {
        throw std::runtime_error("GCOS GPU retrieve kernel failed: " + std::string(e.what()));
    }
}

#else
// Fallback implementations when GCOS is not available
int initialize_gcos(const std::string& config_path) {
    throw std::runtime_error("GCOS not available - build with BUILD_WITH_GCOS=1");
}

void cleanup_gcos() {
    throw std::runtime_error("GCOS not available");
}

py::object create_gcos_file(const std::string& filename, uint64_t size) {
    throw std::runtime_error("GCOS not available");
}

py::object create_gcos_dma_buffer(size_t size, int device) {
    throw std::runtime_error("GCOS not available");
}

uint64_t gpu_file_write(py::object file_obj, py::object dma_buf, uint64_t size, uint32_t device_id) {
    throw std::runtime_error("GCOS not available");
}

uint64_t gpu_file_read(py::object file_obj, py::object dma_buf, uint64_t size, uint32_t device_id) {
    throw std::runtime_error("GCOS not available");
}

int init_allocator() {
    throw std::runtime_error("GCOS not available");
}

void destroy_allocator() {
    throw std::runtime_error("GCOS not available");
}

void gpu_store_kv_kernel(
    const std::vector<torch::Tensor>& kv_caches,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& storage_addresses,
    int num_tokens,
    int num_layers,
    int hidden_dim,
    py::object gpu_queues,
    int offset
) {
    throw std::runtime_error("GCOS not available");
}

void gpu_retrieve_kv_kernel(
    const std::vector<torch::Tensor>& kv_caches,
    const torch::Tensor& slot_mapping, 
    const torch::Tensor& storage_addresses,
    int num_tokens,
    int num_layers,
    int hidden_dim,
    py::object gpu_queues
) {
    throw std::runtime_error("GCOS not available");
}
#endif

PYBIND11_MODULE(gcos_ops, m) {
    m.doc() = "GCOS operations for LMCache";
    
    // System operations
    m.def("initialize_gcos", &initialize_gcos, "Initialize GCOS system");
    m.def("cleanup_gcos", &cleanup_gcos, "Cleanup GCOS system");
    
    // File operations
    m.def("create_gcos_file", &create_gcos_file, "Create GCOS file");
    m.def("gpu_file_write", &gpu_file_write, "Write to GCOS file from GPU");
    m.def("gpu_file_read", &gpu_file_read, "Read from GCOS file to GPU");
    
    // DMA operations
    m.def("create_gcos_dma_buffer", &create_gcos_dma_buffer, "Create GCOS DMA buffer");
    
    // Allocator operations
    m.def("init_allocator", &init_allocator, "Initialize GCOS allocator");
    m.def("destroy_allocator", &destroy_allocator, "Destroy GCOS allocator");
    
    // GPU kernel operations
    m.def("gpu_store_kv_kernel", &gpu_store_kv_kernel, "Direct GPU-to-storage KV kernel");
    m.def("gpu_retrieve_kv_kernel", &gpu_retrieve_kv_kernel, "Direct storage-to-GPU KV kernel");
}