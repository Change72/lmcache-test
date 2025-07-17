# GCOS Integration with LMCache

This document provides a comprehensive guide to the GCOS (GPU-Centric OS) integration with LMCache, including configuration, usage patterns, and performance optimization.

## Overview

The GCOS integration enables direct GPU-to-storage operations for KV cache management, bypassing traditional CPU-memory bottlenecks. This provides:

- **Zero-copy transfers**: Direct GPU-to-NVMe storage operations
- **High throughput**: Parallel access to multiple NVMe devices  
- **Low latency**: Reduced data movement overhead
- **Scalability**: Support for multi-GPU and multi-device configurations

## Architecture

### Integration Points

1. **Storage Backend**: `lmcache/v1/storage_backend/gcos_backend.py`
   - Implements `StorageBackendInterface` 
   - Provides async operations with fallback support
   - Manages GCOS device initialization and cleanup

2. **C++ Bindings**: `csrc/gcos_ops.cpp`
   - pybind11 extension exposing GCOS functions to Python
   - Conditional compilation with `#ifdef WITH_GCOS`
   - Functions: initialize, create_file, gpu_file_write, gpu_file_read

3. **Configuration**: `lmcache/v1/config.py`
   - Added `gcos_path` parameter to `LMCacheEngineConfig`
   - Integration with storage manager initialization

4. **Build System**: `setup.py`
   - Environment variable `BUILD_WITH_GCOS=1` enables GCOS support
   - `GCOS_PATH` specifies GCOS installation directory
   - Conditional compilation for optional GCOS features

## Configuration

### Environment Setup

```bash
# Set GCOS installation path
export GCOS_PATH=/path/to/gcos

# Required directory structure in GCOS_PATH:
# include/          - GCOS headers (gcos.h)
# lib/             - GCOS libraries (libgcos.so)
# build/lib/       - Alternative library location
```

### GCOS System Configuration

Create `gcos.json` in your project root:

```json
{
  "disk": {
    "number": 4,
    "paths": ["/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3"],
    "namespace_number": 2,
    "block_size": 512,
    "capacity_bytes": 1099511627776,
    "reserve_max_queue_pair_num": 256
  },
  "namespace": {
    "cpu": {
      "qps": 32,
      "queue_depth": 512
    },
    "gpu": {
      "num": 1,
      "qps": 64,
      "queue_depth": 1024
    }
  },
  "zone_chunk_size_kb": {
    "sb": 128,
    "meta": 512,
    "data": 8192
  }
}
```

### LMCache Configuration

```python
import lmcache.v1 as lmcache
from lmcache.config import LMCacheEngineMetadata

# Basic GCOS configuration
config = lmcache.LMCacheEngineConfig(
    chunk_size=512,
    gcos_path="/mnt/gcos_cache",  # Enables GCOS backend
    local_cpu=False,              # Use GCOS directly
    max_local_cpu_size=0.0,
    save_decode_cache=True,
    extra_config={
        "gcos_config": "gcos.json",
        "use_cufile": True,
        "use_direct_io": True
    }
)

# Model metadata
metadata = LMCacheEngineMetadata(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    fmt="huggingface",
    world_size=1,
    worker_id=0
)

# Build cache engine
cache_engine = lmcache.LMCacheEngineBuilder.build(config, metadata)
```

## Usage Patterns

### Basic Storage and Retrieval

```python
from lmcache.utils import CacheEngineKey
import torch

# Create cache key
cache_key = CacheEngineKey(
    fmt="huggingface",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    world_size=1,
    worker_id=0,
    chunk_hash="example_prompt_001"
)

# Store KV caches (direct GPU-to-storage via GCOS)
kv_caches = []  # List of [k_cache, v_cache] tensors for each layer
tokens = ["Hello", "world", "test"] * 170  # ~512 tokens

cache_engine.store(cache_key, kv_caches, tokens)

# Retrieve KV caches (direct storage-to-GPU via GCOS)
retrieved_kv, retrieved_tokens = cache_engine.retrieve(cache_key)
```

### vLLM Integration

```python
class VLLMGCOSIntegration:
    def setup_cache_engine(self):
        config = lmcache.LMCacheEngineConfig(
            chunk_size=512,
            gcos_path="/fast_nvme/vllm_cache",
            local_cpu=False,
            use_layerwise=True,  # vLLM compatibility
            extra_config={
                "gcos_config": "gcos.json",
                "vllm_integration": True,
                "batch_size": 32,
                "prefetch_size": 3
            }
        )
        
        metadata = LMCacheEngineMetadata(
            model_name=self.model_name,
            fmt="vllm",
            world_size=torch.cuda.device_count(),
            worker_id=0
        )
        
        return lmcache.LMCacheEngineBuilder.build(config, metadata)
    
    def store_conversation(self, conversation_id, kv_caches, tokens):
        cache_key = CacheEngineKey(
            fmt="vllm",
            model_name=self.model_name,
            world_size=torch.cuda.device_count(),
            worker_id=0,
            chunk_hash=f"conversation_{conversation_id}"
        )
        
        self.cache_engine.store(cache_key, kv_caches, tokens)
        return cache_key
    
    def prefetch_conversation(self, conversation_id):
        # Async prefetching for better performance
        cache_key = self.create_cache_key(conversation_id)
        future = self.cache_engine.storage_manager.get_non_blocking(cache_key)
        return future
```

### Multi-GPU Configuration

```python
# Multi-GPU GCOS configuration
config = lmcache.LMCacheEngineConfig(
    chunk_size=1024,
    gcos_path="/fast_storage/gcos_cache",
    local_cpu=False,
    max_local_cpu_size=0.0,
    use_layerwise=True,
    extra_config={
        "gcos_config": "examples/gcos_config_examples/multi_gpu_gcos.json",
        "use_cufile": True,
        "use_direct_io": True,
        "gcos_queue_num": 256,
        "gcos_queue_depth": 2048
    }
)

metadata = LMCacheEngineMetadata(
    model_name="meta-llama/Llama-3.1-70B-Instruct",
    fmt="vllm",
    world_size=8,  # 8 GPUs
    worker_id=0
)
```

## Storage Backend Architecture

### Initialization Flow

1. **Storage Manager Creation**: `storage_manager.py:76-84`
   - Calls `CreateStorageBackends()` from `__init__.py`
   - Creates GCOS backend if `config.gcos_path` is set

2. **GCOS Backend Initialization**: `gcos_backend.py:__init__`
   - Loads GCOS configuration from `gcos_config` file
   - Initializes GCOS system with `gcos_ops.initialize_gcos()`
   - Sets up GPU memory allocation and device management

3. **Cache Engine Integration**: `cache_engine.py:111-117`
   - Storage manager handles backend routing
   - GCOS backend participates in storage/retrieval operations
   - Automatic fallback to other backends if GCOS unavailable

### Operation Flow

#### Store Operation
1. Cache engine calls `storage_manager.batched_put()`
2. Storage manager routes to GCOS backend
3. GCOS backend calls `_save_gcos()` with GPU pointers
4. Direct GPU-to-storage transfer via `gcos_ops.gpu_file_write()`

#### Retrieve Operation  
1. Cache engine calls `storage_manager.batched_get()`
2. Storage manager routes to GCOS backend
3. GCOS backend calls `_load_gcos()` with GPU pointers
4. Direct storage-to-GPU transfer via `gcos_ops.gpu_file_read()`

## Performance Optimization

### Hardware Configuration

1. **NVMe Devices**:
   - Use high-performance NVMe SSDs (PCIe 4.0+)
   - Configure multiple devices for parallel access
   - Ensure proper NUMA affinity

2. **GPU Setup**:
   - Enable GPUDirect for optimal performance  
   - Use CUDA streams for async operations
   - Configure appropriate queue depths

3. **System Configuration**:
   - Enable large page support
   - Configure CPU isolation for GCOS threads
   - Optimize PCIe slot placement

### Software Tuning

1. **Queue Parameters**:
   ```json
   "gpu": {
     "qps": 128,        // Increase for higher throughput
     "queue_depth": 2048 // Increase for better batching
   }
   ```

2. **Chunk Sizes**:
   - Align with typical KV cache sizes
   - Larger chunks for better throughput
   - Smaller chunks for lower latency

3. **Memory Management**:
   - Disable CPU cache (`local_cpu=False`)
   - Use direct I/O when possible
   - Enable layerwise processing for large models

## Testing

### Unit Tests

```bash
# Build with GCOS support
BUILD_WITH_GCOS=1 pip install -e .

# Run GCOS tests
python tests/v1/test_gcos.py
```

### Integration Testing

```bash
# Test basic functionality
python examples/gcos_usage_examples/basic_gcos_usage.py

# Test vLLM integration
python examples/gcos_usage_examples/vllm_gcos_integration.py
```

### Performance Benchmarking

```bash
# Enable GCOS for benchmarks
export BUILD_WITH_GCOS=1
export GCOS_PATH=/path/to/gcos

# Run performance tests
python -m lmcache.v1.benchmark.gcos_benchmark
```

## Troubleshooting

### Common Issues

1. **Build Errors**:
   - Ensure `GCOS_PATH` points to valid installation
   - Check that `gcos.h` and `libgcos.so` exist
   - Verify GCC/CUDA compatibility

2. **Runtime Errors**:
   - Verify GCOS devices are available (`ls /dev/libnvm*`)
   - Check device permissions
   - Ensure GCOS daemon is running

3. **Performance Issues**:
   - Monitor queue utilization
   - Check PCIe bandwidth usage
   - Verify NUMA configuration
   - Profile GPU memory transfers

### Debug Configuration

```python
# Enable debug logging
config = lmcache.LMCacheEngineConfig(
    gcos_path="/tmp/gcos_debug",
    extra_config={
        "gcos_config": "gcos.json",
        "debug_mode": True,
        "log_level": "DEBUG"
    }
)
```

### Fallback Behavior

The GCOS backend gracefully falls back to other storage backends when:
- GCOS is not available (`BUILD_WITH_GCOS=0`)
- GCOS initialization fails
- Device errors occur during operation
- Performance degradation is detected

## Migration Guide

### From GDS Backend

```python
# Old GDS configuration
config = lmcache.LMCacheEngineConfig(
    gds_path="/mnt/gds_cache",
    # ... other config
)

# New GCOS configuration  
config = lmcache.LMCacheEngineConfig(
    gcos_path="/mnt/gcos_cache",  # Change path parameter
    extra_config={
        "gcos_config": "gcos.json"  # Add GCOS config
    }
    # ... same other config
)
```

### API Compatibility

The GCOS backend maintains full API compatibility with existing LMCache code:
- Same `store()` and `retrieve()` methods
- Same cache key formats
- Same async operation patterns
- Same error handling interfaces

## Future Enhancements

1. **Advanced Features**:
   - RDMA support for multi-node configurations
   - Compression integration
   - Advanced caching policies

2. **Performance Optimizations**:
   - Adaptive queue management
   - Intelligent prefetching
   - Multi-stream operations

3. **Management Tools**:
   - Configuration validation utilities
   - Performance monitoring dashboards
   - Automated tuning recommendations

## References

- [GCOS Documentation](../storage/README.md)
- [LMCache Architecture](docs/architecture.md)
- [Performance Benchmarks](benchmarks/gcos_results.md)
- [Configuration Examples](examples/gcos_config_examples/)