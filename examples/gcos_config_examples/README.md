# GCOS Configuration Examples for LMCache

This directory contains example GCOS configuration files for different deployment scenarios with LMCache.

## Configuration Files

### 1. `basic_gcos.json`
- **Use case**: Development, testing, single NVMe device
- **Hardware**: 1 NVMe device, 1 GPU
- **Features**: Basic configuration with minimal resource allocation
- **Best for**: Local development, proof of concept

### 2. `multi_gpu_gcos.json` 
- **Use case**: Production deployment, high-performance computing
- **Hardware**: 8 NVMe devices, 8 GPUs
- **Features**: Multi-GPU support, high concurrency, performance optimization
- **Best for**: Large-scale LLM serving, tensor parallel workloads

### 3. `../gcos.json` (Root level)
- **Use case**: Standard production deployment
- **Hardware**: 4 NVMe devices, 1-4 GPUs
- **Features**: Balanced configuration for typical production workloads
- **Best for**: Most production LMCache deployments

## Configuration Parameters

### Disk Configuration
- **number**: Number of NVMe devices
- **paths**: Device paths (e.g., `/dev/libnvm0`)
- **namespace_number**: Number of namespaces per device
- **block_size**: Block size in bytes (typically 512)
- **capacity_bytes**: Total storage capacity
- **reserve_max_queue_pair_num**: Maximum queue pairs

### Namespace Configuration
- **cpu.qps**: CPU queue pairs
- **cpu.queue_depth**: CPU queue depth
- **gpu.num**: Number of GPUs
- **gpu.qps**: GPU queue pairs
- **gpu.queue_depth**: GPU queue depth

### Zone Chunk Sizes (in KB)
- **sb**: Superblock zone chunk size
- **meta**: Metadata zone chunk size  
- **data**: Data zone chunk size (for KV cache storage)

## Usage with LMCache

### Python Configuration
```python
import lmcache.v1 as lmcache

config = lmcache.LMCacheEngineConfig(
    gcos_path="/path/to/cache/storage",
    extra_config={
        "gcos_config": "examples/gcos_config_examples/basic_gcos.json"
    }
)

cache_engine = lmcache.LMCacheEngine(config)
```

### YAML Configuration
```yaml
# lmcache_config.yaml
gcos_path: "/mnt/gcos_cache"
extra_config:
  gcos_config: "examples/gcos_config_examples/multi_gpu_gcos.json"
```

## Hardware Requirements

### Minimum (basic_gcos.json)
- 1x NVMe SSD with GCOS driver support
- 1x CUDA-capable GPU
- 16GB+ system RAM
- PCIe 3.0+ connectivity

### Recommended (multi_gpu_gcos.json)
- 8x High-performance NVMe SSDs
- 8x High-end GPUs (A100, H100, etc.)
- 128GB+ system RAM
- PCIe 4.0+ with GPUDirect support
- NUMA-aware memory allocation

## Performance Tuning

### For High Throughput
- Increase `gpu.qps` and `gpu.queue_depth`
- Use larger `data` zone chunk sizes (16MB+)
- Enable `gpu_direct_rdma` if supported

### For Low Latency
- Reduce queue depths for faster response
- Use smaller chunk sizes for fine-grained access
- Increase number of completion threads

### For Memory Efficiency
- Adjust zone chunk sizes based on typical KV cache sizes
- Balance CPU vs GPU queue allocation
- Tune `reserve_max_queue_pair_num` based on workload

## Troubleshooting

### Common Issues
1. **Device not found**: Verify NVMe device paths exist
2. **Permission denied**: Ensure proper device permissions
3. **Queue allocation failed**: Reduce queue pair numbers
4. **Performance issues**: Check PCIe bandwidth and NUMA configuration

### Verification
```bash
# Check GCOS device availability
ls -la /dev/libnvm*

# Verify GPU access
nvidia-smi

# Test configuration
BUILD_WITH_GCOS=1 python tests/v1/test_gcos.py
```