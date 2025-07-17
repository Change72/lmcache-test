# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LMCache is an LLM serving engine extension designed to reduce Time To First Token (TTFT) and increase throughput, particularly for long-context scenarios. It achieves this by storing and reusing KV caches across various storage locations (GPU, CPU DRAM, Local Disk) for any reusable text segments.

## Build System & Dependencies

### Installation & Building
```bash
# Install from source with CUDA extensions
pip install -e .

# For HIP/ROCm builds
BUILD_WITH_HIP=1 pip install -e .

# Install development dependencies
pip install -r requirements/build.txt
pip install -r requirements/test.txt
pip install -r requirements/lint.txt
```

### Testing
```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/v1/
pytest tests/test_cache_engine.py

# Run with coverage
pytest --cov=lmcache
```

### Code Quality
The project uses pre-commit hooks for code quality management:
```bash
pip install -r requirements/lint.txt
pre-commit install
```

Pre-commit hooks will automatically run linting, formatting, and other checks before each commit.

## Architecture Overview

### Dual Version System
LMCache maintains two major versions:
- **V0 (Legacy)**: Original implementation in `lmcache/` root
- **V1 (Current)**: Enhanced implementation in `lmcache/v1/`

### Core Components

#### Cache Engine (`lmcache/cache_engine.py`, `lmcache/v1/cache_engine.py`)
Central orchestrator for KV cache management, handling storage, retrieval, and blending operations.

#### Storage Backends (`lmcache/storage_backend/`, `lmcache/v1/storage_backend/`)
Abstraction layer supporting multiple storage types:
- **Local**: CPU/GPU memory, local disk
- **Remote**: Redis, InfiniStore, MoonCake, external connectors
- **Hybrid**: Combination of local and remote storage
- **Specialized**: GDS, WEKA, NIXL for high-performance scenarios

#### Configuration (`lmcache/config.py`, `lmcache/v1/config.py`)
Centralized configuration management supporting:
- YAML file configuration
- Environment variable overrides
- Runtime configuration updates

#### Integration Layer (`lmcache/integration/`)
- **vLLM integration**: `vllm/` directory with adapters for different vLLM versions
- **SGLang integration**: `sglang/` directory with SGLang-specific adapters

#### Compute Layer (`lmcache/v1/compute/`)
- **Attention mechanisms**: Flash attention and metadata handling
- **Blending**: KV cache blending for RAG and multi-round scenarios
- **Model support**: LLaMA and other transformer architectures

### Key Features

#### KV Cache Blending
Advanced feature for combining cached segments from different contexts (e.g., RAG scenarios).
Configuration via `enable_blending`, `blend_recompute_ratio`, `blend_min_tokens`.

#### Compression Support
CacheGen compression for efficient storage and transmission of KV caches.
Implemented in `serde/` modules with CUDA kernels in `csrc/`.

#### Distributed Support
- **Disaggregated Prefill**: Separate prefill and decode phases across different instances
- **P2P KV Sharing**: Direct cache sharing between inference instances
- **Multi-GPU**: Tensor parallelism support

## Common Development Tasks

### Adding New Storage Backends
1. Implement `AbstractBackend` interface in appropriate version directory
2. Add connector implementation in `storage_backend/connector/`
3. Update factory methods and configuration parsing
4. Add comprehensive tests in `tests/test_backends.py`

### Integration with New Serving Engines
1. Create adapter in `lmcache/integration/[engine_name]/`
2. Implement engine-specific connection logic
3. Add configuration templates in `examples/`
4. Document usage patterns

### Performance Optimization
- Profile using built-in observability tools (`lmcache/observability.py`)
- Optimize CUDA kernels in `csrc/` directory
- Tune compression algorithms in `serde/` modules

## Configuration Examples

### Basic Local Usage
```yaml
chunk_size: 256
local_device: "cuda"
max_local_cache_size: 5
remote_url: null
```

### Distributed Setup
```yaml
chunk_size: 512
local_device: "cuda"
max_local_cache_size: 20
remote_url: "redis://redis-server:6379"
remote_serde: "cachegen"
pipelined_backend: true
```

### Blending Configuration
```yaml
enable_blending: true
blend_recompute_ratio: 0.15
blend_min_tokens: 256
blend_separator: "[BLEND_SEP]"
```

## Debugging & Logging

The project uses structured logging via `lmcache/logging.py`. Enable debug mode:
```python
from lmcache.config import GlobalConfig
GlobalConfig.set_debug(True)
```

## Entry Points

The project provides several command-line tools:
- `lmcache_server`: V1 storage server
- `lmcache_v0_server`: Legacy storage server  
- `lmcache_controller`: V1 API controller

## Development Notes

- CUDA extensions require PyTorch 2.7.0 compatibility
- Use absolute imports with proper First Party/Third Party organization
- Follow the existing dataclass pattern for configuration objects
- Maintain backward compatibility when modifying V0 APIs
- Add comprehensive tests for any new storage backend implementations