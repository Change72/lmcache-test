# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GCOS GPU Connector for direct GPU-initiated storage access."""

import time
from typing import List, Optional, Union, Dict, Any
import torch
import ctypes
import numpy as np

from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryObj
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

try:
    # Import GCOS native library
    import lmcache.csrc.gcos_ops as gcos_ops
    GCOS_AVAILABLE = True
except ImportError:
    logger.warning("GCOS native library not available, falling back to standard operations")
    GCOS_AVAILABLE = False


class GCOSGPUConnector(GPUConnectorInterface):
    """
    GPU connector optimized for GCOS direct storage access.
    
    This connector enables direct GPU-initiated storage operations using the
    GCOS/BaM architecture, bypassing CPU involvement in the storage control path.
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = True,
        **kwargs,
    ):
        """
        Initialize GCOS GPU Connector.
        
        Args:
            hidden_dim_size (int): Size of hidden dimension
            num_layers (int): Number of transformer layers
            use_gpu (bool): Whether to use GPU memory (always True for GCOS)
            **kwargs: Additional configuration parameters
        """
        super().__init__()
        
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        
        # GCOS-specific configuration
        self.chunk_size = kwargs.get('chunk_size', 256)
        self.dtype = kwargs.get('dtype', torch.float16)
        self.device = kwargs.get('device', torch.device('cuda:0'))
        
        # GCOS queue configuration
        self.num_queues = kwargs.get('num_gcos_queues', 128)
        self.queue_depth = kwargs.get('gcos_queue_depth', 1024)
        self.alignment = kwargs.get('gcos_alignment', 4096)  # 4KB alignment
        
        # Initialize GCOS components
        self.gcos_controllers: List[Any] = []
        self.gpu_queues: List[Any] = []
        self.dma_buffers: Dict[int, torch.Tensor] = {}
        
        # Performance monitoring
        self.stats = {
            'total_reads': 0,
            'total_writes': 0,
            'total_bytes_read': 0,
            'total_bytes_written': 0,
            'avg_read_latency': 0.0,
            'avg_write_latency': 0.0,
        }
        
        if GCOS_AVAILABLE:
            self._initialize_gcos()
        else:
            logger.warning("GCOS not available, connector will use fallback operations")

    def _initialize_gcos(self):
        """Initialize GCOS controllers and GPU queues."""
        try:
            # Initialize GCOS controllers
            nvme_paths = [f"/dev/libnvm{i}" for i in range(4)]  # Default 4 NVMe devices
            self.gcos_controllers = gcos_ops.initialize_controllers(nvme_paths)
            
            # Create GPU queue pairs for high-concurrency access
            self.gpu_queues = gcos_ops.create_gpu_queues(
                controllers=self.gcos_controllers,
                num_queues=self.num_queues,
                queue_depth=self.queue_depth,
                device=self.device
            )
            
            logger.info(f"GCOS initialized with {len(self.gcos_controllers)} controllers "
                       f"and {len(self.gpu_queues)} GPU queues")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCOS: {e}")
            raise RuntimeError(f"GCOS initialization failed: {e}")

    def get_shape(self, num_tokens: int) -> torch.Size:
        """
        Get the shape for KV cache tensors.
        
        Args:
            num_tokens (int): Number of tokens
            
        Returns:
            torch.Size: Shape of the KV cache tensor
        """
        return torch.Size([
            self.num_layers, 
            2,  # K and V
            num_tokens, 
            self.hidden_dim_size
        ])

    @_lmcache_nvtx_annotate
    def from_gpu(
        self,
        memory_obj: MemoryObj,
        start: int,
        end: int,
        **kwargs
    ) -> None:
        """
        Transfer KV cache data from GPU memory to storage using GCOS.
        
        Args:
            memory_obj (MemoryObj): Memory object to store data
            start (int): Start token index
            end (int): End token index
            **kwargs: Additional arguments including slot_mapping and kvcaches
        """
        kvcaches = kwargs.get('kvcaches')
        slot_mapping = kwargs.get('slot_mapping')
        offset = kwargs.get('offset', 0)
        
        if not kvcaches or slot_mapping is None:
            raise ValueError("kvcaches and slot_mapping are required for GCOS transfer")
        
        num_tokens = end - start
        if num_tokens <= 0:
            return
            
        start_time = time.perf_counter()
        
        try:
            if GCOS_AVAILABLE:
                self._gcos_gpu_to_storage(memory_obj, kvcaches, slot_mapping, start, end, offset)
            else:
                self._fallback_gpu_to_storage(memory_obj, kvcaches, slot_mapping, start, end, offset)
                
        except Exception as e:
            logger.error(f"GCOS from_gpu failed: {e}")
            # Fallback to standard operation
            self._fallback_gpu_to_storage(memory_obj, kvcaches, slot_mapping, start, end, offset)
        
        # Update statistics
        transfer_time = time.perf_counter() - start_time
        self.stats['total_writes'] += 1
        self.stats['total_bytes_written'] += memory_obj.get_size()
        self.stats['avg_write_latency'] = (
            (self.stats['avg_write_latency'] * (self.stats['total_writes'] - 1) + transfer_time)
            / self.stats['total_writes']
        )

    @_lmcache_nvtx_annotate
    def to_gpu(
        self,
        memory_obj: MemoryObj,
        start: int,
        end: int,
        **kwargs
    ) -> None:
        """
        Transfer KV cache data from storage to GPU memory using GCOS.
        
        Args:
            memory_obj (MemoryObj): Memory object containing data
            start (int): Start token index
            end (int): End token index
            **kwargs: Additional arguments including slot_mapping and kvcaches
        """
        kvcaches = kwargs.get('kvcaches')
        slot_mapping = kwargs.get('slot_mapping')
        
        if not kvcaches or slot_mapping is None:
            raise ValueError("kvcaches and slot_mapping are required for GCOS transfer")
        
        num_tokens = end - start
        if num_tokens <= 0:
            return
            
        start_time = time.perf_counter()
        
        try:
            if GCOS_AVAILABLE:
                self._gcos_storage_to_gpu(memory_obj, kvcaches, slot_mapping, start, end)
            else:
                self._fallback_storage_to_gpu(memory_obj, kvcaches, slot_mapping, start, end)
                
        except Exception as e:
            logger.error(f"GCOS to_gpu failed: {e}")
            # Fallback to standard operation
            self._fallback_storage_to_gpu(memory_obj, kvcaches, slot_mapping, start, end)
        
        # Update statistics
        transfer_time = time.perf_counter() - start_time
        self.stats['total_reads'] += 1
        self.stats['total_bytes_read'] += memory_obj.get_size()
        self.stats['avg_read_latency'] = (
            (self.stats['avg_read_latency'] * (self.stats['total_reads'] - 1) + transfer_time)
            / self.stats['total_reads']
        )

    def _gcos_gpu_to_storage(
        self,
        memory_obj: MemoryObj,
        kvcaches: List[torch.Tensor],
        slot_mapping: torch.Tensor,
        start: int,
        end: int,
        offset: int
    ) -> None:
        """Direct GPU-to-storage transfer using GCOS."""
        num_tokens = end - start
        
        # Prepare aligned storage addresses
        storage_addresses = self._get_storage_addresses(memory_obj, start, end)
        
        # Launch GPU kernel for direct storage writes
        gcos_ops.gpu_store_kv_kernel(
            kv_caches=kvcaches,
            slot_mapping=slot_mapping[start:end],
            storage_addresses=storage_addresses,
            num_tokens=num_tokens,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim_size,
            gpu_queues=self.gpu_queues,
            offset=offset
        )
        
        # Ensure all GPU operations complete
        torch.cuda.synchronize()

    def _gcos_storage_to_gpu(
        self,
        memory_obj: MemoryObj,
        kvcaches: List[torch.Tensor],
        slot_mapping: torch.Tensor,
        start: int,
        end: int
    ) -> None:
        """Direct storage-to-GPU transfer using GCOS."""
        num_tokens = end - start
        
        # Prepare aligned storage addresses
        storage_addresses = self._get_storage_addresses(memory_obj, start, end)
        
        # Launch GPU kernel for direct storage reads
        gcos_ops.gpu_retrieve_kv_kernel(
            kv_caches=kvcaches,
            slot_mapping=slot_mapping[start:end],
            storage_addresses=storage_addresses,
            num_tokens=num_tokens,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim_size,
            gpu_queues=self.gpu_queues
        )
        
        # Ensure all GPU operations complete
        torch.cuda.synchronize()

    def _get_storage_addresses(
        self,
        memory_obj: MemoryObj,
        start: int,
        end: int
    ) -> torch.Tensor:
        """Get aligned storage addresses for token range."""
        num_tokens = end - start
        base_addr = memory_obj.get_base_address()
        token_size = self._get_token_storage_size()
        
        addresses = torch.zeros(num_tokens, dtype=torch.int64, device=self.device)
        for i in range(num_tokens):
            token_offset = (start + i) * token_size
            aligned_addr = self._align_address(base_addr + token_offset)
            addresses[i] = aligned_addr
            
        return addresses

    def _get_token_storage_size(self) -> int:
        """Calculate storage size per token."""
        # Size = num_layers * 2 (K,V) * hidden_dim * dtype_size
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        return self.num_layers * 2 * self.hidden_dim_size * dtype_size

    def _align_address(self, addr: int) -> int:
        """Align address to GCOS requirements (4KB alignment)."""
        return (addr + self.alignment - 1) & ~(self.alignment - 1)

    def _fallback_gpu_to_storage(
        self,
        memory_obj: MemoryObj,
        kvcaches: List[torch.Tensor],
        slot_mapping: torch.Tensor,
        start: int,
        end: int,
        offset: int
    ) -> None:
        """Fallback GPU-to-storage transfer using standard operations."""
        logger.debug("Using fallback GPU-to-storage transfer")
        
        # Extract KV data from GPU memory
        kv_data = self._extract_kv_data(kvcaches, slot_mapping, start, end)
        
        # Copy to CPU and then to storage
        cpu_data = kv_data.cpu()
        memory_obj.store_data(cpu_data.numpy(), offset)

    def _fallback_storage_to_gpu(
        self,
        memory_obj: MemoryObj,
        kvcaches: List[torch.Tensor],
        slot_mapping: torch.Tensor,
        start: int,
        end: int
    ) -> None:
        """Fallback storage-to-GPU transfer using standard operations."""
        logger.debug("Using fallback storage-to-GPU transfer")
        
        # Load data from storage to CPU
        cpu_data = memory_obj.load_data()
        
        # Convert to GPU tensor
        gpu_data = torch.from_numpy(cpu_data).to(self.device)
        
        # Insert into KV caches
        self._insert_kv_data(gpu_data, kvcaches, slot_mapping, start, end)

    def _extract_kv_data(
        self,
        kvcaches: List[torch.Tensor],
        slot_mapping: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """Extract KV data from GPU memory."""
        num_tokens = end - start
        kv_shape = self.get_shape(num_tokens)
        extracted_data = torch.zeros(kv_shape, dtype=self.dtype, device=self.device)
        
        for layer_idx, kv_cache in enumerate(kvcaches):
            for token_idx in range(num_tokens):
                slot_idx = slot_mapping[start + token_idx]
                if slot_idx >= 0:  # Valid slot
                    # Extract K and V for this token
                    extracted_data[layer_idx, 0, token_idx] = kv_cache[0, slot_idx]  # K
                    extracted_data[layer_idx, 1, token_idx] = kv_cache[1, slot_idx]  # V
                    
        return extracted_data

    def _insert_kv_data(
        self,
        kv_data: torch.Tensor,
        kvcaches: List[torch.Tensor],
        slot_mapping: torch.Tensor,
        start: int,
        end: int
    ) -> None:
        """Insert KV data into GPU memory."""
        num_tokens = end - start
        
        for layer_idx, kv_cache in enumerate(kvcaches):
            for token_idx in range(num_tokens):
                slot_idx = slot_mapping[start + token_idx]
                if slot_idx >= 0:  # Valid slot
                    # Insert K and V for this token
                    kv_cache[0, slot_idx] = kv_data[layer_idx, 0, token_idx]  # K
                    kv_cache[1, slot_idx] = kv_data[layer_idx, 1, token_idx]  # V

    def batched_from_gpu(
        self,
        memory_objs: List[MemoryObj],
        starts: List[int],
        ends: List[int],
        **kwargs
    ) -> None:
        """
        Batch transfer from GPU to storage.
        
        Args:
            memory_objs (List[MemoryObj]): List of memory objects
            starts (List[int]): List of start indices
            ends (List[int]): List of end indices
            **kwargs: Additional arguments
        """
        for memory_obj, start, end in zip(memory_objs, starts, ends):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def batched_to_gpu(
        self,
        memory_objs: List[MemoryObj],
        starts: List[int],
        ends: List[int],
        **kwargs
    ) -> None:
        """
        Batch transfer from storage to GPU.
        
        Args:
            memory_objs (List[MemoryObj]): List of memory objects
            starts (List[int]): List of start indices
            ends (List[int]): List of end indices
            **kwargs: Additional arguments
        """
        for memory_obj, start, end in zip(memory_objs, starts, ends):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get performance statistics."""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            'total_reads': 0,
            'total_writes': 0,
            'total_bytes_read': 0,
            'total_bytes_written': 0,
            'avg_read_latency': 0.0,
            'avg_write_latency': 0.0,
        }

    def cleanup(self) -> None:
        """Clean up GCOS resources."""
        if GCOS_AVAILABLE and self.gcos_controllers:
            try:
                gcos_ops.cleanup_controllers(self.gcos_controllers)
                gcos_ops.cleanup_gpu_queues(self.gpu_queues)
                logger.info("GCOS resources cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up GCOS resources: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()