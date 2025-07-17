# Standard
from pathlib import Path
import asyncio
import os
import shutil
import tempfile
import threading

# Third Party
import safetensors
import torch
import pytest

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import CuFileMemoryAllocator
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.gcos_backend import pack_metadata, unpack_metadata


def test_gcos_backend_metadata():
    """Test GCOS backend metadata packing and unpacking functionality."""
    # This is a sanity check that packing and unpacking works. We can add
    # more tensor types to be sure.
    for [tensor, expected_nbytes] in [(torch.randn(3, 10), 120)]:
        r = pack_metadata(tensor, lmcache_version="1")
        size, dtype, nbytes, meta = unpack_metadata(r)
        assert size == tensor.size()
        assert dtype == tensor.dtype
        assert expected_nbytes == nbytes
        assert meta["lmcache_version"] == "1"

        # Make sure that safetensors can load this
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "test.safetensors")
            with open(temp_file_path, "wb") as f:
                f.write(r)
                f.write(b" " * nbytes)

            with safetensors.safe_open(temp_file_path, framework="pt") as f:
                tensor = f.get_tensor("kvcache")
                assert size == tensor.size()
                assert dtype == tensor.dtype
                assert expected_nbytes == nbytes


def test_gcos_backend_sanity():
    """Test basic GCOS backend functionality including put, get, and contains operations."""
    BASE_DIR = Path(__file__).parent
    GCOS_DIR = "/tmp/gcos/test-cache"
    TEST_KEY = CacheEngineKey(
        fmt="vllm",
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        world_size=8,
        worker_id=0,
        chunk_hash="e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406c157",
    )
    BACKEND_NAME = "GcosBackend"

    try:
        os.makedirs(GCOS_DIR, exist_ok=True)
        config_gcos = LMCacheEngineConfig.from_file(BASE_DIR / "data/gcos.yaml")
        assert config_gcos.gcos_path == GCOS_DIR

        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        backends = CreateStorageBackends(
            config_gcos,
            None,
            thread_loop,
            LMCacheEngineBuilder._Create_memory_allocator(config_gcos, None),
        )
        assert len(backends) == 2  # LocalCPUBackend + GcosBackend
        assert BACKEND_NAME in backends

        gcos_backend = backends[BACKEND_NAME]
        assert gcos_backend is not None
        assert gcos_backend.memory_allocator is not None
        assert isinstance(gcos_backend.memory_allocator, CuFileMemoryAllocator)

        # Test that the key doesn't exist initially
        assert not gcos_backend.contains(TEST_KEY, False)
        assert not gcos_backend.exists_in_put_tasks(TEST_KEY)

        # Test put operation
        memory_obj = gcos_backend.memory_allocator.allocate(
            [2048, 2048], dtype=torch.uint8
        )
        future = gcos_backend.submit_put_task(TEST_KEY, memory_obj)
        assert future is not None
        assert gcos_backend.exists_in_put_tasks(TEST_KEY)
        assert not gcos_backend.contains(TEST_KEY, False)
        
        # Wait for put to complete
        future.result()
        assert gcos_backend.contains(TEST_KEY, False)
        assert not gcos_backend.exists_in_put_tasks(TEST_KEY)

        # Test blocking get operation
        returned_memory_obj = gcos_backend.get_blocking(TEST_KEY)
        assert returned_memory_obj is not None
        assert returned_memory_obj.get_size() == memory_obj.get_size()
        assert returned_memory_obj.get_shape() == memory_obj.get_shape()
        assert returned_memory_obj.get_dtype() == memory_obj.get_dtype()

        # Test non-blocking get operation
        future = gcos_backend.get_non_blocking(TEST_KEY)
        assert future is not None
        returned_memory_obj = future.result()
        assert returned_memory_obj is not None
        assert returned_memory_obj.get_size() == memory_obj.get_size()
        assert returned_memory_obj.get_shape() == memory_obj.get_shape()
        assert returned_memory_obj.get_dtype() == memory_obj.get_dtype()
        
        # Clean up memory objects
        memory_obj.ref_count_down()
        returned_memory_obj.ref_count_down()
        
    finally:
        if thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread.is_alive():
            thread.join()
        # We rmtree AFTER we ensure that the thread loop is done.
        # This way we don't hit any race conditions in rmtree()
        # where temp files are renamed while we try to unlink them.
        # We also take care of any other errors with ignore_errors=True
        # so if we want to run tests in parallel in the future they
        # don't make each other fail.
        if os.path.exists(GCOS_DIR):
            shutil.rmtree(GCOS_DIR, ignore_errors=True)


def test_gcos_backend_multiple_keys():
    """Test GCOS backend with multiple cache keys."""
    GCOS_DIR = "/tmp/gcos/test-cache-multi"
    BASE_DIR = Path(__file__).parent
    
    TEST_KEYS = [
        CacheEngineKey(
            fmt="vllm",
            model_name="meta-llama/Llama-3.1-70B-Instruct",
            world_size=8,
            worker_id=0,
            chunk_hash=f"test_hash_{i:08x}",
        )
        for i in range(3)
    ]
    BACKEND_NAME = "GcosBackend"

    try:
        os.makedirs(GCOS_DIR, exist_ok=True)
        config_gcos = LMCacheEngineConfig(
            chunk_size=256,
            gcos_path=GCOS_DIR,
            local_cpu=False,
            max_local_cpu_size=1.0,
            save_decode_cache=True,
            extra_config={"gcos_config": "gcos.json"}
        )

        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        backends = CreateStorageBackends(
            config_gcos,
            None,
            thread_loop,
            LMCacheEngineBuilder._Create_memory_allocator(config_gcos, None),
        )
        
        gcos_backend = backends[BACKEND_NAME]
        
        # Store multiple objects
        memory_objs = []
        futures = []
        
        for i, key in enumerate(TEST_KEYS):
            memory_obj = gcos_backend.memory_allocator.allocate(
                [1024, 512], dtype=torch.float16
            )
            memory_objs.append(memory_obj)
            future = gcos_backend.submit_put_task(key, memory_obj)
            futures.append(future)
        
        # Wait for all puts to complete
        for future in futures:
            future.result()
        
        # Verify all keys exist
        for key in TEST_KEYS:
            assert gcos_backend.contains(key, False)
        
        # Retrieve all objects and verify
        for i, key in enumerate(TEST_KEYS):
            returned_obj = gcos_backend.get_blocking(key)
            assert returned_obj is not None
            assert returned_obj.get_size() == memory_objs[i].get_size()
            assert returned_obj.get_shape() == memory_objs[i].get_shape()
            assert returned_obj.get_dtype() == memory_objs[i].get_dtype()
            returned_obj.ref_count_down()
        
        # Clean up
        for memory_obj in memory_objs:
            memory_obj.ref_count_down()
            
    finally:
        if thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread.is_alive():
            thread.join()
        if os.path.exists(GCOS_DIR):
            shutil.rmtree(GCOS_DIR, ignore_errors=True)


def test_gcos_backend_error_handling():
    """Test GCOS backend error handling for invalid operations."""
    GCOS_DIR = "/tmp/gcos/test-cache-error"
    BACKEND_NAME = "GcosBackend"
    
    INVALID_KEY = CacheEngineKey(
        fmt="vllm",
        model_name="nonexistent/model",
        world_size=1,
        worker_id=0,
        chunk_hash="nonexistent_hash",
    )

    try:
        os.makedirs(GCOS_DIR, exist_ok=True)
        config_gcos = LMCacheEngineConfig(
            chunk_size=256,
            gcos_path=GCOS_DIR,
            local_cpu=False,
            max_local_cpu_size=1.0,
            save_decode_cache=True,
        )

        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        backends = CreateStorageBackends(
            config_gcos,
            None,
            thread_loop,
            LMCacheEngineBuilder._Create_memory_allocator(config_gcos, None),
        )
        
        gcos_backend = backends[BACKEND_NAME]
        
        # Test getting non-existent key
        assert not gcos_backend.contains(INVALID_KEY, False)
        returned_obj = gcos_backend.get_blocking(INVALID_KEY)
        assert returned_obj is None
        
        # Test non-blocking get on non-existent key
        future = gcos_backend.get_non_blocking(INVALID_KEY)
        assert future is None
        
    finally:
        if thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread.is_alive():
            thread.join()
        if os.path.exists(GCOS_DIR):
            shutil.rmtree(GCOS_DIR, ignore_errors=True)


@pytest.mark.skipif(
    not os.environ.get("BUILD_WITH_GCOS", "0") == "1",
    reason="GCOS not built - set BUILD_WITH_GCOS=1 to enable"
)
def test_gcos_backend_native_operations():
    """Test GCOS backend with native GCOS operations (requires GCOS build)."""
    GCOS_DIR = "/tmp/gcos/test-cache-native"
    BACKEND_NAME = "GcosBackend"
    
    TEST_KEY = CacheEngineKey(
        fmt="vllm",
        model_name="test/model",
        world_size=1,
        worker_id=0,
        chunk_hash="native_test_hash",
    )

    try:
        os.makedirs(GCOS_DIR, exist_ok=True)
        config_gcos = LMCacheEngineConfig(
            chunk_size=256,
            gcos_path=GCOS_DIR,
            local_cpu=False,
            max_local_cpu_size=1.0,
            save_decode_cache=True,
            extra_config={"gcos_config": "gcos.json"}
        )

        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        backends = CreateStorageBackends(
            config_gcos,
            None,
            thread_loop,
            LMCacheEngineBuilder._Create_memory_allocator(config_gcos, None),
        )
        
        gcos_backend = backends[BACKEND_NAME]
        
        # Verify GCOS is initialized
        assert gcos_backend.gcos_initialized, "GCOS should be initialized for native operations"
        assert gcos_backend.gcos_ops is not None, "GCOS operations should be available"
        
        # Test with native GCOS operations
        memory_obj = gcos_backend.memory_allocator.allocate(
            [512, 256], dtype=torch.float32
        )
        
        future = gcos_backend.submit_put_task(TEST_KEY, memory_obj)
        future.result()
        
        assert gcos_backend.contains(TEST_KEY, False)
        
        returned_obj = gcos_backend.get_blocking(TEST_KEY)
        assert returned_obj is not None
        assert returned_obj.get_size() == memory_obj.get_size()
        
        # Clean up
        memory_obj.ref_count_down()
        returned_obj.ref_count_down()
        
    finally:
        if thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread.is_alive():
            thread.join()
        if os.path.exists(GCOS_DIR):
            shutil.rmtree(GCOS_DIR, ignore_errors=True)


if __name__ == "__main__":
    # Run basic tests that don't require GCOS build
    test_gcos_backend_metadata()
    print("✓ Metadata test passed")
    
    test_gcos_backend_sanity()
    print("✓ Sanity test passed")
    
    test_gcos_backend_multiple_keys()
    print("✓ Multiple keys test passed")
    
    test_gcos_backend_error_handling()
    print("✓ Error handling test passed")
    
    if os.environ.get("BUILD_WITH_GCOS", "0") == "1":
        test_gcos_backend_native_operations()
        print("✓ Native operations test passed")
    else:
        print("⚠ Native operations test skipped (BUILD_WITH_GCOS=0)")
    
    print("All tests passed! 🎉")