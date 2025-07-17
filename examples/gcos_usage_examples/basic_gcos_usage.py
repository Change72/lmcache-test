#!/usr/bin/env python3
"""
Basic GCOS Backend Usage Example for LMCache

This example demonstrates how to use the GCOS backend with LMCache
for high-performance KV cache storage with direct GPU access.
"""

# Standard
import os
import tempfile
import torch

# Third Party
import torch

# First Party
import lmcache.v1 as lmcache
from lmcache.config import LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey


def basic_gcos_example():
    """Basic example showing GCOS backend usage."""
    print("🚀 Basic GCOS Backend Example")
    
    # GCOS cache directory
    cache_dir = "/tmp/gcos_cache_example"
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # 1. Configure LMCache with GCOS backend
        config = lmcache.LMCacheEngineConfig(
            chunk_size=256,
            gcos_path=cache_dir,  # Enable GCOS backend
            local_cpu=False,      # Disable local CPU cache
            max_local_cpu_size=1.0,
            save_decode_cache=True,
            extra_config={
                "gcos_config": "gcos.json"  # GCOS system config
            }
        )
        
        # 2. Create metadata
        metadata = LMCacheEngineMetadata(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            fmt="huggingface",
            world_size=1,
            worker_id=0
        )
        
        # 3. Build cache engine (automatically uses GCOS backend)
        cache_engine = lmcache.LMCacheEngineBuilder.build(config, metadata)
        
        print(f"✅ LMCache engine created with GCOS backend")
        print(f"   Cache path: {cache_dir}")
        print(f"   Storage backends: {list(cache_engine.storage_manager.storage_backends.keys())}")
        
        # 4. Create sample KV cache data
        batch_size = 4
        num_layers = 32
        num_heads = 32
        head_dim = 128
        seq_len = 512
        
        # Simulate KV caches (List of tensors for each layer)
        kv_caches = []
        for layer in range(num_layers):
            # Each layer has K and V tensors
            k_cache = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                                dtype=torch.float16, device='cuda')
            v_cache = torch.randn(batch_size, num_heads, seq_len, head_dim,
                                dtype=torch.float16, device='cuda')
            kv_caches.append([k_cache, v_cache])
        
        # 5. Create cache key
        cache_key = CacheEngineKey(
            fmt="huggingface",
            model_name="meta-llama/Llama-3.1-8B-Instruct", 
            world_size=1,
            worker_id=0,
            chunk_hash="example_prompt_hash_001"
        )
        
        # 6. Store KV caches (uses GCOS for direct GPU-to-storage)
        print(f"💾 Storing KV caches to GCOS...")
        tokens = ["Hello", "world", "this", "is", "a", "test"] * 85  # ~512 tokens
        
        cache_engine.store(
            cache_key,
            kv_caches,
            tokens
        )
        print(f"✅ KV caches stored successfully")
        
        # 7. Retrieve KV caches (uses GCOS for direct storage-to-GPU)
        print(f"📖 Retrieving KV caches from GCOS...")
        
        retrieved_kv, retrieved_tokens = cache_engine.retrieve(cache_key)
        
        if retrieved_kv is not None:
            print(f"✅ KV caches retrieved successfully")
            print(f"   Retrieved {len(retrieved_kv)} layers")
            print(f"   Retrieved {len(retrieved_tokens)} tokens")
            print(f"   First layer K shape: {retrieved_kv[0][0].shape}")
            print(f"   First layer V shape: {retrieved_kv[0][1].shape}")
        else:
            print(f"❌ Failed to retrieve KV caches")
        
        # 8. Show storage backend information
        gcos_backend = cache_engine.storage_manager.storage_backends.get("GcosBackend")
        if gcos_backend:
            print(f"📊 GCOS Backend Info:")
            print(f"   GCOS initialized: {getattr(gcos_backend, 'gcos_initialized', False)}")
            print(f"   Cache contains key: {gcos_backend.contains(cache_key)}")
            if hasattr(gcos_backend, 'stats'):
                stats = gcos_backend.get_stats()
                print(f"   Stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in GCOS example: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)


def advanced_gcos_example():
    """Advanced example showing GCOS with GPU connector integration."""
    print("\n🔧 Advanced GCOS Backend Example")
    
    cache_dir = "/tmp/gcos_cache_advanced"
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Advanced configuration with performance tuning
        config = lmcache.LMCacheEngineConfig(
            chunk_size=512,  # Larger chunks for better performance
            gcos_path=cache_dir,
            local_cpu=False,
            max_local_cpu_size=0.0,  # Disable CPU cache completely
            save_decode_cache=True,
            use_layerwise=True,  # Enable layerwise processing
            extra_config={
                "gcos_config": "examples/gcos_config_examples/multi_gpu_gcos.json",
                "use_cufile": True,
                "use_direct_io": True,
                "gcos_queue_num": 128,
                "gcos_queue_depth": 1024
            }
        )
        
        metadata = LMCacheEngineMetadata(
            model_name="meta-llama/Llama-3.1-70B-Instruct",
            fmt="vllm",
            world_size=8,
            worker_id=0
        )
        
        # Build with custom GPU connector
        cache_engine = lmcache.LMCacheEngineBuilder.build(config, metadata)
        
        print(f"✅ Advanced LMCache engine created")
        
        # Test multiple cache operations
        cache_keys = [
            CacheEngineKey(
                fmt="vllm",
                model_name="meta-llama/Llama-3.1-70B-Instruct",
                world_size=8,
                worker_id=0,
                chunk_hash=f"advanced_prompt_{i:03d}"
            )
            for i in range(3)
        ]
        
        # Store multiple KV caches
        for i, key in enumerate(cache_keys):
            print(f"💾 Storing cache {i+1}/3...")
            
            # Different sized KV caches
            seq_len = 256 + i * 128
            kv_caches = []
            for layer in range(8):  # Reduced for example
                k_cache = torch.randn(2, 64, seq_len, 128, 
                                    dtype=torch.float16, device='cuda')
                v_cache = torch.randn(2, 64, seq_len, 128,
                                    dtype=torch.float16, device='cuda')
                kv_caches.append([k_cache, v_cache])
            
            tokens = [f"token_{j}" for j in range(seq_len)]
            cache_engine.store(key, kv_caches, tokens)
        
        print(f"✅ All caches stored")
        
        # Retrieve and verify
        for i, key in enumerate(cache_keys):
            print(f"📖 Retrieving cache {i+1}/3...")
            retrieved_kv, retrieved_tokens = cache_engine.retrieve(key)
            
            if retrieved_kv:
                expected_seq_len = 256 + i * 128
                actual_seq_len = retrieved_kv[0][0].shape[2]
                print(f"   ✅ Retrieved seq_len: {actual_seq_len} (expected: {expected_seq_len})")
            else:
                print(f"   ❌ Failed to retrieve cache {i+1}")
        
        # Test GCOS-specific features
        gcos_backend = cache_engine.storage_manager.storage_backends.get("GcosBackend")
        if gcos_backend and hasattr(gcos_backend, 'gcos_ops'):
            print(f"🔧 GCOS Native Features:")
            print(f"   Native operations available: {gcos_backend.gcos_ops is not None}")
            print(f"   Base pointer: {getattr(gcos_backend, 'gcos_base_pointer', 'None')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in advanced example: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)


def configuration_examples():
    """Show different GCOS configuration patterns."""
    print("\n⚙️  GCOS Configuration Examples")
    
    # 1. Development configuration
    dev_config = lmcache.LMCacheEngineConfig(
        chunk_size=256,
        gcos_path="/tmp/gcos_dev",
        local_cpu=False,
        max_local_cpu_size=1.0,
        save_decode_cache=True,
        extra_config={
            "gcos_config": "examples/gcos_config_examples/basic_gcos.json"
        }
    )
    print(f"📝 Development config: {dev_config.gcos_path}")
    
    # 2. Production configuration
    prod_config = lmcache.LMCacheEngineConfig(
        chunk_size=512,
        gcos_path="/mnt/gcos_cache",
        local_cpu=False,
        max_local_cpu_size=0.0,
        save_decode_cache=True,
        use_layerwise=True,
        extra_config={
            "gcos_config": "gcos.json",
            "use_cufile": True,
            "use_direct_io": True
        }
    )
    print(f"🏭 Production config: {prod_config.gcos_path}")
    
    # 3. Multi-GPU configuration
    multi_gpu_config = lmcache.LMCacheEngineConfig(
        chunk_size=1024,
        gcos_path="/fast_storage/gcos_cache",
        local_cpu=False,
        max_local_cpu_size=0.0,
        save_decode_cache=True,
        use_layerwise=True,
        extra_config={
            "gcos_config": "examples/gcos_config_examples/multi_gpu_gcos.json",
            "use_cufile": True,
            "use_direct_io": True,
            "gcos_queue_num": 256,
            "gcos_queue_depth": 2048
        }
    )
    print(f"🔥 Multi-GPU config: {multi_gpu_config.gcos_path}")
    
    return [dev_config, prod_config, multi_gpu_config]


if __name__ == "__main__":
    print("🎯 GCOS Backend Usage Examples for LMCache\n")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. GCOS examples require GPU.")
        exit(1)
    
    # Check if GCOS is built
    gcos_available = os.environ.get("BUILD_WITH_GCOS", "0") == "1"
    if not gcos_available:
        print("⚠️  GCOS not built. Set BUILD_WITH_GCOS=1 for full functionality.")
        print("   Running in fallback mode...\n")
    else:
        print("✅ GCOS build detected. Running with native GCOS support.\n")
    
    success = True
    
    # Run examples
    success &= basic_gcos_example()
    success &= advanced_gcos_example()
    
    # Show configurations
    configs = configuration_examples()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 All GCOS examples completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Configure your GCOS devices in gcos.json")
        print("   2. Set appropriate cache paths and sizes") 
        print("   3. Tune queue parameters for your workload")
        print("   4. Monitor performance with GCOS metrics")
    else:
        print("❌ Some examples failed. Check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure GCOS devices are available")
        print("   2. Check gcos.json configuration")
        print("   3. Verify GPU and storage permissions")
        print("   4. Build with BUILD_WITH_GCOS=1 for full features")
    
    print(f"{'='*60}")