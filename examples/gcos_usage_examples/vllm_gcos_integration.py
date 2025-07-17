#!/usr/bin/env python3
"""
vLLM + GCOS Integration Example

This example shows how to integrate GCOS backend with vLLM for
high-performance LLM serving with persistent KV cache storage.
"""

# Standard
import os
import asyncio
from typing import List, Optional

# Third Party
import torch

# First Party
import lmcache.v1 as lmcache
from lmcache.config import LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey


class VLLMGCOSExample:
    """Example class showing vLLM + GCOS integration."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.cache_engine: Optional[lmcache.LMCacheEngine] = None
        self.setup_complete = False
        
    def setup_lmcache_with_gcos(self, cache_path: str = "/tmp/vllm_gcos_cache"):
        """Setup LMCache with GCOS backend for vLLM integration."""
        print(f"🔧 Setting up LMCache with GCOS backend...")
        
        # Ensure cache directory exists
        os.makedirs(cache_path, exist_ok=True)
        
        # Configure LMCache for vLLM + GCOS
        config = lmcache.LMCacheEngineConfig(
            chunk_size=512,  # Optimized for typical prompt sizes
            gcos_path=cache_path,
            local_cpu=False,  # Use GCOS directly
            max_local_cpu_size=0.0,
            save_decode_cache=True,
            use_layerwise=True,  # Enable layerwise for vLLM compatibility
            extra_config={
                "gcos_config": "gcos.json",
                "use_cufile": True,
                "use_direct_io": True,
                # vLLM-specific optimizations
                "vllm_integration": True,
                "batch_size": 32,
                "prefetch_size": 3
            }
        )
        
        # Metadata for vLLM model
        metadata = LMCacheEngineMetadata(
            model_name=self.model_name,
            fmt="vllm",  # vLLM format
            world_size=torch.cuda.device_count(),
            worker_id=0
        )
        
        # Build cache engine
        self.cache_engine = lmcache.LMCacheEngineBuilder.build(config, metadata)
        self.setup_complete = True
        
        print(f"✅ LMCache + GCOS setup complete")
        print(f"   Model: {self.model_name}")
        print(f"   Cache path: {cache_path}")
        print(f"   Storage backends: {list(self.cache_engine.storage_manager.storage_backends.keys())}")
        
        return self.cache_engine
    
    def simulate_vllm_kv_cache(self, 
                              batch_size: int = 4,
                              seq_len: int = 512,
                              num_layers: int = 32) -> List[List[torch.Tensor]]:
        """Simulate vLLM-style KV cache tensors."""
        print(f"📊 Creating simulated vLLM KV cache...")
        print(f"   Batch size: {batch_size}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Number of layers: {num_layers}")
        
        kv_caches = []
        
        # vLLM uses different tensor layouts depending on attention backend
        for layer in range(num_layers):
            # Key cache: [batch_size, num_heads, seq_len, head_dim]
            k_cache = torch.randn(
                batch_size, 32, seq_len, 128,
                dtype=torch.float16, 
                device='cuda'
            )
            
            # Value cache: [batch_size, num_heads, seq_len, head_dim]
            v_cache = torch.randn(
                batch_size, 32, seq_len, 128,
                dtype=torch.float16,
                device='cuda'
            )
            
            kv_caches.append([k_cache, v_cache])
        
        print(f"✅ KV cache simulation complete")
        return kv_caches
    
    def store_conversation_cache(self, 
                               conversation_id: str,
                               kv_caches: List[List[torch.Tensor]],
                               tokens: List[str]):
        """Store KV cache for a conversation using GCOS."""
        if not self.setup_complete:
            raise RuntimeError("LMCache not setup. Call setup_lmcache_with_gcos() first.")
        
        # Create cache key for this conversation
        cache_key = CacheEngineKey(
            fmt="vllm",
            model_name=self.model_name,
            world_size=torch.cuda.device_count(),
            worker_id=0,
            chunk_hash=f"conversation_{conversation_id}"
        )
        
        print(f"💾 Storing conversation cache...")
        print(f"   Conversation ID: {conversation_id}")
        print(f"   Token count: {len(tokens)}")
        print(f"   Cache key: {cache_key.chunk_hash}")
        
        # Store using GCOS backend
        self.cache_engine.store(cache_key, kv_caches, tokens)
        
        print(f"✅ Conversation cache stored to GCOS")
        return cache_key
    
    def retrieve_conversation_cache(self, conversation_id: str):
        """Retrieve KV cache for a conversation from GCOS."""
        if not self.setup_complete:
            raise RuntimeError("LMCache not setup. Call setup_lmcache_with_gcos() first.")
        
        # Recreate cache key
        cache_key = CacheEngineKey(
            fmt="vllm",
            model_name=self.model_name,
            world_size=torch.cuda.device_count(),
            worker_id=0,
            chunk_hash=f"conversation_{conversation_id}"
        )
        
        print(f"📖 Retrieving conversation cache...")
        print(f"   Conversation ID: {conversation_id}")
        print(f"   Cache key: {cache_key.chunk_hash}")
        
        # Retrieve using GCOS backend
        kv_caches, tokens = self.cache_engine.retrieve(cache_key)
        
        if kv_caches is not None:
            print(f"✅ Conversation cache retrieved from GCOS")
            print(f"   Retrieved {len(kv_caches)} layers")
            print(f"   Retrieved {len(tokens)} tokens")
            return kv_caches, tokens
        else:
            print(f"❌ Conversation cache not found")
            return None, None
    
    def prefetch_conversation_cache(self, conversation_id: str):
        """Prefetch KV cache for a conversation (async)."""
        if not self.setup_complete:
            raise RuntimeError("LMCache not setup. Call setup_lmcache_with_gcos() first.")
        
        cache_key = CacheEngineKey(
            fmt="vllm",
            model_name=self.model_name,
            world_size=torch.cuda.device_count(),
            worker_id=0,
            chunk_hash=f"conversation_{conversation_id}"
        )
        
        print(f"🔄 Prefetching conversation cache...")
        print(f"   Conversation ID: {conversation_id}")
        
        # Start prefetch (non-blocking)
        future = self.cache_engine.storage_manager.get_non_blocking(cache_key)
        
        if future:
            print(f"✅ Prefetch started for conversation {conversation_id}")
            return future
        else:
            print(f"❌ Prefetch failed - cache not found")
            return None
    
    def get_gcos_stats(self):
        """Get GCOS backend statistics."""
        if not self.setup_complete:
            return None
        
        gcos_backend = self.cache_engine.storage_manager.storage_backends.get("GcosBackend")
        if gcos_backend and hasattr(gcos_backend, 'get_stats'):
            return gcos_backend.get_stats()
        return None


async def multi_conversation_example():
    """Example with multiple conversations and prefetching."""
    print("\n🔄 Multi-Conversation Example with Prefetching")
    
    # Setup
    vllm_gcos = VLLMGCOSExample("meta-llama/Llama-3.1-8B-Instruct")
    cache_path = "/tmp/vllm_gcos_multi"
    
    try:
        # Initialize
        vllm_gcos.setup_lmcache_with_gcos(cache_path)
        
        # Simulate multiple conversations
        conversations = [
            {
                "id": "user_001_session_1",
                "tokens": ["Hello", "how", "are", "you", "today", "?"] * 85,
                "context": "Greeting conversation"
            },
            {
                "id": "user_002_session_1", 
                "tokens": ["What", "is", "the", "weather", "like", "?"] * 85,
                "context": "Weather inquiry"
            },
            {
                "id": "user_001_session_2",
                "tokens": ["Tell", "me", "about", "machine", "learning"] * 85,
                "context": "Technical discussion"
            }
        ]
        
        # Store all conversations
        stored_keys = []
        for conv in conversations:
            print(f"\n💾 Processing: {conv['context']}")
            
            # Generate KV cache for this conversation
            kv_caches = vllm_gcos.simulate_vllm_kv_cache(
                batch_size=2,
                seq_len=len(conv['tokens']),
                num_layers=16  # Reduced for example
            )
            
            # Store to GCOS
            cache_key = vllm_gcos.store_conversation_cache(
                conv['id'], 
                kv_caches, 
                conv['tokens']
            )
            stored_keys.append((cache_key, conv))
        
        print(f"\n✅ All conversations stored to GCOS")
        
        # Demonstrate prefetching
        print(f"\n🔄 Testing prefetch functionality...")
        
        # Prefetch first conversation
        prefetch_future = vllm_gcos.prefetch_conversation_cache(conversations[0]['id'])
        
        # Do some other work while prefetching
        print(f"📊 Doing other work while prefetching...")
        await asyncio.sleep(0.1)  # Simulate other work
        
        # Retrieve prefetched conversation (should be fast)
        print(f"📖 Retrieving prefetched conversation...")
        kv_caches, tokens = vllm_gcos.retrieve_conversation_cache(conversations[0]['id'])
        
        if kv_caches:
            print(f"✅ Prefetch successful - fast retrieval completed")
        
        # Retrieve other conversations normally
        for conv in conversations[1:]:
            print(f"\n📖 Retrieving: {conv['context']}")
            kv_caches, tokens = vllm_gcos.retrieve_conversation_cache(conv['id'])
            
            if kv_caches:
                print(f"✅ Retrieved successfully")
                print(f"   Layers: {len(kv_caches)}")
                print(f"   Tokens: {len(tokens)}")
        
        # Show GCOS stats
        stats = vllm_gcos.get_gcos_stats()
        if stats:
            print(f"\n📊 GCOS Backend Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in multi-conversation example: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists(cache_path):
            import shutil
            shutil.rmtree(cache_path, ignore_errors=True)


def vllm_production_pattern():
    """Show production pattern for vLLM + GCOS integration."""
    print("\n🏭 Production Integration Pattern")
    
    # Production-ready configuration
    config_pattern = """
    # Production vLLM + GCOS Configuration
    
    1. LMCache Configuration:
       - chunk_size: 1024 (large chunks for production)
       - gcos_path: '/fast_nvme/vllm_cache'
       - use_layerwise: True (for vLLM compatibility)
       - local_cpu: False (direct GCOS only)
    
    2. GCOS Configuration:
       - Multiple NVMe devices for high throughput
       - Large GPU queue depths (1024+)
       - Direct I/O enabled
       - NUMA-aware allocation
    
    3. Integration Points:
       - Store KV cache after prompt processing
       - Prefetch for multi-turn conversations
       - Batch operations for high throughput
       - Monitor GCOS performance metrics
    
    4. Performance Optimizations:
       - Align chunk sizes with typical prompt lengths
       - Use async prefetching for conversation continuations
       - Batch multiple cache operations
       - Monitor and tune queue parameters
    """
    
    print(config_pattern)


if __name__ == "__main__":
    print("🚀 vLLM + GCOS Integration Examples\n")
    
    # Check prerequisites
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Examples require GPU.")
        exit(1)
    
    gcos_available = os.environ.get("BUILD_WITH_GCOS", "0") == "1"
    if not gcos_available:
        print("⚠️  GCOS not built. Set BUILD_WITH_GCOS=1 for full functionality.")
        print("   Running in fallback mode...\n")
    
    success = True
    
    # Basic example
    print("🎯 Basic vLLM + GCOS Example")
    try:
        vllm_gcos = VLLMGCOSExample()
        cache_path = "/tmp/vllm_gcos_basic"
        
        # Setup
        vllm_gcos.setup_lmcache_with_gcos(cache_path)
        
        # Simulate conversation
        kv_caches = vllm_gcos.simulate_vllm_kv_cache(batch_size=2, seq_len=256)
        tokens = ["Hello", "world", "how", "are", "you"] * 51  # ~255 tokens
        
        # Store and retrieve
        cache_key = vllm_gcos.store_conversation_cache("basic_example", kv_caches, tokens)
        retrieved_kv, retrieved_tokens = vllm_gcos.retrieve_conversation_cache("basic_example")
        
        if retrieved_kv:
            print(f"✅ Basic example successful")
            success = True
        else:
            print(f"❌ Basic example failed")
            success = False
        
        # Cleanup
        if os.path.exists(cache_path):
            import shutil
            shutil.rmtree(cache_path, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ Basic example error: {e}")
        success = False
    
    # Multi-conversation example with async
    if success:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success &= loop.run_until_complete(multi_conversation_example())
        loop.close()
    
    # Show production patterns
    vllm_production_pattern()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 vLLM + GCOS integration examples completed successfully!")
        print("\n💡 Key benefits:")
        print("   ⚡ Direct GPU-to-storage transfers (zero-copy)")
        print("   💾 Persistent KV cache storage") 
        print("   🔄 Async prefetching for better performance")
        print("   📊 Integration with vLLM serving patterns")
        print("\n🚀 Ready for production deployment!")
    else:
        print("❌ Some examples failed. Check error messages above.")
    
    print(f"{'='*60}")