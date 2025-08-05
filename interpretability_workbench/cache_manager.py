"""
Cache manager for performance optimization
"""
import time
import threading
from typing import Dict, Any, Optional, Callable
from collections import OrderedDict
import pickle
from pathlib import Path
import logging

cache_logger = logging.getLogger("cache")


class LRUCache:
    """Least Recently Used cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                # Add new
                self.cache[key] = value
                # Remove oldest if cache is full
                if len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)


class CacheManager:
    """Manages caching for various components"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory caches
        self.feature_cache = LRUCache(max_size=500)
        self.provenance_cache = LRUCache(max_size=200)
        self.inference_cache = LRUCache(max_size=100)
        self.analysis_cache = LRUCache(max_size=300)
        
        # Cache statistics
        self.stats = {
            "feature_cache_hits": 0,
            "feature_cache_misses": 0,
            "provenance_cache_hits": 0,
            "provenance_cache_misses": 0,
            "inference_cache_hits": 0,
            "inference_cache_misses": 0,
            "analysis_cache_hits": 0,
            "analysis_cache_misses": 0
        }
        self.stats_lock = threading.Lock()
    
    def _get_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            key_parts.append(str(arg))
        
        # Add keyword arguments (sorted for consistency)
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}={value}")
        
        return "_".join(key_parts)
    
    def get_feature(self, feature_id: str) -> Optional[Dict[str, Any]]:
        """Get feature from cache"""
        cache_key = self._get_cache_key("feature", feature_id)
        result = self.feature_cache.get(cache_key)
        
        with self.stats_lock:
            if result is not None:
                self.stats["feature_cache_hits"] += 1
            else:
                self.stats["feature_cache_misses"] += 1
        
        return result
    
    def put_feature(self, feature_id: str, feature_data: Dict[str, Any]) -> None:
        """Put feature in cache"""
        cache_key = self._get_cache_key("feature", feature_id)
        self.feature_cache.put(cache_key, feature_data)
    
    def get_provenance(self, feature_id: str, upstream_layers: int = 2, downstream_layers: int = 1) -> Optional[Dict[str, Any]]:
        """Get provenance analysis from cache"""
        cache_key = self._get_cache_key("provenance", feature_id, upstream_layers, downstream_layers)
        result = self.provenance_cache.get(cache_key)
        
        with self.stats_lock:
            if result is not None:
                self.stats["provenance_cache_hits"] += 1
            else:
                self.stats["provenance_cache_misses"] += 1
        
        return result
    
    def put_provenance(self, feature_id: str, provenance_data: Dict[str, Any], upstream_layers: int = 2, downstream_layers: int = 1) -> None:
        """Put provenance analysis in cache"""
        cache_key = self._get_cache_key("provenance", feature_id, upstream_layers, downstream_layers)
        self.provenance_cache.put(cache_key, provenance_data)
    
    def get_inference(self, text: str, max_length: int = 50) -> Optional[Dict[str, Any]]:
        """Get inference result from cache"""
        cache_key = self._get_cache_key("inference", text[:100], max_length)  # Truncate text for key
        result = self.inference_cache.get(cache_key)
        
        with self.stats_lock:
            if result is not None:
                self.stats["inference_cache_hits"] += 1
            else:
                self.stats["inference_cache_misses"] += 1
        
        return result
    
    def put_inference(self, text: str, inference_result: Dict[str, Any], max_length: int = 50) -> None:
        """Put inference result in cache"""
        cache_key = self._get_cache_key("inference", text[:100], max_length)
        self.inference_cache.put(cache_key, inference_result)
    
    def get_analysis(self, analysis_type: str, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Get analysis result from cache"""
        cache_key = self._get_cache_key(analysis_type, *args, **kwargs)
        result = self.analysis_cache.get(cache_key)
        
        with self.stats_lock:
            if result is not None:
                self.stats["analysis_cache_hits"] += 1
            else:
                self.stats["analysis_cache_misses"] += 1
        
        return result
    
    def put_analysis(self, analysis_type: str, analysis_result: Dict[str, Any], *args, **kwargs) -> None:
        """Put analysis result in cache"""
        cache_key = self._get_cache_key(analysis_type, *args, **kwargs)
        self.analysis_cache.put(cache_key, analysis_result)
    
    def clear_all(self) -> None:
        """Clear all caches"""
        self.feature_cache.clear()
        self.provenance_cache.clear()
        self.inference_cache.clear()
        self.analysis_cache.clear()
        cache_logger.info("All caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Add cache sizes
        stats.update({
            "feature_cache_size": self.feature_cache.size(),
            "provenance_cache_size": self.provenance_cache.size(),
            "inference_cache_size": self.inference_cache.size(),
            "analysis_cache_size": self.analysis_cache.size()
        })
        
        # Calculate hit rates
        total_feature = stats["feature_cache_hits"] + stats["feature_cache_misses"]
        total_provenance = stats["provenance_cache_hits"] + stats["provenance_cache_misses"]
        total_inference = stats["inference_cache_hits"] + stats["inference_cache_misses"]
        total_analysis = stats["analysis_cache_hits"] + stats["analysis_cache_misses"]
        
        stats.update({
            "feature_cache_hit_rate": stats["feature_cache_hits"] / total_feature if total_feature > 0 else 0,
            "provenance_cache_hit_rate": stats["provenance_cache_hits"] / total_provenance if total_provenance > 0 else 0,
            "inference_cache_hit_rate": stats["inference_cache_hits"] / total_inference if total_inference > 0 else 0,
            "analysis_cache_hit_rate": stats["analysis_cache_hits"] / total_analysis if total_analysis > 0 else 0
        })
        
        return stats
    
    def save_to_disk(self, cache_name: str) -> None:
        """Save cache to disk"""
        cache_file = self.cache_dir / f"{cache_name}.pkl"
        
        if cache_name == "feature":
            cache_data = dict(self.feature_cache.cache)
        elif cache_name == "provenance":
            cache_data = dict(self.provenance_cache.cache)
        elif cache_name == "inference":
            cache_data = dict(self.inference_cache.cache)
        elif cache_name == "analysis":
            cache_data = dict(self.analysis_cache.cache)
        else:
            raise ValueError(f"Unknown cache name: {cache_name}")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        cache_logger.info(f"Saved {cache_name} cache to {cache_file}")
    
    def load_from_disk(self, cache_name: str) -> None:
        """Load cache from disk"""
        cache_file = self.cache_dir / f"{cache_name}.pkl"
        
        if not cache_file.exists():
            cache_logger.warning(f"Cache file not found: {cache_file}")
            return
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            if cache_name == "feature":
                self.feature_cache.cache.update(cache_data)
            elif cache_name == "provenance":
                self.provenance_cache.cache.update(cache_data)
            elif cache_name == "inference":
                self.inference_cache.cache.update(cache_data)
            elif cache_name == "analysis":
                self.analysis_cache.cache.update(cache_data)
            else:
                raise ValueError(f"Unknown cache name: {cache_name}")
            
            cache_logger.info(f"Loaded {cache_name} cache from {cache_file}")
        except Exception as e:
            cache_logger.error(f"Failed to load {cache_name} cache: {e}")


# Global cache manager instance
cache_manager = CacheManager()


def cached_feature(func: Callable) -> Callable:
    """Decorator to cache feature-related function results"""
    def wrapper(*args, **kwargs):
        # Generate cache key
        cache_key = cache_manager._get_cache_key("feature_func", func.__name__, *args, **kwargs)
        
        # Try to get from cache
        result = cache_manager.feature_cache.get(cache_key)
        if result is not None:
            return result
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Cache result
        cache_manager.feature_cache.put(cache_key, result)
        
        return result
    
    return wrapper


def cached_provenance(func: Callable) -> Callable:
    """Decorator to cache provenance-related function results"""
    def wrapper(*args, **kwargs):
        # Generate cache key
        cache_key = cache_manager._get_cache_key("provenance_func", func.__name__, *args, **kwargs)
        
        # Try to get from cache
        result = cache_manager.provenance_cache.get(cache_key)
        if result is not None:
            return result
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Cache result
        cache_manager.provenance_cache.put(cache_key, result)
        
        return result
    
    return wrapper 