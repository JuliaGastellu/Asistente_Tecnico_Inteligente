import hashlib
import time
from typing import Any, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class QueryCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: dict[str, dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _get_key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        key = self._get_key(query)
        if key in self.cache:
            item = self.cache[key]
            if time.time() - item["timestamp"] < self.ttl_seconds:
                self.hits += 1
                logger.info(f"Cache HIT for query: {query[:50]}...")
                return item["result"]
            else:
                del self.cache[key]
        
        self.misses += 1
        return None

    def set(self, query: str, result: Any):
        key = self._get_key(query)
        self.cache[key] = {
            "result": result,
            "timestamp": time.time()
        }

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, int]:
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses
        }

_cache = QueryCache()

def get_cache() -> QueryCache:
    return _cache
