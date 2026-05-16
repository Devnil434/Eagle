from redis import asyncio as aioredis
import os

_redis_client = None

async def get_redis_client() -> aioredis.Redis:
    """Get or create Redis client from REDIS_URL env var."""
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _redis_client = await aioredis.from_url(redis_url, decode_responses=True)
    return _redis_client

async def close_redis():
    """Close Redis connection on app shutdown."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None