# service_dicts_consumer.py
import os
import json
import asyncio
import redis.asyncio as aioredis
from fastapi import FastAPI

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_CHANNEL = os.getenv("DICTS_CHANNEL", "dicts_updates")  # отдельный канал для словарей

app = FastAPI(title="Dictionaries Consumer Service")

@app.on_event("startup")
async def startup_event():
    app.state.redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}")
    app.state.task = asyncio.create_task(redis_listener())

@app.on_event("shutdown")
async def shutdown_event():
    app.state.task.cancel()
    await app.state.redis.close()

async def redis_listener():
    pubsub = app.state.redis.pubsub()
    await pubsub.subscribe(REDIS_CHANNEL)
    async for message in pubsub.listen():
        if message["type"] == "message":
            data = json.loads(message["data"])
            await handle_dicts_update_message(data)

async def handle_dicts_update_message(message: dict):
    # Заглушка обработки обновления словарей
    print(f"[Dicts Consumer] Received dictionary update: {message}")
