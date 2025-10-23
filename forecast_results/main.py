# import asyncio
# import json
# import logging
# import os
# from typing import Any, Dict

# import redis.asyncio as redis
# from motor.motor_asyncio import AsyncIOMotorClient

# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
# )
# log = logging.getLogger("forecast_consumer")


# REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# REDIS_CHANNEL = "forecast_events"

# MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
# MONGO_DB = os.getenv("MONGO_DB", "forecasts")
# MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "results")


# async def consume_forecasts():
#     redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
#     pubsub = redis_client.pubsub()
#     await pubsub.subscribe(REDIS_CHANNEL)
#     log.info(f"Subscribed to Redis channel: {REDIS_CHANNEL}")

#     mongo_client = AsyncIOMotorClient(MONGO_URI)
#     db = mongo_client[MONGO_DB]
#     collection = db[MONGO_COLLECTION]

#     async for message in pubsub.listen():
#         if message is None or message["type"] != "message":
#             continue
#         try:
#             event: Dict[str, Any] = json.loads(message["data"])
#             # Insert into MongoDB
#             await collection.insert_one(event)
#             log.info(f"Inserted forecast event into MongoDB: {event}")
#         except Exception as e:
#             log.error(f"❌ Failed to process message: {e}", exc_info=True)


# if __name__ == "__main__":
#     try:
#         asyncio.run(consume_forecasts())
#     except KeyboardInterrupt:
#         log.info("Consumer stopped manually")
