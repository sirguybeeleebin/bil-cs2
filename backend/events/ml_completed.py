import json
import logging
from aio_pika import IncomingMessage
import aiosqlite
from backend.models import MLResult  # Pydantic модель для валидации

log = logging.getLogger(__name__)

async def consume_ml_completed(queue, db: aiosqlite.Connection):
    """
    Консьюмер для ml_completed.
    DI: queue и db передаются извне.
    """
    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            await process_ml_message(message, db)


async def process_ml_message(message: IncomingMessage, db: aiosqlite.Connection):
    async with message.process():
        try:
            data = json.loads(message.body)
            hash_id = data["hash_id"]
            metrics = json.dumps(data["metrics"])
            pickle_path = f"data/ml_results/{hash_id}.pickle"

            ml_result = MLResult(hash_id=hash_id, metrics=metrics, path=pickle_path)

            await db.execute(
                "INSERT INTO ml (hash_id, metrics, path) VALUES (?, ?, ?)",
                (ml_result.hash_id, ml_result.metrics, ml_result.path)
            )
            await db.commit()
            log.info(f"Inserted ML results for split {hash_id} into ml")
        except Exception as e:
            log.error(f"Failed to process ml_completed message: {e}")
