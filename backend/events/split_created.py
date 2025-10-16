import json
import logging
from aio_pika import IncomingMessage
import aiosqlite
from backend.models import TrainTestSplit

log = logging.getLogger(__name__)

async def consume_split_created(queue, db: aiosqlite.Connection):
    """
    Консьюмер для split_created.
    DI: queue и db передаются извне.
    """
    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            await process_split_message(message, db)


async def process_split_message(message: IncomingMessage, db: aiosqlite.Connection):
    async with message.process():
        try:
            data = json.loads(message.body)
            hash_id = data["hash_id"]
            file_path = f"data/train_test_splits/{hash_id}.json"

            split = TrainTestSplit(hash_id=hash_id, path=file_path)

            await db.execute(
                "INSERT INTO train_test_splits (hash_id, path) VALUES (?, ?)",
                (split.hash_id, split.path)
            )
            await db.commit()
            log.info(f"Inserted new split {hash_id} into train_test_splits")
        except Exception as e:
            log.error(f"Failed to process split_created message: {e}")
