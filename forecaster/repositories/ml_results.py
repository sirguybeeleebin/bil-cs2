import logging

import asyncpg

log = logging.getLogger("cs2_forecaster")


class MLResultsRepository:
    def __init__(self, pg_pool: asyncpg.pool.Pool):
        self.pg_pool = pg_pool

    async def upsert(
        self, task_id: str, predictor_path: str, metrics_path: str
    ) -> bool:
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO ml_results(
                        ml_result_id, predictor_path, metrics_path, created_at, updated_at
                    )
                    VALUES($1, $2, $3, now(), now())
                    ON CONFLICT (ml_result_id) DO UPDATE
                    SET predictor_path = EXCLUDED.predictor_path,
                        metrics_path = EXCLUDED.metrics_path,
                        updated_at = now()
                    """,
                    task_id,
                    predictor_path,
                    metrics_path,
                )
            log.info(f"✅ ML result upserted: task_id={task_id}")
            return True
        except asyncpg.PostgresError as e:  # <-- catch only DB-related errors
            log.error(f"❌ Failed to upsert ML result: {e}", exc_info=True)
            return False
        except Exception as e:  # <-- catch unexpected errors separately
            log.exception(f"❌ Unexpected error during ML result upsert: {e}")
            return False
