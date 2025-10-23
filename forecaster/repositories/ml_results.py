import asyncpg
import logging
from pathlib import Path

log = logging.getLogger("cs2_forecaster")


class MLResultsRepository:
    def __init__(self, pg_pool: asyncpg.pool.Pool):
        self.pg_pool = pg_pool

    async def upsert(self, task_id: str, predictor_path: str, metrics_path: str) -> bool:
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
        except Exception as e:
            log.error(f"❌ Failed to upsert ML result: {e}", exc_info=True)
            return False

    async def get_latest_predictor_path(self) -> Path | None:       
        try:
            async with self.pg_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT predictor_path FROM ml_results ORDER BY updated_at DESC LIMIT 1"
                )
                if not row:
                    log.warning("⚠️ No ML predictor found in database")
                    return None
                path = Path(row["predictor_path"])
                if not path.exists():
                    log.error(f"❌ Predictor file not found: {path}")
                    return None
                log.info(f"✅ Latest predictor path found: {path}")
                return path
        except Exception as e:
            log.error(f"❌ Failed to fetch latest predictor path: {e}", exc_info=True)
            return None
