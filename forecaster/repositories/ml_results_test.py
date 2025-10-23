import asyncio
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from forecaster.repositories.ml_results import MLResultsRepository  # исправь импорт


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def pg_pool():
    with PostgresContainer("postgres:16-alpine") as postgres:
        dsn = postgres.get_connection_url().replace("+psycopg2", "")
        pool = await asyncpg.create_pool(dsn=dsn)

        # создаем таблицу ml_results
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_results (
                    ml_result_id TEXT PRIMARY KEY,
                    predictor_path TEXT NOT NULL,
                    metrics_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT now(),
                    updated_at TIMESTAMP DEFAULT now()
                )
            """)

        yield pool
        await pool.close()


@pytest.mark.asyncio
async def test_upsert(pg_pool, tmp_path: Path):
    repo = MLResultsRepository(pg_pool)

    predictor_file = tmp_path / "predictor.joblib"
    metrics_file = tmp_path / "metrics.json"
    predictor_file.write_text("fake_model")
    metrics_file.write_text("{'accuracy':0.99}")

    # тест вставки
    result = await repo.upsert("task_1", str(predictor_file), str(metrics_file))
    assert result is True

    async with pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM ml_results WHERE ml_result_id='task_1'"
        )
        assert row["predictor_path"] == str(predictor_file)

    # тест обновления
    predictor_file_v2 = tmp_path / "predictor_v2.joblib"
    predictor_file_v2.write_text("updated_model")
    result2 = await repo.upsert("task_1", str(predictor_file_v2), str(metrics_file))
    assert result2 is True

    async with pg_pool.acquire() as conn:
        row2 = await conn.fetchrow(
            "SELECT * FROM ml_results WHERE ml_result_id='task_1'"
        )
        assert row2["predictor_path"] == str(predictor_file_v2)
