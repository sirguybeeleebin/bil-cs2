import asyncio
import logging

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from dictionaries.repositories.map import MapRepository  # adjust import path

# ------------------------
# Setup logging
# ------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------
# Fix for pytest-asyncio scope mismatch
# ------------------------
@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for the module."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ------------------------
# PostgreSQL container fixture
# ------------------------
@pytest_asyncio.fixture(scope="module")
async def postgres_container():
    logger.info("Starting PostgreSQL container...")
    container = PostgresContainer("postgres:15")
    container.start()
    # Remove +psycopg2 suffix for asyncpg
    dsn = container.get_connection_url().replace("+psycopg2", "")
    logger.info(f"PostgreSQL container DSN: {dsn}")

    # Wait until Postgres is ready
    pool = None
    for i in range(30):  # retry up to 30 seconds
        try:
            pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1;")
            logger.info("PostgreSQL is ready!")
            break
        except Exception as e:
            logger.debug(f"Attempt {i + 1}: PostgreSQL not ready yet ({e})")
            await asyncio.sleep(1)
    else:
        raise RuntimeError("PostgreSQL container did not start in time")

    # Ensure tables exist
    logger.info("Creating 'maps' table if it does not exist...")
    async with pool.acquire() as conn:
        await conn.execute("""
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS maps (
            map_uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            map_id INT UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
        );
        """)
    logger.info("'maps' table ready.")

    yield pool

    logger.info("Closing PostgreSQL pool...")
    await pool.close()
    logger.info("Stopping PostgreSQL container...")
    container.stop()


# ------------------------
# MapRepository fixture
# ------------------------
@pytest_asyncio.fixture
async def map_repo(postgres_container: asyncpg.Pool):
    logger.info("Initializing MapRepository...")
    repo = MapRepository(pool=postgres_container)
    yield repo
    logger.info("Cleaning up 'maps' table after test...")
    async with postgres_container.acquire() as conn:
        await conn.execute("TRUNCATE TABLE maps;")
    logger.info("'maps' table cleaned.")


# ------------------------
# Tests
# ------------------------
@pytest.mark.asyncio
async def test_upsert_and_get_by_name(map_repo: MapRepository):
    logger.info("Running test_upsert_and_get_by_name...")
    data = [
        {"map_id": 1, "name": "Test Map 1"},
        {"map_id": 2, "name": "Test Map 2"},
    ]

    logger.info("Upserting maps...")
    result = await map_repo.upsert(data)
    assert result is True
    logger.info("Upsert successful.")

    logger.info("Testing get_by_name for existing map...")
    map1 = await map_repo.get_by_name("Test Map 1")
    assert map1 is not None
    assert map1["map_id"] == 1
    assert map1["name"] == "Test Map 1"
    assert "created_at" in map1
    assert "updated_at" in map1
    logger.info("get_by_name returned expected result.")

    logger.info("Testing get_by_name for non-existing map...")
    missing = await map_repo.get_by_name("Nonexistent Map")
    assert missing is None
    logger.info("get_by_name correctly returned None for missing map.")


@pytest.mark.asyncio
async def test_search_by_name(map_repo: MapRepository):
    logger.info("Running test_search_by_name...")
    data = [
        {"map_id": 3, "name": "Alpha Map"},
        {"map_id": 4, "name": "Beta Map"},
        {"map_id": 5, "name": "Gamma Map"},
    ]
    logger.info("Upserting maps for search test...")
    await map_repo.upsert(data)
    logger.info("Upsert for search test completed.")

    logger.info("Searching maps containing 'Map'...")
    results = await map_repo.search_by_name("Map", limit=10)
    assert len(results) == 3
    logger.info(f"search_by_name('Map') returned {len(results)} results.")

    logger.info("Searching maps containing 'Alpha'...")
    results_alpha = await map_repo.search_by_name("Alpha", limit=10)
    assert len(results_alpha) == 1
    assert results_alpha[0]["name"] == "Alpha Map"
    logger.info("search_by_name('Alpha') returned correct result.")

    logger.info("Searching maps containing 'Delta' (should be empty)...")
    results_none = await map_repo.search_by_name("Delta", limit=10)
    assert results_none == []
    logger.info("search_by_name('Delta') returned empty list as expected.")
