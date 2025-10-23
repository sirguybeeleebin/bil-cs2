import asyncio
import logging

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from dictionaries.repositories.player import (  # adjust import path
    PlayerRepository,
    make_player_repository,
)

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
    dsn = container.get_connection_url().replace("+psycopg2", "")
    logger.info(f"PostgreSQL container DSN: {dsn}")

    # Wait until Postgres is ready
    pool = None
    for i in range(30):
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

    # Ensure players table exists
    logger.info("Creating 'players' table if it does not exist...")
    async with pool.acquire() as conn:
        await conn.execute("""
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS players (
            player_uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            player_id INT UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
        );
        """)
    logger.info("'players' table ready.")

    yield pool

    logger.info("Closing PostgreSQL pool...")
    await pool.close()
    logger.info("Stopping PostgreSQL container...")
    container.stop()


# ------------------------
# PlayerRepository fixture
# ------------------------
@pytest_asyncio.fixture
async def player_repo(postgres_container: asyncpg.Pool):
    logger.info("Initializing PlayerRepository...")
    repo = make_player_repository(pool=postgres_container)
    yield repo
    logger.info("Cleaning up 'players' table after test...")
    async with postgres_container.acquire() as conn:
        await conn.execute("TRUNCATE TABLE players;")
    logger.info("'players' table cleaned.")


# ------------------------
# Tests
# ------------------------
@pytest.mark.asyncio
async def test_upsert_and_get_by_name(player_repo: PlayerRepository):
    logger.info("Running test_upsert_and_get_by_name for PlayerRepository...")
    data = [
        {"player_id": 1, "name": "Player One"},
        {"player_id": 2, "name": "Player Two"},
    ]

    logger.info("Upserting players...")
    result = await player_repo.upsert(data)
    assert result is True
    logger.info("Upsert successful.")

    logger.info("Testing get_by_name for existing player...")
    player = await player_repo.get_by_name("Player One")
    assert player is not None
    assert player["player_id"] == 1
    assert player["name"] == "Player One"
    assert "created_at" in player
    assert "updated_at" in player
    logger.info("get_by_name returned expected result.")

    logger.info("Testing get_by_name for non-existing player...")
    missing = await player_repo.get_by_name("Nonexistent Player")
    assert missing is None
    logger.info("get_by_name correctly returned None for missing player.")


@pytest.mark.asyncio
async def test_search_by_name(player_repo: PlayerRepository):
    logger.info("Running test_search_by_name for PlayerRepository...")
    data = [
        {"player_id": 3, "name": "Alice"},
        {"player_id": 4, "name": "Bob"},
        {"player_id": 5, "name": "Charlie"},
    ]
    logger.info("Upserting players for search test...")
    await player_repo.upsert(data)
    logger.info("Upsert for search test completed.")

    logger.info("Searching players containing 'a'...")
    results = await player_repo.search_by_name("a", limit=10)
    assert len(results) >= 2  # Alice and Charlie contain 'a'
    logger.info(f"search_by_name('a') returned {len(results)} results.")

    logger.info("Searching players containing 'Alice'...")
    results_alice = await player_repo.search_by_name("Alice", limit=10)
    assert len(results_alice) == 1
    assert results_alice[0]["name"] == "Alice"
    logger.info("search_by_name('Alice') returned correct result.")

    logger.info("Searching players containing 'Zoe' (should be empty)...")
    results_none = await player_repo.search_by_name("Zoe", limit=10)
    assert results_none == []
    logger.info("search_by_name('Zoe') returned empty list as expected.")
