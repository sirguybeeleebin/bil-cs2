import asyncio
import logging

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from dictionaries.repositories.team import (  # adjust import path
    TeamRepository,
    make_team_repository,
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

    # Ensure teams table exists
    logger.info("Creating 'teams' table if it does not exist...")
    async with pool.acquire() as conn:
        await conn.execute("""
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS teams (
            team_uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            team_id INT UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
        );
        """)
    logger.info("'teams' table ready.")

    yield pool

    logger.info("Closing PostgreSQL pool...")
    await pool.close()
    logger.info("Stopping PostgreSQL container...")
    container.stop()


# ------------------------
# TeamRepository fixture
# ------------------------
@pytest_asyncio.fixture
async def team_repo(postgres_container: asyncpg.Pool):
    logger.info("Initializing TeamRepository...")
    repo = make_team_repository(pool=postgres_container)
    yield repo
    logger.info("Cleaning up 'teams' table after test...")
    async with postgres_container.acquire() as conn:
        await conn.execute("TRUNCATE TABLE teams;")
    logger.info("'teams' table cleaned.")


# ------------------------
# Tests
# ------------------------
@pytest.mark.asyncio
async def test_upsert_and_get_by_name(team_repo: TeamRepository):
    logger.info("Running test_upsert_and_get_by_name for TeamRepository...")
    data = [
        {"team_id": 1, "name": "Team Alpha"},
        {"team_id": 2, "name": "Team Beta"},
    ]

    logger.info("Upserting teams...")
    result = await team_repo.upsert(data)
    assert result is True
    logger.info("Upsert successful.")

    logger.info("Testing get_by_name for existing team...")
    team = await team_repo.get_by_name("Team Alpha")
    assert team is not None
    assert team["team_id"] == 1
    assert team["name"] == "Team Alpha"
    assert "created_at" in team
    assert "updated_at" in team
    logger.info("get_by_name returned expected result.")

    logger.info("Testing get_by_name for non-existing team...")
    missing = await team_repo.get_by_name("Nonexistent Team")
    assert missing is None
    logger.info("get_by_name correctly returned None for missing team.")


@pytest.mark.asyncio
async def test_search_by_name(team_repo: TeamRepository):
    logger.info("Running test_search_by_name for TeamRepository...")
    data = [
        {"team_id": 3, "name": "Red Dragons"},
        {"team_id": 4, "name": "Blue Sharks"},
        {"team_id": 5, "name": "Green Wolves"},
    ]
    logger.info("Upserting teams for search test...")
    await team_repo.upsert(data)
    logger.info("Upsert for search test completed.")

    logger.info("Searching teams containing 'a'...")
    results = await team_repo.search_by_name("a", limit=10)
    assert len(results) >= 2  # Red Dragons and Blue Sharks contain 'a'
    logger.info(f"search_by_name('a') returned {len(results)} results.")

    logger.info("Searching teams containing 'Red'...")
    results_red = await team_repo.search_by_name("Red", limit=10)
    assert len(results_red) == 1
    assert results_red[0]["name"] == "Red Dragons"
    logger.info("search_by_name('Red') returned correct result.")

    logger.info("Searching teams containing 'Yellow' (should be empty)...")
    results_none = await team_repo.search_by_name("Yellow", limit=10)
    assert results_none == []
    logger.info("search_by_name('Yellow') returned empty list as expected.")
