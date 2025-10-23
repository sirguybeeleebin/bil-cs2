# tests/test_etl_pg.py
import asyncio
import json
import logging

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from etl.etl import load_cs2_data  # твоя функция ETL с asyncpg.Pool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------
# Фикстура для asyncio event loop
# ------------------------
@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ------------------------
# Фикстура PostgreSQL контейнера
# ------------------------
@pytest_asyncio.fixture(scope="module")
async def postgres_container():
    logger.info("🚀 Запуск PostgreSQL контейнера...")
    container = PostgresContainer("postgres:15")
    container.start()
    dsn = container.get_connection_url().replace("+psycopg2", "")

    # Ждём, пока PostgreSQL готов
    pool = None
    for _ in range(30):
        try:
            pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1;")
            break
        except Exception:
            await asyncio.sleep(1)
    else:
        raise RuntimeError("PostgreSQL container did not start in time")

    # Создаём таблицы
    async with pool.acquire() as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS maps (
            map_id INT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS teams (
            team_id INT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS players (
            player_id INT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        );
        """)

    yield pool

    logger.info("🛑 Закрываем пул и останавливаем контейнер...")
    await pool.close()
    container.stop()


# ------------------------
# Очистка таблиц после каждого теста
# ------------------------
@pytest_asyncio.fixture
async def clean_db(postgres_container: asyncpg.Pool):
    yield
    async with postgres_container.acquire() as conn:
        await conn.execute("TRUNCATE TABLE maps, teams, players RESTART IDENTITY;")


# ------------------------
# Тесты
# ------------------------
@pytest.mark.asyncio
async def test_load_cs2_data_inserts(postgres_container, clean_db, tmp_path):
    # Создаём тестовый JSON файл
    game_data = {
        "map": {"id": 1, "name": "Test Map"},
        "players": [
            {
                "player": {"id": 101, "name": "Player One"},
                "team": {"id": 201, "name": "Team A"},
            },
            {
                "player": {"id": 102, "name": "Player Two"},
                "team": {"id": 202, "name": "Team B"},
            },
        ],
    }
    test_file = tmp_path / "game1.json"
    test_file.write_text(json.dumps(game_data), encoding="utf-8")

    result = await load_cs2_data(str(tmp_path), postgres_container)
    assert result["total"] == 1
    assert result["success"] == 1
    assert result["error"] == 0

    # Проверяем данные в БД
    async with postgres_container.acquire() as conn:
        map_row = await conn.fetchrow("SELECT * FROM maps WHERE map_id=1")
        assert map_row["name"] == "Test Map"

        team_rows = await conn.fetch("SELECT * FROM teams")
        names = {t["name"] for t in team_rows}
        assert names == {"Team A", "Team B"}

        player_rows = await conn.fetch("SELECT * FROM players")
        player_names = {p["name"] for p in player_rows}
        assert player_names == {"Player One", "Player Two"}


@pytest.mark.asyncio
async def test_load_multiple_files(postgres_container, clean_db, tmp_path):
    # Создаём два файла игр
    games = [
        {
            "map": {"id": 1, "name": "Map One"},
            "players": [
                {"player": {"id": 101, "name": "P1"}, "team": {"id": 201, "name": "T1"}}
            ],
        },
        {
            "map": {"id": 2, "name": "Map Two"},
            "players": [
                {"player": {"id": 102, "name": "P2"}, "team": {"id": 202, "name": "T2"}}
            ],
        },
    ]
    for i, g in enumerate(games, start=1):
        f = tmp_path / f"game{i}.json"
        f.write_text(json.dumps(g), encoding="utf-8")

    result = await load_cs2_data(str(tmp_path), postgres_container)
    assert result["total"] == 2
    assert result["success"] == 2
    assert result["error"] == 0

    # Проверяем данные в БД
    async with postgres_container.acquire() as conn:
        map_rows = await conn.fetch("SELECT * FROM maps")
        assert len(map_rows) == 2
        team_rows = await conn.fetch("SELECT * FROM teams")
        assert len(team_rows) == 2
        player_rows = await conn.fetch("SELECT * FROM players")
        assert len(player_rows) == 2
