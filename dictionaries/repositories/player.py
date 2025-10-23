import datetime
import logging
from typing import Dict, List, Optional

import asyncpg

log: logging.Logger = logging.getLogger(__name__)


class PlayerRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool: asyncpg.Pool = pool

    async def get_by_name(self, name: str) -> Optional[Dict]:
        query = """
        SELECT player_uuid, player_id, name, created_at, updated_at
        FROM players
        WHERE name = $1
        LIMIT 1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, name)
            if row:
                log.info(f"Найден игрок с именем: {name}")
                return dict(row)
            log.info(f"Игрок с именем {name} не найден")
            return None

    async def search_by_name(self, name_part: str, limit: int = 10) -> List[Dict]:
        query = """
        SELECT player_uuid, player_id, name, created_at, updated_at
        FROM players
        WHERE name ILIKE '%' || $1 || '%'
        ORDER BY updated_at DESC
        LIMIT $2
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, name_part, limit)
            results = [dict(row) for row in rows]
            log.info(f"Найдено {len(results)} игроков по поиску: {name_part}")
            return results

    async def upsert(self, data: List[Dict]) -> bool:
        if not data:
            log.warning("Попытка сохранить пустой список игроков")
            return False

        now = datetime.datetime.now(datetime.timezone.utc)
        async with self.pool.acquire() as conn:
            for d in data:
                query = """
                INSERT INTO players (player_id, name, created_at, updated_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (player_id) DO UPDATE
                SET name = EXCLUDED.name,
                    updated_at = EXCLUDED.updated_at
                """
                await conn.execute(
                    query, d["player_id"], d["name"], d.get("created_at", now), now
                )

        log.info(f"Успешно сохранено/обновлено {len(data)} игроков")
        return True


def make_player_repository(pool: asyncpg.Pool) -> PlayerRepository:
    return PlayerRepository(pool)
