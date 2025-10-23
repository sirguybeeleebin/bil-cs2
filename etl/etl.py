# etl/etl.py
import json
import logging
import os
from collections.abc import Generator
from typing import List, Optional, Tuple

import asyncpg

log = logging.getLogger("cs2_etl")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


# ----------------------------
# Генератор JSON-файлов
# ----------------------------
def _generate_game_raw(
    path_to_games_raw_dir: str,
) -> Generator[Tuple[str, dict], None, None]:
    log.info(f"🔍 Сканирование директории с играми: {path_to_games_raw_dir}")
    for root, _, files in os.walk(path_to_games_raw_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                log.info(f"📂 Успешно загружен файл игры: {file_path}")
                yield file_path, data
            except Exception as e:
                log.warning(f"⚠️ Не удалось загрузить файл {file_path}: {e}")


# ----------------------------
# Entity extractors
# ----------------------------
def _extract_map(game: dict) -> Optional[dict]:
    try:
        return {"map_id": game["map"]["id"], "name": game["map"]["name"]}
    except Exception as e:
        log.warning(f"⚠️ Не удалось получить данные карты: {e}")
        return None


def _extract_teams(game: dict) -> List[dict]:
    teams_dict = {}
    for p in game.get("players", []):
        team = p.get("team")
        if team and team.get("id") and team.get("name"):
            teams_dict[team["id"]] = {"team_id": team["id"], "name": team["name"]}
    return list(teams_dict.values())


def _extract_players(game: dict) -> List[dict]:
    players_dict = {}
    for p in game.get("players", []):
        player = p.get("player")
        if player and player.get("id") and player.get("name"):
            players_dict[player["id"]] = {
                "player_id": player["id"],
                "name": player["name"],
            }
    return list(players_dict.values())


# ----------------------------
# Insert functions
# ----------------------------
async def _insert_map(conn: asyncpg.Connection, map_data: dict):
    if not map_data:
        return
    await conn.execute(
        """
        INSERT INTO maps(map_id, name)
        VALUES($1, $2)
        ON CONFLICT (map_id) DO UPDATE SET name = EXCLUDED.name, updated_at = now()
        """,
        map_data["map_id"],
        map_data["name"],
    )


async def _insert_teams(conn: asyncpg.Connection, teams: List[dict]):
    if not teams:
        return
    for t in teams:
        await conn.execute(
            """
            INSERT INTO teams(team_id, name)
            VALUES($1, $2)
            ON CONFLICT (team_id) DO UPDATE SET name = EXCLUDED.name, updated_at = now()
            """,
            t["team_id"],
            t["name"],
        )


async def _insert_players(conn: asyncpg.Connection, players: List[dict]):
    if not players:
        return
    for p in players:
        await conn.execute(
            """
            INSERT INTO players(player_id, name)
            VALUES($1, $2)
            ON CONFLICT (player_id) DO UPDATE SET name = EXCLUDED.name, updated_at = now()
            """,
            p["player_id"],
            p["name"],
        )


# ----------------------------
# Main ETL function
# ----------------------------
async def load_cs2_data(
    path_to_games_raw_dir: str, pool: asyncpg.Pool
) -> dict[str, int]:
    log.info("🚀 Запуск процесса загрузки данных CS2 в PostgreSQL (asyncpg)")

    total, success, error = 0, 0, 0

    async with pool.acquire() as conn:
        for file_path, game in _generate_game_raw(path_to_games_raw_dir):
            total += 1
            log.info(f"⚙️ Обработка файла игры #{total}: {file_path}")
            try:
                map_data = _extract_map(game)
                teams = _extract_teams(game)
                players = _extract_players(game)

                await _insert_map(conn, map_data)
                await _insert_teams(conn, teams)
                await _insert_players(conn, players)

                success += 1
                log.info(f"✅ Игра #{total} успешно загружена в PostgreSQL")
            except Exception as e:
                error += 1
                log.error(
                    f"❌ Ошибка при обработке файла {file_path}: {e}", exc_info=True
                )

    log.info("📊 Загрузка завершена")
    log.info(f"Всего файлов: {total}, Успешно: {success}, Ошибки: {error}")

    return {"total": total, "success": success, "error": error}
