import json
import logging
import os
from collections.abc import Generator

from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import execute_values

# -------------------------------------------------
# Logging setup
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("cs2_etl")


# -------------------------------------------------
# Game JSON generator
# -------------------------------------------------
def _generate_game_raw(path_to_games_raw_dir: str) -> Generator[dict, None, None]:
    """Yield raw CS2 game JSON files one by one."""
    log.info(f"🔍 Сканирование директории с играми: {path_to_games_raw_dir}")
    for root, _, files in os.walk(path_to_games_raw_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                log.info(f"📂 Успешно загружен файл игры: {file_path}")
                yield data
            except Exception as e:
                log.warning(f"⚠️ Не удалось загрузить файл {file_path}: {e}")


# -------------------------------------------------
# Extract all entities in a single pass
# -------------------------------------------------
def _extract_entities(game: dict) -> tuple[dict | None, list[dict], list[dict]]:
    """
    Extract map, unique teams and unique players in one pass.
    Returns (map_data, teams, players)
    """
    # Map
    map_data = None
    try:
        map_data = {"map_id": game["map"]["id"], "name": game["map"]["name"]}
    except Exception as e:
        log.warning(f"⚠️ Не удалось получить данные карты: {e}")

    # Teams and players deduplicated
    teams_dict = {}
    players_dict = {}

    for p in game.get("players", []):
        # Teams
        team = p.get("team")
        if team and team.get("id") and team.get("name"):
            teams_dict[team["id"]] = {"team_id": team["id"], "name": team["name"]}

        # Players
        player = p.get("player")
        if player and player.get("id") and player.get("name"):
            players_dict[player["id"]] = {
                "player_id": player["id"],
                "name": player["name"],
            }

    return map_data, list(teams_dict.values()), list(players_dict.values())


# -------------------------------------------------
# Load data into PostgreSQL
# -------------------------------------------------
def load_cs2_data_to_postgres(
    path_to_games_raw_dir: str, conn: PGConnection
) -> dict[str, int]:
    """Main ETL loader — creates tables and loads parsed data into PostgreSQL."""
    log.info("🚀 Запуск процесса загрузки данных CS2 в PostgreSQL")

    total, success, error = 0, 0, 0
    cursor = conn.cursor()

    # 1️⃣ Create tables if not exist
    log.info("🧱 Проверка и создание таблиц (если отсутствуют)...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS maps (
            map_id INT PRIMARY KEY,
            name TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id INT PRIMARY KEY,
            name TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id INT PRIMARY KEY,
            name TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    log.info("✅ Таблицы готовы для загрузки данных")

    # 2️⃣ Process files
    for game in _generate_game_raw(path_to_games_raw_dir):
        total += 1
        log.info(f"⚙️ Обработка игры #{total}")
        try:
            map_data, teams, players = _extract_entities(game)

            # --- Insert map
            if map_data:
                cursor.execute(
                    """
                    INSERT INTO maps (map_id, name)
                    VALUES (%s, %s)
                    ON CONFLICT (map_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        updated_at = NOW();
                """,
                    (map_data["map_id"], map_data["name"]),
                )

            # --- Insert teams
            if teams:
                execute_values(
                    cursor,
                    """
                    INSERT INTO teams (team_id, name)
                    VALUES %s
                    ON CONFLICT (team_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        updated_at = NOW();
                """,
                    [(t["team_id"], t["name"]) for t in teams],
                )

            # --- Insert players
            if players:
                execute_values(
                    cursor,
                    """
                    INSERT INTO players (player_id, name)
                    VALUES %s
                    ON CONFLICT (player_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        updated_at = NOW();
                """,
                    [(p["player_id"], p["name"]) for p in players],
                )

            conn.commit()
            success += 1
            log.info(f"✅ Игра #{total} успешно загружена")
        except Exception as e:
            conn.rollback()
            error += 1
            log.error(f"❌ Ошибка при обработке игры #{total}: {e}", exc_info=True)

    # 3️⃣ Summary
    cursor.close()
    log.info("📊 Загрузка завершена")
    log.info(f"Всего файлов: {total}, Успешно: {success}, Ошибки: {error}")

    return {"total": total, "success": success, "error": error}
