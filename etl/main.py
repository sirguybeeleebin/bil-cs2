import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Generator, List, Optional

import pika
from clickhouse_driver import Client
from dateutil.parser import parse
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


class Settings(BaseSettings):
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_PORT: int = 9000
    CLICKHOUSE_USER: str = "cs2_user"
    CLICKHOUSE_PASSWORD: str = "cs2_password"
    CLICKHOUSE_DB: str = "cs2_db"
    CLICKHOUSE_DROP_TABLE: bool = True    

    PATH_TO_GAMES_RAW_DIR: str = "data/games_raw"    

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")


def generate_game_raw(
    path_to_games_raw_dir: str,
) -> Generator[dict[str, Any], None, None]:
    for filename in os.listdir(path_to_games_raw_dir):
        try:
            with open(
                os.path.join(path_to_games_raw_dir, filename), "r", encoding="utf-8"
            ) as f:
                yield json.load(f)
        except Exception as e:
            log.warning(f"Ошибка чтения файла {filename}: {e}")


def flatten_game_raw(game_raw: dict) -> Optional[List[dict]]:
    try:
        game_flatten = {
            "game_id": int(game_raw["id"]),
            "begin_at": parse(game_raw["begin_at"]),
            "map_id": int(game_raw["map"]["id"]),
            "league_id": int(game_raw["match"]["league"]["id"]),
            "serie_id": int(game_raw["match"]["serie"]["id"]),
            "tier_id": {"s": 1, "a": 2, "b": 3, "c": 4, "d": 5}.get(
                game_raw["match"]["serie"].get("tier"), 0
            ),
            "tournament_id": int(game_raw["match"]["tournament"]["id"]),
        }

        team_players: dict[int, list[int]] = defaultdict(list)
        player_stat: dict[int, dict] = {}

        for p in game_raw.get("players", []):
            team_id = p["team"]["id"]
            player_id = p["player"]["id"]
            team_players[team_id].append(player_id)
            player_stat[player_id] = {
                "kills": int(p.get("kills", 0)),
                "deaths": int(p.get("deaths", 0)),
                "assists": int(p.get("assists", 0)),
                "headshots": int(p.get("headshots", 0)),
                "flash_assists": int(p.get("flash_assists", 0)),
                "first_kills_diff": int(p.get("first_kills_diff", 0)),
                "k_d_diff": int(p.get("k_d_diff", 0)),
                "adr": float(p.get("adr", 0)),
                "kast": float(p.get("kast", 0)),
                "rating": float(p.get("rating", 0)),
            }

        if len(team_players) != 2:
            return None

        t1_id, t2_id = list(team_players.keys())
        team_pair = {t1_id: t2_id, t2_id: t1_id}

        records: list[dict[str, Any]] = []

        for t_id in [t1_id, t2_id]:
            t_opp_id = team_pair[t_id]
            p_ids = team_players[t_id]
            p_opp_ids = team_players[t_opp_id]

            for r in game_raw.get("rounds", []):
                if not all(
                    k in r for k in ("ct", "terrorists", "winner_team", "round")
                ):
                    continue
                if (
                    r["ct"] not in [t1_id, t2_id]
                    or r["terrorists"] not in [t1_id, t2_id]
                    or r["winner_team"] not in [t1_id, t2_id]
                ):
                    continue

                for p_id in p_ids:
                    for p_opp_id in p_opp_ids:
                        rec = game_flatten.copy()
                        rec.update(player_stat.get(p_id, {}))
                        rec.update(
                            {
                                "team_id": t_id,
                                "team_opponent_id": t_opp_id,
                                "player_id": p_id,
                                "player_opponent_id": p_opp_id,
                                "round_id": int(r["round"]),
                                "round_outcome_id": int(
                                    {
                                        "eliminated": 1,
                                        "defused": 2,
                                        "exploded": 3,
                                        "timeout": 4,
                                    }.get(r.get("outcome"), 0)
                                ),
                                "round_is_ct": int(r["ct"] == t_id),
                                "round_win": int(r["winner_team"] == t_id),
                            }
                        )
                        records.append(rec)
        return records or None
    except Exception as e:
        log.warning(f"Ошибка обработки игры {game_raw.get('id')}: {e}")
        return None


def create_table(client: Client, drop: bool = True):
    if drop:
        client.execute("DROP TABLE IF EXISTS cs2_db.games_flatten")
    ddl = """
    CREATE TABLE IF NOT EXISTS cs2_db.games_flatten
    (
        game_id UInt64,
        begin_at DateTime,
        map_id UInt32,
        league_id UInt32,
        serie_id UInt32,
        tier_id UInt8,
        tournament_id UInt32,
        team_id UInt32,
        team_opponent_id UInt32,
        player_id UInt32,
        player_opponent_id UInt32,
        round_id UInt32,
        round_outcome_id UInt8,
        round_is_ct UInt8,
        round_win UInt8,
        kills UInt32,
        deaths UInt32,
        assists UInt32,
        headshots UInt32,
        flash_assists UInt32,
        first_kills_diff Float32,
        k_d_diff Float32,
        adr Float32,
        kast Float32,
        rating Float32
    )
    ENGINE = MergeTree()
    PARTITION BY toYYYYMM(begin_at)
    ORDER BY (begin_at, game_id, player_id, map_id, league_id, serie_id, tier_id, tournament_id, team_id, team_opponent_id, player_opponent_id, round_id);
    """
    client.execute(ddl)
    
    ddl2 = """
    CREATE VIEW cs2_db.player_cumulative_view AS
        SELECT
            player_id,
            begin_at,
            game_id,
            kills,
            deaths,
            assists,
            headshots,
            flash_assists,
            adr,
            kast,
            rating,
            k_d_diff,
            first_kills_diff,
            cumulative_kills,
            cumulative_deaths,
            cumulative_assists,
            cumulative_headshots,
            cumulative_flash_assists,
            cumulative_game_count,
            max_round_id,
            -- Per-round ratios
            cumulative_kills / max_round_id AS kills_per_round,
            cumulative_deaths / max_round_id AS deaths_per_round,
            cumulative_assists / max_round_id AS assists_per_round,
            cumulative_headshots / max_round_id AS headshots_per_round,
            cumulative_flash_assists / max_round_id AS flash_assists_per_round,
            -- Per-game ratios
            cumulative_kills / cumulative_game_count AS kills_per_game,
            cumulative_deaths / cumulative_game_count AS deaths_per_game,
            cumulative_assists / cumulative_game_count AS assists_per_game,
            cumulative_headshots / cumulative_game_count AS headshots_per_game,
            cumulative_flash_assists / cumulative_game_count AS flash_assists_per_game,
            cumulative_adr / cumulative_game_count AS adr_per_game,
            cumulative_kast / cumulative_game_count AS kast_per_game,
            cumulative_rating / cumulative_game_count AS rating_per_game,
            cumulative_k_d_diff / cumulative_game_count AS k_d_diff_per_game,
            cumulative_first_kills_diff / cumulative_game_count AS first_kills_diff_per_game
        FROM
        (
            SELECT
                player_id,
                begin_at,
                game_id,
                kills,
                deaths,
                assists,
                headshots,
                flash_assists,
                adr,
                kast,
                rating,
                k_d_diff,
                first_kills_diff,
                max_round_id,
                SUM(kills) OVER (
                    PARTITION BY player_id
                    ORDER BY begin_at ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_kills,
                SUM(deaths) OVER (
                    PARTITION BY player_id
                    ORDER BY begin_at ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_deaths,
                SUM(assists) OVER (
                    PARTITION BY player_id
                    ORDER BY begin_at ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_assists,
                SUM(headshots) OVER (
                    PARTITION BY player_id
                    ORDER BY begin_at ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_headshots,
                SUM(flash_assists) OVER (
                    PARTITION BY player_id
                    ORDER BY begin_at ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_flash_assists,
                SUM(adr) OVER (
                    PARTITION BY player_id
                    ORDER BY begin_at ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_adr,
                SUM(kast) OVER (
                    PARTITION BY player_id
                    ORDER BY begin_at ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_kast,
                SUM(rating) OVER (
                    PARTITION BY player_id
                    ORDER BY begin_at ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_rating,
                SUM(k_d_diff) OVER (
                    PARTITION BY player_id
                    ORDER BY begin_at ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_k_d_diff,
                SUM(first_kills_diff) OVER (
                    PARTITION BY player_id
                    ORDER BY begin_at ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_first_kills_diff,
                COUNT(*) OVER (
                    PARTITION BY player_id
                    ORDER BY begin_at ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_game_count
            FROM
            (
                SELECT
                    player_id,
                    begin_at,
                    game_id,
                    AVG(kills) AS kills,
                    AVG(deaths) AS deaths,
                    AVG(assists) AS assists,
                    AVG(headshots) AS headshots,
                    AVG(flash_assists) AS flash_assists,
                    AVG(adr) AS adr,
                    AVG(kast) AS kast,
                    AVG(rating) AS rating,
                    AVG(k_d_diff) AS k_d_diff,
                    AVG(first_kills_diff) AS first_kills_diff,
                    MAX(round_id) AS max_round_id
                FROM cs2_db.games_flatten
                GROUP BY player_id, begin_at, game_id
            ) AS subquery
        ) AS cumulative_data
        ORDER BY player_id, begin_at;
    """
    client.execute(ddl)


def insert_games_flat(
    client: Client, records: List[dict], table: str = "games_flatten"
):
    if not records:
        return
    columns = list(records[0].keys())
    values = [[rec.get(col) for col in columns] for rec in records]
    client.execute(f"INSERT INTO {table} ({', '.join(columns)}) VALUES", values)


def get_settings() -> Settings:
    parser = argparse.ArgumentParser(description="ETL JSON games into ClickHouse.")
    parser.add_argument("--env-file", type=str, default=".env")
    args = parser.parse_args()
    env_path = Path(args.env_file)
    if env_path.exists():
        log.info(f"Загрузка конфигурации из {env_path}")
        return Settings(_env_file=env_path)
    log.warning(f"Файл env не найден в {env_path}, используются значения по умолчанию")
    return Settings()


def get_all_game_ids(client: Client) -> list[int]:
    return [
        int(row[0])
        for row in client.execute(
            "SELECT DISTINCT game_id FROM games_flatten ORDER BY begin_at ASC"
        )
    ]


def main():
    settings = get_settings()

    client = Client(
        host=settings.CLICKHOUSE_HOST,
        port=settings.CLICKHOUSE_PORT,
        user=settings.CLICKHOUSE_USER,
        password=settings.CLICKHOUSE_PASSWORD,
        database=settings.CLICKHOUSE_DB,
    )

    try:
        create_table(client, drop=settings.CLICKHOUSE_DROP_TABLE)
    except Exception:
        log.exception("Ошибка при создании таблицы ClickHouse")
        return

    total_records = 0

    for game_raw in generate_game_raw(settings.PATH_TO_GAMES_RAW_DIR):
        flat_records = flatten_game_raw(game_raw)
        if flat_records:
            try:
                insert_games_flat(client, flat_records)
            except Exception:
                log.exception("Ошибка при вставке записей в ClickHouse")
                return
            total_records += len(flat_records)

    log.info(f"В ClickHouse вставлено {total_records} записей")

    


if __name__ == "__main__":
    main()
