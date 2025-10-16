import os
import json
import logging
from dateutil.parser import parse
from collections import defaultdict
from typing import Generator, Any
from pathlib import Path
from clickhouse_driver import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def generate_game_raw(path_to_games_raw_dir: str) -> Generator[dict[str, Any], None, None]:
    for filename in os.listdir(path_to_games_raw_dir):
        try:
            with open(os.path.join(path_to_games_raw_dir, filename), "r", encoding="utf-8") as f:
                yield json.load(f)
        except Exception as e:
            log.warning(f"Error reading file {filename}: {e}")


def flatten_game_raw(game_raw: dict) -> list[dict] | None:
    try:
        game_flatten = {
            "game_id": int(game_raw["id"]),
            "begin_at": parse(game_raw["begin_at"]),
            "map_id": int(game_raw["map"]["id"]),
            "league_id": int(game_raw["match"]["league"]["id"]),
            "serie_id": int(game_raw["match"]["serie"]["id"]),
            "tier_id": {"s": 1, "a": 2, "b": 3, "c": 4, "d": 5}.get(game_raw["match"]["serie"].get("tier"), 0),
            "tournament_id": int(game_raw["match"]["tournament"]["id"]),
        }

        team_players: dict[int, list[int]] = defaultdict(list)
        player_stat: dict[int, dict] = {}

        for p in game_raw.get("players", []):
            team_id = p["team"]["id"]
            player_id = p["player"]["id"]
            team_players[team_id].append(player_id)
            player_stat[player_id] = {
                "kills": p.get("kills", 0),
                "deaths": p.get("deaths", 0),
                "assists": p.get("assists", 0),
                "headshots": p.get("headshots", 0),
                "flash_assists": p.get("flash_assists", 0),
                "first_kills_diff": p.get("first_kills_diff", 0),
                "k_d_diff": p.get("k_d_diff", 0),
                "adr": p.get("adr", 0),
                "kast": p.get("kast", 0),
                "rating": p.get("rating", 0),
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
                if not all(k in r for k in ("ct", "terrorists", "winner_team", "round")):
                    continue
                if r["ct"] not in [t1_id, t2_id] or r["terrorists"] not in [t1_id, t2_id] or r["winner_team"] not in [t1_id, t2_id]:
                    continue

                for p_id in p_ids:
                    for p_opp_id in p_opp_ids:
                        rec = game_flatten.copy()
                        rec.update(player_stat.get(p_id, {}))
                        rec.update({
                            "team_id": t_id,
                            "team_opponent_id": t_opp_id,
                            "player_id": p_id,
                            "player_opponent_id": p_opp_id,
                            "round_id": int(r["round"]),
                            "round_outcome_id": int({"eliminated": 1, "defused": 2, "exploded": 3, "timeout": 4}.get(r.get("outcome"), 0)),
                            "round_is_ct": int(r["ct"] == t_id),
                            "round_win": int(r["winner_team"] == t_id)
                        })
                        records.append(rec)

        return records or None
    except Exception as e:
        log.warning(f"Error flattening game {game_raw.get('id')}: {e}")
        return None


def create_table(client: Client, drop: bool = False) -> None:
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


def insert_games_flat_to_clickhouse(client: Client, records: list[dict], table: str = "games_flatten") -> None:
    if not records:
        return
    columns = list(records[0].keys())
    values = [[rec.get(col) for col in columns] for rec in records]
    client.execute(f"INSERT INTO {table} ({', '.join(columns)}) VALUES", values)


def extract_from_json_load_to_clickhouse(
    clickhouse_host: str = "localhost",
    clickhouse_port: int = 9000,
    clickhouse_user: str = "cs2_user",
    clickhouse_password: str = "cs2_password",
    clickhouse_db: str = "cs2_db",
    drop_table: bool = True,
    path_to_games_raw_dir: str = "data/games_raw",
):
    """ETL JSON games into ClickHouse with explicit arguments."""
    client = Client(
        host=clickhouse_host,
        port=clickhouse_port,
        user=clickhouse_user,
        password=clickhouse_password,
        database=clickhouse_db
    )
    log.info("Connected to ClickHouse")

    create_table(client, drop=drop_table)

    total, success, error, total_inserted = 0, 0, 0, 0
    for game_raw in generate_game_raw(path_to_games_raw_dir):
        total += 1
        flat_records = flatten_game_raw(game_raw)
        if flat_records:
            try:
                insert_games_flat_to_clickhouse(client, flat_records)
                inserted_count = len(flat_records)
                total_inserted += inserted_count
                success += 1
                log.info(f"Inserted {inserted_count} rows for game_id {flat_records[0]['game_id']}")
            except Exception as e:
                error += 1
                log.error(f"Error inserting game {flat_records[0].get('game_id')}: {e}")
        else:
            error += 1
        log.info(f"Progress: total={total}, success={success}, error={error}, total_inserted={total_inserted}")

    log.info("JSON processing completed")
    log.info("Process finished")


if __name__ == "__main__":
    extract_from_json_load_to_clickhouse()
