import argparse
import json
import logging
import os
from collections import defaultdict

import clickhouse_connect
import pandas as pd
from dateutil.parser import parse

# === Настройка логирования ===
logging.basicConfig(
    level=logging.INFO,  # Можно поставить DEBUG для подробных логов
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cs2_loader")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load raw CS2 games data into ClickHouse"
    )
    parser.add_argument(
        "--path_to_games_raw_dir",
        type=str,
        default="data/games_raw",
        help="Путь к директории с JSON файлами игр",
    )
    parser.add_argument(
        "--clickhouse_host",
        type=str,
        default="localhost",
        help="ClickHouse host (по умолчанию localhost)",
    )
    parser.add_argument(
        "--clickhouse_port",
        type=int,
        default=8123,
        help="ClickHouse HTTP порт (по умолчанию 8123)",
    )
    parser.add_argument(
        "--clickhouse_user",
        type=str,
        default="cs2_user",
        help="Имя пользователя ClickHouse (по умолчанию cs2_user)",
    )
    parser.add_argument(
        "--clickhouse_password",
        type=str,
        default="cs2_password",
        help="Пароль пользователя ClickHouse (по умолчанию cs2_password)",
    )
    parser.add_argument(
        "--clickhouse_db",
        type=str,
        default="cs2_db",
        help="Имя базы данных ClickHouse (по умолчанию cs2_db)",
    )
    parser.add_argument(
        "--drop_table",
        type=bool,
        default=True,
        help="Удалить таблицу перед загрузкой (по умолчанию False)",
    )
    return parser.parse_args()


def validate_game_raw(game: dict) -> bool:
    log.debug(f"Validating game id={game.get('id')}")
    try:
        int(game["id"])
        parse(game["begin_at"])
        team_players = defaultdict(set)
        for p in game["players"]:
            t_id = int(p["team"]["id"])
            p_id = int(p["player"]["id"])
            team_players[t_id].add(p_id)
        if len(team_players) != 2 or any(
            len(pids) != 5 for pids in team_players.values()
        ):
            log.warning(f"Game id={game.get('id')} failed team validation")
            return False
        t_ids = list(team_players.keys())
        rounds = []
        for r in game["rounds"]:
            if r["round"] is None:
                return False
            if (
                r["ct"] not in t_ids
                or r["terrorists"] not in t_ids
                or r["winner_team"] not in t_ids
            ):
                return False
            rounds.append(r["round"])
        if min(rounds) != 1 or max(rounds) < 16:
            return False
        return True
    except Exception as e:
        log.error(f"Validation exception for game id={game.get('id')}: {e}")
        return False


def get_dict(d: dict, key: str) -> dict:
    """Безопасно извлекает подсловарь"""
    v = d.get(key)
    return v if v else {}


def flatten_game(game: dict) -> list[dict]:
    log.debug(f"Flattening game id={game.get('id')}")
    game_flatten = {
        "game_id": int(game["id"]),
        "begin_at": parse(game["begin_at"]),
        "map_id": get_dict(game, "map").get("id", 0),
        "league_id": get_dict(get_dict(game, "match"), "league").get("id", 0),
        "serie_id": get_dict(get_dict(game, "match"), "serie").get("id", 0),
        "tier_id": {"s": 1, "a": 2, "b": 3, "c": 4, "d": 5}.get(
            get_dict(get_dict(game, "match"), "serie").get("tier"), 0
        ),
        "tournament_id": get_dict(get_dict(game, "match"), "tournament").get("id", 0),
    }

    team_players = defaultdict(list)
    player_stats = {}

    for p in game["players"]:
        team_players[p["team"]["id"]].append(p["player"]["id"])
        player_stats[p["player"]["id"]] = {
            "kills": p.get("kills", 0),
            "deaths": p.get("deaths", 0),
            "assists": p.get("assists", 0),
            "headshots": p.get("headshots", 0),
            "flash_assists": p.get("flash_assists", 0),
            "k_d_diff": p.get("k_d_diff", 0),
            "first_kills_diff": p.get("first_kills_diff", 0),
            "adr": p.get("adr", 0),
            "kast": p.get("kast", 0),
            "rating": p.get("rating", 0),
        }

    t_ids = list(team_players.keys())
    team_pair = {t_ids[0]: t_ids[1], t_ids[1]: t_ids[0]}

    L = []
    for t_id, p_ids in team_players.items():
        t_opp_id = team_pair[t_id]
        p_opp_ids = team_players[t_opp_id]
        for p_id in p_ids:
            for p_opp_id in p_opp_ids:
                c = game_flatten.copy()
                c["team_id"] = t_id
                c["team_opponent_id"] = t_opp_id
                c["player_id"] = p_id
                c["player_opponent_id"] = p_opp_id
                c.update(player_stats[p_id])

                for r in game["rounds"]:
                    c2 = c.copy()
                    c2["round_id"] = r["round"]
                    c2["round_is_ct"] = int(r["ct"] == t_id)
                    c2["round_outcome"] = {
                        "exploded": 1,
                        "defused": 2,
                        "timeout": 3,
                        "eliminated": 4,
                    }.get(r["outcome"], 0)
                    c2["round_win"] = int(r["winner_team"] == t_id)
                    L.append(c2)
    log.debug(f"Flattened game id={game.get('id')} into {len(L)} rows")
    return L


def load_to_clickhouse(
    client,
    games_flatten: list[dict],
    table_name: str = "games_flat",
    database: str = "cs2_db",
):
    full_table_name = f"{database}.{table_name}"
    log.info(f"Ensuring table exists: {full_table_name}")

    # Создаём таблицу если не существует
    client.command(f"""
        CREATE TABLE IF NOT EXISTS {full_table_name} (
            begin_at DateTime,
            game_id Int64,
            round_id Int64,            
            map_id Int64,
            league_id Int64,
            serie_id Int64,
            tier_id Int64,
            tournament_id Int64,
            team_id Int64,
            team_opponent_id Int64,
            player_id Int64,
            player_opponent_id Int64,
            kills Int64,
            deaths Int64,
            assists Int64,
            headshots Int64,
            flash_assists Int64,
            k_d_diff Int64,
            first_kills_diff Int64,
            adr Float64,
            kast Float64,
            rating Float64,
            round_is_ct Int64,
            round_outcome Int64,
            round_win Int64
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(begin_at)
        ORDER BY (begin_at, game_id, round_id)
        SETTINGS index_granularity = 8192
    """)

    # Преобразуем в DataFrame
    df = pd.DataFrame.from_records(games_flatten)

    # === Приводим типы к соответствию с ClickHouse ===
    int_cols = [
        "game_id",
        "round_id",
        "map_id",
        "league_id",
        "serie_id",
        "tier_id",
        "tournament_id",
        "team_id",
        "team_opponent_id",
        "player_id",
        "player_opponent_id",
        "kills",
        "deaths",
        "assists",
        "headshots",
        "flash_assists",
        "k_d_diff",
        "first_kills_diff",
        "round_is_ct",
        "round_outcome",
        "round_win",
    ]
    float_cols = ["adr", "kast", "rating"]
    df[int_cols] = df[int_cols].fillna(0).astype("int64")
    df[float_cols] = df[float_cols].fillna(0.0).astype("float64")
    df["begin_at"] = pd.to_datetime(df["begin_at"])

    log.debug(f"Data types after conversion:\n{df.dtypes}")

    log.info(f"Loading {len(df)} rows into table '{full_table_name}'")
    columns = [
        "begin_at",
        "game_id",
        "round_id",
        "map_id",
        "league_id",
        "serie_id",
        "tier_id",
        "tournament_id",
        "team_id",
        "team_opponent_id",
        "player_id",
        "player_opponent_id",
        "kills",
        "deaths",
        "assists",
        "headshots",
        "flash_assists",
        "k_d_diff",
        "first_kills_diff",
        "adr",
        "kast",
        "rating",
        "round_is_ct",
        "round_outcome",
        "round_win",
    ]

    # Преобразуем DataFrame в список списков для вставки
    data = df[columns].values.tolist()
    log.debug(f"Prepared data for insertion. Sample row: {data[0] if data else 'N/A'}")

    # Вставляем
    client.insert(full_table_name, data, column_names=columns)
    log.info(f"Inserted {len(df)} rows into '{full_table_name}'")


def generate_game_raw(path_to_dir: str):
    log.info(f"Generating games from directory: {path_to_dir}")
    if not os.path.isdir(path_to_dir):
        raise FileNotFoundError(f"Directory '{path_to_dir}' not found")

    files = [f for f in os.listdir(path_to_dir)]
    log.info(f"Found {len(files)} JSON files")
    if not files:
        log.warning(f"No JSON files found in {path_to_dir}")
        return

    for fname in files:
        path = os.path.join(path_to_dir, fname)
        try:
            with open(path, "r") as f:
                game = json.load(f)
            log.debug(f"Loaded file: {fname}")
            yield fname, game
        except Exception as e:
            log.error(f"Failed to read {fname}: {e}")


if __name__ == "__main__":
    log.info("Starting CS2 ClickHouse loader")
    args = parse_args()

    client = clickhouse_connect.get_client(
        host=args.clickhouse_host,
        port=args.clickhouse_port,
        username=args.clickhouse_user,
        password=args.clickhouse_password,
        database=args.clickhouse_db,
    )
    log.info(
        f"Connected to ClickHouse at {args.clickhouse_host}:{args.clickhouse_port}"
    )

    table_name = "games_flat"

    if args.drop_table:
        log.info(f"Dropping table '{table_name}' as requested")
        try:
            client.command(f"DROP TABLE IF EXISTS {table_name}")
            log.info(f"Table '{table_name}' dropped")
        except Exception as e:
            log.error(f"Failed to drop table '{table_name}': {e}")

    total, success, error = 0, 0, 0
    for fname, game in generate_game_raw(args.path_to_games_raw_dir):
        total += 1
        log.info(f"Processing file [{total}]: {fname}")
        try:
            if validate_game_raw(game):
                games_flatten = flatten_game(game)
                load_to_clickhouse(client, games_flatten, table_name)
                success += 1
                log.info(f"Successfully processed file [{total}/{success}]")
            else:
                log.warning(f"Game in file [{total}] failed validation, skipped")
                error += 1
        except Exception as e:
            log.error(f"Exception processing file [{total}]: {e}")
            error += 1
        log.info(f"Stats so far: total={total}, success={success}, error={error}")

    log.info("Loading complete")
    log.info(f"Final stats: total={total}, success={success}, error={error}")
