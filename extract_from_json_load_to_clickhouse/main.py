import os
import json
import logging
import argparse
from datetime import datetime
from dateutil.parser import parse
from collections import defaultdict
from typing import Generator, Any, List, Optional
from pathlib import Path
from clickhouse_driver import Client
from pydantic_settings import BaseSettings, SettingsConfigDict
import pika

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)




class Settings(BaseSettings):
    # ================= ClickHouse =================
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 9000
    clickhouse_user: str = "cs2_user"
    clickhouse_password: str = "cs2_password"
    clickhouse_db: str = "cs2_db"

    # ================= RabbitMQ =================
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/%2F"
    rabbitmq_exchange: str = "cs2"  # общий exchange для всех событий
    rabbitmq_exchange_type: str = "direct"

    # ETL
    rabbitmq_queue_etl: str = "cs2.etl_consumer"
    rabbitmq_routing_key_etl: str = "cs2.etl_completed"

    # Train/Test Split
    rabbitmq_queue_split: str = "cs2.split_consumer"
    rabbitmq_routing_key_split: str = "cs2.split_created"

    # ML pipeline
    rabbitmq_queue_ml: str = "cs2.ml_consumer"
    rabbitmq_routing_key_ml: str = "cs2.ml_completed"

    # ================= Other =================
    output_dir: str = "data/train_test_splits"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )



def _get_settings() -> Settings:
    parser = argparse.ArgumentParser(description="ETL JSON games into ClickHouse.")
    parser.add_argument("--env-file", type=str, default=".env")
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if env_path.exists():
        log.info(f"Loading configuration from {env_path}")
        return Settings(_env_file=env_path)
    log.warning(f"Env file not found at {env_path}, using defaults")
    return Settings()


def _generate_game_raw(path_to_games_raw_dir: str) -> Generator[dict[str, Any], None, None]:
    """Генератор для чтения JSON файлов с сырыми играми."""
    for filename in os.listdir(path_to_games_raw_dir):
        try:
            with open(os.path.join(path_to_games_raw_dir, filename), "r", encoding="utf-8") as f:
                yield json.load(f)
        except Exception as e:
            log.warning(f"Error reading file {filename}: {e}")


def _flatten_game_raw(game_raw: dict) -> Optional[List[dict]]:
    """Флаттенинг JSON игры в список записей для ClickHouse."""
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


def _create_table(client: Client, drop: bool = False):
    """Создание таблицы ClickHouse, опционально удаляя старую."""
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


def _insert_games_flat(client: Client, records: List[dict], table: str = "games_flatten"):
    """Вставка записей в ClickHouse."""
    if not records:
        return
    columns = list(records[0].keys())
    values = [[rec.get(col) for col in columns] for rec in records]
    client.execute(f"INSERT INTO {table} ({', '.join(columns)}) VALUES", values)


def _publish_event(channel, exchange: str, routing_key: str, payload: dict):
    """Публикация события через уже настроенный канал."""
    try:
        message = json.dumps(payload)
        channel.basic_publish(exchange=exchange, routing_key=routing_key, body=message)
        log.info(f"Event published: {payload}")
    except Exception as e:
        log.error(f"Failed to publish event: {e}")


# ==================== ETL ====================
def process_game_raws(client: Client, channel_rabbit, path_to_games_raw_dir: str, drop_table: bool, settings: Settings):
    _create_table(client, drop=drop_table)
    total, success, error, total_inserted = 0, 0, 0, 0

    for game_raw in _generate_game_raw(path_to_games_raw_dir):
        total += 1
        flat_records = _flatten_game_raw(game_raw)
        if flat_records:
            try:
                _insert_games_flat(client, flat_records)
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

    # Публикация события
    if channel_rabbit:
        payload = {
            "event": "etl.completed",
            "total_games": total,
            "success_games": success,
            "failed_games": error,
            "total_rows_inserted": total_inserted,
            "timestamp": datetime.now().isoformat(),
        }
        _publish_event(channel_rabbit, settings.rabbitmq_exchange, settings.rabbitmq_routing_key, payload)


# ==================== Main ====================
def main():
    settings = _get_settings()

    # ClickHouse client
    client = Client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        user=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_db,
    )
    log.info("Connected to ClickHouse")

    # RabbitMQ connection & channel
    params = pika.URLParameters(settings.rabbitmq_url)
    connection = pika.BlockingConnection(params)
    channel_rabbit = connection.channel()
    channel_rabbit.exchange_declare(
        exchange=settings.rabbitmq_exchange,
        exchange_type=settings.rabbitmq_exchange_type,
        durable=True
    )
    # Queue из settings
    channel_rabbit.queue_declare(queue=settings.rabbitmq_queue, durable=True)
    channel_rabbit.queue_bind(
        queue=settings.rabbitmq_queue,
        exchange=settings.rabbitmq_exchange,
        routing_key=settings.rabbitmq_routing_key
    )
    log.info("Connected to RabbitMQ")

    # Запуск ETL
    process_game_raws(client, channel_rabbit, settings.path_to_games_raw_dir, settings.drop_table, settings)

    connection.close()
    log.info("Process finished")


if __name__ == "__main__":
    main()
