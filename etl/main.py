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

    RABBITMQ_USER: str = "cs2_user"
    RABBITMQ_PASSWORD: str = "cs2_password"
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_AMQP_PORT: int = 5672
    RABBITMQ_EXCHANGE: str = "cs2_exchange"
    RABBITMQ_EXCHANGE_TYPE: str = "direct"
    RABBITMQ_ROUTING_KEY_ETL: str = "cs2.etl_completed"

    PATH_TO_GAMES_RAW_DIR: str = "data/games_raw"

    @property
    def RABBITMQ_URL(self) -> str:
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_AMQP_PORT}/%2F"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def generate_game_raw(
    path_to_games_raw_dir: str,
) -> Generator[dict[str, Any], None, None]:
    for filename in os.listdir(path_to_games_raw_dir)[:1000]:
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

    # --- Query distinct game_ids ordered by begin_at ---
    try:
        processed_game_ids = get_all_game_ids()
        log.info(f"Всего уникальных игр: {len(processed_game_ids)}")
        log.info(f"Первые 10 game_id по begin_at: {processed_game_ids[:10]}")
    except Exception:
        log.exception("Ошибка при выборке game_id из ClickHouse")
        return

    # --- Publish message to RabbitMQ ---
    connection = None
    try:
        credentials = pika.PlainCredentials(
            settings.RABBITMQ_USER, settings.RABBITMQ_PASSWORD
        )
        parameters = pika.ConnectionParameters(
            host=settings.RABBITMQ_HOST,
            port=settings.RABBITMQ_AMQP_PORT,
            credentials=credentials,
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.exchange_declare(
            exchange=settings.RABBITMQ_EXCHANGE,
            exchange_type=settings.RABBITMQ_EXCHANGE_TYPE,
            durable=True,
        )

        message = {"processed_games": processed_game_ids}
        channel.basic_publish(
            exchange=settings.RABBITMQ_EXCHANGE,
            routing_key=settings.RABBITMQ_ROUTING_KEY_ETL,
            body=str(message),
        )
        log.info(
            f"Сообщение ETL опубликовано в {settings.RABBITMQ_ROUTING_KEY_ETL}: {message}"
        )
    except Exception:
        log.exception("Ошибка при публикации сообщения RabbitMQ")
    finally:
        if connection and connection.is_open:
            connection.close()


if __name__ == "__main__":
    main()
