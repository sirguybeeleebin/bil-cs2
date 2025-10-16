import os
import json
import hashlib
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from clickhouse_driver import Client
from pydantic_settings import BaseSettings, SettingsConfigDict
import pika

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Settings(BaseSettings):
    # ================= ClickHouse =================
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 9000
    clickhouse_user: str = "cs2_user"
    clickhouse_password: str = "cs2_password"
    clickhouse_db: str = "cs2_db"

    # ================= Output =================
    output_dir: str = "data/train_test_splits"

    # ================= RabbitMQ =================
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/%2F"
    rabbitmq_exchange: str = "cs2"
    rabbitmq_exchange_type: str = "direct"

    # Очередь для подписки (ETL завершен)
    rabbitmq_consume_queue: str = "cs2_etl_completed_queue"
    rabbitmq_consume_routing_key: str = "cs2.etl_completed"

    # Очередь и ключ для публикации события (сплит создан)
    rabbitmq_publish_queue: str = "cs2_split_created_queue"
    rabbitmq_publish_routing_key: str = "cs2.split_created"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


def _get_settings() -> Settings:
    parser = argparse.ArgumentParser(description="Train/test split consumer.")
    parser.add_argument("--env-file", type=str, default=".env")
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if env_path.exists():
        log.info(f"Loading configuration from {env_path}")
        return Settings(_env_file=env_path)
    log.warning(f"Env file not found at {env_path}, using defaults")
    return Settings()


# ==================== Helpers ====================
def _fetch_game_ids(client: Client) -> list[int]:
    query = "SELECT DISTINCT game_id FROM games_flatten ORDER BY begin_at ASC"
    return [row[0] for row in client.execute(query)]


def _create_split_file(game_ids: list[int], output_dir: Path) -> tuple[Path | None, str]:
    test_game_ids = game_ids[-100:]
    train_game_ids = game_ids[:-100]

    hash_input = ",".join(str(gid) for gid in game_ids)
    hash_id = hashlib.md5(hash_input.encode("utf-8")).hexdigest()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{hash_id}.json"

    if out_file.exists():
        log.info(f"Train/test split {hash_id} already exists, skipping creation")
        return None, hash_id

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train": train_game_ids,
        "test": test_game_ids,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info(f"Train/test split saved to {out_file} (train={len(train_game_ids)}, test={len(test_game_ids)})")
    return out_file, hash_id


def _publish_split_event(channel, exchange: str, routing_key: str, hash_id: str):
    try:
        message = json.dumps({
            "event": "split.created",
            "hash_id": hash_id,
            "timestamp": datetime.now().isoformat()
        })
        channel.basic_publish(exchange=exchange, routing_key=routing_key, body=message)
        log.info(f"Published new split hash {hash_id} to exchange '{exchange}' with routing key '{routing_key}'")
    except Exception as e:
        log.error(f"Failed to publish split event: {e}")


# ==================== Consumer ====================
def _process_etl_completed(body: bytes, client: Client, channel_rabbit, settings: Settings):
    try:
        payload = json.loads(body)
        log.info(f"ETL payload: {payload}")
    except Exception as e:
        log.error(f"Invalid ETL message: {e}")
        return

    game_ids = _fetch_game_ids(client)
    log.info(f"Fetched {len(game_ids)} unique game_ids from ClickHouse")

    out_file, hash_id = _create_split_file(game_ids, Path(settings.output_dir))
    if out_file:
        _publish_split_event(channel_rabbit, settings.rabbitmq_exchange, settings.rabbitmq_publish_routing_key, hash_id)


def on_etl_completed(ch, method, properties, body, client: Client, channel_rabbit, settings: Settings):
    log.info("Received ETL completion event")
    _process_etl_completed(body, client, channel_rabbit, settings)
    ch.basic_ack(delivery_tag=method.delivery_tag)


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

    # Queue для подписки
    channel_rabbit.queue_declare(queue=settings.rabbitmq_consume_queue, durable=True)
    channel_rabbit.queue_bind(
        exchange=settings.rabbitmq_exchange,
        queue=settings.rabbitmq_consume_queue,
        routing_key=settings.rabbitmq_consume_routing_key
    )

    log.info("Waiting for ETL completed events...")
    channel_rabbit.basic_consume(
        queue=settings.rabbitmq_consume_queue,
        on_message_callback=lambda ch, method, props, body: on_etl_completed(
            ch, method, props, body, client, channel_rabbit, settings
        )
    )

    channel_rabbit.start_consuming()


if __name__ == "__main__":
    main()
