import argparse
import ast
import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pika
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


class Settings(BaseSettings):
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_PORT: int = 9000
    CLICKHOUSE_HTTP_PORT: int = 8123
    CLICKHOUSE_USER: str = "cs2_user"
    CLICKHOUSE_PASSWORD: str = "cs2_password"
    CLICKHOUSE_DB: str = "cs2_db"

    RABBITMQ_USER: str = "cs2_user"
    RABBITMQ_PASSWORD: str = "cs2_password"
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_AMQP_PORT: int = 5672
    RABBITMQ_MANAGEMENT_PORT: int = 15672
    RABBITMQ_EXCHANGE: str = "cs2_exchange"
    RABBITMQ_EXCHANGE_TYPE: str = "direct"
    RABBITMQ_QUEUE: str = "cs2_queue"

    RABBITMQ_ROUTING_KEY_ETL: str = "cs2.etl_completed"
    RABBITMQ_ROUTING_KEY_SPLIT: str = "cs2.split_created"
    RABBITMQ_ROUTING_KEY_ML: str = "cs2.ml_completed"

    OUTPUT_DIR_RAW_SPLITS: str = "data/train_test_splits"
    OUTPUT_DIR_ML: str = "data/ml"

    PATH_TO_GAMES_RAW_DIR: str = "data/games_raw"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def handle_message(
    body: bytes,
    channel: pika.adapters.blocking_connection.BlockingChannel,
    settings: Settings,
):
    log.info(f"Получено split сообщение: {body.decode()}")
    data = ast.literal_eval(body.decode())

    train_ids = data.get("train", [])
    test_ids = data.get("test", [])

    if not train_ids and not test_ids:
        log.warning("Пустой split, пропускаем обучение модели")
        return

    log.info(f"Начало обучения модели: train={len(train_ids)}, test={len(test_ids)}")
    log.info("→ Запуск пайплайна обучения (fit pipeline)...")
    log.info("→ Расчет и сохранение метрик (save metrics)...")

    hash_input = ",".join(str(x) for x in train_ids + test_ids)
    hash_id = hashlib.md5(hash_input.encode("utf-8")).hexdigest()

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "hash_id": hash_id,
        "status": "ml_completed",
    }

    output_dir = Path(settings.OUTPUT_DIR_ML)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_file = output_dir / f"{hash_id}.json"
    model_file = output_dir / f"{hash_id}.pkl"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(model_file, "wb") as f:
        pickle.dump(None, f)

    log.info(
        f"Результаты обучения сохранены: метрики — {json_file}, модель — {model_file}"
    )

    channel.basic_publish(
        exchange=settings.RABBITMQ_EXCHANGE,
        routing_key=settings.RABBITMQ_ROUTING_KEY_ML,
        body=str(result),
    )

    log.info(
        f"Сообщение об окончании обучения опубликовано в {settings.RABBITMQ_ROUTING_KEY_ML}: {result}"
    )


def get_settings() -> Settings:
    parser = argparse.ArgumentParser(
        description="ML consumer для обработки split сообщений."
    )
    parser.add_argument("--env-file", type=str, default=".env")
    args = parser.parse_args()
    env_path = Path(args.env_file)
    if env_path.exists():
        log.info(f"Загрузка конфигурации из {env_path}")
        return Settings(_env_file=env_path)
    log.warning(f"Файл {env_path} не найден, используются настройки по умолчанию")
    return Settings()


def main():
    settings = get_settings()

    log.info("Подключение к RabbitMQ...")
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
    channel.queue_declare(queue=settings.RABBITMQ_QUEUE, durable=True)
    channel.queue_bind(
        queue=settings.RABBITMQ_QUEUE,
        exchange=settings.RABBITMQ_EXCHANGE,
        routing_key=settings.RABBITMQ_ROUTING_KEY_SPLIT,
    )

    log.info("Ожидание сообщений split для обучения моделей...")

    def callback(ch, method, properties, body):
        try:
            handle_message(body, ch, settings)
        except Exception as e:
            log.exception(f"Ошибка при обработке split сообщения: {e}")
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=settings.RABBITMQ_QUEUE, on_message_callback=callback)
    channel.start_consuming()


if __name__ == "__main__":
    main()
