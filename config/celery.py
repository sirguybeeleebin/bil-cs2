import logging
import os
from celery import Celery, chain
from celery.signals import worker_ready
from backend.tasks import parse, train_test_split, run_ml_pipeline, generate_pipeline_id
from django.conf import settings

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Django settings
# -------------------------------------------------------------------
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# -------------------------------------------------------------------
# Celery app
# -------------------------------------------------------------------
app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()


@worker_ready.connect
def run_pipeline_chain(sender, **kwargs):
    """
    Запускает полный ML pipeline как Celery chain:
    parse -> train_test_split -> run_ml_pipeline
    с уникальным pipeline_id.
    """
    pipeline_id = generate_pipeline_id()
    log.info(f"[PIPELINE] Запуск ML pipeline с ID: {pipeline_id}")

    # Передаем pipeline_id на каждый шаг chain
    pipeline = chain(
        parse.s(pipeline_id=pipeline_id),                        # parse возвращает valid_game_ids_path
        train_test_split.s(),             # принимает valid_game_ids_path + pipeline_id
        run_ml_pipeline.s()               # принимает train_test_path + pipeline_id
    )

    # Асинхронное выполнение
    pipeline.apply_async()
    log.info("[PIPELINE] Цепочка ML pipeline успешно отправлена в очередь.")
