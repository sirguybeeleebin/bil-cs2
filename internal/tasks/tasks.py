# app/tasks.py
from celery import shared_task
from django.apps import apps
from django.conf import settings
import logging
import pickle
from datetime import datetime

log = logging.getLogger(__name__)

@shared_task(bind=True)
def run_etl_task(self):
    try:
        task_id = self.request.id
        log.info(f"[{task_id}] Запуск ETL задачи")
        etl_service = apps.get_app_config("internal").etl_service
        etl_service.start(settings.PATH_TO_GAMES_RAW_DIR)
        log.info(f"[{task_id}] ETL задача успешно выполнена")
        return {"status": "success"}
    except Exception as e:
        log.error(f"[{task_id}] ETL задача завершилась с ошибкой: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}


@shared_task(bind=True)
def run_ml_task(self):
    try:
        task_id = self.request.id
        log.info(f"[{task_id}] Запуск ML задачи")
        ml_service = apps.get_app_config("internal").ml_service
        results = ml_service.start(settings.PATH_TO_GAMES_RAW_DIR)
        file_path = settings.PATH_TO_ML_RESULTS_DIR / f"{task_id}.pickle"
        with open(file_path, "wb") as f:
            pickle.dump({"results": results, "created_at": datetime.now().isoformat()}, f)
        log.info(f"[{task_id}] ML задача успешно выполнена, результаты сохранены: {file_path}")
        return {"status": "success", "path": str(file_path)}
    except Exception as e:
        log.error(f"[{task_id}] ML задача завершилась с ошибкой: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}
