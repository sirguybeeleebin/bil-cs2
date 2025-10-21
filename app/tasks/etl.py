from celery import shared_task
import logging
from django.conf import settings
from internal.services.etl_service import ETLService

log = logging.getLogger(__name__)

def make_run_etl_task(svc: ETLService):
    @shared_task(bind=True, name="internal.tasks.run_etl_task")
    def run_etl_task(self):
        task_id = self.request.id
        try:
            log.info(f"[{task_id}] Запуск ETL задачи")
            svc.start(settings.PATH_TO_GAMES_RAW_DIR)
            log.info(f"[{task_id}] ETL задача успешно выполнена")
            return {"status": "success"}
        except Exception as e:
            log.error(f"[{task_id}] ETL задача завершилась с ошибкой: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
    return run_etl_task
