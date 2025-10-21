from celery import shared_task
from app.services import etl_service, ml_service
import logging

log = logging.getLogger(__name__)


@shared_task
def run_etl_task():
    try:
        log.info("Запуск ETL задачи")
        etl_service.start()
        log.info("ETL задача успешно выполнена")
        return {"status": "success"}
    except Exception as e:
        log.error(f"ETL задача завершилась с ошибкой: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}


@shared_task
def run_ml_task():
    try:
        log.info("Запуск ML задачи")
        pipeline_path = ml_service.start()
        if pipeline_path is None:
            log.warning("ML пайплайн не завершился успешно")
            return {"status": "failed", "error": "ML пайплайн не завершился успешно"}

        log.info(f"ML задача успешно выполнена, пайплайн сохранён: {pipeline_path}")
        return {
            "status": "success",
            "pipeline_path": pipeline_path,
        }
    except Exception as e:
        log.error(f"ML задача завершилась с ошибкой: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}
