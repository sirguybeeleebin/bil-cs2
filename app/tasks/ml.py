from celery import shared_task
import logging, pickle
from datetime import datetime
from django.conf import settings
from internal.services.ml_service import MLService

log = logging.getLogger(__name__)

def make_run_ml_task(svc: MLService):
    @shared_task(bind=True, name="internal.tasks.run_ml_task")
    def run_ml_task(self):
        task_id = self.request.id
        try:
            log.info(f"[{task_id}] Запуск ML задачи")
            results = svc.start(settings.PATH_TO_GAMES_RAW_DIR)
            file_path = settings.PATH_TO_ML_RESULTS_DIR / f"{task_id}.pickle"
            with open(file_path, "wb") as f:
                pickle.dump(
                    {"results": results, "created_at": datetime.now().isoformat()}, f
                )
            log.info(
                f"[{task_id}] ML задача успешно выполнена, результаты сохранены: {file_path}"
            )
            return {"status": "success", "path": str(file_path)}
        except Exception as e:
            log.error(f"[{task_id}] ML задача завершилась с ошибкой: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
    return run_ml_task
