import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable

import joblib
from celery import shared_task

from config.settings import PATH_TO_ML_RESULTS_DIR

log = logging.getLogger(__name__)


def make_ml_pipeline_task(
    ml_pipeline_repository,
    ml_pipeline_metrics_repository,
    train_model_fn: Callable[..., tuple[Any, dict]],
):
    @shared_task(name="run_ml_pipeline_task")
    def ml_pipeline_task(
        path_to_games_raw_dir: str,
        test_size: int = 100,
        n_splits: int = 10,
        n_iters: int = 10,
        random_state: int = 42,
    ):
        log.info("Запуск ML пайплайна")
        try:
            pipeline_id = uuid.uuid4()

            ml_pipeline, metrics = train_model_fn(
                path_to_games_raw_dir=path_to_games_raw_dir,
                test_size=test_size,
                n_splits=n_splits,
                n_iters=n_iters,
                random_state=random_state,
            )

            results_dir = Path(PATH_TO_ML_RESULTS_DIR)
            results_dir.mkdir(parents=True, exist_ok=True)

            pipeline_file = results_dir / f"{pipeline_id}.joblib"
            metrics_file = results_dir / f"{pipeline_id}.json"

            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            joblib.dump(ml_pipeline, pipeline_file)

            log.info(
                f"Файлы пайплайна и метрик сохранены: {pipeline_file}, {metrics_file}"
            )

            pipeline = ml_pipeline_repository.upsert(
                {
                    "pipeline_id": pipeline_id,
                    "path_to_pipeline_file": str(pipeline_file),
                }
            )
            if not pipeline:
                log.error(f"Не удалось сохранить ML пайплайн в БД: {pipeline_file}")
                raise RuntimeError("Сохранение ML пайплайна не удалось")

            metrics = ml_pipeline_metrics_repository.upsert(
                {
                    "metrics_id": uuid.uuid4(),
                    "pipeline_id": pipeline_id,
                    "path_to_metrics_file": str(metrics_file),
                }
            )
            if not metrics:
                log.error(
                    f"Не удалось сохранить метрики ML пайплайна в БД: {metrics_file}"
                )
                raise RuntimeError("Сохранение метрик ML пайплайна не удалось")

            log.info("ML пайплайн успешно выполнен и сохранён")
            return {"status": "готово", "pipeline_id": str(pipeline_id)}

        except Exception as e:
            log.exception("Ошибка при выполнении ML пайплайна")
            return {"status": "ошибка", "message": str(e)}

    return ml_pipeline_task
