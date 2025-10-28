import json
import logging
import os
import uuid
from pathlib import Path
from typing import Callable, Optional
from uuid import UUID
<<<<<<< HEAD

=======
>>>>>>> origin/main
import joblib
from celery import shared_task
from django.conf import settings

from backend.di import (
    map_repo,
    ml_result_metrics_repo,
    ml_result_repo,
    player_repo,
    team_repo,
)
from backend.repositories import (
    MapRepository,
    MLPipelineMetricRepository,
    MLPipelineRepository,
    PlayerRepository,
    TeamRepository,
)

log = logging.getLogger(__name__)


def make_update_dictionaries_task(
    update_dictionaries_func: Callable,
    games_raw_dir: str = settings.GAMES_RAW_DIR,
    maps_dir: str = settings.MAPS_DIR,
    teams_dir: str = settings.TEAMS_DIR,
    players_dir: str = settings.PLAYERS_DIR,
    task_name: str = "backend.tasks.update_dictionaries_task",
) -> Callable[..., None]:
    @shared_task(name=task_name)
    def task(*args, **kwargs) -> None:
        log.info("Запуск задачи обновления словарей...")
        update_dictionaries_func(
            games_raw_dir=games_raw_dir,
            maps_dir=maps_dir,
            teams_dir=teams_dir,
            players_dir=players_dir,
        )
        log.info("Задача обновления словарей завершена.")

    return task


def make_load_dictionaries_task(
    maps_dir: str = settings.MAPS_DIR,
    teams_dir: str = settings.TEAMS_DIR,
    players_dir: str = settings.PLAYERS_DIR,
    map_repository: MapRepository = map_repo,
    team_repository: TeamRepository = team_repo,
    player_repository: PlayerRepository = player_repo,
    task_name: str = "backend.tasks.load_dictionaries_task",
) -> Callable[..., None]:
    @shared_task(name=task_name)
    def task(*args, **kwargs) -> None:
        log.info("Запуск задачи загрузки словарей в БД...")

        for filename in os.listdir(maps_dir):
            if filename.endswith(".json"):
                with open(os.path.join(maps_dir, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                map_result = map_repository.upsert(
                    map_id=data["map_id"], name=data["name"]
                )
                if map_result:
                    log.info(f"Создана/обновлена карта: {map_result['name']}")

        for filename in os.listdir(teams_dir):
            if filename.endswith(".json"):
                with open(
                    os.path.join(teams_dir, filename), "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)
                team_result = team_repository.upsert(
                    team_id=data["team_id"], name=data["name"]
                )
                if team_result:
                    log.info(f"Создана/обновлена команда: {team_result['name']}")

        for filename in os.listdir(players_dir):
            if filename.endswith(".json"):
                with open(
                    os.path.join(players_dir, filename), "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)
                player_result = player_repository.upsert(
                    player_id=data["player_id"], name=data["name"]
                )
                if player_result:
                    log.info(f"Создан/обновлен игрок: {player_result['name']}")

        log.info("Загрузка словарей в БД завершена.")

    return task


def make_train_model_task(
    train_model_func: Callable,
    games_raw_dir: str = settings.GAMES_RAW_DIR,
    ml_results_dir: str = settings.ML_RESULTS_DIR,
    test_size: float = settings.TEST_SIZE,
    n_splits: int = settings.N_SPLITS,
    random_state: int = settings.RANDOM_STATE,
    ml_pipeline_repository: MLPipelineRepository = ml_result_repo,
    ml_pipeline_metrics_repository: MLPipelineMetricRepository = ml_result_metrics_repo,
<<<<<<< HEAD
    task_name: str = "backend.tasks.train_model_task",
=======
>>>>>>> origin/main
) -> Callable[..., dict]:
    @shared_task(name=task_name)
    def task(*args, **kwargs) -> dict:
        try:
            log.info("Начало обучения ML модели...")
            ml_pipeline, metrics = train_model_func(
                games_raw_dir=games_raw_dir,
                test_size=test_size,
                n_splits=n_splits,
                random_state=random_state,
            )
            log.info("Обучение модели завершено успешно.")

            os.makedirs(ml_results_dir, exist_ok=True)
            pipeline_id = str(uuid.uuid4())

            pipeline_path = Path(ml_results_dir) / f"{pipeline_id}.joblib"
            metrics_path = Path(ml_results_dir) / f"{pipeline_id}.json"

            joblib.dump(ml_pipeline, pipeline_path)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            log.info(f"ML модель сохранена: {pipeline_path}")
            log.info(f"Метрики модели сохранены: {metrics_path}")

<<<<<<< HEAD
=======
            # Используем новые репозитории
>>>>>>> origin/main
            ml_result: Optional[dict] = ml_pipeline_repository.upsert(
                pipeline_file=pipeline_path, metrics_file=metrics_path
            )

            if ml_result:
                ml_pipeline_metrics_repository.upsert(
                    ml_pipeline_id=UUID(ml_result["ml_pipeline_id"]),
                    roc_auc=metrics["roc_auc"],
                    f1=metrics["f1"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    accuracy=metrics["accuracy"],
                    tp=metrics["tp"],
                    tn=metrics["tn"],
                    fp=metrics["fp"],
                    fn=metrics["fn"],
                )
                log.info(
                    f"Метрики модели сохранены в БД для MLPipeline ID: {ml_result['ml_pipeline_id']}"
                )

            return {
                "pipeline_id": pipeline_id,
                "metrics": metrics,
                "db_id": ml_result["ml_pipeline_id"] if ml_result else None,
            }

        except Exception as e:
            log.error(f"Ошибка при обучении модели: {e}")
            raise e

    return task
