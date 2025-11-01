import logging
from pathlib import Path
from uuid import uuid4

from celery import shared_task
from django.conf import settings

log = logging.getLogger(__name__)


@shared_task(name="update_dictionaries_task")
def update_dictionaries_task():
    from backend.di import dictionary_service

    log.info("Запуск обновления словарей")
    dictionary_service.load_dictionaries_from_json(Path(settings.PATH_TO_GAMES_RAW_DIR))
    log.info("Обновление словарей завершено")


@shared_task(name="train_model_task")
def train_model_task():
    from backend.di import (
        train_metric_repository,
        train_result_repository,
        train_test_split_repository,
    )
    from ml.train import train

    path_to_games = Path(settings.PATH_TO_GAMES_RAW_DIR)
    models_dir = Path(settings.PATH_TO_TRAINED_MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Начало обучения модели. Игры: {path_to_games}")

    (
        model,
        metrics,
        path_to_model,
        split_hash,
        game_ids_train,
        game_ids_test,
        begin_at_min,
        begin_at_max,
    ) = train(path_to_games, models_dir)

    if model is None:
        log.warning("Обучение модели не выполнено")
        return

    split_obj_dict = train_test_split_repository.save(
        train_test_split_hash=split_hash,
        game_ids_train=game_ids_train,
        game_ids_test=game_ids_test,
        begin_at_min=begin_at_min,
        begin_at_max=begin_at_max,
    )
    if not split_obj_dict:
        log.error("Ошибка сохранения сплита")
        return
    log.info(f"Сплит сохранён: {split_hash}")

    train_result_id = uuid4()
    train_result_dict = train_result_repository.save(
        train_result_id=train_result_id,
        train_test_split_hash=split_hash,
        path_to_model=str(path_to_model),
    )
    if not train_result_dict:
        log.error("Ошибка сохранения результатов обучения")
        return
    log.info(f"Результаты обучения сохранены: train_result_id={train_result_id}")

    train_metric_id = uuid4()
    metric_dict = train_metric_repository.save(
        train_metric_id=train_metric_id,
        train_result_id=train_result_id,
        auc=metrics.get("auc"),
        f1=metrics.get("f1"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        accuracy=metrics.get("accuracy"),
        tp=metrics.get("tp"),
        tn=metrics.get("tn"),
        fp=metrics.get("fp"),
        fn=metrics.get("fn"),
    )
    if metric_dict:
        log.info(f"Метрики сохранены для train_result_id={train_result_id}")
    else:
        log.error("Ошибка сохранения метрик")

    log.info("Задача обучения завершена")


@shared_task(name="inference_trained_model")
def inference_trained_model(X: list[int]) -> dict:
    from backend.di import forecaster_service

    try:
        log.info("Запуск прогнозирования через ForecasterService")
        forecast_id = forecaster_service.forecast(
            map_id=X[0],
            team1_id=X[1],
            team2_id=X[2],
            team1_player1_id=X[3],
            team1_player2_id=X[4],
            team1_player3_id=X[5],
            team1_player4_id=X[6],
            team1_player5_id=X[7],
            team2_player1_id=X[8],
            team2_player2_id=X[9],
            team2_player3_id=X[10],
            team2_player4_id=X[11],
            team2_player5_id=X[12],
        )
        log.info(f"Прогноз поставлен в очередь, forecast_id={forecast_id}")
        return {"forecast_id": forecast_id}
    except Exception as e:
        log.error(f"Ошибка прогнозирования: {e}")
        return {"error": str(e)}
