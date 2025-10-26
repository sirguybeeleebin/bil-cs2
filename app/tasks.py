import json
import logging
import os
import uuid
from pathlib import Path
import httpx
import joblib
from celery import shared_task

from app.di import (
    map_repository,
    ml_pipeline_metrics_repository,
    ml_pipeline_repository,
    player_repository,
    prediction_repository,
    team_repository,
)
from app.dictionaries.dictionaries import (
    get_map_dict,
    get_players_dicts,
    get_teams_dicts,
)
from app.ml.ml_pipeline import run_ml_pipeline
from app.ws import send_prediction_to_ws
from config.settings import PATH_TO_ML_RESULTS_DIR

log = logging.getLogger(__name__)


@shared_task
def fill_dictionaries_task(path_to_games_raw_dir: str):
    if not os.path.exists(path_to_games_raw_dir):
        log.error(f"Директория {path_to_games_raw_dir} не существует")
        return {"status": "ошибка", "message": f"{path_to_games_raw_dir} не существует"}

    log.info(f"Начало загрузки данных из {path_to_games_raw_dir}")
    filenames = os.listdir(path_to_games_raw_dir)
    total = len(filenames)
    for idx, filename in enumerate(filenames):
        log.info(f"Обработка файла {idx + 1}/{total}: {filename}")
        file_path = os.path.join(path_to_games_raw_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            map_data = get_map_dict(data)
            if map_data:
                result = map_repository.upsert(map_data)
                if not result:
                    log.error(f"Не удалось создать/обновить карту: {map_data}")

            teams = get_teams_dicts(data)
            for team in teams:
                result = team_repository.upsert(team)
                if not result:
                    log.error(f"Не удалось создать/обновить команду: {team}")

            players = get_players_dicts(data)
            for player in players:
                result = player_repository.upsert(player)
                if not result:
                    log.error(f"Не удалось создать/обновить игрока: {player}")

        except (json.JSONDecodeError, FileNotFoundError, TypeError, KeyError) as e:
            log.error(f"Ошибка при обработке файла {file_path}: {e}")
            continue

    log.info("Загрузка данных завершена")
    return {"status": "готово"}


@shared_task(name="run_ml_pipeline_task")
def run_ml_pipeline_task(    
    path_to_games_raw_dir: str,
    test_size: int = 100,
    n_splits: int = 10,
    n_iters: int = 10,
    random_state: int = 42,
):
    log.info("Запуск ML пайплайна")

    try:
        pipeline_id = uuid.uuid4()

        ml_pipeline, metrics = run_ml_pipeline(
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
        log.info(f"Файлы пайплайна и метрик сохранены: {pipeline_file}, {metrics_file}")

        pipeline = ml_pipeline_repository.upsert(
            {"pipeline_id": pipeline_id, "path_to_pipeline_file": str(pipeline_file)}
        )
        if not pipeline:
            log.error(f"Не удалось сохранить ML пайплайн в БД: {pipeline_file}")
            raise RuntimeError("ML pipeline upsert failed")

        metrics = ml_pipeline_metrics_repository.upsert(
            {
                "metrics_id": uuid.uuid4(),
                "pipeline_id": pipeline_id,
                "path_to_metrics_file": str(metrics_file),
            }
        )
        if not metrics:
            log.error(f"Не удалось сохранить метрики ML пайплайна в БД: {metrics_file}")
            raise RuntimeError("ML pipeline metrics upsert failed")

        log.info("ML пайплайн успешно выполнен и сохранён")
        return {"status": "готово", "pipeline_id": str(pipeline_id)}

    except Exception as e:
        log.exception("Ошибка при выполнении ML пайплайна")
        return {"status": "ошибка", "message": str(e)}


@shared_task
def run_inference(prediction_id: str):
    try:
        prediction = prediction_repository.get_by_id(prediction_id)
        if not prediction:
            log.error(f"Prediction with id {prediction_id} not found")
            return {"status": "ошибка", "message": "Prediction not found"}
        send_prediction_to_ws(prediction)
        log.info(f"Prediction {prediction_id} sent to WebSocket")
        return {"status": "готово"}
    except Exception as e:
        log.error(f"Ошибка при выполнении inference для {prediction_id}: {e}")
        return {"status": "ошибка", "message": str(e)}


@shared_task
def send_log_to_fluentd(client: httpx.Client, fluentd_url: str, log_data: dict):
    try:
        client.post(fluentd_url, json=log_data)
    except Exception as e:
        log.error(f"Ошибка при логировании запроса и ответа {log_data['request_id']}: {e}")
        
    

