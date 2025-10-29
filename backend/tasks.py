import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import joblib
from celery import chain, shared_task
from dateutil.parser import parse
from django.conf import settings

from backend.di import (
    map_repo,
    ml_result_metrics_repo,
    ml_result_repo,
    player_repo,
    team_repo,
)
from train_model.train_model import train_model

log = logging.getLogger(__name__)


def _validate_game(game: dict[str, Any]) -> bool:
    try:
        parse(game["begin_at"])
        int(game["map"]["id"])
        team_players: dict[Any, list[Any]] = defaultdict(list)
        for p in game["players"]:
            team_players[p["team"]["id"]].append(p["player"]["id"])
        if len(team_players) != 2:
            return False
        for _, p_ids in team_players.items():
            if len(set(p_ids)) != 5:
                return False
        t1_id, t2_id = list(team_players.keys())
        rounds: list[int] = []
        for r in game["rounds"]:
            if r["round"] is None:
                continue
            if r["winner_team"] not in (t1_id, t2_id):
                return False
            rounds.append(r["round"])
        if min(rounds, default=1) != 1 or max(rounds, default=0) < 16:
            return False
        return True
    except Exception as e:
        log.warning(f"Ошибка при валидации игры {game.get('id', 'неизвестно')}: {e}")
        return False


@shared_task(name="backend.tasks.update_dictionaries_task")
def update_dictionaries_task():
    log.info("Запуск задачи обновления словарей...")
    games_raw_dir = settings.GAMES_RAW_DIR
    maps_dir = settings.MAPS_DIR
    teams_dir = settings.TEAMS_DIR
    players_dir = settings.PLAYERS_DIR
    for directory in [maps_dir, teams_dir, players_dir]:
        os.makedirs(directory, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    for filename in os.listdir(games_raw_dir):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(games_raw_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.error(f"Ошибка чтения {filename}: {e}")
            continue
        if not _validate_game(data):
            continue
        map_data = {
            "map_id": data["map"]["id"],
            "name": data["map"]["name"],
            "updated_at": now_iso,
        }
        map_file = os.path.join(maps_dir, f"{map_data['map_id']}.json")
        with open(map_file, "w", encoding="utf-8") as f:
            json.dump(map_data, f, ensure_ascii=False, indent=2)
        for p in data["players"]:
            team_data = {
                "team_id": p["team"]["id"],
                "name": p["team"]["name"],
                "updated_at": now_iso,
            }
            team_file = os.path.join(teams_dir, f"{team_data['team_id']}.json")
            with open(team_file, "w", encoding="utf-8") as f:
                json.dump(team_data, f, ensure_ascii=False, indent=2)
        for p in data["players"]:
            player_data = {
                "player_id": p["player"]["id"],
                "name": p["player"]["name"],
                "updated_at": now_iso,
            }
            player_file = os.path.join(players_dir, f"{player_data['player_id']}.json")
            with open(player_file, "w", encoding="utf-8") as f:
                json.dump(player_data, f, ensure_ascii=False, indent=2)
    log.info("Задача обновления словарей завершена.")


@shared_task(name="backend.tasks.load_dictionaries_task")
def load_dictionaries_task():
    log.info("Запуск задачи загрузки словарей в БД...")
    for filename in os.listdir(settings.MAPS_DIR):
        if filename.endswith(".json"):
            with open(
                os.path.join(settings.MAPS_DIR, filename), "r", encoding="utf-8"
            ) as f:
                data = json.load(f)
            map_repo.upsert(map_id=data["map_id"], name=data["name"])
    for filename in os.listdir(settings.TEAMS_DIR):
        if filename.endswith(".json"):
            with open(
                os.path.join(settings.TEAMS_DIR, filename), "r", encoding="utf-8"
            ) as f:
                data = json.load(f)
            team_repo.upsert(team_id=data["team_id"], name=data["name"])
    for filename in os.listdir(settings.PLAYERS_DIR):
        if filename.endswith(".json"):
            with open(
                os.path.join(settings.PLAYERS_DIR, filename), "r", encoding="utf-8"
            ) as f:
                data = json.load(f)
            player_repo.upsert(player_id=data["player_id"], name=data["name"])
    log.info("Загрузка словарей завершена.")


@shared_task(name="backend.tasks.update_and_load_dictionaries_task")
def update_and_load_dictionaries_task():
    try:
        log.info("Запуск цепочки задач: обновление словарей -> загрузка в БД")
        workflow = chain(update_dictionaries_task.si(), load_dictionaries_task.si())
        workflow.apply_async()
        return {"status": "Цепочка задач запущена"}
    except Exception as e:
        log.exception(f"Ошибка при запуске цепочки задач: {e}")
        raise


@shared_task(name="backend.tasks.train_model_task")
def train_model_task():
    try:
        log.info("Начало обучения ML модели...")

        ml_pipeline, metrics = train_model(
            games_raw_dir=settings.GAMES_RAW_DIR,
            test_size=settings.TEST_SIZE,
            n_splits=settings.N_SPLITS,
            random_state=settings.RANDOM_STATE,
        )

        pipeline_id = str(uuid.uuid4())

        ml_results_dir = Path(settings.ML_RESULTS_DIR)
        ml_results_dir.mkdir(parents=True, exist_ok=True)

        pipeline_path = ml_results_dir / f"{pipeline_id}.joblib"
        metrics_path = ml_results_dir / f"{pipeline_id}.json"
        joblib.dump(ml_pipeline, pipeline_path)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        ml_result = ml_result_repo.upsert(
            ml_pipeline_id=pipeline_id,
            pipeline_file_path=str(pipeline_path),
            metrics_file_path=str(metrics_path),
        )

        if ml_result:
            ml_result_metrics_repo.upsert(
                ml_pipeline_metric_id=uuid4(),
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

        return {
            "pipeline_id": pipeline_id,
            "metrics": metrics,
        }

    except Exception:
        log.exception("Ошибка при обучении модели")
        raise


@lru_cache(maxsize=1)
def _load_latest_pipeline():
    latest_pipeline = ml_result_repo.get_latest_pipeline()
    if not latest_pipeline:
        raise ValueError("ML pipeline not found")

    pipeline_path = Path(latest_pipeline["pipeline_file_path"])
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline file not found at {pipeline_path}")

    log.info(f"Loading ML pipeline from {pipeline_path}")
    return joblib.load(pipeline_path)


@shared_task(name="backend.tasks.ml_forecast_inference_task", bind=True)
def ml_forecast_inference_task(self, forecast_input: dict[str, Any]) -> dict[str, Any]:
    try:
        ml_pipeline = _load_latest_pipeline()

        team_players = defaultdict(list)
        team_players[forecast_input["team1_id"]].extend(
            [
                forecast_input["team1_player1_id"],
                forecast_input["team1_player2_id"],
                forecast_input["team1_player3_id"],
                forecast_input["team1_player4_id"],
                forecast_input["team1_player5_id"],
            ]
        )
        team_players[forecast_input["team2_id"]].extend(
            [
                forecast_input["team2_player1_id"],
                forecast_input["team2_player2_id"],
                forecast_input["team2_player3_id"],
                forecast_input["team2_player4_id"],
                forecast_input["team2_player5_id"],
            ]
        )

        sorted_team_ids = sorted(team_players.keys())
        features = [forecast_input["map_id"], *sorted_team_ids]

        for team_id in sorted_team_ids:
            sorted_players = sorted(team_players[team_id])
            features.extend(sorted_players)

        result = ml_pipeline.predict_proba([features])[0]

        if sorted_team_ids[0] == forecast_input["team1_id"]:
            team1_prob, team2_prob = result[0], result[1]
        else:
            team1_prob, team2_prob = result[1], result[0]

        return {
            "task_id": self.request.id,
            "team1_id": sorted_team_ids[0],
            "team2_id": sorted_team_ids[1],
            "team1_win_probability": float(team1_prob),
            "team2_win_probability": float(team2_prob),
        }

    except Exception as e:
        log.exception(f"Ошибка при прогнозе ML: {e}")
        raise
