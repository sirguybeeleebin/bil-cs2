import os
import json
import uuid
import logging
import joblib
from dotenv import load_dotenv
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from train_model.data_loader import get_game_ids, get_X_y
from train_model.feature_extractors import (
    BagEncoder,
    ColumnSelectorArray,
    PlayerEloEncoder,
    PlayerStatisticSumExtractor,
    PlayerMapStatisticSumExtractor,
)
from train_model.feature_selectors import LogitL1FeatureSelector
from train_model.metrics import get_metrics
from train_model.stacker import MLStacker, OOFPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def main():    
    load_dotenv()

    GAMES_RAW_DIR = os.getenv("GAMES_RAW_DIR", "data/games_raw")
    ML_RESULTS_DIR = os.getenv("ML_RESULTS_DIR", "data/ml_results")
    TEST_SIZE = int(os.getenv("TEST_SIZE", 100))
    N_SPLITS = int(os.getenv("N_SPLITS", 10))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
    
    TASK_ID = str(uuid.uuid4())

    log.info("Конфигурация:")
    log.info(f"GAMES_RAW_DIR  = {GAMES_RAW_DIR}")
    log.info(f"ML_RESULTS_DIR = {ML_RESULTS_DIR}")
    log.info(f"TEST_SIZE      = {TEST_SIZE}")
    log.info(f"N_SPLITS       = {N_SPLITS}")
    log.info(f"RANDOM_STATE   = {RANDOM_STATE}")
    log.info(f"TASK_ID        = {TASK_ID}")

    log.info("Начало обучения ML модели")

    game_ids = get_game_ids(GAMES_RAW_DIR)
    game_ids_train = game_ids[:-TEST_SIZE]
    game_ids_test = game_ids[-TEST_SIZE:]

    X_train, y_train = get_X_y(game_ids_train, path_to_dir=GAMES_RAW_DIR)
    X_test, y_test = get_X_y(game_ids_test, path_to_dir=GAMES_RAW_DIR)

    map_col = [0]
    team_cols = [1, 2]
    player_cols = list(range(3, 13))
    player_stats_keys = ["kills", "deaths", "assists", "flash_assists", "headshots"]

    map_pipeline = (
        "map_features",
        Pipeline([
            ("select", ColumnSelectorArray(map_col)),
            ("onehot", OneHotEncoder(sparse_output=False)),
            ("l1_select", LogitL1FeatureSelector()),
        ])
    )

    team_pipeline = (
        "team_features",
        Pipeline([
            ("select", ColumnSelectorArray(team_cols)),
            ("bag", BagEncoder()),
            ("l1_select", LogitL1FeatureSelector()),
        ])
    )

    player_pipeline = (
        "player_features",
        Pipeline([
            ("select", ColumnSelectorArray(player_cols)),
            ("bag", BagEncoder()),
            ("l1_select", LogitL1FeatureSelector()),
        ])
    )

    player_elo_pipeline = (
        "player_elo_features",
        Pipeline([
            ("select", ColumnSelectorArray(player_cols)),
            ("elo", PlayerEloEncoder()),
            ("scale", MinMaxScaler()),
            ("l1_select", LogitL1FeatureSelector()),
        ])
    )

    player_stats_pipelines = [
        (f"player_stat_{key}",
         Pipeline([
             ("select", ColumnSelectorArray(player_cols)),
             ("stat", PlayerStatisticSumExtractor(game_ids=game_ids_train, key=key)),
             ("scale", MinMaxScaler()),
             ("l1_select", LogitL1FeatureSelector()),
         ])
        )
        for key in player_stats_keys
    ]   
    

    player_map_stats_pipelines = [
        (f"player_map_stat_{key}",
        Pipeline([
            ("select", ColumnSelectorArray(map_col+player_cols)),  
            ("stat", PlayerMapStatisticSumExtractor(game_ids=game_ids_train, key=key)),
            ("scale", MinMaxScaler()),
            ("l1_select", LogitL1FeatureSelector()),
        ])
        )
        for key in player_stats_keys
    ]

    all_pipelines = [
        map_pipeline,
        team_pipeline,
        player_pipeline,
        player_elo_pipeline,
    ]
    all_pipelines += player_stats_pipelines 
    all_pipelines += player_map_stats_pipelines

    ml_pipeline = MLStacker(
        all_pipelines,
        oof_predictor=OOFPredictor(n_splits=N_SPLITS, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
    )

    log.info("Обучение модели...")
    ml_pipeline.fit(X_train, y_train)

    log.info("Предсказание на тесте...")
    y_pred_test_proba = ml_pipeline.predict_proba(X_test)
    metrics = get_metrics(y_test, y_pred_test_proba)

    log.info("Метрики модели:\n%s", json.dumps(metrics, ensure_ascii=False, indent=4))

    os.makedirs(ML_RESULTS_DIR, exist_ok=True)

    model_path = os.path.join(ML_RESULTS_DIR, f"{TASK_ID}.joblib")
    metrics_path = os.path.join(ML_RESULTS_DIR, f"{TASK_ID}.json")

    joblib.dump(ml_pipeline, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    log.info(f"Модель сохранена: {model_path}")
    log.info(f"Метрики сохранены: {metrics_path}")
    log.info("Обучение завершено успешно")


if __name__ == "__main__":
    main()
