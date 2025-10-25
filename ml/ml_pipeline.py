from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ml.data_loader import get_game_ids, get_X_y
from ml.feature_extractors import (
    BagEncoder,
    ColumnSelectorArray,
    PlayerEloEncoder,
    PlayerStatisticSumExtractor,
)
from ml.metrics import get_metrics
from ml.stacker import MLStacker, OOFPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ml_pipeline(
    data_path: str = "data/games_raw",
    test_size: int = 100,
    n_splits: int = 10,
    n_iters: int = 10,
    random_state: int = 42,
) -> tuple[MLStacker, dict]:
    game_ids = get_game_ids(data_path)
    game_ids_train = game_ids[:-test_size]
    game_ids_test = game_ids[-test_size:]

    X_train, y_train = get_X_y(game_ids_train, path_to_dir=data_path)
    X_test, y_test = get_X_y(game_ids_test, path_to_dir=data_path)

    map_col = [0]
    team_cols = [1, 2]
    player_cols = list(range(3, 13))
    player_stats_keys = ["kills", "deaths", "assists", "flash_assists", "headshots"]

    map_pipeline = (
        "map_features",
        Pipeline(
            [
                ("select", ColumnSelectorArray(map_col)),
                ("onehot", OneHotEncoder(sparse_output=False)),
            ]
        ),
    )

    team_pipeline = (
        "team_features",
        Pipeline(
            [
                ("select", ColumnSelectorArray(team_cols)),
                ("bag", BagEncoder()),
            ]
        ),
    )

    player_pipeline = (
        "player_features",
        Pipeline(
            [
                ("select", ColumnSelectorArray(player_cols)),
                ("bag", BagEncoder()),
            ]
        ),
    )

    player_elo_pipeline = (
        "player_elo_features",
        Pipeline(
            [
                ("select", ColumnSelectorArray(player_cols)),
                ("elo", PlayerEloEncoder(k_factor=32, base_elo=1000)),
                ("scale", MinMaxScaler()),
            ]
        ),
    )

    player_stats_pipelines = [
        (
            f"player_stat_{key}",
            Pipeline(
                [
                    ("select", ColumnSelectorArray(player_cols)),
                    (
                        "stat",
                        PlayerStatisticSumExtractor(game_ids=game_ids_train, key=key),
                    ),
                    ("scale", MinMaxScaler()),
                ]
            ),
        )
        for key in player_stats_keys
    ]

    all_pipelines = [
        map_pipeline,
        team_pipeline,
        player_pipeline,
        player_elo_pipeline,
    ] + player_stats_pipelines

    oof_predictor = OOFPredictor(n_splits=n_splits, random_state=random_state)
    ml_pipeline = MLStacker(
        all_pipelines,
        oof_predictor=oof_predictor,
        n_iters=n_iters,
        random_state=random_state,
    )

    ml_pipeline.fit(X_train, y_train)
    y_pred_test_proba = ml_pipeline.predict_proba(X_test)
    metrics = get_metrics(y_test, y_pred_test_proba)

    return ml_pipeline, metrics


if __name__ == "__main__":
    load_dotenv()

    DATA_PATH = os.getenv("DATA_PATH", "data/games_raw")
    ML_RESULTS_PATH = os.getenv("ML_RESULTS_PATH", "data/ml_results")
    TEST_SIZE = int(os.getenv("TEST_SIZE", 100))
    N_SPLITS = int(os.getenv("N_SPLITS", 10))
    N_ITERS = int(os.getenv("N_ITERS", 10))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

    Path(ML_RESULTS_PATH).mkdir(parents=True, exist_ok=True)

    pipeline, metrics = run_ml_pipeline(
        data_path=DATA_PATH,
        test_size=TEST_SIZE,
        n_splits=N_SPLITS,
        n_iters=N_ITERS,
        random_state=RANDOM_STATE,
    )

    pipeline_uuid = str(uuid.uuid4())
    metrics_path = Path(ML_RESULTS_PATH) / f"{pipeline_uuid}.json"
    pipeline_path = Path(ML_RESULTS_PATH) / f"{pipeline_uuid}.joblib"

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=4)
    joblib.dump(pipeline, pipeline_path)

    logger.info("Pipeline saved to %s", pipeline_path)
    logger.info("Metrics saved to %s", metrics_path)
    logger.info("Metrics: %s", json.dumps(metrics, indent=4))
