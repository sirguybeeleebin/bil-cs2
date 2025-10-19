import warnings
import argparse
import os
import hashlib
import json
import pickle
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from dotenv import load_dotenv

from ml.feature_extraction import (
    ColumnSelector,
    PlayerBagEncoder,
    PlayerEloEncoder,
    TeamBagEncoder,
)
from ml.feature_selection import RecursiveL1Selector
from ml.load_data import get_game_ids, get_X_y
from ml.metrics import get_metrics

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def get_settings():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_file", type=str, default=".env", help="Path to .env file"
    )
    args, _ = parser.parse_known_args()
    load_dotenv(args.env_file)
    settings = {
        "PATH_TO_GAMES_RAW": os.getenv("PATH_TO_GAMES_RAW_DIR", "data/games_raw"),
        "TEST_SIZE": int(os.getenv("TEST_SIZE", 100)),
        "RANDOM_STATE": int(os.getenv("RANDOM_STATE", 42)),
        "PATH_TO_ML_RESULTS": os.getenv("PATH_TO_ML_RESULTS", "data/ml"),
        "N_SPLITS": int(os.getenv("N_SPLITS", 10)),
    }
    return settings


def run(
    path_to_game_raw_dir: str = "data/games_raw",
    path_to_results_dir: str = "data/ml",
    test_size: int = 100,
    n_splits: int = 10,
    l1_c: float = 1.0,
    logreg_c: float = 1.0,
    random_state: int = 42,
):
    log.info("Loading game IDs...")
    game_ids = get_game_ids(path_to_game_raw_dir)
    game_ids_train, game_ids_test = game_ids[:-test_size], game_ids[-test_size:]
    log.info(f"Total games: {len(game_ids)}, train: {len(game_ids_train)}, test: {len(game_ids_test)}")

    X_train, y_train = get_X_y(path_to_game_raw_dir, game_ids_train)
    X_test, y_test = get_X_y(path_to_game_raw_dir, game_ids_test)

    ONE_HOT_COLS = [0, 1]
    TEAM_COLS = [2, 3]
    PLAYER_COLS = list(range(4, X_train.shape[1]))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    log.info("Building pipeline...")
    pipeline = Pipeline(
        [
            (
                "encoder",
                FeatureUnion(
                    [
                        (
                            "one_hot",
                            Pipeline(
                                [
                                    ("select_cols", ColumnSelector(ONE_HOT_COLS)),
                                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                        ),
                        (
                            "team_bag",
                            Pipeline(
                                [
                                    ("select_teams", ColumnSelector(TEAM_COLS)),
                                    ("team_encoder", TeamBagEncoder()),
                                ]
                            ),
                        ),
                        (
                            "player_bag",
                            Pipeline(
                                [
                                    ("select_players", ColumnSelector(PLAYER_COLS)),
                                    ("player_encoder", PlayerBagEncoder()),
                                ]
                            ),
                        ),
                        (
                            "player_elo",
                            Pipeline(
                                [
                                    ("select_players", ColumnSelector(PLAYER_COLS)),
                                    ("elo_encoder", PlayerEloEncoder()),
                                    ("scaler", MinMaxScaler()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("l1", RecursiveL1Selector(C=l1_c, cv=tscv)),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear", C=logreg_c, random_state=random_state
                ),
            ),
        ]
    )

    log.info("Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = get_metrics(y_test, y_test_pred, y_test_proba)

    log.info(f"Creating results directory: {path_to_results_dir}")
    os.makedirs(path_to_results_dir, exist_ok=True)

    log.info("Generating hash ID for game_ids...")
    game_ids_bytes = ",".join(map(str, game_ids)).encode()
    hash_id = hashlib.md5(game_ids_bytes).hexdigest()
    log.info(f"Hash ID: {hash_id}")

    metrics_path = os.path.join(path_to_results_dir, f"{hash_id}.json")
    log.info(f"Saving metrics to {metrics_path}...")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    pipeline_path = os.path.join(path_to_results_dir, f"{hash_id}.pickle")
    log.info(f"Saving pipeline to {pipeline_path}...")
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    log.info("Results saved successfully.")
    
    return metrics


def main():
    settings = get_settings()
    metrics = run(
        path_to_game_raw_dir=settings["PATH_TO_GAMES_RAW"],
        path_to_results_dir=settings["PATH_TO_ML_RESULTS"],
        test_size=settings["TEST_SIZE"],
        n_splits=settings["N_SPLITS"],
        random_state=settings["RANDOM_STATE"],
    )

    log.info(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
