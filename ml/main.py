import warnings
import argparse
import os
import hashlib
import json
import pickle
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from dotenv import load_dotenv

from ml.feature_extraction import (
    ColumnSelector,
    PlayerBagEncoder,
    PlayerEloEncoder,
    PlayerMapEloEncoder,
    TeamBagEncoder,
    TimeFeatureExtractor,
    PlayerKillsSumFeatureExtractor,
)
from ml.feature_selection import RecursiveL1Selector, RecursiveCVFeatureSelection
from ml.load_data import get_game_ids, get_X_y
from ml.metrics import get_metrics

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# =============================================
# Settings
# =============================================
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
    random_state: int = 42,
):
    log.info("Загрузка ID игр...")
    game_ids = get_game_ids(path_to_game_raw_dir)
    game_ids_train, game_ids_test = game_ids[:-test_size], game_ids[-test_size:]
    log.info(f"Всего игр: {len(game_ids)}, обучение: {len(game_ids_train)}, тест: {len(game_ids_test)}")

    X_train, y_train = get_X_y(path_to_game_raw_dir, game_ids_train)
    X_test, y_test = get_X_y(path_to_game_raw_dir, game_ids_test)

    BEGIN_AT_COL = [0]  
    MAP_ID_COL = [1]  
    ONE_HOT_COLS = [1, 2]
    TEAM_COLS = [3, 4]
    PLAYER_COLS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]    
    
    tscv = TimeSeriesSplit(n_splits=n_splits)

    log.info("Построение пайплайна признаков и селекции...")
    feature_pipeline = Pipeline(
        [
            (
                "encoder",
                FeatureUnion(
                    [
                        ("one_hot", Pipeline([
                            ("select_cols", ColumnSelector(ONE_HOT_COLS)),
                            ("ohe", OneHotEncoder(handle_unknown="ignore"))
                        ])),
                        ("team_bag", Pipeline([
                            ("select_teams", ColumnSelector(TEAM_COLS)),
                            ("team_encoder", TeamBagEncoder())
                        ])),
                        ("player_bag", Pipeline([
                            ("select_players", ColumnSelector(PLAYER_COLS)),
                            ("player_encoder", PlayerBagEncoder())
                        ])),
                        ("player_elo", Pipeline([
                            ("select_players", ColumnSelector(PLAYER_COLS)),
                            ("elo_encoder", PlayerEloEncoder()),
                            ("scaler", MinMaxScaler())
                        ])),
                        ("player_map_elo", Pipeline([
                            ("select_players", ColumnSelector(MAP_ID_COL + PLAYER_COLS)),
                            ("elo_encoder", PlayerMapEloEncoder()),
                            ("scaler", MinMaxScaler())
                        ])),                        
                        ("time_features", Pipeline([
                            ("select_time", ColumnSelector(BEGIN_AT_COL)),
                            ("time_feat", TimeFeatureExtractor()),
                            ("time_union", FeatureUnion([
                                ("timestamp_scaled", Pipeline([
                                    ("select_ts", ColumnSelector([0])),
                                    ("scaler", MinMaxScaler())
                                ])),
                                ("time_ohe", ColumnTransformer([
                                    ("year", OneHotEncoder(handle_unknown="ignore"), [1]),
                                    ("month", OneHotEncoder(handle_unknown="ignore"), [2]),
                                    ("day", OneHotEncoder(handle_unknown="ignore"), [3]),
                                    ("weekday", OneHotEncoder(handle_unknown="ignore"), [4]),
                                    ("hour", OneHotEncoder(handle_unknown="ignore"), [5]),
                                ], remainder="drop")),
                            ])),
                        ])),                        
                    ]
                ),
            ),
            ("l1", RecursiveL1Selector(C=l1_c, cv=tscv)),
            ("recursive_cv", RecursiveCVFeatureSelection(tscv=tscv, verbose=2))
        ]
    )

    log.info("Обучение feature pipeline (кодировщики + L1 + RecursiveCV)...")
    feature_pipeline.fit(X_train, y_train)

    # Use only the selected features from RFECV
    X_train_final = feature_pipeline.transform(X_train)
    X_test_final = feature_pipeline.transform(X_test)

    log.info(f"Размер после рекурсивного отбора признаков: {X_train_final.shape}")

    # ===== Final Logistic Regression with GridSearchCV =====
    param_grid = {"C": np.logspace(-2, 2, 50)}
    grid_search_final = GridSearchCV(
        estimator=LogisticRegression(solver="liblinear", random_state=random_state),
        param_grid=param_grid,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2
    )
    grid_search_final.fit(X_train_final, y_train)
    final_clf = grid_search_final.best_estimator_

    log.info(f"Лучший C для финальной логистической регрессии: {grid_search_final.best_params_['C']}")
    log.info(f"CV ROC AUC: {grid_search_final.best_score_:.4f}")

    y_test_proba = final_clf.predict_proba(X_test_final)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    metrics = get_metrics(y_test, y_test_pred, y_test_proba)
    log.info(f"Метрики на тесте: {metrics}")

    os.makedirs(path_to_results_dir, exist_ok=True)
    game_ids_bytes = ",".join(map(str, game_ids)).encode()
    hash_id = hashlib.md5(game_ids_bytes).hexdigest()

    metrics_path = os.path.join(path_to_results_dir, f"{hash_id}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    pipeline_path = os.path.join(path_to_results_dir, f"{hash_id}.pickle")
    with open(pipeline_path, "wb") as f:
        pickle.dump(
            {
                "feature_pipeline": feature_pipeline,
                "final_clf": final_clf,
            },
            f,
        )

    log.info("Результаты успешно сохранены.")
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
