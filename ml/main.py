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
from sklearn.feature_selection import RFECV
import numpy as np
from dotenv import load_dotenv

from ml.feature_extraction import (
    ColumnSelector,
    PlayerBagEncoder,
    PlayerEloEncoder,
    TeamBagEncoder,
    TimeFeatureExtractor,
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
    random_state: int = 42,
):
    log.info("Загрузка ID игр...")
    game_ids = get_game_ids(path_to_game_raw_dir)
    game_ids_train, game_ids_test = game_ids[:-test_size], game_ids[-test_size:]
    log.info(f"Всего игр: {len(game_ids)}, обучение: {len(game_ids_train)}, тест: {len(game_ids_test)}")

    X_train, y_train = get_X_y(path_to_game_raw_dir, game_ids_train)
    X_test, y_test = get_X_y(path_to_game_raw_dir, game_ids_test)

    BEGIN_AT_COL = [0]
    ONE_HOT_COLS = [1, 2]
    TEAM_COLS = [3, 4]
    PLAYER_COLS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    log.info("Построение пайплайна признаков и L1 селектора...")
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
        ]
    )

    log.info("Обучение feature pipeline (кодировщики + L1 селектор)...")
    feature_pipeline.fit(X_train, y_train)
    X_train_sel = feature_pipeline.transform(X_train)
    X_test_sel = feature_pipeline.transform(X_test)
    log.info(f"Размер после L1 селекции: {X_train_sel.shape}")

    log.info("Запуск RFECV для отбора признаков...")
    base_logreg = LogisticRegression(solver="liblinear", random_state=random_state)
    rfecv = RFECV(
        estimator=base_logreg,
        step=1,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2,
    )
    rfecv.fit(X_train_sel, y_train)
    X_train_final = rfecv.transform(X_train_sel)
    X_test_final = rfecv.transform(X_test_sel)
    log.info(f"Оптимальное количество признаков после RFECV: {rfecv.n_features_}")

    log.info("Запуск GridSearchCV для подбора C в логистической регрессии...")
    param_grid = {"C": np.logspace(-2, 2, 100)}
    final_logreg = GridSearchCV(
        estimator=LogisticRegression(solver="liblinear", random_state=random_state),
        param_grid=param_grid,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    final_logreg.fit(X_train_final, y_train)
    log.info(f"Лучший C: {final_logreg.best_params_['C']} (ROC AUC CV: {final_logreg.best_score_:.4f})")

    y_test_proba = final_logreg.predict_proba(X_test_final)[:, 1]  
    y_test_pred = (y_test_proba >= 0.5).astype(int)                

    metrics = get_metrics(y_test, y_test_pred, y_test_proba)
    log.info(f"Метрики на тесте: {metrics}")

    log.info(f"Создание директории для результатов: {path_to_results_dir}")
    os.makedirs(path_to_results_dir, exist_ok=True)

    log.info("Генерация хэша для набора игр...")
    game_ids_bytes = ",".join(map(str, game_ids)).encode()
    hash_id = hashlib.md5(game_ids_bytes).hexdigest()
    log.info(f"Hash ID: {hash_id}")

    metrics_path = os.path.join(path_to_results_dir, f"{hash_id}.json")
    log.info(f"Сохранение метрик в {metrics_path}...")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    pipeline_path = os.path.join(path_to_results_dir, f"{hash_id}.pickle")
    log.info(f"Сохранение пайплайна и финальной модели в {pipeline_path}...")
    with open(pipeline_path, "wb") as f:
        pickle.dump(
            {
                "feature_pipeline": feature_pipeline,
                "rfecv": rfecv,
                "final_clf": final_logreg.best_estimator_,
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
