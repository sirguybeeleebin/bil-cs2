import logging
from typing import List

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler

from ml.feature_extractors import (
    BagEncoder,
    ColumnSelectorArray,
    PlayerEloEncoder,
    PlayerMapStatSumExtractor,
    PlayerRoundFirstHalfWinSumExtractor,
    PlayerRoundWinSumExtractor,
    PlayerStatSumExtractor,
)
from ml.feature_selectors import LogitL1CVFeatureSelector

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def train_model(
    path_to_dir: str,
    game_ids_train: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    log.info("Начало обучения модели...")

    MAP_ID_COLS: List[int] = [0]
    TEAM_ID_COLS: List[int] = [1, 2]
    PLAYER_ID_COLS: List[int] = list(range(3, 13))
    PLAYER_STAT_KEYS: List[str] = [
        "kills",
        "deaths",
        "assists",
        "headshots",
        "flash_assists",
    ]

    TS_CV_SPLITS: int = 10
    LOGIT_SOLVER: str = "liblinear"
    LOGIT_RANDOM_STATE: int = 42
    RFECV_STEP: int = 1
    RFECV_SCORING: str = "roc_auc"
    RFECV_N_JOBS: int = -1
    RFECV_VERBOSE: int = 1
    L1_C: float = 0.3
    PLAYER_ELO_K_FACTOR: int = 32
    PLAYER_ELO_BASE: int = 1000

    TS_CV = TimeSeriesSplit(n_splits=TS_CV_SPLITS)

    feature_pipelines = []

    log.info("Создание пайплайнов признаков для мешка карт.")
    feature_pipelines.append(
        (
            "map_pipeline",
            Pipeline(
                [
                    ("map_selector", ColumnSelectorArray(MAP_ID_COLS)),
                    ("map_bag", BagEncoder()),
                    (
                        "rfecv",
                        RFECV(
                            estimator=LogisticRegression(solver=LOGIT_SOLVER),
                            step=RFECV_STEP,
                            cv=TS_CV,
                            scoring=RFECV_SCORING,
                            n_jobs=RFECV_N_JOBS,
                            verbose=RFECV_VERBOSE,
                        ),
                    ),
                ]
            ),
        )
    )

    log.info("Создание пайплайнов признаков для мешка команд.")
    feature_pipelines.append(
        (
            "team_pipeline",
            Pipeline(
                [
                    ("team_selector", ColumnSelectorArray(TEAM_ID_COLS)),
                    ("team_bag", BagEncoder()),
                    ("l1", LogitL1CVFeatureSelector(C=L1_C)),
                    (
                        "rfecv",
                        RFECV(
                            estimator=LogisticRegression(solver=LOGIT_SOLVER),
                            step=RFECV_STEP,
                            cv=TS_CV,
                            scoring=RFECV_SCORING,
                            n_jobs=RFECV_N_JOBS,
                            verbose=RFECV_VERBOSE,
                        ),
                    ),
                ]
            ),
        )
    )

    log.info("Создание пайплайнов признаков для мешка игроков.")
    feature_pipelines.append(
        (
            "player_pipeline",
            Pipeline(
                [
                    ("player_selector", ColumnSelectorArray(PLAYER_ID_COLS)),
                    ("player_bag", BagEncoder()),
                    ("l1", LogitL1CVFeatureSelector(C=L1_C)),
                    (
                        "rfecv",
                        RFECV(
                            estimator=LogisticRegression(solver=LOGIT_SOLVER),
                            step=RFECV_STEP,
                            cv=TS_CV,
                            scoring=RFECV_SCORING,
                            n_jobs=RFECV_N_JOBS,
                            verbose=RFECV_VERBOSE,
                        ),
                    ),
                ]
            ),
        )
    )

    log.info("Создание пайплайнов эло игроков.")
    feature_pipelines.append(
        (
            "player_elo_pipeline",
            Pipeline(
                [
                    ("player_selector", ColumnSelectorArray(PLAYER_ID_COLS)),
                    (
                        "player_elo",
                        PlayerEloEncoder(
                            k_factor=PLAYER_ELO_K_FACTOR, base_elo=PLAYER_ELO_BASE
                        ),
                    ),
                    ("scaler", MinMaxScaler()),
                    (
                        "rfecv",
                        RFECV(
                            estimator=LogisticRegression(solver=LOGIT_SOLVER),
                            step=RFECV_STEP,
                            cv=TS_CV,
                            scoring=RFECV_SCORING,
                            n_jobs=RFECV_N_JOBS,
                            verbose=RFECV_VERBOSE,
                        ),
                    ),
                ]
            ),
        )
    )

    for stat_key in PLAYER_STAT_KEYS:
        log.info(f"Создание пайплайна статистики игроков: {stat_key}")
        feature_pipelines.append(
            (
                f"player_stat_{stat_key}_pipeline",
                Pipeline(
                    [
                        ("player_selector", ColumnSelectorArray(PLAYER_ID_COLS)),
                        (
                            "player_stat",
                            PlayerStatSumExtractor(
                                game_ids=game_ids_train,
                                stat_key=stat_key,
                                path_to_dir=path_to_dir,
                            ),
                        ),
                        ("scaler", MinMaxScaler()),
                        (
                            "rfecv",
                            RFECV(
                                estimator=LogisticRegression(solver=LOGIT_SOLVER),
                                step=RFECV_STEP,
                                cv=TS_CV,
                                scoring=RFECV_SCORING,
                                n_jobs=RFECV_N_JOBS,
                                verbose=RFECV_VERBOSE,
                            ),
                        ),
                    ]
                ),
            )
        )

    for stat_key in PLAYER_STAT_KEYS:
        log.info(f"Создание пайплайна статистики игроков по картам: {stat_key}")
        feature_pipelines.append(
            (
                f"player_map_stat_{stat_key}_pipeline",
                Pipeline(
                    [
                        (
                            "player_map_selector",
                            ColumnSelectorArray(MAP_ID_COLS + PLAYER_ID_COLS),
                        ),
                        (
                            "player_map_stat",
                            PlayerMapStatSumExtractor(
                                game_ids=game_ids_train,
                                stat_key=stat_key,
                                path_to_dir=path_to_dir,
                            ),
                        ),
                        ("scaler", MinMaxScaler()),
                        (
                            "rfecv",
                            RFECV(
                                estimator=LogisticRegression(solver=LOGIT_SOLVER),
                                step=RFECV_STEP,
                                cv=TS_CV,
                                scoring=RFECV_SCORING,
                                n_jobs=RFECV_N_JOBS,
                                verbose=RFECV_VERBOSE,
                            ),
                        ),
                    ]
                ),
            )
        )

    log.info("Создание пайплайнов подсчета побед по раундам.")
    feature_pipelines.append(
        (
            "player_round_win_pipeline",
            Pipeline(
                [
                    ("player_selector", ColumnSelectorArray(PLAYER_ID_COLS)),
                    (
                        "player_round_win",
                        PlayerRoundWinSumExtractor(game_ids=game_ids_train),
                    ),
                    ("scaler", MinMaxScaler()),
                    (
                        "rfecv",
                        RFECV(
                            estimator=LogisticRegression(solver=LOGIT_SOLVER),
                            step=RFECV_STEP,
                            cv=TS_CV,
                            scoring=RFECV_SCORING,
                            n_jobs=RFECV_N_JOBS,
                            verbose=RFECV_VERBOSE,
                        ),
                    ),
                ]
            ),
        )
    )

    feature_pipelines.append(
        (
            "player_round_first_half_win_pipeline",
            Pipeline(
                [
                    ("player_selector", ColumnSelectorArray(PLAYER_ID_COLS)),
                    (
                        "player_first_half_win",
                        PlayerRoundFirstHalfWinSumExtractor(game_ids=game_ids_train),
                    ),
                    ("scaler", MinMaxScaler()),
                    (
                        "rfecv",
                        RFECV(
                            estimator=LogisticRegression(solver=LOGIT_SOLVER),
                            step=RFECV_STEP,
                            cv=TS_CV,
                            scoring=RFECV_SCORING,
                            n_jobs=RFECV_N_JOBS,
                            verbose=RFECV_VERBOSE,
                        ),
                    ),
                ]
            ),
        )
    )

    log.info("Объединение всех пайплайнов признаков.")
    features_pipeline = FeatureUnion(feature_pipelines)

    log.info("Создание финального пайплайна модели.")
    final_pipeline = Pipeline(
        [
            ("features", features_pipeline),            
            (
                "rfecv",
                RFECV(
                    estimator=LogisticRegression(solver=LOGIT_SOLVER),
                    step=RFECV_STEP,
                    cv=TS_CV,
                    scoring=RFECV_SCORING,
                    n_jobs=RFECV_N_JOBS,
                    verbose=RFECV_VERBOSE,
                ),
            ),
            (
                "logit",
                LogisticRegression(
                    solver=LOGIT_SOLVER, random_state=LOGIT_RANDOM_STATE
                ),
            ),
        ]
    )

    log.info("Начало обучения финального пайплайна...")
    final_pipeline.fit(X_train, y_train)
    log.info("Обучение завершено.")

    return final_pipeline
