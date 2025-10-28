import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from train_model.data_loader import get_game_ids, get_X_y
from train_model.feature_extractors import (
    BagEncoder,
    ColumnSelectorArray,
    PlayerEloEncoder,
    PlayerMapStatisticSumExtractor,
    PlayerStatisticSumExtractor,
)
from train_model.feature_selectors import LogitL1FeatureSelector
from train_model.metrics import get_metrics
from train_model.stacker import OOFPredictor, Stacker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def train_model(
    games_raw_dir: str,
    test_size: int = 100,
    n_splits: int = 10,
    random_state: int = 42,
):
    log.info("Начало обучения ML модели")

    # Load data
    game_ids = get_game_ids(games_raw_dir)
    game_ids_train = game_ids[:-test_size]
    game_ids_test = game_ids[-test_size:]

    X_train, y_train = get_X_y(game_ids_train, path_to_dir=games_raw_dir)
    X_test, y_test = get_X_y(game_ids_test, path_to_dir=games_raw_dir)

    # Feature columns
    map_col = [0]
    team_cols = [1, 2]
    player_cols = list(range(3, 13))
    player_stats_keys = ["kills", "deaths", "assists", "flash_assists", "headshots"]

    # Pipelines
    map_pipeline = (
        "map_features",
        Pipeline(
            [
                ("select", ColumnSelectorArray(map_col)),
                ("onehot", OneHotEncoder(sparse_output=False)),
                ("l1_select", LogitL1FeatureSelector()),
            ]
        ),
    )

    team_pipeline = (
        "team_features",
        Pipeline(
            [
                ("select", ColumnSelectorArray(team_cols)),
                ("bag", BagEncoder()),
                ("l1_select", LogitL1FeatureSelector()),
            ]
        ),
    )

    player_pipeline = (
        "player_features",
        Pipeline(
            [
                ("select", ColumnSelectorArray(player_cols)),
                ("bag", BagEncoder()),
                ("l1_select", LogitL1FeatureSelector()),
            ]
        ),
    )

    player_elo_pipeline = (
        "player_elo_features",
        Pipeline(
            [
                ("select", ColumnSelectorArray(player_cols)),
                ("elo", PlayerEloEncoder()),
                ("scale", MinMaxScaler()),
                ("l1_select", LogitL1FeatureSelector()),
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
                    ("l1_select", LogitL1FeatureSelector()),
                ]
            ),
        )
        for key in player_stats_keys
    ]

    player_map_stats_pipelines = [
        (
            f"player_map_stat_{key}",
            Pipeline(
                [
                    ("select", ColumnSelectorArray(map_col + player_cols)),
                    (
                        "stat",
                        PlayerMapStatisticSumExtractor(
                            game_ids=game_ids_train, key=key
                        ),
                    ),
                    ("scale", MinMaxScaler()),
                    ("l1_select", LogitL1FeatureSelector()),
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
        *player_stats_pipelines,
        *player_map_stats_pipelines,
    ]

    # Stacker pipeline
    ml_pipeline = Stacker(
        all_pipelines,
        oof_predictor=OOFPredictor(n_splits=n_splits, random_state=random_state),
        random_state=random_state,
    )

    # Train model
    log.info("Обучение модели...")
    ml_pipeline.fit(X_train, y_train)

    # Predict and evaluate
    log.info("Предсказание на тесте...")
    y_pred_test_proba = ml_pipeline.predict_proba(X_test)
    metrics = get_metrics(y_test, y_pred_test_proba)
    log.info("Метрики модели:\n%s", metrics)

    log.info("Обучение завершено успешно")
    return ml_pipeline, metrics
