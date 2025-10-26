import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from app.ml.data_loader import get_game_ids, get_X_y
from app.ml.feature_extractors import (
    BagEncoder,
    ColumnSelectorArray,
    PlayerEloEncoder,
    PlayerStatisticSumExtractor,
)
from app.ml.metrics import get_metrics
from app.ml.stacker import MLStacker, OOFPredictor

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_ml_pipeline(
    path_to_games_raw_dir: str = "data/games_raw",
    test_size: int = 100,
    n_splits: int = 10,
    n_iters: int = 10,
    random_state: int = 42,
) -> tuple[MLStacker, dict]:
    log.info("Начало выполнения ML пайплайна")

    log.info(f"Загрузка идентификаторов игр из {path_to_games_raw_dir}")
    game_ids = get_game_ids(path_to_games_raw_dir)
    game_ids_train = game_ids[:-test_size]
    game_ids_test = game_ids[-test_size:]
    log.info(
        f"Количество тренировочных игр: {len(game_ids_train)}, тестовых: {len(game_ids_test)}"
    )

    log.info("Формирование X и y для тренировочного набора")
    X_train, y_train = get_X_y(game_ids_train, path_to_dir=path_to_games_raw_dir)
    log.info(f"Размер X_train: {len(X_train)}, y_train: {len(y_train)}")

    log.info("Формирование X и y для тестового набора")
    X_test, y_test = get_X_y(game_ids_test, path_to_dir=path_to_games_raw_dir)
    log.info(f"Размер X_test: {len(X_test)}, y_test: {len(y_test)}")

    log.info("Настройка пайплайнов признаков")
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

    log.info("Создание MLStacker")
    oof_predictor = OOFPredictor(n_splits=n_splits, random_state=random_state)
    ml_pipeline = MLStacker(
        all_pipelines,
        oof_predictor=oof_predictor,
        n_iters=n_iters,
        random_state=random_state,
    )

    log.info("Обучение ML пайплайна")
    ml_pipeline.fit(X_train, y_train)

    log.info("Предсказание на тестовом наборе")
    y_pred_test_proba = ml_pipeline.predict_proba(X_test)

    log.info("Вычисление метрик")
    metrics = get_metrics(y_test, y_pred_test_proba)
    log.info(f"Метрики: {metrics}")

    log.info("ML пайплайн успешно выполнен")
    return ml_pipeline, metrics
