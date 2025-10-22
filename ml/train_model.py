import logging
import warnings

from ml.data_loader import get_game_ids_ordered_by_begin_at, get_X_y
from ml.feature_extractors import (
    ColumnSelector,
    PlayerBagEncoder,
    PlayerEloEncoder,
    TeamBagEncoder,
)
from ml.metrics import get_metrics
from ml.predictor import Team1WinProbabilityPredictor

warnings.filterwarnings("ignore")

# Настройка базового логгера
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train_model(path_to_games_raw_dir: str):
    log.info("Начало обучения модели...")

    map_id_col = [0]
    team_cols = [1, 2]
    player_cols = list(range(3, 13))

    column_selector_cls = ColumnSelector
    player_elo_encoder = PlayerEloEncoder()
    player_bag_encoder = PlayerBagEncoder()
    team_bag_encoder = TeamBagEncoder()
    predictor = Team1WinProbabilityPredictor(
        column_selector_cls=column_selector_cls,
        player_elo_encoder=player_elo_encoder,
        player_bag_encoder=player_bag_encoder,
        team_bag_encoder=team_bag_encoder,
    )

    log.info("Загрузка идентификаторов игр...")
    game_ids = get_game_ids_ordered_by_begin_at(path_to_games_raw_dir)
    game_ids_train, game_ids_test = (
        game_ids[:-100],
        game_ids[-100:],
    )
    log.info(
        f"Количество обучающих игр: {len(game_ids_train)}, тестовых игр: {len(game_ids_test)}"
    )

    X_train, y_train = get_X_y(path_to_games_raw_dir, game_ids_train)
    X_test, y_test = get_X_y(path_to_games_raw_dir, game_ids_test)

    log.info("Обучение модели...")
    predictor.fit(
        X_train,
        y_train,
        map_id_col=map_id_col,
        team_cols=team_cols,
        player_cols=player_cols,
    )

    log.info("Предсказание на тестовой выборке...")
    y_pred = predictor.predict(X_test)
    y_pred_proba = predictor.predict_proba(X_test)[:, 1]

    metrics = get_metrics(y_test, y_pred, y_pred_proba)
    log.info(f"Обучение завершено. Метрики: {metrics}")

    return predictor, metrics
