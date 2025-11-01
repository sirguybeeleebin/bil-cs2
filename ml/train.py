import hashlib
import logging
from pathlib import Path
from uuid import uuid4

import joblib

from ml.dataset_loader import build_dataset, get_game_ids
from ml.metrics import get_metrics
from ml.train_model import train_model

log = logging.getLogger(__name__)


def train(path_to_games: Path, models_dir: Path):
    # Получаем список всех игр
    game_ids, begin_at_min, begin_at_max = get_game_ids(path_to_games)
    if not game_ids:
        log.warning("Не найдено игр")
        return None, None, None, None, None, None

    # Разделяем на обучающий и тестовый набор
    game_ids_train, game_ids_test = game_ids[:-100], game_ids[-100:]
    log.info(
        f"Размер тренировочного датасета: {len(game_ids_train)}, Размер тестового датасета: {len(game_ids_test)}"
    )

    # Вычисляем хеш сплита
    split_hash = hashlib.sha256(
        ",".join(map(str, game_ids)).encode("utf-8")
    ).hexdigest()
    log.info(f"Сплит хеш: {split_hash}")

    # Строим датасеты
    X_train, y_train = build_dataset(path_to_games, game_ids_train)
    X_test, y_test = build_dataset(path_to_games, game_ids_test)

    # Обучение модели
    log.info("Обучение модели...")
    model = train_model(path_to_games, game_ids_train, X_train, y_train)

    # Сохраняем модель на диск
    train_result_id = uuid4()
    path_to_model = models_dir / f"{train_result_id}.joblib"
    joblib.dump(model, path_to_model)
    log.info(f"Модель сохранена: {path_to_model}")

    # Предсказания и метрики
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = get_metrics(y_test, y_pred_proba)
    log.info(f"Метрики: {metrics}")

    log.info("Задача обучения завершена")

    return (
        model,
        metrics,
        path_to_model,
        split_hash,
        game_ids_train,
        game_ids_test,
        begin_at_min,
        begin_at_max,
    )
