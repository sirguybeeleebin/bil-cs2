import os
import json
import logging
import uuid
from collections import defaultdict
from dateutil.parser import parse as parse_date
from celery import shared_task
from django.conf import settings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
import pickle
from sklearn.model_selection import TimeSeriesSplit


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# -------------------------------------------------------------------
# ML Pipeline
# -------------------------------------------------------------------
class MLPipeline:
    def __init__(self):
        self.team_map = {}
        self.player_map = {}
        self.model = LogisticRegression(solver="liblinear")
        self.selected_features_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """Обучение модели и отбор признаков через permutation importance."""
        log.info("[MLPipeline] Начало обучения модели")

        # Собираем уникальные команды и игроков
        unique_teams = set()
        unique_players = set()
        for row in X:
            t1, t2, *players = row
            unique_teams.update([t1, t2])
            unique_players.update(players)

        # Создаём индексы признаков
        self.team_map = {team_id: idx for idx, team_id in enumerate(sorted(unique_teams))}
        self.player_map = {player_id: idx for idx, player_id in enumerate(sorted(unique_players))}

        # Преобразование данных и первичное обучение
        X_transformed = self._transform(X)
        self.model.fit(X_transformed, np.array(y))

        log.info(f"[MLPipeline] Обучение завершено ({len(self.team_map)} команд, {len(self.player_map)} игроков)")

        # Отбор признаков
        self.selected_features_ = self._select_features(X, y)

        # Переобучаем модель только на выбранных признаках
        X_selected = X_transformed[:, self.selected_features_]
        self.model.fit(X_selected, np.array(y))
        log.info(f"[MLPipeline] Модель переобучена на {self.selected_features_.sum()} важных признаках")

    def _transform(self, X):
        """Преобразование данных в bag-of-teams + bag-of-players."""
        n_samples = len(X)
        n_team = len(self.team_map)
        n_player = len(self.player_map)
        bag_of_teams = np.zeros((n_samples, n_team), dtype=np.int8)
        bag_of_players = np.zeros((n_samples, n_player), dtype=np.int8)

        for i, row in enumerate(X):
            t1, t2, *players = row
            t1_idx = self.team_map.get(t1)
            t2_idx = self.team_map.get(t2)
            if t1_idx is not None:
                bag_of_teams[i, t1_idx] = 1
            if t2_idx is not None:
                bag_of_teams[i, t2_idx] = -1

            for p_id in players[:5]:
                p_idx = self.player_map.get(p_id)
                if p_idx is not None:
                    bag_of_players[i, p_idx] = 1
            for p_id in players[5:]:
                p_idx = self.player_map.get(p_id)
                if p_idx is not None:
                    bag_of_players[i, p_idx] = -1

        return np.hstack([bag_of_teams, bag_of_players])

    def predict_proba(self, X):
        """Возвращает вероятности победы первой команды (использует выбранные признаки)."""
        X_transformed = self._transform(X)
        if self.selected_features_ is not None:
            X_transformed = X_transformed[:, self.selected_features_]
        return self.model.predict_proba(X_transformed)[:, 1]

    def _select_features(
        self,
        X,
        y,
        cv: int = 5,
        scoring: str = "roc_auc",
        n_repeats: int = 1,
        random_state: int = 42,
    ):
        """
        Приватный метод итеративного отбора признаков через permutation importance с кросс-валидацией.
        Работает до тех пор, пока маска признаков не стабилизируется.
        """
        log.info("[MLPipeline] Запуск итеративного отбора признаков с CV (permutation importance)")

        X_transformed = self._transform(X)
        selected_mask = np.ones(X_transformed.shape[1], dtype=bool)
        iteration = 0

        kf = TimeSeriesSplit(n_splits=cv)

        while True:
            iteration += 1
            X_current = X_transformed[:, selected_mask]
            importances_list = []

            for train_idx, valid_idx in kf.split(X_current):
                X_train, X_valid = X_current[train_idx], X_current[valid_idx]
                y_train, y_valid = np.array(y)[train_idx], np.array(y)[valid_idx]

                # Обучение модели на текущем фолде
                self.model.fit(X_train, y_train)

                # Permutation importance для текущего фолда
                result = permutation_importance(
                    self.model,
                    X_valid,
                    y_valid,
                    scoring=scoring,
                    n_repeats=n_repeats,
                    random_state=random_state,
                )
                importances_list.append(result.importances_mean)

            # Усредняем важность по фолдам
            mean_importances = np.mean(importances_list, axis=0)
            self.feature_importances_ = mean_importances
            important = mean_importances > 0
            n_selected = important.sum()

            log.info(f"[MLPipeline] Итерация {iteration}: {n_selected}/{selected_mask.sum()} признаков важны")

            # Условия остановки
            if n_selected == 0:
                log.warning("[MLPipeline] Ни один признак не важен — остановка.")
                break
            if n_selected == selected_mask.sum():
                log.info("[MLPipeline] Все оставшиеся признаки важны — завершение.")
                break

            # Обновляем маску — сохраняем только важные признаки
            new_mask = np.zeros_like(selected_mask)
            new_mask[np.where(selected_mask)[0][important]] = True

            if np.array_equal(new_mask, selected_mask):
                log.info("[MLPipeline] Маска признаков стабилизировалась — завершение отбора.")
                break

            selected_mask = new_mask

        log.info(f"[MLPipeline] Финально отобрано {selected_mask.sum()} признаков из {X_transformed.shape[1]}")
        return selected_mask



# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def generate_pipeline_id() -> str:
    return uuid.uuid4().hex


def validate_game_raw(game: dict | None) -> bool:
    """Проверка структуры и полноты данных игры."""
    try:
        int(game["id"])
        parse_date(game["begin_at"])
        team_players = defaultdict(set)
        for p in game["players"]:
            t_id = int(p["team"]["id"])
            p_id = int(p["player"]["id"])
            team_players[t_id].add(p_id)
        if len(team_players) != 2 or any(len(pids) != 5 for pids in team_players.values()):
            return False
        rounds = game.get("rounds", [])
        round_numbers = [r.get("round") for r in rounds if r.get("round")]
        if not round_numbers or min(round_numbers) != 1 or max(round_numbers) < 16:
            return False
        return True
    except Exception as e:
        log.warning(f"Validation failed for game {game.get('id', 'unknown')}: {e}")
        return False


# -------------------------------------------------------------------
# Tasks
# -------------------------------------------------------------------
@shared_task(name="backend.tasks.parse")
def parse(pipeline_id: str) -> str:
    """Валидация и сохранение списка валидных игр."""
    raw_dir = settings.GAMES_RAW_DIR
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".json"))
    valid_game_ids = []

    log.info(f"[PARSE] Начинаем обработку {len(files)} файлов в {raw_dir}")
    for idx, filename in enumerate(files):
        filepath = os.path.join(raw_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data and validate_game_raw(data):
                valid_game_ids.append(data["id"])
            else:
                log.warning(f"[PARSE] ({idx}/{len(files)}) Файл не прошел валидацию: {filename}")
        except Exception as e:
            log.warning(f"[PARSE] ({idx}/{len(files)}) Ошибка загрузки {filename}: {e}")
            continue

    os.makedirs(settings.GAMES_VALID_DIR, exist_ok=True)
    with open(os.path.join(settings.GAMES_VALID_DIR, f"{pipeline_id}.json"), "w", encoding="utf-8") as f:
        json.dump(valid_game_ids, f, indent=4, default=str)

    log.info(f"[PARSE] Сохранено {len(valid_game_ids)} валидных ID игр для пайплайна {pipeline_id}")
    return pipeline_id


@shared_task(name="backend.tasks.train_test_split")
def train_test_split(pipeline_id: str) -> str:
    """Создаёт train/test сплит из валидных ID, читая по game_id."""
    valid_game_ids_path = os.path.join(settings.GAMES_VALID_DIR, f"{pipeline_id}.json")
    with open(valid_game_ids_path, "r", encoding="utf-8") as f:
        valid_game_ids = json.load(f)

    raw_dir = settings.GAMES_RAW_DIR
    temp_list = []

    for game_id in valid_game_ids:
        filepath = os.path.join(raw_dir, f"{game_id}.json")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                game = json.load(f)
        except Exception as e:
            log.warning(f"[TRAIN_TEST_SPLIT] Ошибка загрузки игры {game_id}: {e}")
            continue

        dd = defaultdict(list)
        for p in game["players"]:
            dd[p["team"]["id"]].append(p["player"]["id"])
        t1_id, t2_id = sorted(dd.keys())
        features = [t1_id, t2_id] + sorted(dd[t1_id]) + sorted(dd[t2_id])
        target = int(t1_id == pd.DataFrame.from_records(game["rounds"])["winner_team"].value_counts().idxmax())
        temp_list.append({"game_id": game_id, "input": features, "output": target, "begin_at": game["begin_at"]})

    temp_list.sort(key=lambda x: x["begin_at"])
    X = [tuple(d["input"]) for d in temp_list]
    y = [d["output"] for d in temp_list]
    game_ids = [d["game_id"] for d in temp_list]

    test_size = getattr(settings, "TEST_SIZE", 100)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    train_ids, test_ids = game_ids[:-test_size], game_ids[-test_size:]

    os.makedirs(settings.ML_INPUT_DIR, exist_ok=True)
    with open(os.path.join(settings.ML_INPUT_DIR, f"{pipeline_id}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_ids": train_ids, "test_ids": test_ids,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test
        }, f, indent=4, default=str)

    log.info(f"[TRAIN_TEST_SPLIT] Train/test сплит сохранен для пайплайна {pipeline_id}")
    return pipeline_id


@shared_task(name="backend.tasks.run_ml_pipeline")
def run_ml_pipeline(pipeline_id: str) -> str:
    """Финальное обучение модели и вычисление метрик."""
    train_test_path = os.path.join(settings.ML_INPUT_DIR, f"{pipeline_id}.json")
    with open(train_test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ml_pipeline = MLPipeline()
    log.info(f"[RUN_ML_PIPELINE] Начинаем обучение модели для пайплайна {pipeline_id}")
    ml_pipeline.fit(data["X_train"], data["y_train"])

    y_pred_proba = ml_pipeline.predict_proba(data["X_test"])
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(data["y_test"], y_pred_proba)
    f1 = f1_score(data["y_test"], y_pred)
    acc = accuracy_score(data["y_test"], y_pred)
    cm = confusion_matrix(data["y_test"], y_pred)

    log.info(f"[RUN_ML_PIPELINE] Метрики для {pipeline_id}: AUC={auc:.4f}, F1={f1:.4f}, ACC={acc:.4f}")
    log.info(f"[RUN_ML_PIPELINE] Матрица ошибок:\n{cm}")

    os.makedirs(settings.ML_RESULTS_DIR, exist_ok=True)
    with open(os.path.join(settings.ML_RESULTS_DIR, f"{pipeline_id}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "pipeline_id": pipeline_id,
            "metrics": {"auc": auc, "f1": f1, "accuracy": acc, "confusion_matrix": cm.tolist()},
            "feature_importances": ml_pipeline.feature_importances_.tolist() if ml_pipeline.feature_importances_ is not None else None,
            "y_true": data["y_test"],
            "y_pred": y_pred.tolist(),
            "y_pred_proba": y_pred_proba.tolist(),
            "train_ids": data["train_ids"],
            "test_ids": data["test_ids"]
        }, f, indent=4, default=str)

    with open(os.path.join(settings.ML_RESULTS_DIR, f"{pipeline_id}.pickle"), "wb") as f:
        pickle.dump(ml_pipeline, f)

    log.info(f"[RUN_ML_PIPELINE] Пайплайн {pipeline_id} завершен успешно ✅")
    return pipeline_id
