import logging
from collections import defaultdict
from dateutil.parser import parse
import numpy as np
from pathlib import Path
import hashlib
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from app.repositories import (
    game_raw_repository,
    ml_results_repository,
    map_repo,
    team_repo,
    player_repo,
    JsonRepository,
    PickleRepository,
    MapRepository,
    TeamRepository,
    PlayerRepository
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ETLService:
    def __init__(self, json_repo: JsonRepository, map_repo: MapRepository,
                 team_repo: TeamRepository, player_repo: PlayerRepository):
        self.json_repo = json_repo
        self.map_repo = map_repo
        self.team_repo = team_repo
        self.player_repo = player_repo
        log.info("ETLService инициализирован")

    def start(self):
        log.info("Начало ETL процесса")
        for game in self.json_repo.generate():
            log.info(f"Обработка игры {game.get('id')}")
            map_data = self._extract_map(game)
            if map_data:
                self.map_repo.upsert(map_data)
                log.info(f"Обновлена информация о карте: {map_data}")

            teams_data = self._extract_teams(game)
            for team in teams_data:
                self.team_repo.upsert(team)
                log.info(f"Обновлена информация о команде: {team}")

            players_data = self._extract_players(game)
            for player in players_data:
                self.player_repo.upsert(player)
                log.info(f"Обновлена информация о игроке: {player}")

    def _extract_map(self, game: dict) -> dict | None:
        try:
            return {"map_id": game["map"]["id"], "name": game["map"]["name"]}
        except KeyError:
            log.warning("Игра не содержит данных карты")
            return None

    def _extract_teams(self, game: dict) -> list[dict]:
        teams = []
        try:
            seen_ids = set()
            for p in game.get("players", []):
                team = p.get("team")
                if team and team.get("id") not in seen_ids:
                    teams.append({"team_id": team["id"], "name": team["name"]})
                    seen_ids.add(team["id"])
        except Exception as e:
            log.error(f"Ошибка при извлечении команд: {e}")
        return teams

    def _extract_players(self, game: dict) -> list[dict]:
        players = []
        try:
            seen_ids = set()
            for p in game.get("players", []):
                player = p.get("player")
                if player and player.get("id") not in seen_ids:
                    players.append({"player_id": player["id"], "name": player["name"]})
                    seen_ids.add(player["id"])
        except Exception as e:
            log.error(f"Ошибка при извлечении игроков: {e}")
        return players


class MLService:
    def __init__(
        self, 
        games_raw_repository: JsonRepository, 
        ml_results_repository: PickleRepository,
    ):
        self.games_raw_repository = games_raw_repository
        self.ml_results_repository = ml_results_repository
        self.team_bag_encoder = self.TeamBagEncoder()
        self.player_bag_encoder = self.PlayerBagEncoder()
        self.player_elo_encode = self.PlayerEloEncoder()
        self.test_size = 100
        self.C_grid = np.array([0.08, 0.09, 0.1, 0.2, 0.3])
        self.scoring = "roc_auc"
        self.n_splits = 10
        self.cv = TimeSeriesSplit(self.n_splits)
        self.verbose = 2
        self.feature_pipeline: Pipeline | None = None
        self.best_model = None
        self.grid_search_final = GridSearchCV(
            LogisticRegression(solver="liblinear"),
            param_grid={"C": np.linspace(0.01, 1, 100)},
            cv=self.cv, scoring=self.scoring,
            n_jobs=-1, verbose=self.verbose
        )
        log.info("MLService инициализирован")

    def start(self) -> Path | None:
        log.info("Запуск ML пайплайна")
        try:
            game_ids = self._get_game_ids()
            if not game_ids:
                log.warning("Не найдено подходящих игр")
                return None
            log.info(f"Найдено {len(game_ids)} валидных игр")

            game_ids_train, game_ids_test = game_ids[:-self.test_size], game_ids[-self.test_size:]
            X_train, y_train = self._get_X_y(game_ids_train)
            X_test, y_test = self._get_X_y(game_ids_test)
            log.info("Данные разделены на train и test")

            self.feature_pipeline = self._create_feature_extraction_pipeline(
                team_cols=[0, 1], player_cols=list(range(2, 12))
            )
            self.feature_pipeline.fit(X_train, y_train)
            log.info("Фичи обучены")
            X_train_transformed = self.feature_pipeline.transform(X_train)
            X_test_transformed = self.feature_pipeline.transform(X_test)

            mask = self._select_features_with_logit_and_cv(
                X_train_transformed, y_train, 
                Cs=self.C_grid, scoring=self.scoring, 
                cv=self.cv, verbose=self.verbose
            )
            X_train_selected = X_train_transformed[:, mask]
            X_test_selected = X_test_transformed[:, mask]
            log.info("Выбраны лучшие признаки")

            self.grid_search_final.fit(X_train_selected, y_train)
            best_model = self.grid_search_final.best_estimator_
            self.best_model = best_model
            log.info("Лучшая модель обучена")

            y_test_proba = best_model.predict_proba(X_test_selected)[:, 1]
            y_test_pred = (y_test_proba >= 0.5).astype(int)

            metrics = self._get_metrics(y_test, y_test_pred, y_test_proba)
            log.info(f"Метрики на тестовой выборке: {metrics}")

            game_ids_bytes = ",".join(map(str, game_ids)).encode()
            hash_id = hashlib.md5(game_ids_bytes).hexdigest()

            pipeline_path = self.ml_results_repository.save({
                "hash_id": hash_id,
                "feature_pipeline": self.feature_pipeline,
                "feature_selection_mask": mask,
                "best_model": best_model,
                "metrics": metrics
            })
            log.info("ML пайплайн завершен и сохранен")
            return pipeline_path
        except Exception as e:
            log.error(f"Ошибка в ML пайплайне: {e}")
            return None

    def _validate_game(self, game: dict) -> bool:
        try:
            parse(game["begin_at"])
            team_players: dict[int, list[int]] = defaultdict(list)
            for p in game.get("players", []):
                team_players[p["team"]["id"]].append(p["player"]["id"])
            if len(team_players) != 2:
                return False
            for p_ids in team_players.values():
                if len(set(p_ids)) != 5:
                    return False
            team_ids = list(team_players.keys())
            rounds: list[int] = []
            for r in game.get("rounds", []):
                if (r.get("round") is None or
                        r.get("ct") not in team_ids or
                        r.get("terrorists") not in team_ids or
                        r.get("winner_team") not in team_ids):
                    return False
                rounds.append(r["round"])
            if min(rounds) != 1 or max(rounds) < 16:
                return False
            return True
        except Exception as e:
            log.warning(f"Ошибка при валидации игры {game.get('id')}: {e}")
            return False

    def _get_game_ids(self) -> list[int]:
        log.info("Получение ID игр")
        game_ids_valid, game_begin_at_valid = [], []
        for game in self.games_raw_repository.generate():
            if self._validate_game(game):
                game_ids_valid.append(game["id"])
                game_begin_at_valid.append(parse(game["begin_at"]))
        sorted_idx = np.argsort(game_begin_at_valid)
        log.info(f"Найдено {len(game_ids_valid)} валидных игр после сортировки")
        return np.array(game_ids_valid)[sorted_idx].tolist()

    def _get_X_y(self, game_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        log.info("Формирование X и y")
        X, y = [], []
        for game_id in game_ids:
            try:
                game = self.games_raw_repository.get(game_id)
                team_players: dict[int, list[int]] = defaultdict(list)
                for p in game.get("players", []):
                    team_players[p["team"]["id"]].append(p["player"]["id"])
                t1_id, t2_id = sorted(team_players.keys())
                X.append([t1_id, t2_id] + sorted(team_players[t1_id]) + sorted(team_players[t2_id]))
                team_win_count = {t1_id: 0, t2_id: 0}
                for r in game.get("rounds", []):
                    team_win_count[r["winner_team"]] += 1
                y.append(int(team_win_count[t1_id] > team_win_count[t2_id]))
            except Exception as e:
                log.warning(f"Ошибка при обработке игры {game_id}: {e}")
                continue
        log.info("X и y сформированы")
        return np.array(X), np.array(y)

    def _create_feature_extraction_pipeline(self, team_cols: list[int], player_cols: list[int]):
        log.info("Создание пайплайна для извлечения признаков")
        return Pipeline([
            ("encoder", FeatureUnion([
                ("team_bag", Pipeline([
                    ("select_team", self.ColumnSelector(team_cols)),
                    ("encode_team", self.team_bag_encoder)
                ])),
                ("player_bag", Pipeline([
                    ("select_player", self.ColumnSelector(player_cols)),
                    ("encode_player", self.player_bag_encoder)
                ])),
                ("player_elo", Pipeline([
                    ("select_player", self.ColumnSelector(player_cols)),
                    ("elo_encoder", self.player_elo_encode)
                ]))
            ]))
        ])

    def _get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
        try:
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_true, y_proba),
                "confusion_matrix": cm.tolist(),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn)
            }
            return metrics
        except Exception as e:
            log.error(f"Ошибка при вычислении метрик: {e}")
            return {}

    class ColumnSelector(BaseEstimator, TransformerMixin):
        def __init__(self, columns: list[int]):
            self.columns = columns
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X = np.asarray(X)
            return X[:, self.columns]

    class PlayerBagEncoder(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            X = np.asarray(X)
            uniques = np.unique(X.flatten())
            self.player_dict = {player: idx for idx, player in enumerate(uniques)}
            return self
        def transform(self, X):
            X = np.asarray(X)
            n_samples, n_features = X.shape[0], len(self.player_dict)
            rows, cols, data = [], [], []
            for i, row in enumerate(X):
                for j, player in enumerate(row):
                    col_idx = self.player_dict.get(player)
                    if col_idx is not None:
                        rows.append(i)
                        cols.append(col_idx)
                        data.append(1 if j < len(row) // 2 else -1)
            return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features), dtype=int)

    class TeamBagEncoder(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            X = np.asarray(X)
            uniques = np.unique(X.flatten())
            self.team_dict = {team: idx for idx, team in enumerate(uniques)}
            return self
        def transform(self, X):
            X = np.asarray(X)
            n_samples, n_features = X.shape[0], len(self.team_dict)
            rows, cols, data = [], [], []
            for i, row in enumerate(X):
                for j, team in enumerate(row):
                    col_idx = self.team_dict.get(team)
                    if col_idx is not None:
                        rows.append(i)
                        cols.append(col_idx)
                        data.append(1 if j == 0 else -1)
            return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features), dtype=int)

    class PlayerEloEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, k_factor=32, base_elo=1000):
            self.k_factor = k_factor
            self.base_elo = base_elo
            self.elo_dict_ = {}
            self.X_elo_train_ = None
        def _expected_score(self, rating_a, rating_b):
            return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        def _augment_X(self, row):
            x1, x2 = np.sort(row[:5]), np.sort(row[5:])
            features = []
            mean1, mean2 = np.mean(x1), np.mean(x2)
            features.extend([mean1, mean2, mean1, -mean2, mean1 - mean2])
            features.extend([x1[i] - x2[j] for i in range(5) for j in range(5)])
            return np.array(features, dtype=float)
        def fit(self, X, y):
            X, y = np.asarray(X), np.asarray(y)
            X_elo = []
            for row, outcome in zip(X, y):
                elos_before = [self.elo_dict_.get(pid, self.base_elo) for pid in row]
                X_elo.append(elos_before)
                avg1, avg2 = np.mean(elos_before[:5]), np.mean(elos_before[5:])
                exp1 = self._expected_score(avg1, avg2)
                score1, score2 = int(outcome == 1), int(outcome == 0)
                for pid in row[:5]:
                    self.elo_dict_[pid] = self.elo_dict_.get(pid, self.base_elo) + self.k_factor * (score1 - exp1)
                for pid in row[5:]:
                    self.elo_dict_[pid] = self.elo_dict_.get(pid, self.base_elo) + self.k_factor * (score2 - (1 - exp1))
            self.X_elo_train_ = np.array(X_elo, dtype=float)
            return self
        def transform(self, X):
            X_out = np.copy(self.X_elo_train_) if X.shape == self.X_elo_train_.shape else np.array(
                [[self.elo_dict_.get(pid, self.base_elo) for pid in row] for row in X], dtype=float)
            return np.array([self._augment_X(row) for row in X_out], dtype=float)


etl_service = ETLService(
    json_repo=game_raw_repository,
    map_repo=map_repo,
    team_repo=team_repo,
    player_repo=player_repo
)

ml_service = MLService(
    games_raw_repository=game_raw_repository,
    ml_results_repository=ml_results_repository
)
