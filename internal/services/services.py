import json
import logging
import os
import warnings
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from dateutil.parser import parse
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

from internal.repositories import MapRepository, PlayerRepository, TeamRepository

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


class ETLService:
    def __init__(
        self,
        map_repo: MapRepository,
        team_repo: TeamRepository,
        player_repo: PlayerRepository,
    ):
        self.map_repo = map_repo
        self.team_repo = team_repo
        self.player_repo = player_repo

    def start(self, path_to_games_raw_dir: str):
        log.info("Начало ETL процесса")

        if not os.path.exists(path_to_games_raw_dir):
            log.error(f"Директория не найдена: {path_to_games_raw_dir}")
            return

        for fnm in os.listdir(path_to_games_raw_dir):
            pth = os.path.join(path_to_games_raw_dir, fnm)
            if not os.path.isfile(pth):
                continue

            try:
                with open(pth, "r", encoding="utf-8") as f:
                    game = json.load(f)
            except Exception as e:
                log.error(f"Ошибка при загрузке файла {pth}: {e}")
                continue

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

    def _extract_map(self, game: dict) -> Optional[dict]:
        try:
            return {"map_id": game["map"]["id"], "name": game["map"]["name"]}
        except KeyError:
            log.warning("Игра не содержит данных карты")
            return None

    def _extract_teams(self, game: dict) -> List[dict]:
        teams: List[dict] = []
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

    def _extract_players(self, game: dict) -> List[dict]:
        players: List[dict] = []
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
    def __init__(self):
        self.team_bag_encoder = self.TeamBagEncoder()
        self.player_bag_encoder = self.PlayerBagEncoder()
        self.player_elo_encode = self.PlayerEloEncoder()
        self.test_size = 100
        self.n_jobs = 1
        self.C_grid = np.array([0.08, 0.09, 0.1, 0.2, 0.3])
        self.scoring = "precision"
        self.n_splits = 10
        self.cv = TimeSeriesSplit(self.n_splits)
        self.verbose = 2
        self.feature_pipeline: Pipeline | None = None
        self.best_model = None
        self.grid_search_final = GridSearchCV(
            LogisticRegression(solver="liblinear"),
            param_grid={"C": np.linspace(0.01, 1, 100)},
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        log.info("MLService инициализирован")

    def start(self, path_to_games_raw_dir: str) -> List[dict]:
        log.info("Запуск ML пайплайна")

        game_id_dicts = self._get_game_ids(path_to_games_raw_dir)
        if not game_id_dicts:
            log.warning("Не найдено подходящих игр")
            return []

        results = []
        map_ids = ["all"] + [k for k in list(game_id_dicts.keys()) if k != "all"]

        for map_id in map_ids:
            game_ids = game_id_dicts[map_id]
            try:
                game_ids_train, game_ids_test = (
                    game_ids[: -self.test_size],
                    game_ids[-self.test_size :],
                )
                X_train, y_train = self._get_X_y(path_to_games_raw_dir, game_ids_train)
                X_test, y_test = self._get_X_y(path_to_games_raw_dir, game_ids_test)

                log.info(f"Данные разделены: train={len(X_train)}, test={len(X_test)}")

                self.feature_pipeline = self._create_feature_extraction_pipeline(
                    team_cols=[0, 1], player_cols=list(range(2, 12))
                )
                self.feature_pipeline.fit(X_train, y_train)
                X_train_transformed = self.feature_pipeline.transform(X_train)
                X_test_transformed = self.feature_pipeline.transform(X_test)

                mask = self._select_features_with_logit_and_cv(
                    X_train_transformed,
                    y_train,
                    Cs=self.C_grid,
                    scoring=self.scoring,
                    cv=self.cv,
                    verbose=self.verbose,
                )
                X_train_selected = X_train_transformed[:, mask]
                X_test_selected = X_test_transformed[:, mask]

                self.grid_search_final.fit(X_train_selected, y_train)
                best_model = self.grid_search_final.best_estimator_
                self.best_model = best_model

                y_test_proba = best_model.predict_proba(X_test_selected)[:, 1]
                y_test_pred = (y_test_proba >= 0.5).astype(int)
                metrics = self._get_metrics(y_test, y_test_pred, y_test_proba)

                log.info(f"Метрики: {metrics}")

                results.append(
                    {
                        "map_id": map_id,
                        "feature_pipeline": self.feature_pipeline,
                        "feature_selection_mask": mask,
                        "best_model": best_model,
                        "game_ids_train": game_ids_train,
                        "game_ids_test": game_ids_test,
                        "X_train": X_train_selected,
                        "X_test": X_test_selected,
                        "y_train": y_train,
                        "y_test": y_test,
                        "metrics": metrics,
                    }
                )

                log.info(f"ML пайплайн для карты {map_id} завершен")

            except Exception as e:
                log.error(f"Ошибка при обработке карты {map_id}: {e}")
                continue

        return results

    def _validate_game(self, game: dict) -> bool:
        try:
            parse(game["begin_at"])
            int(game["map"]["id"])
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
                if (
                    r.get("round") is None
                    or r.get("ct") not in team_ids
                    or r.get("terrorists") not in team_ids
                    or r.get("winner_team") not in team_ids
                ):
                    return False
                rounds.append(r["round"])
            if min(rounds) != 1 or max(rounds) < 16:
                return False
            return True
        except Exception as e:
            log.warning(f"Ошибка при валидации игры {game.get('id')}: {e}")
            return False

    def _get_game_ids(self, path_to_games_raw_dir: str) -> defaultdict:
        log.info("Получение ID игр")
        game_ids_valid = defaultdict(list)
        game_begin_at_valid = defaultdict(list)
        for fnm in os.listdir(path_to_games_raw_dir):
            try:
                with open(
                    os.path.join(path_to_games_raw_dir, fnm), "r", encoding="utf-8"
                ) as f:
                    game = json.load(f)
                if not game:
                    continue
                if self._validate_game(game):
                    map_id = game["map"]["id"]
                    game_ids_valid[map_id].append(game["id"])
                    game_ids_valid["all"].append(game["id"])
                    game_begin_at_valid[map_id].append(parse(game["begin_at"]))
                    game_begin_at_valid["all"].append(parse(game["begin_at"]))
            except Exception:
                continue
        for map_id in game_ids_valid:
            sorted_idx = np.argsort(game_begin_at_valid[map_id])
            game_ids_valid[map_id] = [game_ids_valid[map_id][i] for i in sorted_idx]
        total_games = sum(len(ids) for ids in game_ids_valid.values())
        log.info(f"Найдено {total_games} валидных игр после сортировки по картам")
        return game_ids_valid

    def _get_X_y(
        self, path_to_games_raw_dir: str, game_ids: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for game_id in game_ids:
            file_path = os.path.join(path_to_games_raw_dir, f"{game_id}.json")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    game = json.load(f)
            except Exception as e:
                log.error(f"Ошибка при загрузке игры {game_id}: {e}")
                continue
            team_players: dict[int, list[int]] = defaultdict(list)
            for p in game.get("players", []):
                team_players[p["team"]["id"]].append(p["player"]["id"])
            t1_id, t2_id = sorted(team_players.keys())
            X.append(
                [t1_id, t2_id]
                + sorted(team_players[t1_id])
                + sorted(team_players[t2_id])
            )
            team_win_count = {t1_id: 0, t2_id: 0}
            for r in game.get("rounds", []):
                team_win_count[r["winner_team"]] += 1
            y.append(int(team_win_count[t1_id] > team_win_count[t2_id]))
        log.info(f"X и y сформированы: {len(X)} образцов")
        return np.array(X), np.array(y)

    def _create_feature_extraction_pipeline(
        self, team_cols: list[int], player_cols: list[int]
    ):
        log.info("Создание пайплайна для извлечения признаков")
        return Pipeline(
            [
                (
                    "encoder",
                    FeatureUnion(
                        [
                            (
                                "team_bag",
                                Pipeline(
                                    [
                                        ("select_team", self.ColumnSelector(team_cols)),
                                        ("encode_team", self.team_bag_encoder),
                                    ]
                                ),
                            ),
                            (
                                "player_bag",
                                Pipeline(
                                    [
                                        (
                                            "select_player",
                                            self.ColumnSelector(player_cols),
                                        ),
                                        ("encode_player", self.player_bag_encoder),
                                    ]
                                ),
                            ),
                            (
                                "player_elo",
                                Pipeline(
                                    [
                                        (
                                            "select_player",
                                            self.ColumnSelector(player_cols),
                                        ),
                                        ("elo_encoder", self.player_elo_encode),
                                    ]
                                ),
                            ),
                        ]
                    ),
                )
            ]
        )

    def _select_features_with_logit_and_cv(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        Cs: np.ndarray = np.array([0.08, 0.09, 0.1, 0.2, 0.3]),
        scoring: str = "accuracy",
        random_state: int = 42,
        cv: TimeSeriesSplit = TimeSeriesSplit(10),
        verbose: int = 0,
        n_jobs: int = 1,
    ) -> np.ndarray:
        scores = []
        masks_per_C = []
        for C in Cs:
            logit = LogisticRegression(
                C=C,
                penalty="l1",
                solver="liblinear",
                random_state=random_state,
            )
            fold_masks = []
            for tr_idx, _ in cv.split(X_train, y_train):
                logit.fit(X_train[tr_idx], y_train[tr_idx])
                fold_masks.append(logit.coef_.flatten() != 0)
            mask_C = np.mean(fold_masks, axis=0) > 0.5
            masks_per_C.append(mask_C)

            score = cross_val_score(
                LogisticRegression(random_state=random_state, solver="liblinear"),
                X_train[:, mask_C],
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
            ).mean()
            scores.append(score)

        best_idx = np.argmax(scores)
        best_mask = masks_per_C[best_idx]

        logit_final = LogisticRegression(
            random_state=random_state, solver="liblinear", max_iter=1000
        )
        X_train_selected = X_train[:, best_mask]

        rfecv = RFECV(
            estimator=logit_final,
            step=1,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        rfecv.fit(X_train_selected, y_train)

        final_mask = np.zeros_like(best_mask, dtype=bool)
        final_mask[best_mask] = rfecv.support_

        return final_mask

    def _get_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> dict:
        try:
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            metrics = {
                "accuracy": round(accuracy_score(y_true, y_pred), 2),
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 2),
                "recall": round(recall_score(y_true, y_pred, zero_division=0), 2),
                "f1": round(f1_score(y_true, y_pred, zero_division=0), 2),
                "roc_auc": round(roc_auc_score(y_true, y_proba), 2),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
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
            return sparse.csr_matrix(
                (data, (rows, cols)), shape=(n_samples, n_features), dtype=int
            )

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
            return sparse.csr_matrix(
                (data, (rows, cols)), shape=(n_samples, n_features), dtype=int
            )

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
                    self.elo_dict_[pid] = self.elo_dict_.get(
                        pid, self.base_elo
                    ) + self.k_factor * (score1 - exp1)
                for pid in row[5:]:
                    self.elo_dict_[pid] = self.elo_dict_.get(
                        pid, self.base_elo
                    ) + self.k_factor * (score2 - (1 - exp1))
            self.X_elo_train_ = np.array(X_elo, dtype=float)
            return self

        def transform(self, X):
            X = np.asarray(X)
            if self.X_elo_train_ is not None and X.shape == self.X_elo_train_.shape:
                X_out = self.X_elo_train_
            else:
                X_out = np.array(
                    [
                        [self.elo_dict_.get(pid, self.base_elo) for pid in row]
                        for row in X
                    ],
                    dtype=float,
                )
            return np.array([self._augment_X(row) for row in X_out], dtype=float)
