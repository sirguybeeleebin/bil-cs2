from __future__ import annotations

import json
import logging
import os
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator, TransformerMixin

log = logging.getLogger("train_model")
logging.basicConfig(level=logging.INFO)


class ColumnSelectorArray(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[int]) -> None:
        self.columns: list[int] = columns

    def fit(self, X: np.ndarray, y=None) -> ColumnSelectorArray:
        log.info(f"ColumnSelectorArray: fit на {X.shape[0]} строках")
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        log.info(f"ColumnSelectorArray: transform с колонками {self.columns}")
        return X[:, self.columns]


class BagEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X: np.ndarray, y=None) -> BagEncoder:
        arr = X
        uniques = np.unique(arr.flatten())
        self.dict_: dict[int, int] = {v: i for i, v in enumerate(uniques)}
        self.n_features_: int = len(self.dict_)
        log.info(
            f"BagEncoder: создан словарь с {self.n_features_} уникальными значениями"
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> csr_matrix:
        arr = X
        n_rows, n_cols = arr.shape
        split_idx = n_cols // 2
        bag = lil_matrix((n_rows, self.n_features_), dtype=np.int8)
        for i in range(n_rows):
            for j in range(n_cols):
                val = arr[i, j]
                idx = self.dict_.get(val)
                if idx is None:
                    continue
                bag[i, idx] = 1 if j < split_idx else -1
        log.info(f"BagEncoder: трансформация массива размером {X.shape}")
        return csr_matrix(bag)


class PlayerEloEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, k_factor: float = 32, base_elo: float = 1000) -> None:
        self.k_factor: float = k_factor
        self.base_elo: float = base_elo
        self.elo_dict_: dict[int, float] = {}
        self.X_elo_train_: np.ndarray | None = None

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def _augment_X(self, row: np.ndarray) -> np.ndarray:
        x1, x2 = np.sort(row[:5]), np.sort(row[5:])
        features: list[float] = []
        mean1, mean2 = np.mean(x1), np.mean(x2)
        features.extend([mean1, mean2, mean1, -mean2, mean1 - mean2])
        features.extend([x1[i] - x2[j] for i in range(5) for j in range(5)])
        return np.array(features, dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> PlayerEloEncoder:
        X_elo: list[list[float]] = []
        total = X.shape[0]
        log.info(f"PlayerEloEncoder: обучение на {total} матчах")
        for row_idx, (row, outcome) in enumerate(zip(X, y)):
            elos_before = [self.elo_dict_.get(pid, self.base_elo) for pid in row]
            X_elo.append(elos_before)
            avg1, avg2 = np.mean(elos_before[:5]), np.mean(elos_before[5:])
            exp1 = self._expected_score(avg1, avg2)
            score1, score2 = int(outcome == 1), int(outcome == 0)

            for i, pid in enumerate(row[:5]):
                old_elo = self.elo_dict_.get(pid, self.base_elo)
                self.elo_dict_[pid] = old_elo + self.k_factor * (score1 - exp1)
                log.info(
                    f"Игра {row_idx + 1}/{total}, левый игрок {pid}: Elo до={old_elo:.2f}, Elo после={self.elo_dict_[pid]:.2f}"
                )

            for i, pid in enumerate(row[5:]):
                old_elo = self.elo_dict_.get(pid, self.base_elo)
                self.elo_dict_[pid] = old_elo + self.k_factor * (score2 - (1 - exp1))
                log.info(
                    f"Игра {row_idx + 1}/{total}, правый игрок {pid}: Elo до={old_elo:.2f}, Elo после={self.elo_dict_[pid]:.2f}"
                )

            log.info(
                f"Игра {row_idx + 1}/{total}: avg1={avg1:.2f}, avg2={avg2:.2f}, expected1={exp1:.2f}, исход={outcome}"
            )

        self.X_elo_train_ = np.array(X_elo, dtype=float)
        log.info("PlayerEloEncoder: обучение завершено")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        log.info(f"PlayerEloEncoder: трансформация массива размером {X.shape}")
        if self.X_elo_train_ is not None and X.shape == self.X_elo_train_.shape:
            X_out = self.X_elo_train_
        else:
            X_out = np.array(
                [[self.elo_dict_.get(pid, self.base_elo) for pid in row] for row in X],
                dtype=float,
            )
        augmented = np.array([self._augment_X(row) for row in X_out], dtype=float)
        log.info(
            f"PlayerEloEncoder: трансформация завершена, выходной массив размером {augmented.shape}"
        )
        return augmented


class PlayerStatisticSumExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        game_ids: list[int],
        path_to_dir: str = "data/games_raw",
        key: str = "kills",
    ) -> None:
        self.game_ids: list[int] = game_ids
        self.path_to_dir: str = path_to_dir
        self.key: str = key
        self.player_stat_dict: dict[int, float] = {}

    def fit(self, X: np.ndarray, y=None) -> PlayerStatisticSumExtractor:
        X_out: list[list[float]] = []
        total = X.shape[0]
        log.info(
            f"PlayerStatisticSumExtractor: обучение на {X.shape[0]} матчах, ключ '{self.key}'"
        )
        for row_idx, row in enumerate(X):
            X_out.append([self.player_stat_dict.get(pid, 0.0) for pid in row])
            if row_idx < len(self.game_ids):
                game_id = self.game_ids[row_idx]
                try:
                    with open(
                        os.path.join(self.path_to_dir, f"{game_id}.json"),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        game = json.load(f)
                    for p in game["players"]:
                        p_id = p["player"]["id"]
                        current = self.player_stat_dict.get(p_id, 0.0)
                        current += p.get(self.key, 0.0) or 0.0
                        self.player_stat_dict[p_id] = current
                    log.info(
                        f"Игра {row_idx + 1}/{total}: обновлены статистики игроков для игры {game_id}"
                    )
                except Exception as e:
                    log.error(
                        f"PlayerStatisticSumExtractor: ошибка чтения игры {game_id}: {e}"
                    )
        self.X_train_ = np.array(X_out, dtype=float)
        log.info(
            f"PlayerStatisticSumExtractor: обучение завершено для ключа '{self.key}'"
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X_out: list[np.ndarray] = []
        for row in X:
            stats = [self.player_stat_dict.get(pid, 0.0) for pid in row]
            X_out.append(self._augment(np.array(stats, dtype=float)))
        log.info(
            f"PlayerStatisticSumExtractor: трансформация массива размером {X.shape}"
        )
        return np.array(X_out, dtype=float)

    def _augment(self, row: np.ndarray) -> np.ndarray:
        left_team, right_team = row[:5], row[5:]
        left_sorted, right_sorted = np.sort(left_team), np.sort(right_team)
        mean_left, mean_right = np.mean(left_sorted), np.mean(right_sorted)
        features: list[float] = [
            *left_sorted,
            *right_sorted,
            mean_left,
            mean_right,
            mean_left - mean_right,
        ]
        for i in range(5):
            for j in range(5):
                features.append(left_sorted[i] - right_sorted[j])
        return np.array(features, dtype=float)


class PlayerMapStatisticSumExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        game_ids: list[int],
        path_to_dir: str = "data/games_raw",
        key: str = "kills",
    ) -> None:
        self.game_ids: list[int] = game_ids
        self.path_to_dir: str = path_to_dir
        self.key: str = key
        self.player_stat_dict: dict[int, dict[int, float]] = {}

    def fit(self, X: np.ndarray, y=None) -> PlayerMapStatisticSumExtractor:
        X_out: list[list[float]] = []
        total = X.shape[0]
        log.info(
            f"PlayerMapStatisticSumExtractor: обучение на {total} матчах, ключ '{self.key}'"
        )

        for row_idx, row in enumerate(X):
            map_id = row[0]  # предполагаем, что map_id в первой колонке
            player_ids = row[1:]
            stats_row: list[float] = []

            if row_idx < len(self.game_ids):
                game_id = self.game_ids[row_idx]
                try:
                    with open(
                        os.path.join(self.path_to_dir, f"{game_id}.json"),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        game = json.load(f)
                    map_id = game["map"]["id"]  # берем карту из файла, на всякий случай
                    if map_id not in self.player_stat_dict:
                        self.player_stat_dict[map_id] = {}
                    for p in game["players"]:
                        p_id = p["player"]["id"]
                        current = self.player_stat_dict[map_id].get(p_id, 0.0)
                        current += p.get(self.key, 0.0) or 0.0
                        self.player_stat_dict[map_id][p_id] = current
                        stats_row.append(self.player_stat_dict[map_id][p_id])
                    log.info(
                        f"Игра {row_idx + 1}/{total}: обновлены статистики игроков для игры {game_id} на карте {map_id}"
                    )
                except Exception as e:
                    log.error(
                        f"PlayerMapStatisticSumExtractor: ошибка чтения игры {game_id}: {e}"
                    )

            # Если нет данных для карты, используем нули
            if not stats_row:
                stats_row = [0.0 for _ in player_ids]

            X_out.append(stats_row)

        self.X_train_ = np.array(X_out, dtype=float)
        log.info(
            f"PlayerMapStatisticSumExtractor: обучение завершено для ключа '{self.key}'"
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X_out: list[np.ndarray] = []
        X.shape[0]

        for row_idx, row in enumerate(X):
            map_id = row[0]
            player_ids = row[1:]
            if map_id in self.player_stat_dict:
                stats = [
                    self.player_stat_dict[map_id].get(pid, 0.0) for pid in player_ids
                ]
            else:
                stats = [0.0 for _ in player_ids]
            X_out.append(self._augment(np.array(stats, dtype=float)))

        log.info(
            f"PlayerMapStatisticSumExtractor: трансформация массива размером {X.shape}"
        )
        return np.array(X_out, dtype=float)

    def _augment(self, row: np.ndarray) -> np.ndarray:
        left_team, right_team = row[:5], row[5:]
        left_sorted, right_sorted = np.sort(left_team), np.sort(right_team)
        mean_left, mean_right = np.mean(left_sorted), np.mean(right_sorted)
        features: list[float] = [
            *left_sorted,
            *right_sorted,
            mean_left,
            mean_right,
            mean_left - mean_right,
        ]
        for i in range(5):
            for j in range(5):
                features.append(left_sorted[i] - right_sorted[j])
        return np.array(features, dtype=float)


class PlayerRoundWinSumExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        game_ids: list[int],
        path_to_dir: str = "data/games_raw",
    ) -> None:
        self.game_ids: list[int] = game_ids
        self.path_to_dir: str = path_to_dir
        self.player_round_dict: dict[int, float] = {}

    def fit(self, X: np.ndarray, y=None) -> "PlayerRoundWinSumExtractor":
        X_out: list[list[float]] = []
        total = X.shape[0]

        for row_idx, row in enumerate(X):
            X_out.append([self.player_round_dict.get(pid, 0.0) for pid in row])

            if row_idx < len(self.game_ids):
                game_id = self.game_ids[row_idx]
                try:
                    with open(
                        os.path.join(self.path_to_dir, f"{game_id}.json"),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        game = json.load(f)

                    team_players = {
                        p["player"]["id"]: p["team"]["id"] for p in game["players"]
                    }
                    win_round_counts = Counter(r["winner_team"] for r in game["rounds"])

                    for p_id in row:
                        t_id = team_players[p_id]
                        count = win_round_counts.get(t_id, 0)
                        current = self.player_round_dict.get(p_id, 0.0)
                        current += count  # <-- was incorrectly adding t_id
                        self.player_round_dict[p_id] = current

                    log.info(
                        f"Игра {row_idx + 1}/{total}: обновлены статистики игроков для игры {game_id}"
                    )
                except Exception as e:
                    log.error(
                        f"PlayerRoundWinSumExtractor: ошибка чтения игры {game_id}: {e}"
                    )

        self.X_train_ = np.array(X_out, dtype=float)
        log.info("PlayerRoundWinSumExtractor: обучение завершено")
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X_out: list[np.ndarray] = []

        for row in X:
            stats = [self.player_round_dict.get(pid, 0.0) for pid in row]
            X_out.append(self._augment(np.array(stats, dtype=float)))

        log.info(
            f"PlayerRoundWinSumExtractor: трансформация массива размером {X.shape}"
        )
        return np.array(X_out, dtype=float)

    def _augment(self, row: np.ndarray) -> np.ndarray:
        left_team, right_team = row[:5], row[5:]
        left_sorted, right_sorted = np.sort(left_team), np.sort(right_team)
        mean_left, mean_right = np.mean(left_sorted), np.mean(right_sorted)

        features: list[float] = [
            *left_sorted,
            *right_sorted,
            mean_left,
            mean_right,
            mean_left - mean_right,
        ]

        for i in range(5):
            for j in range(5):
                features.append(left_sorted[i] - right_sorted[j])

        return np.array(features, dtype=float)


class PlayerRoundWinMapSumExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        game_ids: list[int],
        path_to_dir: str = "data/games_raw",
    ) -> None:
        self.game_ids: list[int] = game_ids
        self.path_to_dir: str = path_to_dir
        self.player_round_dict: dict[int, dict[int, float]] = {}

    def fit(self, X: np.ndarray, y=None) -> "PlayerRoundWinMapSumExtractor":
        X_out: list[list[float]] = []
        total = X.shape[0]
        log.info(f"PlayerRoundWinMapSumExtractor: обучение на {total} матчах")

        for row_idx, row in enumerate(X):
            map_id = row[0]
            player_ids = row[1:]
            X_out.append(
                [
                    self.player_round_dict.get(map_id, {}).get(pid, 0.0)
                    for pid in player_ids
                ]
            )

            if row_idx < len(self.game_ids):
                game_id = self.game_ids[row_idx]
                try:
                    with open(
                        os.path.join(self.path_to_dir, f"{game_id}.json"),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        game = json.load(f)

                    map_id_game = game["map"]["id"]
                    if map_id_game not in self.player_round_dict:
                        self.player_round_dict[map_id_game] = {}

                    team_players = {
                        p["player"]["id"]: p["team"]["id"] for p in game["players"]
                    }
                    win_round_counts = Counter(r["winner_team"] for r in game["rounds"])

                    for p_id in player_ids:
                        t_id = team_players[p_id]
                        count = win_round_counts.get(t_id, 0)
                        current = self.player_round_dict[map_id_game].get(p_id, 0.0)
                        current += count
                        self.player_round_dict[map_id_game][p_id] = current

                    log.info(
                        f"Игра {row_idx + 1}/{total}: обновлены статистики игроков для игры {game_id} на карте {map_id_game}"
                    )
                except Exception as e:
                    log.error(
                        f"PlayerRoundWinMapSumExtractor: ошибка чтения игры {game_id}: {e}"
                    )

        self.X_train_ = np.array(X_out, dtype=float)
        log.info("PlayerRoundWinMapSumExtractor: обучение завершено")
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X_out: list[np.ndarray] = []

        for row in X:
            map_id = row[0]
            player_ids = row[1:]
            if map_id in self.player_round_dict:
                stats = [
                    self.player_round_dict[map_id].get(pid, 0.0) for pid in player_ids
                ]
            else:
                stats = [0.0 for _ in player_ids]
            X_out.append(self._augment(np.array(stats, dtype=float)))

        log.info(
            f"PlayerRoundWinMapSumExtractor: трансформация массива размером {X.shape}"
        )
        return np.array(X_out, dtype=float)

    def _augment(self, row: np.ndarray) -> np.ndarray:
        left_team, right_team = row[:5], row[5:]
        left_sorted, right_sorted = np.sort(left_team), np.sort(right_team)
        mean_left, mean_right = np.mean(left_sorted), np.mean(right_sorted)

        features: list[float] = [
            *left_sorted,
            *right_sorted,
            mean_left,
            mean_right,
            mean_left - mean_right,
        ]

        for i in range(5):
            for j in range(5):
                features.append(left_sorted[i] - right_sorted[j])

        return np.array(features, dtype=float)
