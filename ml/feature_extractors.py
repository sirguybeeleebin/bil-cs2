from __future__ import annotations

import json
import os

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelectorArray(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[int]) -> None:
        self.columns: list[int] = columns

    def fit(self, X: np.ndarray, y=None) -> ColumnSelectorArray:
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return X[:, self.columns]


class BagEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X: np.ndarray, y=None) -> BagEncoder:
        arr = X
        uniques = np.unique(arr.flatten())
        self.dict_: dict[int, int] = {v: i for i, v in enumerate(uniques)}
        self.n_features_: int = len(self.dict_)
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

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.X_elo_train_ is not None and X.shape == self.X_elo_train_.shape:
            X_out = self.X_elo_train_
        else:
            X_out = np.array(
                [[self.elo_dict_.get(pid, self.base_elo) for pid in row] for row in X],
                dtype=float,
            )
        return np.array([self._augment_X(row) for row in X_out], dtype=float)


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
        for row_idx, row in enumerate(X):
            X_out.append([self.player_stat_dict.get(pid, 0.0) for pid in row])
            if row_idx < len(self.game_ids):
                game_id = self.game_ids[row_idx]
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
        self.X_train_ = np.array(X_out, dtype=float)
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X_out: list[np.ndarray] = []
        for row in X:
            stats = [self.player_stat_dict.get(pid, 0.0) for pid in row]
            X_out.append(self._augment(np.array(stats, dtype=float)))
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
