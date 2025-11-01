import json
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelectorArray(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[int]) -> None:
        self.columns = columns

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = np.atleast_2d(X)
        return X[:, self.columns]


class BagEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X: np.ndarray, y=None):
        uniques = np.unique(X.flatten())
        self.dict_: dict[int, int] = {v: i for i, v in enumerate(uniques)}
        self.n_features_ = len(self.dict_)
        return self

    def transform(self, X: np.ndarray, y=None) -> csr_matrix:
        n_rows, n_cols = X.shape
        split_idx = n_cols // 2
        bag = lil_matrix((n_rows, self.n_features_), dtype=np.int8)
        for i in range(n_rows):
            for j in range(n_cols):
                val = X[i, j]
                idx = self.dict_.get(val)
                if idx is None:
                    continue
                bag[i, idx] = 1 if j < split_idx else -1
        return csr_matrix(bag)


class PlayerEloEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, k_factor: float = 32, base_elo: float = 1000):
        self.k_factor = k_factor
        self.base_elo = base_elo
        self.elo_dict_: dict[int, float] = {}
        self.X_train: np.ndarray | None = None

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_augmented: list[list[float]] = []
        for row, outcome in zip(X, y):
            elos_before = [self.elo_dict_.get(pid, self.base_elo) for pid in row]
            avg1, avg2 = np.mean(elos_before[:5]), np.mean(elos_before[5:])
            exp1 = self._expected_score(avg1, avg2)
            score1, score2 = int(outcome == 1), int(outcome == 0)
            for pid in row[:5]:
                old_elo = self.elo_dict_.get(pid, self.base_elo)
                self.elo_dict_[pid] = old_elo + self.k_factor * (score1 - exp1)
            for pid in row[5:]:
                old_elo = self.elo_dict_.get(pid, self.base_elo)
                self.elo_dict_[pid] = old_elo + self.k_factor * (score2 - (1 - exp1))
            X_augmented.append(self._augment(elos_before))
        self.X_train = np.array(X_augmented, dtype=float)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is not None and X.shape == self.X_train.shape:
            return self.X_train
        X_augmented: list[list[float]] = []
        for row in X:
            elos = [self.elo_dict_.get(pid, self.base_elo) for pid in row]
            X_augmented.append(self._augment(elos))
        return np.array(X_augmented, dtype=float)

    def _augment(self, row: np.ndarray) -> np.ndarray:
        left, right = row[:5], row[5:]
        left_sorted, right_sorted = np.sort(left), np.sort(right)
        mean_left, mean_right = np.mean(left_sorted), np.mean(right_sorted)
        features = [
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


class PlayerStatSumExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self, game_ids: list[int], stat_key: str, path_to_dir: str = "data/games_raw"
    ):
        self.game_ids = game_ids
        self.stat_key = stat_key
        self.path_to_dir = path_to_dir
        self.dict: dict[int, float] = {}

    def fit(self, X: np.ndarray, y=None):
        X_out = []
        for row_idx, row in enumerate(X):
            X_out.append([self.dict.get(pid, 0.0) for pid in row])
            if row_idx < len(self.game_ids):
                game_id = self.game_ids[row_idx]
                with open(
                    f"{self.path_to_dir}/{game_id}.json", "r", encoding="utf-8"
                ) as f:
                    game = json.load(f)
                for p in game["players"]:
                    pid = p["player"]["id"]
                    value = float(p.get(self.stat_key) or 0.0)
                    if value is None:
                        value = 0.0
                    self.dict[pid] = self.dict.get(pid, 0.0) + float(value)
        self.X_train = np.array(X_out, dtype=float)
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X_out = []
        if X.shape == self.X_train.shape:
            for row in self.X_train:
                X_out.append(self._augment(row))
            return np.array(X_out, dtype=float)
        for row in X:
            stats = [self.dict.get(pid, 0.0) for pid in row]
            X_out.append(self._augment(np.array(stats, dtype=float)))
        return np.array(X_out, dtype=float)

    def _augment(self, row: np.ndarray) -> np.ndarray:
        left, right = row[:5], row[5:]
        left_sorted, right_sorted = np.sort(left), np.sort(right)
        mean_left, mean_right = np.mean(left_sorted), np.mean(right_sorted)
        features = [
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


class PlayerMapStatSumExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self, game_ids: list[int], stat_key: str, path_to_dir: str = "data/games_raw"
    ):
        self.game_ids = game_ids
        self.stat_key = stat_key
        self.path_to_dir = path_to_dir
        self.dict: dict[int, dict[int, float]] = {}

    def fit(self, X: np.ndarray, y=None):
        X_out = []
        for row_idx, row in enumerate(X):
            map_id = int(row[0])
            stats_row = []
            for pid in row[1:]:
                stats_row.append(self.dict.get(pid, {}).get(map_id, 0.0))
            X_out.append(stats_row)

            if row_idx < len(self.game_ids):
                game_id = self.game_ids[row_idx]
                with open(
                    f"{self.path_to_dir}/{game_id}.json", "r", encoding="utf-8"
                ) as f:
                    game = json.load(f)
                for p in game["players"]:
                    pid = p["player"]["id"]
                    value = float(p.get(self.stat_key) or 0.0)
                    if pid not in self.dict:
                        self.dict[pid] = {}
                    self.dict[pid][map_id] = self.dict[pid].get(map_id, 0.0) + value
        self.X_train = np.array(X_out, dtype=float)
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X_out = []
        for row in X:
            map_id = int(row[0])
            stats = [self.dict.get(pid, {}).get(map_id, 0.0) for pid in row[1:]]
            X_out.append(self._augment(np.array(stats, dtype=float)))
        return np.array(X_out, dtype=float)

    def _augment(self, row: np.ndarray) -> np.ndarray:
        left, right = row[:5], row[5:]
        left_sorted, right_sorted = np.sort(left), np.sort(right)
        mean_left, mean_right = np.mean(left_sorted), np.mean(right_sorted)
        features = [
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
    def __init__(self, game_ids: list[int], path_to_dir: str = "data/games_raw"):
        self.game_ids = game_ids
        self.path_to_dir = path_to_dir
        self.dict: dict[int, float] = {}

    def fit(self, X: np.ndarray, y=None):
        X_out = []
        for row_idx, row in enumerate(X):
            X_out.append([self.dict.get(pid, 0.0) for pid in row])
            if row_idx < len(self.game_ids):
                game_id = self.game_ids[row_idx]
                with open(
                    f"{self.path_to_dir}/{game_id}.json", "r", encoding="utf-8"
                ) as f:
                    game = json.load(f)
                round_win_count = Counter(
                    r["winner_team"] for r in game.get("rounds", [])
                )
                for player in game.get("players", []):
                    team_id = player["team"]["id"]
                    player_id = player["player"]["id"]
                    value = round_win_count.get(team_id, 0.0)
                    self.dict[player_id] = self.dict.get(player_id, 0.0) + float(value)
        self.X_train = np.array(X_out, dtype=float)
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X_out = []
        if hasattr(self, "X_train") and X.shape == self.X_train.shape:
            for row in self.X_train:
                X_out.append(self._augment(row))
            return np.array(X_out, dtype=float)
        for row in X:
            stats = [self.dict.get(pid, 0.0) for pid in row]
            X_out.append(self._augment(np.array(stats, dtype=float)))
        return np.array(X_out, dtype=float)

    def _augment(self, row: np.ndarray) -> np.ndarray:
        left, right = row[:5], row[5:]
        left_sorted, right_sorted = np.sort(left), np.sort(right)
        mean_left, mean_right = np.mean(left_sorted), np.mean(right_sorted)
        features = [
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


class PlayerRoundFirstHalfWinSumExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, game_ids: list[int], path_to_dir: str = "data/games_raw"):
        self.game_ids = game_ids
        self.path_to_dir = path_to_dir
        self.dict: dict[int, float] = {}

    def fit(self, X: np.ndarray, y=None):
        X_out = []
        for row_idx, row in enumerate(X):
            X_out.append([self.dict.get(pid, 0.0) for pid in row])
            if row_idx < len(self.game_ids):
                game_id = self.game_ids[row_idx]
                with open(
                    f"{self.path_to_dir}/{game_id}.json", "r", encoding="utf-8"
                ) as f:
                    game = json.load(f)

                # Only consider rounds with round_number <= 15
                first_half_rounds = [
                    r for r in game.get("rounds", []) if r.get("round", 0) <= 15
                ]
                round_win_count = Counter(r["winner_team"] for r in first_half_rounds)

                for player in game.get("players", []):
                    team_id = player["team"]["id"]
                    player_id = player["player"]["id"]
                    value = round_win_count.get(team_id, 0.0)
                    self.dict[player_id] = self.dict.get(player_id, 0.0) + float(value)

        self.X_train = np.array(X_out, dtype=float)
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X_out = []
        if hasattr(self, "X_train") and X.shape == self.X_train.shape:
            for row in self.X_train:
                X_out.append(self._augment(row))
            return np.array(X_out, dtype=float)

        for row in X:
            stats = [self.dict.get(pid, 0.0) for pid in row]
            X_out.append(self._augment(np.array(stats, dtype=float)))
        return np.array(X_out, dtype=float)

    def _augment(self, row: np.ndarray) -> np.ndarray:
        left, right = row[:5], row[5:]
        left_sorted, right_sorted = np.sort(left), np.sort(right)
        mean_left, mean_right = np.mean(left_sorted), np.mean(right_sorted)
        features = [
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
