import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
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
        n_samples = X.shape[0]
        n_features = len(self.player_dict)
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
        n_samples = X.shape[0]
        n_features = len(self.team_dict)
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
        self.X_shape_ = None

    def _expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def _augment_X(self, row):
        x1 = np.sort(row[:5])
        x2 = np.sort(row[5:])
        features = []
        mean1 = np.mean(x1)
        mean2 = np.mean(x2)
        features.extend([mean1, mean2, mean1, -mean2, mean1 - mean2])
        for i in range(5):
            for j in range(5):
                features.append(x1[i] - x2[j])
        return np.array(features, dtype=float)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X_elo = []
        for row, outcome in zip(X, y):
            elos_before = [self.elo_dict_.get(pid, self.base_elo) for pid in row]
            X_elo.append(elos_before)
            avg1 = np.mean(elos_before[:5])
            avg2 = np.mean(elos_before[5:])
            exp1 = self._expected_score(avg1, avg2)
            score1 = 1 if outcome == 1 else 0
            score2 = 1 - score1
            for pid in row[:5]:
                self.elo_dict_[pid] = self.elo_dict_.get(
                    pid, self.base_elo
                ) + self.k_factor * (score1 - exp1)
            for pid in row[5:]:
                self.elo_dict_[pid] = self.elo_dict_.get(
                    pid, self.base_elo
                ) + self.k_factor * (score2 - (1 - exp1))
        self.X_elo_train_ = np.array(X_elo, dtype=float)
        self.X_shape_ = X.shape
        return self

    def transform(self, X):
        if X.shape == self.X_elo_train_.shape:
            X_out = np.copy(self.X_elo_train_)
        else:
            X_out = np.array(
                [[self.elo_dict_.get(pid, self.base_elo) for pid in row] for row in X],
                dtype=float,
            )
        X_aug = np.array([self._augment_X(row) for row in X_out], dtype=float)
        return X_aug


class PlayerMapEloEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, k_factor=32, base_elo=1000):
        self.k_factor = k_factor
        self.base_elo = base_elo
        self.elo_dicts_ = {}  # {map_id: {player_id: elo}}
        self.X_elo_train_ = None
        self.X_shape_ = None

    def _expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def _augment_X(self, row):
        x1 = np.sort(row[:5])
        x2 = np.sort(row[5:])
        features = []
        mean1 = np.mean(x1)
        mean2 = np.mean(x2)
        features.extend([mean1, mean2, mean1, -mean2, mean1 - mean2])
        for i in range(5):
            for j in range(5):
                features.append(x1[i] - x2[j])
        return np.array(features, dtype=float)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X_elo = []

        for row, outcome in zip(X, y):
            map_id = row[0]
            players = row[1:]

            if map_id not in self.elo_dicts_:
                self.elo_dicts_[map_id] = {}

            elo_dict = self.elo_dicts_[map_id]
            elos_before = [elo_dict.get(pid, self.base_elo) for pid in players]
            X_elo.append(elos_before)

            avg1 = np.mean(elos_before[:5])
            avg2 = np.mean(elos_before[5:])
            exp1 = self._expected_score(avg1, avg2)
            score1 = 1 if outcome == 1 else 0
            score2 = 1 - score1

            # Update team1
            for pid in players[:5]:
                elo_dict[pid] = elo_dict.get(pid, self.base_elo) + self.k_factor * (
                    score1 - exp1
                )
            # Update team2
            for pid in players[5:]:
                elo_dict[pid] = elo_dict.get(pid, self.base_elo) + self.k_factor * (
                    score2 - (1 - exp1)
                )

        self.X_elo_train_ = np.array(X_elo, dtype=float)
        self.X_shape_ = X.shape
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_out = []

        if self.X_elo_train_ is not None and X.shape == self.X_shape_:
            # Use training cache for efficiency
            X_out = np.copy(self.X_elo_train_)
        else:
            for row in X:
                map_id = row[0]
                players = row[1:]
                elo_dict = self.elo_dicts_.get(map_id, {})
                elos = [elo_dict.get(pid, self.base_elo) for pid in players]
                X_out.append(elos)

        X_out = np.array(X_out, dtype=float)
        X_aug = np.array([self._augment_X(row) for row in X_out], dtype=float)
        return X_aug


class PlayerKillsSumFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, path_to_games_raw_dir: str, game_ids: list[int]):
        self.path_to_games_raw_dir = path_to_games_raw_dir
        self.game_ids = game_ids
        self.player_dict: dict[int, int] = {}
        self.X_train: np.ndarray = np.zeros((len(self.game_ids), 10))

    def fit(self, X, y=None):
        for i, game_id in enumerate(self.game_ids):
            self.X_train[i] = [self.player_dict.get(p_id, 0) for p_id in X[i]]
            with open(
                os.path.join(self.path_to_games_raw_dir, f"{game_id}.json"), "r"
            ) as f:
                game = json.load(f)
            player_kills = {p["player"]["id"]: p.get("kills", 0) or 0 for p in game}
            for p_id in X[i]:
                self.player_dict[p_id] = self.player_dict.get(
                    p_id, 0
                ) + player_kills.get(p_id, 0)
        return self

    def transform(self, X):
        if X.shape == self.X_train.shape:
            return self.X_train
        for i, row in enumerate(X):
            X[i] = [self.player_dict.get(p_id, 0) for p_id in row]
        return X


class PlayerStatSumFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, stat_name: str, path_to_games_raw_dir: str, game_ids: list[int]):
        self.stat_name = stat_name
        self.path_to_games_raw_dir = path_to_games_raw_dir
        self.game_ids = game_ids
        self.player_dict: dict[int, int] = {}
        self.X_train: np.ndarray = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=int)
        augmented = []
        for i, game_id in enumerate(self.game_ids):
            stat_row = [self.player_dict.get(p_id, 0) for p_id in X[i]]
            with open(
                os.path.join(self.path_to_games_raw_dir, f"{game_id}.json"), "r"
            ) as f:
                game = json.load(f)
            player_stats = {
                p["player"]["id"]: p.get(self.stat_name, 0) or 0
                for p in game["players"]
            }
            for p_id in X[i]:
                self.player_dict[p_id] = self.player_dict.get(
                    p_id, 0
                ) + player_stats.get(p_id, 0)
            augmented_row = self._augment(stat_row)
            augmented.append(augmented_row)
        self.X_train = np.array(augmented, dtype=float)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=int)
        if X.shape[0] == len(self.game_ids):
            return self.X_train
        X_out = []
        for row in X:
            stat_row = [self.player_dict.get(p_id, 0) for p_id in row]
            augmented_row = self._augment(stat_row)
            X_out.append(augmented_row)
        return np.array(X_out, dtype=float)

    def _augment(self, row):
        team1 = np.sort(row[:5])
        team2 = np.sort(row[5:])
        features = np.zeros(10 + 3 + 25)
        features[:5] = team1
        features[5:10] = team2
        features[10:13] = [
            np.mean(team1),
            np.mean(team2),
            np.mean(team1) - np.mean(team2),
        ]
        idx = 13
        for i in range(5):
            for j in range(5):
                features[idx] = team1[i] - team2[j]
                idx += 1
        return features


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert input to pandas Series of datetime
        if isinstance(X, pd.DataFrame):
            dates = pd.to_datetime(X.iloc[:, 0])
        else:
            dates = pd.to_datetime(np.ravel(X))

        # Ensure 'dates' is a Series (DatetimeIndex has no .dt)
        if isinstance(dates, pd.DatetimeIndex):
            dates = pd.Series(dates)

        # Return timestamp + date components
        features = np.column_stack(
            [
                dates.view("int64") // 10**9,  # UNIX timestamp (seconds)
                dates.dt.year,
                dates.dt.month,
                dates.dt.day,
                dates.dt.dayofweek,
                dates.dt.hour,
            ]
        )
        return features
