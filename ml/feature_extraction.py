import warnings

import numpy as np
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
