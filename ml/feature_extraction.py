import warnings

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

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
    
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
                elo_dict[pid] = elo_dict.get(pid, self.base_elo) + self.k_factor * (score1 - exp1)
            # Update team2
            for pid in players[5:]:
                elo_dict[pid] = elo_dict.get(pid, self.base_elo) + self.k_factor * (score2 - (1 - exp1))

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

    
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim > 1:
            X = X[:, 0]
        if not np.issubdtype(X.dtype, np.datetime64):
            X = X.astype('datetime64[ns]')
        n_samples = len(X)
        out = np.zeros((n_samples, 6), dtype=float)
        out[:, 0] = X.astype('int64') / 1e9       
        years = X.astype('datetime64[Y]').astype(int) + 1970
        out[:, 1] = years
        months = (X.astype('datetime64[M]').astype(int) % 12) + 1
        out[:, 2] = months
        days = X.astype('datetime64[D]') - X.astype('datetime64[M]')
        out[:, 3] = days.astype(int) + 1        
        weekdays = (X.astype('datetime64[D]').view('int64') + 3) % 7
        out[:, 4] = weekdays
        hours = X.astype('datetime64[h]').view('int64') % 24
        out[:, 5] = hours
        return out
    
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
import numpy as np

class PlayerKillsSumFeatureExtractor(BaseEstimator, TransformerMixin):    
    def __init__(self):
        self.player_kills_dict = {}
        self.X_train_kills_ = None

    def fit(self, X, y=None):        
        self.X_train_kills_ = []
        for row in X:
            player_ids = row[:10]
            player_kills = row[10:20]
            row_cum_kills = [self.player_kills_dict.get(pid, 0) for pid in player_ids]
            self.X_train_kills_.append(row_cum_kills)
            for pid, kills in zip(player_ids, player_kills):
                self.player_kills_dict[pid] = self.player_kills_dict.get(pid, 0) + kills        
        self.X_train_kills_ = np.array(self.X_train_kills_)
        return self

    def transform(self, X):        
        if self.X_train_kills_ is not None and X.shape == self.X_train_kills_.shape:
            X_out = self.X_train_kills_
        else:
            X_out = []
            for row in X:
                player_ids = row[:10]
                row_kills = [self.player_kills_dict.get(pid, 0) for pid in player_ids]
                X_out.append(row_kills)
            X_out = np.array(X_out)
        # augment features
        X_aug = np.array([self._augment(row) for row in X_out], dtype=float)
        return X_aug

    def _augment(self, row):        
        left = np.sort(row[:5])
        right = np.sort(row[5:])
        features = np.concatenate([
            left, 
            right, 
            [np.mean(left), np.mean(right), np.mean(left) - np.mean(right)]
        ])
        pairwise_diffs = [left[i] - right[j] for i in range(5) for j in range(5)]
        features = np.concatenate([features, pairwise_diffs])
        return features
