import logging
import warnings
from typing import Optional, Union

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# Настройка логгера
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------- ColumnSelector ----------------
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[int]):
        self.columns: list[int] = columns

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ColumnSelector":
        log.info(f"ColumnSelector: fit завершен")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        log.info(f"ColumnSelector: трансформация данных...")
        X_selected = np.asarray(X)[:, self.columns]
        return X_selected


# ---------------- PlayerBagEncoder ----------------
class PlayerBagEncoder(BaseEstimator, TransformerMixin):
    player_dict: dict[str, int]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PlayerBagEncoder":
        uniques = np.unique(X.flatten())
        self.player_dict = {player: idx for idx, player in enumerate(uniques)}
        log.info(
            f"PlayerBagEncoder: найдено {len(self.player_dict)} уникальных игроков"
        )
        return self

    def transform(self, X: np.ndarray) -> sparse.csr_matrix:
        log.info(f"PlayerBagEncoder: трансформация данных...")
        n_samples, n_features = X.shape[0], len(self.player_dict)
        rows, cols, data = [], [], []

        for i, row in enumerate(X):
            for j, player in enumerate(row):
                col_idx = self.player_dict.get(player)
                if col_idx is not None:
                    rows.append(i)
                    cols.append(col_idx)
                    data.append(1 if j < len(row) // 2 else -1)
            if (i + 1) % 50 == 0:
                log.info(f"PlayerBagEncoder: обработано {i + 1}/{n_samples} строк")

        return sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_samples, n_features), dtype=int
        )


# ---------------- TeamBagEncoder ----------------
class TeamBagEncoder(BaseEstimator, TransformerMixin):
    team_dict: dict[str, int]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TeamBagEncoder":
        uniques = np.unique(X.flatten())
        self.team_dict = {team: idx for idx, team in enumerate(uniques)}
        log.info(f"TeamBagEncoder: найдено {len(self.team_dict)} уникальных команд")
        return self

    def transform(self, X: np.ndarray) -> sparse.csr_matrix:
        log.info("TeamBagEncoder: трансформация данных...")
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
            if (i + 1) % 50 == 0:
                log.info(f"TeamBagEncoder: обработано {i + 1}/{n_samples} строк")

        return sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_samples, n_features), dtype=int
        )


# ---------------- PlayerEloEncoder ----------------
class PlayerEloEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, k_factor: float = 32, base_elo: float = 1000):
        self.k_factor: float = k_factor
        self.base_elo: float = base_elo
        self.elo_dict_: dict[Union[int, str], float] = {}
        self.X_elo_train_: Optional[np.ndarray] = None

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def _augment_X(self, row: np.ndarray) -> np.ndarray:
        x1, x2 = np.sort(row[:5]), np.sort(row[5:])
        features: list[float] = []
        mean1, mean2 = np.mean(x1), np.mean(x2)
        features.extend([mean1, mean2, mean1, -mean2, mean1 - mean2])
        features.extend([x1[i] - x2[j] for i in range(5) for j in range(5)])
        return np.array(features, dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PlayerEloEncoder":
        log.info("PlayerEloEncoder: обучение...")
        X_elo: list[list[float]] = []
        for idx, (row, outcome) in enumerate(zip(X, y), start=1):
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
            if idx % 50 == 0:
                log.info(f"PlayerEloEncoder: обработано {idx}/{len(X)} строк")
        self.X_elo_train_ = np.array(X_elo, dtype=float)
        log.info("PlayerEloEncoder: обучение завершено")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        log.info("PlayerEloEncoder: трансформация данных...")
        if self.X_elo_train_ is not None and X.shape == self.X_elo_train_.shape:
            X_out = self.X_elo_train_
        else:
            X_out = np.array(
                [[self.elo_dict_.get(pid, self.base_elo) for pid in row] for row in X],
                dtype=float,
            )
        X_aug = np.array([self._augment_X(row) for row in X_out], dtype=float)
        return X_aug
