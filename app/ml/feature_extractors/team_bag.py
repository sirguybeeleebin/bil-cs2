from typing import Optional
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

class TeamBagEncoder(BaseEstimator, TransformerMixin):
    team_dict: dict[str, int]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TeamBagEncoder":
        uniques = np.unique(X.flatten())
        self.team_dict = {team: idx for idx, team in enumerate(uniques)}
        return self

    def transform(self, X: np.ndarray) -> sparse.csr_matrix:
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
