from typing import Optional
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

class PlayerBagEncoder(BaseEstimator, TransformerMixin):
    player_dict: dict[str, int]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PlayerBagEncoder":        
        uniques = np.unique(X.flatten())
        self.player_dict = {player: idx for idx, player in enumerate(uniques)}
        return self

    def transform(self, X: np.ndarray) -> sparse.csr_matrix:        
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
