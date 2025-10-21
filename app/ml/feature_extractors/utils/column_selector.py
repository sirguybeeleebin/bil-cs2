import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[int]):
        self.columns: list[int] = columns

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ColumnSelector":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return X[:, self.columns]
