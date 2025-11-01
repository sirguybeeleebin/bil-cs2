import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


class LogitL1CVFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cv=TimeSeriesSplit(10), random_state=42, C=1.0):
        self.cv = cv
        self.C = C
        self.random_state = random_state
        self.selected_idx_: np.ndarray | None = None

    def fit(self, X, y):
        n_features = X.shape[1]
        nonzero_counts = np.zeros(n_features, dtype=int)
        n_splits = self.cv.get_n_splits()
        for train_idx, val_idx in self.cv.split(X, y):
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=self.C,
                random_state=self.random_state,
            )
            model.fit(X_train_fold, y_train_fold)
            coefs = model.coef_.ravel()
            nonzero_counts += (coefs != 0).astype(int)
        threshold = int(np.ceil(n_splits / 2))
        self.selected_idx_ = np.where(nonzero_counts >= threshold)[0]
        return self

    def transform(self, X):
        if self.selected_idx_ is None:
            raise RuntimeError("Fit the selector first")
        return X[:, self.selected_idx_]
