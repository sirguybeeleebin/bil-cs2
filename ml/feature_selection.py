import warnings

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


class RecursiveL1Selector(BaseEstimator, TransformerMixin):
    def __init__(self, C=1, cv=None):
        self.C = C
        self.cv = cv
        self.features_mask_ = None

    def fit(self, X, y):
        X_dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
        mask_all = np.ones(X_dense.shape[1], dtype=bool)
        iteration = 0
        while True:
            iteration += 1
            masks = []
            for train_idx, val_idx in self.cv.split(X_dense[:, mask_all]):
                X_train, y_train = X_dense[train_idx][:, mask_all], y[train_idx]
                model = LogisticRegression(
                    solver="liblinear",
                    penalty="l1",
                    C=self.C,
                    max_iter=1000,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                masks.append(model.coef_[0] != 0)
            majority_mask = np.vstack(masks).mean(axis=0) >= 0.5
            prev_sum = mask_all.sum()
            mask_all[np.where(mask_all)[0][~majority_mask]] = False
            if prev_sum == mask_all.sum():
                break
        self.features_mask_ = mask_all
        return self

    def transform(self, X):
        X_dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
        return X_dense[:, self.features_mask_]
