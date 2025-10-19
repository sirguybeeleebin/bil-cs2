import warnings
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


class RecursiveL1Selector(BaseEstimator, TransformerMixin):
    def __init__(self, C=1, cv=None, random_state: int = 42):
        self.C = C
        self.cv = cv
        self.features_mask_ = None
        self.random_state = random_state

    def fit(self, X, y):
        X_dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
        mask_all = np.ones(X_dense.shape[1], dtype=bool)

        while True:
            if mask_all.sum() == 0:
                break

            masks = []
            for train_idx, _ in self.cv.split(X_dense[:, mask_all]):
                X_train, y_train = X_dense[train_idx][:, mask_all], y[train_idx]
                if X_train.shape[1] == 0:
                    continue
                model = LogisticRegression(
                    solver="liblinear",
                    penalty="l1",
                    C=self.C,
                    max_iter=1000,
                    random_state=self.random_state,
                )
                model.fit(X_train, y_train)
                masks.append(model.coef_[0] != 0)

            if not masks:
                break

            majority_mask = np.vstack(masks).mean(axis=0) >= 0.5
            
            if not majority_mask.any():                
                coef_abs = np.abs(model.coef_[0])
                majority_mask[np.argmax(coef_abs)] = True

            prev_sum = mask_all.sum()
            mask_all[np.where(mask_all)[0][~majority_mask]] = False

            if prev_sum == mask_all.sum():
                break

        self.features_mask_ = mask_all
        return self

    def transform(self, X):
        X_dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
        if self.features_mask_.sum() == 0:
            return np.empty((X_dense.shape[0], 0))
        return X_dense[:, self.features_mask_]

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.feature_selection import RFECV
import numpy as np
import logging

log = logging.getLogger(__name__)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
import numpy as np
import logging

log = logging.getLogger(__name__)

class RecursiveCVFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, tscv=None, C_grid=None, random_state=42, verbose=1):
        self.tscv = tscv
        self.C_grid = C_grid if C_grid is not None else np.logspace(-2, 2, 20)
        self.random_state = random_state
        self.verbose = verbose
        self.rfecv_list_ = []  # store RFECV objects for each iteration
        self.best_C_list_ = []
        self.history_ = []

    def fit(self, X, y):
        X_curr = X
        prev_num_features = -1
        iteration = 0
        self.feature_masks_ = []  # masks for selected features at each iteration

        while True:
            iteration += 1
            if self.verbose:
                log.info(f"=== RecursiveCVFeatureSelection Iteration {iteration} ===")

            # GridSearch for best C
            grid_search = GridSearchCV(
                LogisticRegression(solver="liblinear", random_state=self.random_state),
                param_grid={"C": self.C_grid},
                cv=self.tscv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_curr, y)
            best_C = grid_search.best_params_["C"]
            best_score_cv = grid_search.best_score_
            if self.verbose:
                log.info(f"Best C: {best_C}, CV ROC AUC: {best_score_cv:.4f}")
            self.best_C_list_.append(best_C)

            # RFECV with best C
            clf = LogisticRegression(C=best_C, solver="liblinear", random_state=self.random_state)
            rfecv = RFECV(estimator=clf, step=1, cv=self.tscv, scoring="roc_auc", n_jobs=-1, verbose=self.verbose)
            rfecv.fit(X_curr, y)
            self.rfecv_list_.append(rfecv)

            # Store feature mask
            mask = rfecv.support_
            self.feature_masks_.append(mask)

            # Apply mask for next iteration
            X_curr = X_curr[:, mask]

            # Stop if number of features stabilizes
            if X_curr.shape[1] == prev_num_features:
                if self.verbose:
                    log.info("Number of features stabilized, stopping recursive selection.")
                break

            prev_num_features = X_curr.shape[1]
            self.history_.append({
                "iteration": iteration,
                "best_C": best_C,
                "num_features": X_curr.shape[1],
                "cv_score": best_score_cv
            })

        if self.verbose:
            log.info(f"Recursive selection finished with {X_curr.shape[1]} features.")
        return self

    def transform(self, X):
        if not self.feature_masks_:
            raise ValueError("The estimator has not been fitted yet.")
        X_trans = X
        for mask in self.feature_masks_:
            X_trans = X_trans[:, mask]
        return X_trans
