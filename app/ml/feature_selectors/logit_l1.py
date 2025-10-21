import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import make_scorer, precision_score

class LogitL1CVSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cs=np.logspace(-2, 1, 10),
        cv=TimeSeriesSplit(n_splits=5),
        scoring=precision_score,
        random_state=42
    ):
        self.cs = cs
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.selected_features_ = None
        self.best_C_ = None
        self.best_score_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):        
        n_features = X.shape[1]

        best_score = -np.inf
        best_features = None
        best_C = None

        scorer = make_scorer(self.scoring)

        for C in self.cs:            
            fold_nonzero = np.zeros((self.cv.get_n_splits(X, y), n_features), dtype=int)
            
            for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                model = LogisticRegression(
                    penalty='l1', C=C, solver='liblinear', random_state=self.random_state
                )
                model.fit(X_train, y_train)
                fold_nonzero[fold_idx] = (model.coef_[0] != 0).astype(int)
            
            selected_features = np.where(fold_nonzero.mean(axis=0) > 0.5)[0]
            if len(selected_features) == 0:
                continue
            
            model = LogisticRegression(
                penalty='l1', C=C, solver='liblinear', random_state=self.random_state
            )
            scores = cross_val_score(model, X[:, selected_features], y, cv=self.cv, scoring=scorer)
            mean_score = scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_C = C
                best_features = selected_features

        self.selected_features_ = best_features
        self.best_C_ = best_C
        self.best_score_ = best_score
        return self

    def transform(self, X: np.ndarray):        
        if self.selected_features_ is None:
            raise ValueError("You must fit the selector before transforming.")
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        self.fit(X, y)
        return self.transform(X)
