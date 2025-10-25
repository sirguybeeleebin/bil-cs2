from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline


class OOFPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, n_splits: int = 5, random_state: int = 42) -> None:
        self.n_splits: int = n_splits
        self.random_state: int = random_state
        self.base_model: LogisticRegression = LogisticRegression(
            solver="liblinear", random_state=self.random_state
        )
        self.base_models_: list[LogisticRegression] = []
        self.oof_predictions_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> OOFPredictor:
        if issparse(X):
            X = X.toarray()
        X = np.atleast_2d(X)
        y = np.array(y)
        self.oof_predictions_ = np.zeros(X.shape[0])
        self.base_models_ = []
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in kf.split(X):
            model = clone(self.base_model)
            model.fit(X[train_idx], y[train_idx])
            self.oof_predictions_[val_idx] = model.predict_proba(X[val_idx])[:, 1]
            self.base_models_.append(model)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if issparse(X):
            X = X.toarray()
        X = np.atleast_2d(X)
        preds = np.column_stack(
            [model.predict_proba(X)[:, 1] for model in self.base_models_]
        )
        return np.mean(preds, axis=1)

    def get_oof_predictions(self) -> np.ndarray:
        return self.oof_predictions_


class MLStacker:
    def __init__(
        self,
        pipelines: list[tuple[str, Pipeline]],
        oof_predictor: OOFPredictor,
        n_iters: int = 10,
        random_state: int = 42,
    ) -> None:
        self.pipelines: list[tuple[str, Pipeline]] = pipelines
        self.base_oof_predictor: OOFPredictor = oof_predictor
        self.n_iters: int = n_iters
        self.random_state: int = random_state
        self.oof_preds_train_avg: OrderedDict[str, np.ndarray] = OrderedDict()
        self.oof_models: OrderedDict[str, list[OOFPredictor]] = OrderedDict()
        self.X_meta_train: np.ndarray | None = None
        self.final_model: LogisticRegression = LogisticRegression(
            solver="liblinear", random_state=self.random_state
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> MLStacker:
        for name, pipe in self.pipelines:
            print(f"Processing pipeline: {name}")
            X_train_feat = pipe.fit_transform(X_train, y_train)
            oof_preds_list: list[np.ndarray] = []
            models_list: list[OOFPredictor] = []
            for _ in range(self.n_iters):
                oof_model = clone(self.base_oof_predictor)
                oof_model.fit(X_train_feat, y_train)
                oof_preds_list.append(oof_model.get_oof_predictions())
                models_list.append(oof_model)
            self.oof_preds_train_avg[name] = np.mean(oof_preds_list, axis=0)
            self.oof_models[name] = models_list
        X_meta_train = np.column_stack(
            [self.oof_preds_train_avg[name] for name in self.oof_preds_train_avg]
        )
        tscv = TimeSeriesSplit(n_splits=10)
        self.rfecv_ = RFECV(
            estimator=self.final_model, step=1, cv=tscv, scoring="roc_auc", n_jobs=-1
        )
        self.rfecv_.fit(X_meta_train, y_train)
        X_meta_train_selected = self.rfecv_.transform(X_meta_train)
        self.final_model.fit(X_meta_train_selected, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        meta_features: list[np.ndarray] = []
        for name, pipe in self.pipelines:
            X_test_feat = pipe.transform(X_test)
            preds_list = [
                model.predict_proba(X_test_feat) for model in self.oof_models[name]
            ]
            avg_preds = np.mean(np.column_stack(preds_list), axis=1)
            meta_features.append(avg_preds)
        X_meta_test = np.column_stack(meta_features)
        X_meta_test_selected = self.rfecv_.transform(X_meta_test)
        return self.final_model.predict_proba(X_meta_test_selected)[:, 1]
