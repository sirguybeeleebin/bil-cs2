from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OOFPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, n_splits: int = 10, random_state: int = 42) -> None:
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
        log.info(f"Начало обучения OOFPredictor с {self.n_splits}-fold CV")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            model = clone(self.base_model)
            model.fit(X[train_idx], y[train_idx])
            self.oof_predictions_[val_idx] = model.predict_proba(X[val_idx])[:, 1]
            self.base_models_.append(model)
            log.info(f"Fold {fold}/{self.n_splits} завершен")
        log.info("OOFPredictor обучение завершено")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if issparse(X):
            X = X.toarray()
        X = np.atleast_2d(X)
        preds = np.column_stack(
            [model.predict_proba(X)[:, 1] for model in self.base_models_]
        )
        log.info("OOFPredictor: предсказания выполнены")
        return np.mean(preds, axis=1)

    def get_oof_predictions(self) -> np.ndarray:
        return self.oof_predictions_


class Stacker:
    def __init__(
        self,
        pipelines: list[tuple[str, Pipeline]],
        oof_predictor: OOFPredictor,
        random_state: int = 42,
    ) -> None:
        self.pipelines: list[tuple[str, Pipeline]] = pipelines
        self.base_oof_predictor: OOFPredictor = oof_predictor
        self.random_state: int = random_state
        self.oof_preds_train_avg: OrderedDict[str, np.ndarray] = OrderedDict()
        self.oof_models: OrderedDict[str, list[OOFPredictor]] = OrderedDict()
        self.X_meta_train: np.ndarray | None = None
        self.final_model: LogisticRegression = LogisticRegression(
            solver="liblinear", random_state=self.random_state
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Stacker:
        log.info(f"Начало обучения Stacker с {len(self.pipelines)} пайплайнами")

        # Fit each pipeline and OOF predictor
        for name, pipe in self.pipelines:
            log.info(f"Обработка пайплайна: {name}")
            X_train_feat = pipe.fit_transform(X_train, y_train)
            oof_model = clone(self.base_oof_predictor)
            oof_model.fit(X_train_feat, y_train)
            self.oof_preds_train_avg[name] = oof_model.get_oof_predictions()
            self.oof_models[name] = [oof_model]
            log.info(f"{name} - OOF предсказания выполнены")

        # Form meta-features
        self.X_meta_train = np.column_stack(
            [self.oof_preds_train_avg[name] for name in self.oof_preds_train_avg]
        )
        log.info("Формирование мета-признаков завершено")

        # Train final model on meta-features directly
        self.final_model.fit(self.X_meta_train, y_train)
        log.info("Stacker обучение завершено")
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        log.info(f"Stacker: предсказание для {X_test.shape[0]} примеров")
        meta_features: list[np.ndarray] = []

        for name, pipe in self.pipelines:
            X_test_feat = pipe.transform(X_test)
            preds_list = [
                model.predict_proba(X_test_feat) for model in self.oof_models[name]
            ]
            avg_preds = np.mean(np.column_stack(preds_list), axis=1)
            meta_features.append(avg_preds)
            log.info(f"{name} - предсказания средние по {len(preds_list)} моделям")

        X_meta_test = np.column_stack(meta_features)
        final_preds = self.final_model.predict_proba(X_meta_test)[:, 1]
        log.info("Stacker: предсказания выполнены")
        return final_preds
