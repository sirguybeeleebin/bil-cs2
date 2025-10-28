import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LogitL1FeatureSelector:
    def __init__(
        self,
        C_values=[0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
        max_iter=1000,
        scoring="roc_auc",
        cv=TimeSeriesSplit(10),
        random_state=42,
        n_jobs=-1,
    ):
        self.C_values = C_values
        self.max_iter = max_iter
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.selected_idx_ = None
        self.model_ = None
        self.best_C_ = None
        self.best_score_ = None

    def fit(self, X, y):
        best_score = -np.inf
        best_mask = None
        best_C = None
        best_model = None

        log.info(f"LogitL1FeatureSelector: обучение на массиве {X.shape}")

        for C in self.C_values:
            log.info(f"Проверка C={C}")
            logit = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=C,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            logit.fit(X, y)
            mask = logit.coef_.flatten() != 0
            selected_count = mask.sum()
            log.info(f"C={C}: выбрано {selected_count}/{X.shape[1]} признаков")

            if selected_count == 0:
                log.info(f"C={C}: пропускаем, так как нет выбранных признаков")
                continue

            score = cross_val_score(
                LogisticRegression(
                    solver="liblinear",
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                ),
                X[:, mask],
                y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            ).mean()

            log.info(f"C={C}: CV score ({self.scoring}) = {score:.4f}")

            if score > best_score:
                best_score = score
                best_mask = mask
                best_C = C
                best_model = LogisticRegression(
                    solver="liblinear",
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
                best_model.fit(X[:, best_mask], y)
                log.info(f"Новый лучший C={best_C}, лучший score={best_score:.4f}")

        self.selected_idx_ = (
            np.where(best_mask)[0] if best_mask is not None else np.array([])
        )
        self.model_ = best_model
        self.best_C_ = best_C
        self.best_score_ = best_score

        log.info(
            f"Обучение завершено: лучший C={self.best_C_}, выбрано признаков {len(self.selected_idx_)}/{X.shape[1]}, лучший CV score={self.best_score_:.4f}"
        )
        return self

    def transform(self, X):
        if self.selected_idx_ is None:
            raise ValueError("Сначала вызовите метод fit")
        log.info(
            f"LogitL1FeatureSelector: трансформация массива размером {X.shape} с {len(self.selected_idx_)} выбранными признаками"
        )
        return X[:, self.selected_idx_]
