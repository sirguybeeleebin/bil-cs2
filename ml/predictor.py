import logging
import warnings
from typing import Optional, Type

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Team1WinProbabilityPredictor:
    def __init__(
        self,
        column_selector_cls: Type,
        player_elo_encoder: Optional[object] = None,
        player_bag_encoder: Optional[object] = None,
        team_bag_encoder: Optional[object] = None,
    ):
        self.ColumnSelector = column_selector_cls
        self.player_elo_encoder = player_elo_encoder
        self.player_bag_encoder = player_bag_encoder
        self.team_bag_encoder = team_bag_encoder
        self.best_model: Optional[LogisticRegression] = None
        self.pipeline: Optional[Pipeline] = None
        self.selected_features_mask: Optional[np.ndarray] = None
        self.random_state = 42

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        map_id_col: list[int],
        player_cols: list[int],
        team_cols: Optional[list[int]] = None,
    ):
        log.info("=== Начало построения пайплайна ===")
        union_steps = []

        if map_id_col:
            union_steps.append(
                (
                    "map_id_ohe",
                    Pipeline(
                        [
                            ("select_map", self.ColumnSelector(map_id_col)),
                            (
                                "onehot",
                                OneHotEncoder(
                                    sparse_output=True, handle_unknown="ignore"
                                ),
                            ),
                        ]
                    ),
                )
            )

        if self.player_elo_encoder is not None:
            union_steps.append(
                (
                    "player_elo",
                    Pipeline(
                        [
                            ("select_player", self.ColumnSelector(player_cols)),
                            ("elo_encoder", self.player_elo_encoder),
                            ("scaler", MinMaxScaler()),
                        ]
                    ),
                )
            )

        if self.player_bag_encoder is not None:
            union_steps.append(
                (
                    "player_bag",
                    Pipeline(
                        [
                            ("select_player", self.ColumnSelector(player_cols)),
                            ("bag_encoder", self.player_bag_encoder),
                        ]
                    ),
                )
            )

        if self.team_bag_encoder is not None and team_cols is not None:
            union_steps.append(
                (
                    "team_bag",
                    Pipeline(
                        [
                            ("select_team", self.ColumnSelector(team_cols)),
                            ("bag_encoder", self.team_bag_encoder),
                        ]
                    ),
                )
            )

        self.pipeline = Pipeline([("encoder", FeatureUnion(union_steps))])

        log.info("Фитим пайплайн...")
        X_transformed = self.pipeline.fit_transform(X, y)
        log.info("Форма X после трансформации: %s", X_transformed.shape)

        param_grid = {"C": np.linspace(0.01, 1.0, 10)}
        grid_search_l1 = GridSearchCV(
            estimator=LogisticRegression(
                penalty="l1", solver="liblinear", random_state=self.random_state
            ),
            param_grid=param_grid,
            cv=TimeSeriesSplit(n_splits=10),
            scoring="precision",
            n_jobs=-1,
            verbose=2,
        )
        grid_search_l1.fit(X_transformed, y)
        best_C_l1 = grid_search_l1.best_params_["C"]
        log.info("Лучший C для L1: %s", best_C_l1)

        l1_model = LogisticRegression(
            penalty="l1",
            C=best_C_l1,
            solver="liblinear",
            random_state=self.random_state,
        )
        l1_model.fit(X_transformed, y)
        initial_mask = l1_model.coef_[0] != 0
        log.info("Число признаков после L1 отбора: %d", np.sum(initial_mask))

        X_selected = X_transformed[:, initial_mask]
        rfecv = RFECV(
            estimator=LogisticRegression(
                solver="liblinear", C=best_C_l1, random_state=self.random_state
            ),
            step=1,
            cv=TimeSeriesSplit(n_splits=10),
            scoring="precision",
            n_jobs=-1,
            verbose=2,
        )
        rfecv.fit(X_selected, y)
        self.selected_features_mask = np.zeros(X_transformed.shape[1], dtype=bool)
        self.selected_features_mask[initial_mask] = rfecv.support_
        log.info(
            "Число выбранных признаков после RFECV: %d",
            np.sum(self.selected_features_mask),
        )

        self.best_model = LogisticRegression(
            solver="liblinear", C=best_C_l1, random_state=self.random_state
        )
        self.best_model.fit(X_transformed[:, self.selected_features_mask], y)
        log.info("Обучение завершено.")
        return self

    def predict_proba(self, X: np.ndarray):
        if self.best_model is None:
            raise ValueError("Call fit() first")
        X_transformed = self.pipeline.transform(X)
        return self.best_model.predict_proba(
            X_transformed[:, self.selected_features_mask]
        )

    def predict(self, X: np.ndarray):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
