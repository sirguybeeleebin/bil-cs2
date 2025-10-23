import logging
import warnings
from typing import Type

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ml.feature_extractors import (
    ColumnSelector,
    PlayerBagEncoder,
    PlayerEloEncoder,
    TeamBagEncoder,
)

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Team1WinProbabilityPredictor:
    def __init__(
        self,
        column_selector_cls: Type[ColumnSelector],
        player_elo_encoder: PlayerEloEncoder | None = None,
        player_bag_encoder: PlayerBagEncoder | None = None,
        team_bag_encoder: TeamBagEncoder | None = None,
    ) -> None:
        self.ColumnSelector = column_selector_cls
        self.player_elo_encoder: PlayerEloEncoder | None = player_elo_encoder
        self.player_bag_encoder: PlayerBagEncoder | None = player_bag_encoder
        self.team_bag_encoder: TeamBagEncoder | None = team_bag_encoder
        self.best_model: LogisticRegression | None = None
        self.pipeline: Pipeline | None = None
        self.selected_features_mask: np.ndarray | None = None
        self.random_state: int = 42
        self.cv = TimeSeriesSplit(n_splits=10)
        self.scoring = "precision"

    def _build_pipeline(
        self,
        map_id_col: list[int],
        player_cols: list[int],
        team_cols: list[int] | None = None,
    ) -> None:
        union_steps: list[tuple[str, Pipeline]] = []

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

    def _fit_l1_model(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        param_grid: dict[str, np.ndarray] = {"C": np.linspace(0.01, 1.0, 10)}
        grid_search_l1: GridSearchCV[LogisticRegression] = GridSearchCV(
            estimator=LogisticRegression(
                penalty="l1", solver="liblinear", random_state=self.random_state
            ),
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=2,
        )
        grid_search_l1.fit(X, y)
        best_C_l1: float = grid_search_l1.best_params_["C"]
        log.info("Лучший C для L1: %s", best_C_l1)

        l1_model: LogisticRegression = LogisticRegression(
            penalty="l1",
            C=best_C_l1,
            solver="liblinear",
            random_state=self.random_state,
        )
        l1_model.fit(X, y)
        initial_mask: np.ndarray = l1_model.coef_[0] != 0
        log.info("Число признаков после L1 отбора: %d", np.sum(initial_mask))
        return initial_mask, best_C_l1

    def _fit_rfecv(
        self, X: np.ndarray, y: np.ndarray, initial_mask: np.ndarray, best_C_l1: float
    ) -> None:
        if np.sum(initial_mask) == 0:
            log.warning("No features remain after L1. Skipping RFECV.")
            self.selected_features_mask = np.ones(
                X.shape[1], dtype=bool
            )  # fallback: keep all
            return

        X_selected: np.ndarray = X[:, initial_mask]
        rfecv: RFECV = RFECV(
            estimator=LogisticRegression(
                solver="liblinear", C=best_C_l1, random_state=self.random_state
            ),
            step=1,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1,
        )
        rfecv.fit(X_selected, y)
        self.selected_features_mask = np.zeros(X.shape[1], dtype=bool)
        self.selected_features_mask[initial_mask] = rfecv.support_
        log.info(
            "Число выбранных признаков после RFECV: %d",
            np.sum(self.selected_features_mask),
        )

    def _fit_final_model(self, X: np.ndarray, y: np.ndarray, best_C_l1: float) -> None:
        self.best_model = LogisticRegression(
            solver="liblinear", C=best_C_l1, random_state=self.random_state
        )
        self.best_model.fit(X[:, self.selected_features_mask], y)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        map_id_col: list[int],
        player_cols: list[int],
        team_cols: list[int] | None = None,
    ) -> "Team1WinProbabilityPredictor":
        log.info("Начало построения пайплайна")
        self._build_pipeline(map_id_col, player_cols, team_cols)
        X_transformed: np.ndarray = self.pipeline.fit_transform(X, y)
        log.info("Форма X после трансформации: %s", X_transformed.shape)

        initial_mask, best_C_l1 = self._fit_l1_model(X_transformed, y)
        self._fit_rfecv(X_transformed, y, initial_mask, best_C_l1)
        self._fit_final_model(X_transformed, y, best_C_l1)

        log.info("Обучение завершено.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Call fit() first")
        X_transformed: np.ndarray = self.pipeline.transform(X)
        return self.best_model.predict_proba(
            X_transformed[:, self.selected_features_mask]
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba: np.ndarray = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
