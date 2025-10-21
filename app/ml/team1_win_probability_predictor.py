from typing import Type
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion

# Import your custom feature extractors and selectors
from app.ml.feature_extractors.utils.column_selector import ColumnSelector
from app.ml.feature_extractors.team_bag import TeamBagEncoder
from app.ml.feature_extractors.player_bag import PlayerBagEncoder
from app.ml.feature_extractors.player_elo import PlayerEloEncoder
from app.ml.feature_selectors.logit_l1 import LogitL1CVSelector
from app.ml.feature_selectors.logit_rfe import LogitRFECV
from app.ml.optimizers.logit import LogitOptimizer

from typing import Type
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion

from app.ml.feature_extractors.utils.column_selector import ColumnSelector
from app.ml.feature_extractors.team_bag import TeamBagEncoder
from app.ml.feature_extractors.player_bag import PlayerBagEncoder
from app.ml.feature_extractors.player_elo import PlayerEloEncoder
from app.ml.feature_selectors.logit_l1 import LogitL1CVSelector
from app.ml.feature_selectors.logit_rfe import LogitRFECV
from app.ml.optimizers.logit import LogitOptimizer

class Team1WinProbabilityPredictor:
    def __init__(
        self,
        column_selector_cls: Type[ColumnSelector],
        team_bag_encoder: TeamBagEncoder,
        player_bag_encoder: PlayerBagEncoder,
        player_elo_encoder: PlayerEloEncoder,
        logit_l1_feature_selector: LogitL1CVSelector,
        logit_rfe_feature_selector: LogitRFECV,
        logit_optimizer: LogitOptimizer
    ):
        self.ColumnSelector = column_selector_cls
        self.team_bag_encoder = team_bag_encoder
        self.player_bag_encoder = player_bag_encoder
        self.player_elo_encoder = player_elo_encoder
        self.l1_selector = logit_l1_feature_selector
        self.rfecv_selector = logit_rfe_feature_selector
        self.logit_optimizer = logit_optimizer

        self.best_model = None
        self.pipeline: Pipeline | None = None        

    def fit(self, X: np.ndarray, y: np.ndarray, team_cols: list[int], player_cols: list[int]):
        self.pipeline = Pipeline(
            [
                (
                    "encoder",
                    FeatureUnion(
                        [
                            (
                                "team_bag",
                                Pipeline(
                                    [
                                        ("select_team", self.ColumnSelector(team_cols)),
                                        ("encode_team", self.team_bag_encoder),
                                    ]
                                ),
                            ),
                            (
                                "player_bag",
                                Pipeline(
                                    [
                                        ("select_player", self.ColumnSelector(player_cols)),
                                        ("encode_player", self.player_bag_encoder),
                                    ]
                                ),
                            ),
                            (
                                "player_elo",
                                Pipeline(
                                    [
                                        ("select_player", self.ColumnSelector(player_cols)),
                                        ("elo_encoder", self.player_elo_encoder),
                                    ]
                                ),
                            ),
                        ]
                    ),
                )
            ]
        )

        X_encoded = self.pipeline.fit_transform(X, y)
        X_l1 = self.l1_selector.fit_transform(X_encoded, y)
        X_final_selected = self.rfecv_selector.fit_transform(X_l1, y)

        self.logit_optimizer.fit(X_final_selected, y)
        self.best_model = self.logit_optimizer.best_model_        
        return self

    def predict_proba(self, X: np.ndarray):
        if self.best_model is None:
            raise ValueError("Call fit() first")
        if self.best_model is None:
            raise ValueError("Pipeline not fitted")

        X_encoded = self.pipeline.transform(X)
        X_l1 = self.l1_selector.transform(X_encoded)
        X_final_selected = self.rfecv_selector.transform(X_l1)

        return self.best_model.predict_proba(X_final_selected)

    def predict(self, X: np.ndarray):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
