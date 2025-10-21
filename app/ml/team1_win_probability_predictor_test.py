import pytest
import numpy as np
from unittest.mock import MagicMock
from app.ml.team1_win_probability_predictor import Team1WinProbabilityPredictor
from app.ml.feature_extractors.utils.column_selector import ColumnSelector
from app.ml.feature_extractors.team_bag import TeamBagEncoder
from app.ml.feature_extractors.player_bag import PlayerBagEncoder
from app.ml.feature_extractors.player_elo import PlayerEloEncoder
from app.ml.feature_selectors.logit_l1 import LogitL1CVSelector
from app.ml.feature_selectors.logit_rfe import LogitRFECV
from app.ml.optimizers.logit import LogitOptimizer

# -----------------------
# Fixtures
# -----------------------
@pytest.fixture
def mock_predictor():
    # Mock all feature encoders
    team_encoder = MagicMock(spec=TeamBagEncoder)
    player_encoder = MagicMock(spec=PlayerBagEncoder)
    elo_encoder = MagicMock(spec=PlayerEloEncoder)

    # All transformers return 2D arrays
    team_encoder.fit_transform.return_value = np.array([[1, 0], [0, 1]])
    team_encoder.transform.return_value = np.array([[1, 0], [0, 1]])

    player_encoder.fit_transform.return_value = np.array([[0, 1], [1, 0]])
    player_encoder.transform.return_value = np.array([[0, 1], [1, 0]])

    elo_encoder.fit_transform.return_value = np.array([[1000, 1000], [1000, 1000]])
    elo_encoder.transform.return_value = np.array([[1000, 1000], [1000, 1000]])

    # Mock selectors to pass arrays through
    l1_selector = MagicMock(spec=LogitL1CVSelector)
    l1_selector.fit_transform.side_effect = lambda X, y: X
    l1_selector.transform.side_effect = lambda X: X

    rfecv_selector = MagicMock(spec=LogitRFECV)
    rfecv_selector.fit_transform.side_effect = lambda X, y: X
    rfecv_selector.transform.side_effect = lambda X: X

    # Mock optimizer with a trained model
    optimizer = MagicMock(spec=LogitOptimizer)
    optimizer.best_model_ = MagicMock()
    optimizer.best_model_.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
    optimizer.fit.return_value = optimizer

    predictor = Team1WinProbabilityPredictor(
        column_selector_cls=ColumnSelector,
        team_bag_encoder=team_encoder,
        player_bag_encoder=player_encoder,
        player_elo_encoder=elo_encoder,
        logit_l1_feature_selector=l1_selector,
        logit_rfe_feature_selector=rfecv_selector,
        logit_optimizer=optimizer
    )
    return predictor

# -----------------------
# Tests
# -----------------------
def test_fit_sets_pipeline_and_model(mock_predictor):
    X = np.array([[0, 1], [1, 0]])
    y = np.array([1, 0])
    team_cols = [0]
    player_cols = [1]

    predictor = mock_predictor.fit(X, y, team_cols, player_cols)

    assert predictor.pipeline is not None
    assert predictor.best_model is not None

def test_predict_proba_returns_correct_shape(mock_predictor):
    X = np.array([[0, 1], [1, 0]])
    y = np.array([1, 0])
    team_cols = [0]
    player_cols = [1]

    predictor = mock_predictor.fit(X, y, team_cols, player_cols)
    proba = predictor.predict_proba(X)

    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

def test_predict_returns_binary_labels(mock_predictor):
    X = np.array([[0, 1], [1, 0]])
    y = np.array([1, 0])
    team_cols = [0]
    player_cols = [1]

    predictor = mock_predictor.fit(X, y, team_cols, player_cols)
    y_pred = predictor.predict(X)

    assert y_pred.shape == (2,)
    assert set(np.unique(y_pred)).issubset({0, 1})

def test_predict_proba_raises_if_not_fitted(mock_predictor):
    X = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        mock_predictor.predict_proba(X)

def test_predict_raises_if_not_fitted(mock_predictor):
    X = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        mock_predictor.predict(X)
