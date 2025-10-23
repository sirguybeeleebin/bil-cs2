from unittest.mock import MagicMock

import numpy as np
import pytest

from ml.feature_extractors import (
    ColumnSelector,
    PlayerBagEncoder,
    PlayerEloEncoder,
    TeamBagEncoder,
)
from ml.predictor import Team1WinProbabilityPredictor


@pytest.fixture
def mock_column_selector():
    selector = MagicMock(spec=ColumnSelector)
    selector.return_value.fit_transform = lambda X, y=None: np.array(X)[:, :1]
    selector.return_value.transform = lambda X: np.array(X)[:, :1]
    return selector


@pytest.fixture
def mock_player_elo_encoder():
    encoder = MagicMock(spec=PlayerEloEncoder)
    encoder.fit_transform = lambda X, y=None: np.ones((len(X), 1))
    encoder.transform = lambda X: np.ones((len(X), 1))
    return encoder


@pytest.fixture
def mock_player_bag_encoder():
    encoder = MagicMock(spec=PlayerBagEncoder)
    encoder.fit_transform = lambda X, y=None: np.ones((len(X), 1))
    encoder.transform = lambda X: np.ones((len(X), 1))
    return encoder


@pytest.fixture
def mock_team_bag_encoder():
    encoder = MagicMock(spec=TeamBagEncoder)
    encoder.fit_transform = lambda X, y=None: np.ones((len(X), 1))
    encoder.transform = lambda X: np.ones((len(X), 1))
    return encoder


@pytest.fixture
def sample_data():
    X = np.random.randint(0, 5, size=(20, 5))
    y = np.random.randint(0, 2, size=20)
    return X, y


def test_fit_predict_pipeline(
    mock_column_selector,
    mock_player_elo_encoder,
    mock_player_bag_encoder,
    mock_team_bag_encoder,
    sample_data,
):
    X, y = sample_data
    predictor = Team1WinProbabilityPredictor(
        column_selector_cls=mock_column_selector,
        player_elo_encoder=mock_player_elo_encoder,
        player_bag_encoder=mock_player_bag_encoder,
        team_bag_encoder=mock_team_bag_encoder,
    )

    # Fit the predictor
    predictor.fit(X, y, map_id_col=[0], player_cols=[1, 2], team_cols=[3, 4])

    # Check that pipeline is created
    assert predictor.pipeline is not None
    assert predictor.best_model is not None
    assert predictor.selected_features_mask is not None
    assert predictor.selected_features_mask.dtype == bool

    # Test predict_proba
    proba = predictor.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.all((proba >= 0) & (proba <= 1))

    # Test predict
    preds = predictor.predict(X)
    assert preds.shape == (X.shape[0],)
    assert set(preds).issubset({0, 1})


def test_predict_before_fit_raises(mock_column_selector, sample_data):
    X, _ = sample_data
    predictor = Team1WinProbabilityPredictor(column_selector_cls=mock_column_selector)
    with pytest.raises(ValueError):
        predictor.predict(X)
