import numpy as np
import pytest

from ml.main import MLStackingPipeline, PlayerBagPipeline, TeamBagPipeline


# ---------------------------
# Fixtures with enough samples
# ---------------------------
@pytest.fixture
def player_data():
    # 8 samples, 4 per class to avoid KFold issues
    X_player = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X_player, y


@pytest.fixture
def team_data():
    # 8 samples, 4 per class
    X_team = np.array(
        [
            [100, 200],
            [101, 201],
            [102, 202],
            [103, 203],
            [100, 200],
            [101, 201],
            [102, 202],
            [103, 203],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X_team, y


# ---------------------------
# PlayerBagPipeline tests
# ---------------------------
def test_player_fit_predict_proba(player_data):
    X_player, y = player_data
    pipeline = PlayerBagPipeline(n_folds=2)

    # Fit
    oof = pipeline.fit(X_player, y)
    assert oof.shape == y.shape
    assert np.all((oof >= 0) & (oof <= 1))  # probabilities between 0 and 1

    # Predict
    proba = pipeline.predict_proba(X_player)
    assert proba.shape == y.shape
    assert np.all((proba >= 0) & (proba <= 1))


# ---------------------------
# TeamBagPipeline tests
# ---------------------------
def test_team_fit_predict_proba(team_data):
    X_team, y = team_data
    pipeline = TeamBagPipeline(n_folds=2)

    # Fit
    oof = pipeline.fit(X_team, y)
    assert oof.shape == y.shape
    assert np.all((oof >= 0) & (oof <= 1))

    # Predict
    proba = pipeline.predict_proba(X_team)
    assert proba.shape == y.shape
    assert np.all((proba >= 0) & (proba <= 1))


# ---------------------------
# MLStackingPipeline tests
# ---------------------------
def test_ml_stacking_pipeline_fit_predict(player_data, team_data):
    X_player, y = player_data
    X_team, _ = team_data

    pipeline = MLStackingPipeline(n_folds=2)

    # Fit
    player_oof, team_oof = pipeline.fit(X_player, X_team, y)
    assert player_oof.shape == y.shape
    assert team_oof.shape == y.shape

    # Predict probabilities
    y_proba = pipeline.predict_proba(X_player, X_team)
    assert y_proba.shape == y.shape
    assert np.all((y_proba >= 0) & (y_proba <= 1))

    # Predict classes
    y_pred = pipeline.predict(X_player, X_team)
    assert y_pred.shape == y.shape
    assert set(np.unique(y_pred)).issubset({0, 1})
