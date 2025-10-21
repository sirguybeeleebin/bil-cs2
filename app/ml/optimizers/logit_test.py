import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from app.ml.optimizers.logit import LogitOptimizer

def test_fit_creates_best_params():
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, size=10)

    # Mock Optuna study
    mock_study = MagicMock()
    mock_study.best_params = {"C": 1.0, "penalty": "l2"}

    with patch("optuna.create_study", return_value=mock_study):
        optimizer = LogitOptimizer(cv=None, n_trials=1)
        optimizer.fit(X, y)

    # _best_params should be set
    assert optimizer._best_params is not None
    assert optimizer._best_params["C"] == 1.0
    assert optimizer._best_params["penalty"] == "l2"

def test_get_best_params_raises_without_fit():
    optimizer = LogitOptimizer(cv=None, n_trials=1)
    with pytest.raises(ValueError):
        optimizer.get_best_params()

def test_get_best_params_after_fit_returns_dict():
    X = np.random.rand(5, 2)
    y = np.random.randint(0, 2, size=5)

    mock_study = MagicMock()
    mock_study.best_params = {"C": 0.5, "penalty": "l1"}

    with patch("optuna.create_study", return_value=mock_study):
        optimizer = LogitOptimizer(cv=None, n_trials=1)
        optimizer.fit(X, y)

    best_params = optimizer.get_best_params()
    assert isinstance(best_params, dict)
    assert best_params["C"] == 0.5
    assert best_params["penalty"] == "l1"


