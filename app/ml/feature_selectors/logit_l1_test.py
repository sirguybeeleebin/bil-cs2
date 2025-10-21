import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score
from app.ml.feature_selectors.logit_l1 import LogitL1CVSelector  # замените на свой модуль

@pytest.fixture
def example_data():
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    cv = TimeSeriesSplit(n_splits=5)
    return X, y, cv

def test_fit_selects_features(example_data):
    X, y, cv = example_data
    selector = LogitL1CVSelector(cs=np.logspace(-2, 1, 5), cv=cv, scoring=precision_score, random_state=42)
    selector.fit(X, y)

    # Проверяем, что выбраны признаки
    assert selector.selected_features_ is not None
    assert len(selector.selected_features_) > 0

    # Проверяем, что best_C_ и best_score_ установлены
    assert selector.best_C_ is not None
    assert selector.best_score_ is not None
    assert selector.best_score_ >= 0

def test_transform_and_fit_transform(example_data):
    X, y, cv = example_data
    selector = LogitL1CVSelector(cs=np.logspace(-2, 1, 5), cv=cv, scoring=precision_score, random_state=42)
    
    # Проверяем fit_transform
    X_selected = selector.fit_transform(X, y)
    assert X_selected.shape[0] == X.shape[0]
    assert X_selected.shape[1] == len(selector.selected_features_)

    # Проверяем transform после fit
    X_selected2 = selector.transform(X)
    assert np.array_equal(X_selected, X_selected2)

def test_transform_without_fit_raises(example_data):
    X, y, cv = example_data
    selector = LogitL1CVSelector(cs=np.logspace(-2, 1, 5), cv=cv, scoring=precision_score, random_state=42)
    with pytest.raises(ValueError):
        selector.transform(X)

def test_selector_with_no_features_selected():
    # Случай, когда Cs слишком маленькие -> могут не выбрать ни одной фичи
    X = np.ones((10, 5))
    y = np.array([0,1]*5)
    cv = TimeSeriesSplit(n_splits=2)
    selector = LogitL1CVSelector(cs=[1e-10], cv=cv, scoring=precision_score, random_state=42)
    selector.fit(X, y)
    # Должен пропускать и не ломаться
    assert selector.selected_features_ is None
