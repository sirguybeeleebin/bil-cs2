import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score
from app.ml.feature_selectors.logit_rfe import LogitRFECV  # замените на свой модуль

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
    selector = LogitRFECV(C=1.0, cv=cv, scoring=precision_score, step=1, random_state=42)
    selector.fit(X, y)

    # Проверяем, что RFECV установился
    assert selector.selector_ is not None

    # Проверяем, что выбраны признаки
    assert selector.selector_.support_ is not None
    assert np.any(selector.selector_.support_)

    # Проверяем, что ranking_ и n_features_ корректны
    assert len(selector.selector_.ranking_) == X.shape[1]
    assert selector.selector_.n_features_ > 0

def test_transform_and_fit_transform(example_data):
    X, y, cv = example_data
    selector = LogitRFECV(C=1.0, cv=cv, scoring=precision_score, step=1, random_state=42)

    # Проверяем fit_transform
    X_selected = selector.fit_transform(X, y)
    assert X_selected.shape[0] == X.shape[0]
    assert X_selected.shape[1] == selector.selector_.n_features_

    # Проверяем transform после fit
    X_selected2 = selector.transform(X)
    assert np.array_equal(X_selected, X_selected2)

def test_transform_without_fit_raises(example_data):
    X, y, cv = example_data
    selector = LogitRFECV(C=1.0, cv=cv, scoring=precision_score, step=1, random_state=42)
    with pytest.raises(ValueError):
        selector.transform(X)

def test_selector_with_perfectly_constant_features():
    # Случай, когда все признаки одинаковые -> RFECV выберет минимум признаков
    X = np.ones((10, 5))
    y = np.array([0,1]*5)
    cv = TimeSeriesSplit(n_splits=2)
    selector = LogitRFECV(C=1.0, cv=cv, scoring=precision_score, step=1, random_state=42)
    selector.fit(X, y)
    # Проверяем, что RFECV установился
    assert selector.selector_ is not None
    # Проверяем, что n_features_ >= 1
    assert selector.selector_.n_features_ >= 1
