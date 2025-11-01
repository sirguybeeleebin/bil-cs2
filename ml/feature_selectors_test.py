import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import TimeSeriesSplit

from ml.feature_selectors import LogitL1CVFeatureSelector  # replace with your module


@pytest.fixture
def sample_data():
    # 100 samples, 20 features, binary classification
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=5,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    return X, y


def test_selector_fit_transforms_shape(sample_data):
    X, y = sample_data
    selector = LogitL1CVFeatureSelector(cv=TimeSeriesSplit(n_splits=5), C=1.0)
    selector.fit(X, y)
    Xt = selector.transform(X)

    # Ensure selected_idx_ is a non-empty array
    assert selector.selected_idx_ is not None
    assert len(selector.selected_idx_) > 0

    # Output shape should match the number of selected features
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == len(selector.selected_idx_)


def test_transform_before_fit_raises(sample_data):
    X, _ = sample_data
    selector = LogitL1CVFeatureSelector()
    with pytest.raises(RuntimeError, match="Fit the selector first"):
        selector.transform(X)


def test_selected_features_are_consistent(sample_data):
    X, y = sample_data
    selector = LogitL1CVFeatureSelector(
        cv=TimeSeriesSplit(n_splits=5), C=1.0, random_state=123
    )
    selector.fit(X, y)
    idx_first = selector.selected_idx_.copy()

    # Re-fit should give the same selected features with same random_state
    selector2 = LogitL1CVFeatureSelector(
        cv=TimeSeriesSplit(n_splits=5), C=1.0, random_state=123
    )
    selector2.fit(X, y)
    idx_second = selector2.selected_idx_

    assert np.array_equal(idx_first, idx_second)
