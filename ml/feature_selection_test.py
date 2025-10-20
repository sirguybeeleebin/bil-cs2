import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import TimeSeriesSplit

from ml.feature_selection import select_features_with_logit_and_cv  # replace with actual import


def test_select_features_with_logit_and_cv_basic():
    # Generate a small synthetic dataset
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )

    cv = TimeSeriesSplit(n_splits=5)

    mask = select_features_with_logit_and_cv(
        X_train=X,
        y_train=y,
        Cs=np.array([0.1, 0.5]),
        scoring="roc_auc",
        random_state=42,
        cv=cv,
        verbose=0,
        n_jobs=1,
    )

    # Check type
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool

    # Mask length equals number of features
    assert mask.shape[0] == X.shape[1]

    # At least one feature selected
    assert mask.sum() > 0


def test_select_features_with_logit_and_cv_all_zero_features():
    X = np.zeros((50, 10))
    y = np.random.randint(0, 2, size=50)
    cv = TimeSeriesSplit(n_splits=3)

    try:
        mask = select_features_with_logit_and_cv(
            X_train=X,
            y_train=y,
            Cs=np.array([0.1]),
            scoring="roc_auc",
            random_state=42,
            cv=cv,
            verbose=0,
            n_jobs=1,
        )
        # If it returns, all features are False
        assert np.all(mask == False)
    except ValueError as e:
        # Accept the scikit-learn error as expected
        assert "Found array with 0 feature(s)" in str(e)

