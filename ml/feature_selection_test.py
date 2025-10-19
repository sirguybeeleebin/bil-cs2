import numpy as np
import pytest
from scipy import sparse
from sklearn.model_selection import KFold

from ml.feature_selection import RecursiveL1Selector

def test_recursive_l1_selector_dense():
    X = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
    ])
    y = np.array([1, 1, 0, 0])

    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    selector = RecursiveL1Selector(C=1, cv=cv)
    selector.fit(X, y)
    X_trans = selector.transform(X)

    assert selector.features_mask_.dtype == bool
    assert selector.features_mask_.shape[0] == X.shape[1]
    assert X_trans.shape[1] == selector.features_mask_.sum()
    assert X_trans.shape[1] <= X.shape[1]

def test_recursive_l1_selector_sparse():
    X = sparse.csr_matrix([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
    ])
    y = np.array([1, 1, 0, 0])

    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    selector = RecursiveL1Selector(C=1, cv=cv)
    selector.fit(X, y)
    X_trans = selector.transform(X)

    assert selector.features_mask_.dtype == bool
    assert selector.features_mask_.shape[0] == X.shape[1]
    assert X_trans.shape[1] == selector.features_mask_.sum()
    assert X_trans.shape[1] <= X.shape[1]

def test_recursive_l1_selector_removes_zero_features():
    X = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ])
    y = np.array([1, 1, 0, 0])
    cv = KFold(n_splits=2, shuffle=True, random_state=42)

    selector = RecursiveL1Selector(C=1, cv=cv)
    selector.fit(X, y)

    # Feature 2 should be removed
    assert selector.features_mask_[2] == False  # Use == instead of `is`

    # At least one of feature 0 or 1 should be kept
    assert selector.features_mask_[:2].any()


def test_transform_returns_correct_shape():
    X = np.random.rand(5, 6)
    y = np.array([0, 1, 0, 1, 0])
    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    selector = RecursiveL1Selector(C=1, cv=cv)
    selector.fit(X, y)
    X_trans = selector.transform(X)

    assert X_trans.shape[0] == X.shape[0]
    # Should match number of selected features
    assert X_trans.shape[1] == selector.features_mask_.sum()
