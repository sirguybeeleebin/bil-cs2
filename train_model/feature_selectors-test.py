import numpy as np
import pytest

from train_model.feature_extractors import LogitL1FeatureSelector


def test_logit_l1_feature_selector_fit_transform():
    # synthetic dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    # make first 3 features predictive
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(100) * 0.1 > 0).astype(int)

    selector = LogitL1FeatureSelector(C_values=[0.01, 0.1, 1.0], cv=3)
    selector.fit(X, y)

    # Check that selected indices are not empty
    assert selector.selected_idx_ is not None
    assert len(selector.selected_idx_) > 0
    assert selector.model_ is not None
    assert selector.best_C_ in [0.01, 0.1, 1.0]
    assert selector.best_score_ is not None

    # Transform
    X_trans = selector.transform(X)
    # Check shape: number of columns equals number of selected features
    assert X_trans.shape[0] == X.shape[0]
    assert X_trans.shape[1] == len(selector.selected_idx_)

    # Transform should raise error if fit not called
    selector2 = LogitL1FeatureSelector()
    with pytest.raises(ValueError):
        selector2.transform(X)
