import numpy as np
import pytest

from train_model.feature_selectors import LogitL1FeatureSelector


def test_logit_l1_feature_selector_fit_transform():
    # Synthetic dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    # Make first 3 features predictive
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(100) * 0.1 > 0).astype(int)

    selector = LogitL1FeatureSelector(C_values=[0.01, 0.1, 1.0], cv=3)
    selector.fit(X, y)

    # Check selected indices
    assert selector.selected_idx_ is not None, "Selected indices should not be None"
    assert len(selector.selected_idx_) > 0, "At least one feature should be selected"
    assert selector.model_ is not None, "Fitted model should not be None"
    assert selector.best_C_ in [0.01, 0.1, 1.0], "Best C must be one of tested values"
    assert selector.best_score_ is not None, "Best score should not be None"

    # Check transform
    X_trans = selector.transform(X)
    assert X_trans.shape[0] == X.shape[0], (
        "Number of rows should stay the same after transform"
    )
    assert X_trans.shape[1] == len(selector.selected_idx_), (
        "Number of columns should match selected features"
    )

    # Transform without fit should raise error
    selector2 = LogitL1FeatureSelector()
    with pytest.raises(ValueError):
        selector2.transform(X)


def test_logit_l1_feature_selector_no_features_selected():
    np.random.seed(42)
    X = np.random.randn(50, 5)
    # Ensure at least 2 classes, but no predictive signal
    y = np.zeros(50, dtype=int)
    y[0] = 1  # one positive sample

    selector = LogitL1FeatureSelector(C_values=[0.01, 0.1], cv=2)
    selector.fit(X, y)

    # Should handle zero features gracefully
    assert selector.selected_idx_ is not None
    assert len(selector.selected_idx_) == 0, "No features should be selected"
    assert selector.model_ is None, "No model should be fitted if no features selected"
    assert selector.best_C_ is None, "Best C should be None if no features selected"
    assert selector.best_score_ == -np.inf, (
        "Best score should remain -inf if no features selected"
    )
