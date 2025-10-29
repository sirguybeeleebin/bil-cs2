import numpy as np

from train_model.stacker import OOFPredictor


def test_oof_predictor_fit_predict():
    # Enough samples for 3-fold CV
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([0, 1, 0, 1, 0, 1])

    oof = OOFPredictor(n_splits=3, random_state=42)
    oof.fit(X, y)
    preds = oof.predict_proba(X)

    # Predictions shape
    assert preds.shape[0] == X.shape[0]
    # Predictions in [0,1]
    assert np.all(preds >= 0) and np.all(preds <= 1)
    # OOF predictions
    oof_preds = oof.get_oof_predictions()
    assert oof_preds.shape[0] == X.shape[0]
