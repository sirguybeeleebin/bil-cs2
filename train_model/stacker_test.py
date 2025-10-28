import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.stacker import MLStacker, OOFPredictor


def test_oof_predictor_basic():
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, size=20)

    oof = OOFPredictor(n_splits=5, random_state=42)
    oof.fit(X, y, i=0)
    preds = oof.predict_proba(X)

    assert preds.shape == (X.shape[0],)
    assert np.all((preds >= 0) & (preds <= 1))
    oof_preds = oof.get_oof_predictions()
    assert oof_preds.shape == (X.shape[0],)


@pytest.fixture
def pipelines():
    pipe1 = Pipeline([("scaler", StandardScaler())])
    pipe2 = Pipeline([("scaler", StandardScaler())])
    return [("pipe1", pipe1), ("pipe2", pipe2)]


def test_ml_stacker_fit_predict(pipelines):
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, size=20)
    oof = OOFPredictor(n_splits=4, random_state=42)

    stacker = MLStacker(
        pipelines=pipelines, oof_predictor=oof, n_iters=2, random_state=42
    )
    stacker.fit(X, y)
    preds = stacker.predict_proba(X)

    assert preds.shape == (X.shape[0],)
    assert np.all((preds >= 0) & (preds <= 1))
