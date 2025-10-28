import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_model.feature_extractors import ColumnSelectorArray
from train_model.stacker import OOFPredictor, Stacker


def test_oof_predictor_fit_predict():
    # synthetic dataset
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([0, 0, 1, 1, 0, 1])

    oof = OOFPredictor(n_splits=3, random_state=42)
    oof.fit(X, y)
    preds = oof.predict_proba(X)

    assert preds.shape[0] == X.shape[0]
    assert np.all(preds >= 0) and np.all(preds <= 1)
    oof_preds = oof.get_oof_predictions()
    assert oof_preds.shape[0] == X.shape[0]


def test_ml_stacker_fit_predict():
    # synthetic dataset
    X = np.array([[i] * 10 for i in range(1, 11)])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # simple pipelines
    pipelines = [
        (
            "scale",
            Pipeline(
                [
                    ("select", ColumnSelectorArray([0, 1, 2])),
                    ("scale", StandardScaler()),
                ]
            ),
        ),
        ("first3", Pipeline([("select", ColumnSelectorArray([0, 1, 2]))])),
    ]

    oof_predictor = OOFPredictor(n_splits=2, random_state=42)
    stacker = Stacker(pipelines=pipelines, oof_predictor=oof_predictor)
    stacker.fit(X, y)
    preds = stacker.predict_proba(X)

    assert preds.shape[0] == X.shape[0]
    assert np.all(preds >= 0) and np.all(preds <= 1)


def test_ml_stacker_pipeline_order_preserved():
    X = np.random.randint(0, 10, (6, 5))
    y = np.array([0, 1, 0, 1, 0, 1])

    pipelines = [
        ("pipe1", Pipeline([("sel", ColumnSelectorArray([0, 1]))])),
        ("pipe2", Pipeline([("sel", ColumnSelectorArray([2, 3]))])),
    ]

    oof_predictor = OOFPredictor(n_splits=2)
    stacker = Stacker(pipelines, oof_predictor=oof_predictor)
    stacker.fit(X, y)

    # Check meta features shape
    X_meta = np.column_stack(
        [stacker.oof_preds_train_avg[name] for name, _ in pipelines]
    )
    assert X_meta.shape[1] == len(pipelines)
