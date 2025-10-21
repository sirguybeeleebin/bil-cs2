import numpy as np
import pytest
from app.ml.feature_extractors.utils.column_selector import ColumnSelector  # replace with your actual module path

def test_transform_selects_correct_columns():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    selector = ColumnSelector(columns=[0, 2])
    X_transformed = selector.transform(X)
    
    expected = np.array([
        [1, 3],
        [4, 6],
        [7, 9]
    ])
    np.testing.assert_array_equal(X_transformed, expected)

def test_transform_preserves_shape():
    X = np.random.rand(5, 4)
    selector = ColumnSelector(columns=[1, 3])
    X_transformed = selector.transform(X)
    
    assert X_transformed.shape == (5, 2)

def test_fit_returns_self():
    X = np.random.rand(3, 3)
    selector = ColumnSelector(columns=[0, 1])
    result = selector.fit(X)
    assert result is selector

def test_transform_with_invalid_column_index_raises():
    X = np.random.rand(3, 3)
    selector = ColumnSelector(columns=[0, 4])  # 4 is out of bounds
    with pytest.raises(IndexError):
        selector.transform(X)


