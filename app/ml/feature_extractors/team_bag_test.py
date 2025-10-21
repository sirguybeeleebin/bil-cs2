import numpy as np
from scipy import sparse
import pytest
from app.ml.feature_extractors.team_bag import TeamBagEncoder  # replace 'your_module' with actual module name

def test_fit_creates_team_dict():
    X = np.array([["A", "B"], ["C", "A"]])
    encoder = TeamBagEncoder()
    encoder.fit(X)
    
    assert isinstance(encoder.team_dict, dict)
    assert set(encoder.team_dict.keys()) == {"A", "B", "C"}
    assert all(isinstance(idx, int) for idx in encoder.team_dict.values())

def test_transform_creates_correct_sparse_matrix():
    X = np.array([["A", "B"], ["C", "A"]])
    encoder = TeamBagEncoder()
    encoder.fit(X)
    X_transformed = encoder.transform(X)
    
    assert sparse.issparse(X_transformed)
    assert X_transformed.shape == (2, 3)  # 2 samples, 3 unique teams
    # Convert to dense for checking values
    dense = X_transformed.toarray()
    team_to_idx = encoder.team_dict
    # First row: A +1, B -1
    assert dense[0, team_to_idx["A"]] == 1
    assert dense[0, team_to_idx["B"]] == -1
    # Second row: C +1, A -1
    assert dense[1, team_to_idx["C"]] == 1
    assert dense[1, team_to_idx["A"]] == -1

def test_transform_ignores_unseen_teams():
    X_train = np.array([["A", "B"]])
    X_test = np.array([["A", "D"]])
    encoder = TeamBagEncoder()
    encoder.fit(X_train)
    X_transformed = encoder.transform(X_test)
    
    dense = X_transformed.toarray()
    team_to_idx = encoder.team_dict
    # Only A exists
    assert dense[0, team_to_idx["A"]] == 1
    assert dense.shape[1] == len(team_to_idx)  # Should not add new column for D

if __name__ == "__main__":
    pytest.main()
