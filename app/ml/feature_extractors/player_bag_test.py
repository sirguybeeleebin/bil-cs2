import numpy as np
from scipy import sparse
import pytest
from app.ml.feature_extractors.player_bag import PlayerBagEncoder  # replace with actual module path

def test_fit_creates_player_dict():
    X = np.array([["p1", "p2", "p3", "p4"], ["p5", "p1", "p6", "p7"]])
    encoder = PlayerBagEncoder()
    encoder.fit(X)
    
    assert isinstance(encoder.player_dict, dict)
    expected_players = {"p1", "p2", "p3", "p4", "p5", "p6", "p7"}
    assert set(encoder.player_dict.keys()) == expected_players
    assert all(isinstance(idx, int) for idx in encoder.player_dict.values())

def test_transform_creates_correct_sparse_matrix():
    X = np.array([["p1", "p2", "p3", "p4"], ["p5", "p1", "p6", "p7"]])
    encoder = PlayerBagEncoder()
    encoder.fit(X)
    X_transformed = encoder.transform(X)
    
    assert sparse.issparse(X_transformed)
    assert X_transformed.shape == (2, 7)  # 2 samples, 7 unique players
    dense = X_transformed.toarray()
    player_to_idx = encoder.player_dict

    # Check first row: first half +1, second half -1
    mid = len(X[0]) // 2
    for j, player in enumerate(X[0]):
        expected = 1 if j < mid else -1
        assert dense[0, player_to_idx[player]] == expected

    # Check second row
    mid = len(X[1]) // 2
    for j, player in enumerate(X[1]):
        expected = 1 if j < mid else -1
        assert dense[1, player_to_idx[player]] == expected

def test_transform_ignores_unseen_players():
    X_train = np.array([["p1", "p2", "p3", "p4"]])
    X_test = np.array([["p1", "p5", "p6", "p2"]])
