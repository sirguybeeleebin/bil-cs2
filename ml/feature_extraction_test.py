import numpy as np
import pytest
from scipy import sparse

from ml.feature_extraction import ColumnSelector, PlayerBagEncoder, TeamBagEncoder, PlayerEloEncoder

def test_column_selector():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    selector = ColumnSelector(columns=[0, 2])
    X_selected = selector.fit_transform(X)
    assert X_selected.shape == (2, 2)
    assert (X_selected == np.array([[1, 3], [4, 6]])).all()

def test_player_bag_encoder():
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    encoder = PlayerBagEncoder()
    encoder.fit(X)
    X_transformed = encoder.transform(X)
    assert isinstance(X_transformed, sparse.csr_matrix)
    # Shape: n_samples x n_unique_players
    assert X_transformed.shape[0] == 2
    assert X_transformed.shape[1] == 8
    # First half of row should be 1, second half -1
    data_row0 = X_transformed.toarray()[0]
    for idx, val in enumerate(data_row0):
        if idx < 2:  # first half
            assert val == 1 or val == 0
        else:
            assert val == -1 or val == 0

def test_team_bag_encoder():
    X = np.array([[100, 200], [300, 400]])
    encoder = TeamBagEncoder()
    encoder.fit(X)
    X_transformed = encoder.transform(X)
    assert isinstance(X_transformed, sparse.csr_matrix)
    assert X_transformed.shape[0] == 2
    assert X_transformed.shape[1] == 4
    # First column of each row is 1, second -1
    arr = X_transformed.toarray()
    assert arr[0][0] == 1
    assert arr[0][1] == -1
    assert arr[1][2] == 1
    assert arr[1][3] == -1

def test_player_elo_encoder_fit_transform_consistency():
    # 2 games, 5 players per team
    X = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ])
    y = np.array([1, 0])
    encoder = PlayerEloEncoder(k_factor=32, base_elo=1000)
    encoder.fit(X, y)
    X_transformed = encoder.transform(X)
    # Fix: shape should be n_samples x 30
    assert X_transformed.shape == (2, 30)
    # Values should not be exactly base_elo
    assert not np.all(X_transformed == 1000)
    
    # Transform on new data
    X_new = np.array([[1,2,3,4,5,6,7,8,9,10]])
    X_new_trans = encoder.transform(X_new)
    assert X_new_trans.shape == (1, 30)

def test_player_elo_encoder_values_change():
    # Check that Elo values are updated correctly
    X = np.array([[1,2,3,4,5,6,7,8,9,10]])
    y = np.array([1])
    encoder = PlayerEloEncoder(k_factor=32, base_elo=1000)
    encoder.fit(X, y)
    # Player 1 should have higher Elo than base after win
    assert encoder.elo_dict_[1] > 1000
    # Player 6 (losing team) should have Elo lower than base
    assert encoder.elo_dict_[6] < 1000
