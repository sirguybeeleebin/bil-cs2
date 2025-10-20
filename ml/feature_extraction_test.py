import numpy as np
from scipy import sparse
import pytest
from ml.feature_extraction import ColumnSelector, PlayerBagEncoder, TeamBagEncoder, PlayerEloEncoder


def test_column_selector_basic():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    selector = ColumnSelector(columns=[0, 2])
    X_selected = selector.fit_transform(X)
    assert X_selected.shape == (2, 2)
    assert (X_selected == np.array([[1, 3], [4, 6]])).all()


def test_column_selector_non_contiguous():
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    selector = ColumnSelector(columns=[0, 3])
    X_selected = selector.fit_transform(X)
    assert X_selected.shape == (2, 2)
    assert (X_selected == np.array([[1, 4], [5, 8]])).all()


def test_player_bag_encoder_basic():
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    encoder = PlayerBagEncoder()
    encoder.fit(X)
    X_trans = encoder.transform(X)
    assert isinstance(X_trans, sparse.csr_matrix)
    assert X_trans.shape == (2, 8)
    arr = X_trans.toarray()
    # First half of row 0: positive (team1), second half negative (team2)
    assert all((arr[0][:2] == 1) | (arr[0][:2] == 0))
    assert all((arr[0][2:] == -1) | (arr[0][2:] == 0))


def test_team_bag_encoder_basic():
    X = np.array([[100, 200], [300, 400]])
    encoder = TeamBagEncoder()
    encoder.fit(X)
    X_trans = encoder.transform(X)
    assert X_trans.shape == (2, 4)
    arr = X_trans.toarray()
    # Check first/second columns
    assert arr[0][0] == 1
    assert arr[0][1] == -1
    assert arr[1][2] == 1
    assert arr[1][3] == -1


def test_player_elo_encoder_fit_transform_consistency():
    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    y = np.array([1, 0])
    encoder = PlayerEloEncoder(k_factor=32, base_elo=1000)
    encoder.fit(X, y)
    X_trans = encoder.transform(X)
    # Augmented shape: 10 players -> 5+5 means 30 features
    assert X_trans.shape == (2, 30)
    # Values differ from base Elo
    assert not np.all(X_trans == 1000)

    # Transform new data
    X_new = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    X_new_trans = encoder.transform(X_new)
    assert X_new_trans.shape == (1, 30)


def test_player_elo_encoder_values_update():
    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    y = np.array([1])
    encoder = PlayerEloEncoder(k_factor=32, base_elo=1000)
    encoder.fit(X, y)
    # Check winners > base, losers < base
    for p in range(1, 6):
        assert encoder.elo_dict_[p] > 1000
    for p in range(6, 11):
        assert encoder.elo_dict_[p] < 1000


def test_player_elo_encoder_new_players():
    X_train = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    y_train = np.array([1])
    encoder = PlayerEloEncoder(k_factor=32, base_elo=1000)
    encoder.fit(X_train, y_train)

    # Новые игроки не в тренировке
    X_test = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    
    # Проверяем сырые Elo
    raw_elos = np.array([[encoder.elo_dict_.get(pid, 1000) for pid in row] for row in X_test])
    assert np.all(raw_elos == 1000)

    # Проверяем размер аугментированного вектора
    X_test_trans = encoder.transform(X_test)
    assert X_test_trans.shape == (1, 30)  # 10 игроков + 20 фичей разностей



def test_player_elo_encoder_augment_features():
    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    y = np.array([1])
    encoder = PlayerEloEncoder(k_factor=32, base_elo=1000)
    encoder.fit(X, y)
    X_trans = encoder.transform(X)
    # Augmented features: means and pairwise diffs included
    assert X_trans.shape[1] == 30
