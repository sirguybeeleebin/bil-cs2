import numpy as np
from scipy import sparse

from ml.feature_extractors import (
    ColumnSelector,
    PlayerBagEncoder,
    PlayerEloEncoder,
    TeamBagEncoder,
)


# --------------------------
# ColumnSelector tests
# --------------------------
def test_column_selector():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    selector = ColumnSelector(columns=[0, 2])
    selector.fit(X)
    X_sel = selector.transform(X)
    assert X_sel.shape == (2, 2)
    assert (X_sel == np.array([[1, 3], [4, 6]])).all()


# --------------------------
# PlayerBagEncoder tests
# --------------------------
def test_player_bag_encoder():
    X = np.array([[101, 102, 103, 104, 105, 201, 202, 203, 204, 205]])
    encoder = PlayerBagEncoder()
    encoder.fit(X)
    X_enc = encoder.transform(X)
    assert sparse.issparse(X_enc)
    assert X_enc.shape[1] == 10  # 10 unique players
    assert X_enc[0, encoder.player_dict[101]] == 1
    assert X_enc[0, encoder.player_dict[201]] == -1


# --------------------------
# TeamBagEncoder tests
# --------------------------
def test_team_bag_encoder():
    X = np.array([[1, 2]])
    encoder = TeamBagEncoder()
    encoder.fit(X)
    X_enc = encoder.transform(X)
    assert sparse.issparse(X_enc)
    assert X_enc.shape[1] == 2
    assert X_enc[0, encoder.team_dict[1]] == 1
    assert X_enc[0, encoder.team_dict[2]] == -1


# --------------------------
# PlayerEloEncoder tests
# --------------------------
def test_player_elo_encoder_basic():
    # Two teams: 5 players each
    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    y = np.array([1, 0])  # first game team1 wins, second game team2 wins

    encoder = PlayerEloEncoder(k_factor=32, base_elo=1000)
    encoder.fit(X, y)
    X_transformed = encoder.transform(X)

    # Should return numpy array with shape (n_samples, augmented_features)
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == 2
    # Features = 2 means + 2 means + 1 difference + 25 pairwise differences = 30
    assert (
        X_transformed.shape[1] == 5 * 5 + 5
    )  # 25 pairwise + 5 aggregated features approx

    # Elo values updated after fit
    assert encoder.elo_dict_[1] != 1000
    assert encoder.elo_dict_[6] != 1000


def test_player_elo_encoder_default_values():
    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    y = np.array([1])
    encoder = PlayerEloEncoder()
    encoder.fit(X, y)
    X_transformed = encoder.transform(X)
    # All output values should be floats
    assert np.issubdtype(X_transformed.dtype, np.floating)
