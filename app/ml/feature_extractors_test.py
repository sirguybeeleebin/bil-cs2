import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from app.ml.feature_extractors import (
    BagEncoder,
    ColumnSelectorArray,
    PlayerEloEncoder,
    PlayerStatisticSumExtractor,
)

# -----------------------------
# ColumnSelectorArray
# -----------------------------


def test_column_selector_array_basic():
    X = np.arange(20).reshape(4, 5)
    selector = ColumnSelectorArray(columns=[0, 2, 4])
    selector.fit(X)
    X_out = selector.transform(X)
    assert X_out.shape == (4, 3)
    np.testing.assert_array_equal(X_out, X[:, [0, 2, 4]])


# -----------------------------
# BagEncoder
# -----------------------------


def test_bag_encoder_basic():
    X = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
    encoder = BagEncoder()
    encoder.fit(X)
    X_out = encoder.transform(X)
    assert isinstance(X_out, csr_matrix)
    assert X_out.shape[0] == X.shape[0]
    assert X_out.shape[1] == len(encoder.dict_)


# -----------------------------
# PlayerEloEncoder
# -----------------------------


def test_player_elo_encoder_fit_transform():
    X = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    )
    y = np.array([1, 0])
    encoder = PlayerEloEncoder(k_factor=32, base_elo=1000)
    encoder.fit(X, y)
    X_out = encoder.transform(X)
    assert X_out.shape[0] == X.shape[0]
    assert X_out.shape[1] > 0  # features augmented
    assert np.all(np.isfinite(X_out))


def test_player_elo_encoder_unseen_players():
    X_train = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    y_train = np.array([1])
    encoder = PlayerEloEncoder()
    encoder.fit(X_train, y_train)

    X_test = np.array([[100, 101, 102, 103, 104, 105, 106, 107, 108, 109]])
    X_out = encoder.transform(X_test)
    assert X_out.shape[0] == 1
    assert X_out.shape[1] > 0


# -----------------------------
# PlayerStatisticSumExtractor
# -----------------------------


@pytest.fixture
def game_data(tmp_path):
    game = {
        "id": 1,
        "players": [{"player": {"id": i}, "kills": i} for i in range(1, 11)],
    }
    path = tmp_path / "1.json"
    path.write_text(json.dumps(game))
    return tmp_path, [1]


def test_player_statistic_sum_extractor_fit_transform(game_data):
    tmp_path, game_ids = game_data
    X = np.array([list(range(1, 11))])
    extractor = PlayerStatisticSumExtractor(
        game_ids=game_ids, path_to_dir=str(tmp_path), key="kills"
    )
    extractor.fit(X)
    X_out = extractor.transform(X)
    assert X_out.shape[0] == X.shape[0]
    assert X_out.shape[1] > X.shape[1]  # augmented features
    assert np.all(np.isfinite(X_out))


def test_player_statistic_sum_extractor_missing_key(tmp_path):
    # Game with no 'assists' key
    game = {"id": 1, "players": [{"player": {"id": i}} for i in range(1, 11)]}
    path = tmp_path / "1.json"
    path.write_text(json.dumps(game))

    X = np.array([list(range(1, 11))])
    extractor = PlayerStatisticSumExtractor(
        game_ids=[1], path_to_dir=str(tmp_path), key="assists"
    )
    extractor.fit(X)
    X_out = extractor.transform(X)
    assert X_out.shape[0] == 1
    assert X_out.shape[1] > X.shape[1]
    assert np.all(np.isfinite(X_out))
