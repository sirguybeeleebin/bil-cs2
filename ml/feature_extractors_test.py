import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ml.feature_extractors import (
    BagEncoder,
    ColumnSelectorArray,
    PlayerEloEncoder,
    PlayerMapStatSumExtractor,
    PlayerRoundFirstHalfWinSumExtractor,
    PlayerRoundWinSumExtractor,
    PlayerStatSumExtractor,
)


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def sample_X():
    # 1 map ID, 2 team IDs, 10 player IDs
    return np.array([[101, 10, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])


@pytest.fixture
def sample_y():
    return np.array([1])


@pytest.fixture
def game_json(tmp_path):
    game = {
        "id": 1,
        "map": {"id": 101},
        "begin_at": "2025-11-04T12:00:00Z",
        "players": [
            {"team": {"id": 10}, "player": {"id": i + 1}, "kills": 5, "assists": 2}
            for i in range(5)
        ]
        + [
            {"team": {"id": 20}, "player": {"id": i + 6}, "kills": 3, "assists": 1}
            for i in range(5)
        ],
        "rounds": [
            {
                "round": i + 1,
                "winner_team": 10 if i % 2 == 0 else 20,
                "ct": 10,
                "terrorists": 20,
            }
            for i in range(16)
        ],
    }
    fpath = tmp_path / "1.json"
    fpath.write_text(json.dumps(game))
    return tmp_path


# -----------------------------
# ColumnSelectorArray tests
# -----------------------------
def test_column_selector_array(sample_X):
    selector = ColumnSelectorArray(columns=[0, 3, 4])
    Xt = selector.fit_transform(sample_X)
    assert Xt.shape == (1, 3)
    assert (Xt[0] == np.array([101, 1, 2])).all()


# -----------------------------
# BagEncoder tests
# -----------------------------
def test_bag_encoder(sample_X):
    encoder = BagEncoder()
    encoder.fit(sample_X)
    Xt = encoder.transform(sample_X)
    assert isinstance(Xt, csr_matrix)
    assert Xt.shape[0] == sample_X.shape[0]


# -----------------------------
# PlayerEloEncoder tests
# -----------------------------
def test_player_elo_encoder(sample_X, sample_y):
    encoder = PlayerEloEncoder()
    encoder.fit(sample_X, sample_y)
    Xt = encoder.transform(sample_X)
    assert Xt.shape[0] == sample_X.shape[0]
    # 5 left + 5 right + mean_left + mean_right + mean_diff + 25 pairwise differences = 41
    assert Xt.shape[1] == 41


# -----------------------------
# PlayerStatSumExtractor tests
# -----------------------------
def test_player_stat_sum_extractor(game_json, sample_X):
    extractor = PlayerStatSumExtractor(
        game_ids=[1], stat_key="kills", path_to_dir=str(game_json)
    )
    extractor.fit(sample_X)
    Xt = extractor.transform(sample_X)
    assert Xt.shape[0] == sample_X.shape[0]
    # 5 left + 5 right + mean_left + mean_right + mean_diff + 25 pairwise differences = 41
    assert Xt.shape[1] == 41


# -----------------------------
# PlayerMapStatSumExtractor tests
# -----------------------------
def test_player_map_stat_sum_extractor(game_json, sample_X):
    extractor = PlayerMapStatSumExtractor(
        game_ids=[1], stat_key="kills", path_to_dir=str(game_json)
    )
    extractor.fit(sample_X)
    Xt = extractor.transform(sample_X)
    assert Xt.shape[0] == sample_X.shape[0]


# -----------------------------
# PlayerRoundWinSumExtractor tests
# -----------------------------
def test_player_round_win_sum_extractor(game_json, sample_X):
    extractor = PlayerRoundWinSumExtractor(game_ids=[1], path_to_dir=str(game_json))
    extractor.fit(sample_X)
    Xt = extractor.transform(sample_X)
    assert Xt.shape[0] == sample_X.shape[0]


# -----------------------------
# PlayerRoundFirstHalfWinSumExtractor tests
# -----------------------------
def test_player_round_first_half_win_sum_extractor(game_json, sample_X):
    extractor = PlayerRoundFirstHalfWinSumExtractor(
        game_ids=[1], path_to_dir=str(game_json)
    )
    extractor.fit(sample_X)
    Xt = extractor.transform(sample_X)
    assert Xt.shape[0] == sample_X.shape[0]
