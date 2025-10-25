import json

import numpy as np
import pytest
from scipy.sparse import issparse

from ml.feature_extractors import (
    BagEncoder,
    ColumnSelectorArray,
    PlayerEloEncoder,
    PlayerStatisticSumExtractor,
)


@pytest.fixture
def sample_X():
    # 2 samples, 10 columns (map + 2 teams + 8 players)
    return np.array(
        [[1, 10, 20, 0, 1, 2, 3, 4, 5, 6], [2, 11, 21, 7, 8, 9, 10, 11, 12, 13]],
        dtype=int,
    )


@pytest.fixture
def sample_y():
    return np.array([1, 0], dtype=int)


def test_column_selector_array(sample_X):
    sel = ColumnSelectorArray(columns=[0, 2])
    X_out = sel.fit_transform(sample_X)
    assert X_out.shape == (2, 2)
    np.testing.assert_array_equal(X_out, sample_X[:, [0, 2]])


def test_bag_encoder(sample_X):
    encoder = BagEncoder()
    encoder.fit(sample_X)
    X_out = encoder.transform(sample_X)
    assert issparse(X_out)
    assert X_out.shape[0] == sample_X.shape[0]
    # Number of features should equal number of unique values
    assert X_out.shape[1] == len(encoder.dict_)


def test_player_elo_encoder(sample_X, sample_y):
    encoder = PlayerEloEncoder(k_factor=32, base_elo=1000)
    encoder.fit(sample_X, sample_y)
    X_out = encoder.transform(sample_X)
    assert X_out.shape[0] == sample_X.shape[0]
    assert X_out.shape[1] == 30


def test_player_statistic_sum_extractor(tmp_path):
    # Prepare fake game JSON
    game_id = 100
    game_file = tmp_path / f"{game_id}.json"
    game_data = {
        "players": [
            {"player": {"id": i}, "kills": i, "deaths": i * 0.5, "assists": 1}
            for i in range(10)
        ]
    }
    game_file.write_text(json.dumps(game_data))

    X = np.array([list(range(10))], dtype=int)
    extractor = PlayerStatisticSumExtractor(
        game_ids=[game_id], path_to_dir=str(tmp_path), key="kills"
    )
    extractor.fit(X)
    X_out = extractor.transform(X)
    assert X_out.shape[0] == X.shape[0]
    # Number of features = 10 (players) + 10*10 differences + 2 means + 1 diff = 123? Let's check
    assert np.issubdtype(X_out.dtype, np.floating)
