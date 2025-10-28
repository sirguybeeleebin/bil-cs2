import json

import numpy as np

from train_model.feature_extractors import (
    BagEncoder,
    ColumnSelectorArray,
    PlayerEloEncoder,
    PlayerMapStatisticSumExtractor,
    PlayerStatisticSumExtractor,
)


def test_column_selector_array():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    selector = ColumnSelectorArray(columns=[0, 2])
    selector.fit(X)
    X_trans = selector.transform(X)
    np.testing.assert_array_equal(X_trans, X[:, [0, 2]])


def test_bag_encoder():
    X = np.array([[1, 2, 3, 4], [3, 4, 1, 2]])
    encoder = BagEncoder()
    encoder.fit(X)
    X_trans = encoder.transform(X)
    assert X_trans.shape[0] == 2
    assert X_trans.shape[1] == len(encoder.dict_)


def test_player_elo_encoder():
    X = np.array([[101, 102, 103, 104, 105, 201, 202, 203, 204, 205]])
    y = np.array([1])
    encoder = PlayerEloEncoder(k_factor=10)
    encoder.fit(X, y)
    X_trans = encoder.transform(X)
    assert X_trans.shape[0] == 1
    assert X_trans.shape[1] == 30  # corrected from 35 to 30
    for val in encoder.elo_dict_.values():
        assert val != 1000  # Elo updated


def test_player_statistic_sum_extractor(tmp_path):
    # create a sample JSON game
    game = {
        "id": 1,
        "players": [{"player": {"id": i}, "kills": i * 2} for i in range(1, 11)],
        "map": {"id": 100},
    }
    file_path = tmp_path / "1.json"
    file_path.write_text(json.dumps(game))

    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    extractor = PlayerStatisticSumExtractor(
        game_ids=[1], path_to_dir=str(tmp_path), key="kills"
    )
    extractor.fit(X)
    X_trans = extractor.transform(X)
    assert X_trans.shape[0] == 1
    assert X_trans.shape[1] > 10
    assert all(val > 0 for val in extractor.player_stat_dict.values())


def test_player_map_statistic_sum_extractor(tmp_path):
    # create a sample JSON game
    game = {
        "id": 1,
        "players": [{"player": {"id": i}, "kills": i} for i in range(1, 11)],
        "map": {"id": 100},
    }
    file_path = tmp_path / "1.json"
    file_path.write_text(json.dumps(game))

    X = np.array([[100] + list(range(1, 11))])
    extractor = PlayerMapStatisticSumExtractor(
        game_ids=[1], path_to_dir=str(tmp_path), key="kills"
    )
    extractor.fit(X)
    X_trans = extractor.transform(X)
    assert X_trans.shape[0] == 1
    assert X_trans.shape[1] > 10
    assert extractor.player_stat_dict[100][1] == 1
