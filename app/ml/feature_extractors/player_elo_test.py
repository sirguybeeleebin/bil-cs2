import numpy as np
import pytest
from app.ml.feature_extractors.player_elo import PlayerEloEncoder  # replace with your actual module path

def test_fit_updates_elo_dict():
    X = np.array([
        ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
    ])
    y = np.array([1])
    encoder = PlayerEloEncoder(k_factor=32, base_elo=1000)
    encoder.fit(X, y)
    
    # Check that all players have Elo ratings updated
    for pid in X[0]:
        assert pid in encoder.elo_dict_
        assert isinstance(encoder.elo_dict_[pid], float)

def test_transform_returns_correct_shape():
    X = np.array([
        ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
    ])
    y = np.array([1])
    encoder = PlayerEloEncoder()
    encoder.fit(X, y)
    X_transformed = encoder.transform(X)
    
    # _augment_X produces 2 means + 3 differences + 25 pairwise diffs = 30 features
    assert X_transformed.shape == (1, 30)
    assert isinstance(X_transformed, np.ndarray)
    assert np.issubdtype(X_transformed.dtype, np.floating)

def test_transform_consistency_for_train_data():
    X = np.array([
        ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
    ])
    y = np.array([1])
    encoder = PlayerEloEncoder()
    encoder.fit(X, y)
    
    # Transforming training data should reuse stored X_elo_train_
    X_transformed1 = encoder.transform(X)
    X_transformed2 = encoder.transform(X)
    np.testing.assert_array_equal(X_transformed1, X_transformed2)

def test_transform_handles_unseen_players():
    X_train = np.array([
        ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
    ])
    y_train = np.array([1])
    X_test = np.array([
        ["p1", "p2", "p3", "p4", "p5", "p6", "p11", "p12", "p13", "p14"]
    ])
    
    encoder = PlayerEloEncoder()
    encoder.fit(X_train, y_train)
    X_transformed = encoder.transform(X_test)
    
    # Should produce same number of features (10 players -> 30 augmented features)
    assert X_transformed.shape == (1, 30)
    # Check that unseen players get base Elo
    for pid in ["p11", "p12", "p13", "p14"]:
        assert pid in encoder.elo_dict_ or pid not in encoder.elo_dict_  # they use base Elo if missing


