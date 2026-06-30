import numpy as np

from real2sim_scene_foundry.alignment import SimilarityTransform, estimate_similarity_umeyama


def test_estimate_similarity_recovers_scale_rotation_translation():
    source = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = SimilarityTransform(scale=0.42, rotation=rotation, translation=np.array([0.3, -0.2, 1.1]))
    target = expected.apply(source)

    actual = estimate_similarity_umeyama(source, target)

    assert actual.scale == expected.scale
    np.testing.assert_allclose(actual.rotation, expected.rotation, atol=1e-8)
    np.testing.assert_allclose(actual.translation, expected.translation, atol=1e-8)
    np.testing.assert_allclose(actual.matrix(), expected.matrix(), atol=1e-8)
