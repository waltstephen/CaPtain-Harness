import numpy as np
from scipy.spatial import cKDTree

from captain_harness.adapters.capx.backend import _NOISE_MIN_SAMPLES, _NOISE_RADIUS_M, _filter_point_cloud_noise


def _ball_count_reference(points):
    """The previous implementation (count every neighbour in the ball): the k-th-neighbour version must match it."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < _NOISE_MIN_SAMPLES:
        return np.empty((0, 3), dtype=np.float64)
    counts = cKDTree(points).query_ball_point(points, r=_NOISE_RADIUS_M, return_length=True)
    core = np.asarray(counts) >= _NOISE_MIN_SAMPLES
    if not core.any():
        return np.empty((0, 3), dtype=np.float64)
    distances, _ = cKDTree(points[core]).query(points, k=1)
    return points[np.asarray(distances) <= _NOISE_RADIUS_M]


def _depth_like_cloud(rng, side, pitch, noise, outliers):
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2) * pitch
    surface = np.column_stack([grid, 0.3 + rng.normal(0.0, noise, len(grid))])
    return np.vstack([surface, rng.uniform(-0.05, 0.3, (outliers, 3)), surface[:7]])   # + exact duplicates


def test_filter_matches_ball_count_on_dense_sparse_and_border_clouds():
    rng = np.random.default_rng(7)
    clouds = [
        _depth_like_cloud(rng, 120, 0.001, 0.0015, 300),    # dense mask surface: most points core
        _depth_like_cloud(rng, 40, 0.0018, 0.003, 200),     # sparse: core status near the 10-neighbour border
        _depth_like_cloud(rng, 30, 0.004, 0.001, 50),       # pitch near the radius: few cores
        rng.uniform(0, 0.02, (_NOISE_MIN_SAMPLES - 1, 3)),  # too few points
        rng.uniform(0, 1.0, (500, 3)),                      # isolated points only
    ]
    for cloud in clouds:
        np.testing.assert_array_equal(_filter_point_cloud_noise(cloud), _ball_count_reference(cloud))
