"""Clustering backends operating on decoded spatial and temporal coordinates."""

from numbers import Integral
from typing import Literal

import numpy as np

ClusteringAlgorithm = Literal["legacy", "dbscan", "optics", "agglomerative"]


def validate_clustering_parameters(tw: float, radius: float, algorithm: str, min_samples: int) -> None:
    if algorithm not in ("legacy", "dbscan", "optics", "agglomerative"):
        raise ValueError("clustering_algorithm must be 'legacy', 'dbscan', 'optics', or 'agglomerative'")
    if not np.isfinite(tw) or tw < 0.0015625:
        raise ValueError("time window must be finite and at least one timestamp tick (0.0015625 microseconds)")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if isinstance(min_samples, bool) or not isinstance(min_samples, Integral) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    if algorithm == "optics" and min_samples < 2:
        raise ValueError("min_samples must be at least 2 for OPTICS")


def get_cluster_labels(df, tw_ticks: float, radius: float, algorithm: ClusteringAlgorithm, min_samples: int):
    """Return contiguous labels in input row order, with -1 denoting noise.

    Time is continuous and scaled so that one time window has unit distance,
    alongside pixel coordinates. Subtract timestamps before conversion to float
    to retain small time differences in long acquisitions.
    """
    # Avoid loading scikit-learn for the default Numba backend.
    from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering  # noqa: PLC0415

    n = len(df)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    features = df[["x", "y", "t"]].to_numpy(dtype=np.float64, copy=True)
    features[:, 2] = (df["t"] - df["t"].min()).to_numpy(dtype=np.float64) / tw_ticks
    if not np.isfinite(features).all():
        raise ValueError("clustering coordinates x, y, and t must be finite")
    if algorithm in ("dbscan", "optics") and n < min_samples:
        return np.full(n, -1, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    if algorithm == "dbscan":
        estimator = DBSCAN(eps=radius, min_samples=min_samples)
    elif algorithm == "optics":
        # Extract DBSCAN-style clusters from the OPTICS reachability ordering.
        # Keeping max_eps infinite also handles isolated points without warnings.
        estimator = OPTICS(min_samples=min_samples, cluster_method="dbscan", eps=radius)
    else:
        estimator = AgglomerativeClustering(n_clusters=None, distance_threshold=radius, linkage="single")

    labels = estimator.fit_predict(features).astype(np.int64)
    valid = labels >= 0
    # Centroiding requires dense IDs; noise must never become a centroid.
    _, labels[valid] = np.unique(labels[valid], return_inverse=True)
    return labels
