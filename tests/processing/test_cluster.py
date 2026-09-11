from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tpx3awkward import Tpx3Config
from tpx3awkward.processing import cluster_decoded_df, decode_tpx3_binary, raw_as_numpy
from tpx3awkward.processing.corrections import estimate_energies, timewalk_corr
from tpx3awkward.processing.schemas import empty_cent_df, empty_raw_df

RAW_DATA_DIR = Path(__file__).parents[1] / "data/raw/"
PROC_DATA_DIR = Path(__file__).parents[1] / "data/processed/"
CONFIG_DIR = Path(__file__).parents[1] / "configs"


@pytest.fixture
def decoded_df():
    event_df, _ = decode_tpx3_binary(raw_as_numpy(RAW_DATA_DIR / "raw_test_data_01.tpx3"))
    return event_df


@pytest.fixture
def stable_cdf():
    cdf = pd.read_parquet(PROC_DATA_DIR / "raw_test_data_01_cent.parquet")
    cdf.loc[cdf["xc"] >= 255.5, "xc"] -= 2
    cdf.loc[cdf["yc"] >= 255.5, "yc"] -= 2
    return cdf


@pytest.fixture
def config():
    with Path(CONFIG_DIR / "tpx3_configurations.yaml").open() as f:
        data = yaml.safe_load(f)
    return Tpx3Config.model_validate(data)


def test_cluster_decoded_df(decoded_df, stable_cdf, config):
    current_cdf = cluster_decoded_df(decoded_df, config.time_window, config.radius)

    pd.testing.assert_frame_equal(current_cdf, stable_cdf.drop(columns=["e_sum", "t_corr"]), atol=0.01)


def test_cluster_decoded_df_only_tcorr(decoded_df, stable_cdf, config):
    decoded_df["t_corr"] = timewalk_corr(
        decoded_df["t"].to_numpy(), decoded_df["ToT"].to_numpy(), config.timewalk_b, config.timewalk_c
    )
    current_cdf = cluster_decoded_df(decoded_df, config.time_window, config.radius)

    pd.testing.assert_frame_equal(current_cdf, stable_cdf.drop(columns="e_sum"), atol=0.01)


def test_cluster_decoded_df_only_e(decoded_df, stable_cdf, config):
    decoded_df["e"] = estimate_energies(
        decoded_df["x"].to_numpy(),
        decoded_df["y"].to_numpy(),
        decoded_df["ToT"].to_numpy(),
        config.energy_estimation_parameters,
    )
    current_cdf = cluster_decoded_df(decoded_df, config.time_window, config.radius)

    pd.testing.assert_frame_equal(current_cdf, stable_cdf.drop(columns="t_corr"), atol=0.01)


def test_cluster_decoded_df_e_and_tcorr(decoded_df, stable_cdf, config):
    decoded_df["t_corr"] = timewalk_corr(
        decoded_df["t"].to_numpy(), decoded_df["ToT"].to_numpy(), config.timewalk_b, config.timewalk_c
    )
    decoded_df["e"] = estimate_energies(
        decoded_df["x"].to_numpy(),
        decoded_df["y"].to_numpy(),
        decoded_df["ToT"].to_numpy(),
        config.energy_estimation_parameters,
    )
    current_cdf = cluster_decoded_df(decoded_df, config.time_window, config.radius)

    pd.testing.assert_frame_equal(current_cdf, stable_cdf, atol=0.01)


ALGORITHMS = ["legacy", "dbscan", "optics", "agglomerative"]
NEW_ALGORITHMS = ALGORITHMS[1:]


@pytest.fixture
def separated_hits():
    # Deliberately unsorted and with a non-unique index: labels are positional.
    return pd.DataFrame(
        {
            "x": [20, 0, 21, 1, 100],
            "y": [20, 0, 20, 0, 100],
            "t": np.array([1280, 0, 1290, 10, 2560], dtype=np.uint64),
            "ToT": [10, 10, 30, 30, 50],
            "e": [1.0, 2.0, 3.0, 4.0, 5.0],
            "t_corr": np.array([1275, 0, 1285, 5, 2555], dtype=np.uint64),
        },
        index=[8, 3, 8, 1, 0],
    )


@pytest.mark.parametrize("algorithm", NEW_ALGORITHMS)
def test_backend_centroids_and_noise(separated_hits, algorithm):
    result = cluster_decoded_df(separated_hits, 0.1, 2, clustering_algorithm=algorithm)
    expected_count = 3 if algorithm == "agglomerative" else 2
    assert len(result) == expected_count
    np.testing.assert_allclose(result["xc"].iloc[:2], [0.75, 20.75])
    np.testing.assert_allclose(result["yc"].iloc[:2], [0, 20])
    np.testing.assert_array_equal(result["n"].iloc[:2], [2, 2])
    np.testing.assert_array_equal(result["ToT_sum"].iloc[:2], [40, 40])
    np.testing.assert_array_equal(result["ToT_max"].iloc[:2], [30, 30])
    np.testing.assert_array_equal(result["t"].iloc[:2], [10, 1290])
    np.testing.assert_array_equal(result["t_corr"].iloc[:2], [5, 1285])
    np.testing.assert_allclose(result["e_sum"].iloc[:2], [6, 4])
    labels = separated_hits["cluster_id"].to_numpy()
    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]
    assert (labels[4] == -1) == (algorithm != "agglomerative")
    assert separated_hits.index.tolist() == [8, 3, 8, 1, 0]


@pytest.mark.parametrize("algorithm", NEW_ALGORITHMS)
def test_transitive_clustering(algorithm):
    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0] * 4, "t": [0] * 4, "ToT": [10] * 4})
    result = cluster_decoded_df(df, 0.1, 1.1, clustering_algorithm=algorithm)
    assert result["n"].tolist() == [4]
    assert result["xc"].tolist() == [1.5]


@pytest.mark.parametrize("algorithm", NEW_ALGORITHMS)
def test_continuous_time_and_scaling(algorithm):
    df = pd.DataFrame({"x": [0, 0], "y": [0, 0], "t": [63, 65], "ToT": [10, 20]})
    joined = cluster_decoded_df(df, 0.1, 1.1, clustering_algorithm=algorithm)
    assert joined["n"].tolist() == [2]
    # Same spatial location, but more than radius * tw apart in time.
    df["t"] = [0, 128]
    separated = cluster_decoded_df(df, 0.1, 1.1, clustering_algorithm=algorithm)
    assert separated["n"].tolist() == ([1, 1] if algorithm == "agglomerative" else [])
    joined = cluster_decoded_df(df, 0.2, 1.1, clustering_algorithm=algorithm)
    assert joined["n"].tolist() == [2]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize(("energy", "timewalk"), [(False, False), (True, False), (False, True), (True, True)])
def test_empty_clustering(algorithm, energy, timewalk):
    df = empty_raw_df()
    if energy:
        df["e"] = np.array([], dtype=np.float32)
    if timewalk:
        df["t_corr"] = np.array([], dtype=np.uint64)
    result = cluster_decoded_df(df, 0.3, 3, clustering_algorithm=algorithm)
    pd.testing.assert_frame_equal(result, empty_cent_df(energy, timewalk))
    assert df["cluster_id"].dtype == np.int64


@pytest.mark.parametrize("algorithm", ["dbscan", "optics"])
@pytest.mark.parametrize("size", [1, 2, 4])
def test_all_noise_and_small_samples(algorithm, size):
    df = pd.DataFrame({"x": np.arange(size) * 10, "y": [0] * size, "t": [0] * size, "ToT": [10] * size})
    result = cluster_decoded_df(df, 0.3, 1, clustering_algorithm=algorithm, min_samples=3)
    pd.testing.assert_frame_equal(result, empty_cent_df())
    assert df["cluster_id"].tolist() == [-1] * size


@pytest.mark.parametrize("algorithm", ["dbscan", "agglomerative"])
def test_singleton_retained(algorithm):
    df = pd.DataFrame({"x": [2], "y": [3], "t": [4], "ToT": [10]})
    result = cluster_decoded_df(df, 0.3, 1, clustering_algorithm=algorithm, min_samples=1)
    assert result["n"].tolist() == [1]
    assert result["xc"].tolist() == [2]
    assert df["cluster_id"].tolist() == [0]


def test_dbscan_border_point_does_not_connect_clusters():
    # x=0 is a border point touching two distinct dense groups, but not core.
    df = pd.DataFrame({"x": [0, -1, -1.5, -1.6, -1.7, 1, 1.5, 1.6, 1.7], "y": 0, "t": 0, "ToT": 10})
    result = cluster_decoded_df(df, 0.3, 1.01, clustering_algorithm="dbscan", min_samples=4)
    assert sorted(result["n"].tolist()) == [4, 5]
    assert df["cluster_id"].iloc[0] >= 0
    assert df["cluster_id"].iloc[1] != df["cluster_id"].iloc[5]


@pytest.mark.parametrize("algorithm", NEW_ALGORITHMS)
def test_large_cluster_count(algorithm):
    df = pd.DataFrame({"x": np.linspace(0, 0.1, 300), "y": 0, "t": 0, "ToT": 10})
    result = cluster_decoded_df(df, 0.3, 1, clustering_algorithm=algorithm)
    assert result["n"].tolist() == [300]
    assert result["n"].dtype == np.uint64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clustering_algorithm": "unknown"},
        {"tw": 0},
        {"tw": 0.001},
        {"tw": np.inf},
        {"tw": np.nan},
        {"radius": 0},
        {"radius": -1},
        {"radius": np.inf},
        {"min_samples": 0},
        {"min_samples": 1.5},
        {"min_samples": True},
        {"clustering_algorithm": "optics", "min_samples": 1},
    ],
)
def test_invalid_clustering_parameters(kwargs):
    params = {"tw": 0.3, "radius": 3} | kwargs
    with pytest.raises(ValueError):
        cluster_decoded_df(empty_raw_df(), **params)
