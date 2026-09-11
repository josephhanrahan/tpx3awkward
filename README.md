# tpx3awkward

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![Coverage][coverage-badge]][coverage-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/NSLS2/tpx3awkward/workflows/CI/badge.svg
[actions-link]:             https://github.com/NSLS2/tpx3awkward/actions
[pypi-link]:                https://pypi.org/project/tpx3awkward/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/tpx3awkward
[pypi-version]:             https://img.shields.io/pypi/v/tpx3awkward
[coverage-badge]:           https://codecov.io/github/NSLS2/tpx3awkward/branch/main/graph/badge.svg
[coverage-link]:            https://codecov.io/github/NSLS2/tpx3awkward

<!-- prettier-ignore-end -->

tpx3awkward is a Python package for efficient handling of data produced by the
Timepix family of detectors, this includes:

- Fast Decoding of raw `.tpx3` binary files
- Event clustering and centroiding
- Timewalk correction
- Energy estimation

## Installation

```bash
pip install tpx3awkward
```

## Clustering

Select a clustering algorithm through the existing DataFrame API:

```python
from tpx3awkward import cluster_decoded_df

centroids = cluster_decoded_df(
    decoded_df,
    tw=0.3,
    radius=3,
    clustering_algorithm="dbscan",
    min_samples=2,
)
```

The same options are available through `Tpx3Config` and all conversion functions:

```python
from tpx3awkward import Tpx3Config, convert_tpx3_file

config = Tpx3Config.from_defaults(clustering_algorithm="dbscan", min_samples=2)
convert_tpx3_file("events.tpx3", config=config)

# Keyword overrides are also supported.
convert_tpx3_file("events.tpx3", clustering_algorithm="agglomerative")
```

| Algorithm | Behavior |
| --- | --- |
| `legacy` (default) | Existing greedy clustering using binned timestamps; expects time-sorted input. |
| `dbscan` | Density-based clustering with neighborhood radius `radius` and minimum neighborhood size `min_samples`. A useful starting point for detector hits. |
| `optics` | Builds a reachability ordering and extracts DBSCAN-style clusters at `radius`. Requires `min_samples >= 2`. |
| `agglomerative` | Single-linkage hierarchical clustering cut at `radius`; keeps isolated hits as singleton clusters. |

The new backends use scikit-learn and accept unsorted rows. They use Euclidean
distance in `(x, y, t / tw)` coordinates: `x` and `y` are measured in pixels, and a
time difference of `tw` microseconds contributes one unit of distance. Decoded `t`
values are timestamp ticks (1.5625 ns), converted internally. Unlike `legacy`,
these backends use continuous time rather than time bins. Thus `radius` bounds the
combined spatial-temporal distance, not independent spatial and temporal windows.
DBSCAN and OPTICS include neighbors exactly at `radius`; agglomerative merges
distances strictly below it. OPTICS and hierarchical clustering are best suited
to smaller datasets because their runtime can grow quadratically.

`min_samples` includes the hit itself and is used only by DBSCAN and OPTICS.
Density-based noise is labeled `-1` and omitted from the centroid output. Use
DBSCAN with `min_samples=1` to retain isolated hits. As in the existing API,
`cluster_decoded_df` adds or replaces `cluster_id` on the input DataFrame, with
labels aligned to its original rows.

All backends share ToT-weighted centroiding and optional `e_sum` and `t_corr`
output. Clustering uses raw `t`, while the timestamp and corrected timestamp of
the highest-ToT hit are retained in each centroid. The output columns remain
`t`, `xc`, `yc`, `ToT_max`, `ToT_sum`, and `n`, plus the optional correction columns.
For clusters larger than 255 hits, `n` is promoted from `uint8` to `uint64` to
preserve the hit count. Empty or all-noise inputs return an empty centroid
DataFrame with the same columns and dtypes as other empty pipeline results.
