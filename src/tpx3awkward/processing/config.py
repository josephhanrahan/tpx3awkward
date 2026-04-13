from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .cluster import DEFAULT_CLUSTER_TW


class Tpx3Config(BaseModel):
    """
    Configuration for TPX3 data processing pipeline.

    This includes clustering, corrections, energy estimations, and output formats.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Clustering ---
    time_window: float
    radius: float

    # --- Energy estimation ---
    estimate_energy: bool = False
    energy_estimation_parameters: np.ndarray | None = None

    # --- Timewalk ---
    correct_timewalk: bool = False
    timewalk_b: float | None = None
    timewalk_c: float | None = None

    # --- Trim correction ---
    correct_trim: bool = False
    trim_mask: np.ndarray | None = None

    # --- Misc. ---
    file_extension: str
    add_centroid_cols: bool
    overwrite: bool
    verbose: bool = False

    @field_validator("energy_estimation_parameters", mode="before")
    @classmethod
    def load_energy_estimation_parameters(cls, value: np.ndarray | Path | str):
        if value is None or isinstance(value, np.ndarray):
            return value
        if isinstance(value, (str, Path)):
            path = Path(value)
            try:
                return np.load(path)
            except Exception as e:
                raise ValueError(f"Failed to load energy estimation parameters with path {path}") from e
        raise TypeError("energy_estimation_parameters must be a numpy array, a path, or None")

    @field_validator("trim_mask", mode="before")
    @classmethod
    def load_trim_mask(cls, value: np.ndarray | Path | str):
        if value is None or isinstance(value, np.ndarray):
            return value
        if isinstance(value, (str, Path)):
            path = Path(value)
            try:
                return np.load(path)
            except Exception as e:
                raise ValueError(f"Failed to load trim mask with path {path}") from e
        raise TypeError("trim_mask must be a numpy array, a path, or None")

    @field_validator("file_extension", mode="after")
    @classmethod
    def validate_file_extension(cls, value: str):
        if value not in (".h5", ".parquet"):
            raise ValueError("file_extension must be one of '.h5' or '.parquet'")

    @model_validator(mode="after")
    def validate_dependencies(self):
        if self.estimate_energy and self.energy_estimation_parameters is None:
            raise ValueError("energy_estimation_parameters must be provided when estimate_energy=True")
        if self.correct_timewalk and (self.timewalk_b is None or self.timewalk_c is None):
            raise ValueError("timewalk_b and timewalk_c parameters must be provided when correct_timewalk=True")
        if self.correct_trim and (self.trim_mask is None):
            raise ValueError("trim_mask must be provided when correct_trim=True")
        return self

    @classmethod
    def from_defaults(cls, **overrides) -> "Tpx3Config":
        defaults = {
            "time_window": DEFAULT_CLUSTER_TW,
            "radius": 3,
            "file_extension": ".parquet",
            "add_centroid_cols": True,
            "overwrite": True,
        }

        defaults.update(overrides)
        return cls(**defaults)
