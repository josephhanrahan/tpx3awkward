from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class Tpx3Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # clustering/centroiding configurations
    time_window: float
    radius: float

    # energy estimation configurations
    estimate_energy: bool
    energy_estimation_parameters: np.ndarray | Path | str | None

    # timewalk configurations
    correct_timewalk: bool
    b: float
    c: float

    # trim configurations
    correct_trim: bool
    trim_mask: np.ndarray | Path | str | None

    # misc configurations
    file_extension: str
    add_centroid_cols: bool

    @field_validator("energy_estimation_parameters", mode="after")
    @classmethod
    def load_energy_estimation_parameters(cls, value: np.ndarray | Path | str):
        if isinstance(value, np.ndarray):
            return value

        value = Path(value)
        try:
            return np.load(value)
        except Exception as e:
            raise ValueError(f"Failed to load energy estimation parameters with path {value}") from e
