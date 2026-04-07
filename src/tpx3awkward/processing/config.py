from types import Path

import numpy as np
from pydantic import BaseModel

from .pipeline import f_type


class Tpx3Config(BaseModel):
    # clustering/centroiding configurations
    time_window: float
    radius: float

    # energy estimation configurations
    estimate_energy: bool
    energy_estimation_params: np.ndarray | Path

    # timewalk configurations
    correct_timewalk: bool
    b: float
    c: float

    # trim configurations
    correct_trim: bool
    trim_mask: np.ndarray | Path

    # misc configurations
    file_extension: f_type
    add_centroid_cols: bool
