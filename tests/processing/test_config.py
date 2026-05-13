import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tpx3awkward import Tpx3Config

CONFIG_DIR = Path(__file__).parents[1] / "configs"


def default_tpx3config_dict(**overrides):
    defaults = {
        "time_window": 0.3,
        "radius": 3,
    }
    defaults.update(overrides)
    return defaults


def assert_tpx3config_matches_input(tpx3config, input_dict):
    for field, expected in input_dict.items():
        actual = getattr(tpx3config, field)
        if field in {"energy_estimation_parameters", "trim_mask"}:
            if expected is None:
                assert actual is None
            elif isinstance(expected, Path | str):
                np.testing.assert_array_equal(actual, np.load(expected))
            else:
                np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


def test_tpx3config_init():
    tpx3config_dict = {
        "time_window": 0.3,
        "radius": 3,
        "estimate_energy": True,
        "energy_estimation_parameters": np.load(CONFIG_DIR / "energy_estimation_test_parameters.npy"),
        "correct_timewalk": True,
        "timewalk_b": 167.0,
        "timewalk_c": -0.016,
        "correct_trim": False,  # TODO get an example trim correction file and test this too
        "file_extension": ".parquet",
        "add_centroid_cols": True,
        "overwrite": True,
        "verbose": True,
    }

    tpx3config_constructor = Tpx3Config(**tpx3config_dict)
    tpx3config_model_validate = Tpx3Config.model_validate(tpx3config_dict)

    # assert Tpx3Config objects from Tpx3Config.model_validate and Tpx3Config constructor are the same
    assert tpx3config_constructor == tpx3config_model_validate

    # assert the values used to initalize the config are the same as what's in the Tpx3Config object
    assert_tpx3config_matches_input(tpx3config_constructor, tpx3config_dict)
    assert_tpx3config_matches_input(tpx3config_model_validate, tpx3config_dict)


def test_tpx3config_init_yaml():
    yaml_file_path = CONFIG_DIR / "tpx3_configurations.yaml"

    with Path.open(yaml_file_path) as f:
        tpx3config_dict = yaml.safe_load(f)

    tpx3config = Tpx3Config(**tpx3config_dict)
    assert_tpx3config_matches_input(tpx3config, tpx3config_dict)


def test_tpx3config_init_json():
    json_file_path = CONFIG_DIR / "tpx3_configurations.json"

    with Path.open(json_file_path) as f:
        tpx3config_dict = json.load(f)
    tpx3config_model_validate_json = Tpx3Config.model_validate_json(json_file_path.read_text())
    tpx3config_json_load = Tpx3Config(**tpx3config_dict)

    assert_tpx3config_matches_input(tpx3config_model_validate_json, tpx3config_dict)
    assert_tpx3config_matches_input(tpx3config_json_load, tpx3config_dict)


def test_tpx3config_load_energy_estimation_parameters():
    path_to_energy_estimation_parameters = CONFIG_DIR / "energy_estimation_test_parameters.npy"
    tpx3config_dict = default_tpx3config_dict(
        estimate_energy=True, energy_estimation_parameters=path_to_energy_estimation_parameters
    )
    tpx3config = Tpx3Config.model_validate(tpx3config_dict)

    np.testing.assert_array_equal(tpx3config.energy_estimation_parameters, np.load(path_to_energy_estimation_parameters))


def test_tpx3config_load_energy_estimation_parameters_error():
    # ensure ValueError when energy_estimation_parameters is path and is invalid
    with pytest.raises(ValueError):
        Tpx3Config.model_validate(
            default_tpx3config_dict(estimate_energy=True, energy_estimation_parameters="invalidpath.npy")
        )

    # ensure TypeError when energy_estimation_parameters is not np.ndarray, str, or Path
    with pytest.raises(TypeError):
        Tpx3Config.model_validate(default_tpx3config_dict(estimate_energy=True, energy_estimation_parameters=5))


def test_tpx3config_load_trim_mask(): ...


def test_validate_file_extension():
    tpx3config_dict_h5 = default_tpx3config_dict(file_extension=".h5")
    tpx3config_dict_parquet = default_tpx3config_dict(file_extension=".parquet")
    tpx3config_h5 = Tpx3Config.model_validate(tpx3config_dict_h5)
    tpx3config_parquet = Tpx3Config.model_validate(tpx3config_dict_parquet)

    assert tpx3config_h5.file_extension == ".h5"
    assert tpx3config_parquet.file_extension == ".parquet"

    with pytest.raises(ValueError):
        Tpx3Config.model_validate(default_tpx3config_dict(file_extension=".csv"))


def test_tpx3config_validate_dependencies():
    # ensure ValueError when estimate_energy is True but no energy_estimation_parameters are provided
    with pytest.raises(ValueError, match="energy_estimation_parameters must be provided when estimate_energy=True"):
        Tpx3Config.model_validate(default_tpx3config_dict(estimate_energy=True, energy_estimation_parameters=None))

    # ensure ValueError when correct_timewalk is True but timewalk parameters are missing
    with pytest.raises(ValueError, match="timewalk_b and timewalk_c parameters must be provided when correct_timewalk=True"):
        Tpx3Config.model_validate(default_tpx3config_dict(correct_timewalk=True, timewalk_b=None, timewalk_c=None))

    # ensure ValueError when correct_trim is True but no trim_mas is provided
    with pytest.raises(ValueError, match="trim_mask must be provided when correct_trim=True"):
        Tpx3Config.model_validate(default_tpx3config_dict(correct_trim=True, trim_mask=None))


def test_tpx3config_from_defaults():
    Tpx3Config.from_defaults()


def test_tpx3config_from_defaults_overrides():
    tpx3config = Tpx3Config.from_defaults(file_extension=".h5", verbose=True)

    assert tpx3config.file_extension == ".h5"
    assert tpx3config.verbose
