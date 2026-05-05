from pathlib import Path

from tpx3awkward.processing.decoding import decode_tpx3_binary
from tpx3awkward.processing.files import raw_as_numpy

RAW_DATA_DIR = Path(__file__).parents[1] / "data/raw"


def test_decode_tpx3_binary_missing_messages(capsys):
    path_to_raw_data = RAW_DATA_DIR / "raw_test_data_00.tpx3"
    binary = raw_as_numpy(path_to_raw_data)
    decode_tpx3_binary(binary)
    decode_tpx3_binary_capture = capsys.readouterr()
    assert "Missing messages!" not in decode_tpx3_binary_capture.out


def test_decode_tpx3_binary_serval_4_missing_messages(capsys):
    path_to_raw_data = RAW_DATA_DIR / "serval_4_3/raw_test_data_serval_4_3_0.tpx3"
    binary = raw_as_numpy(path_to_raw_data)
    decode_tpx3_binary(binary)
    decode_tpx3_binary_capture = capsys.readouterr()
    # numba doesn't support raising errors, so we print error messages
    assert "Missing messages!" not in decode_tpx3_binary_capture.out
