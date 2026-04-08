import gc
import logging
import multiprocessing
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from .cluster import DEFAULT_CLUSTER_RADIUS, DEFAULT_CLUSTER_TW, cluster_raw_df
from .decoding import decode_tpx3_binary
from .files import converted_path, raw_as_numpy, save_df, trim_corr_file
from .schemas import empty_cent_df, empty_raw_df

logger = logging.getLogger(__name__)
f_type = SimpleNamespace(HDF=".h5", PARQUET=".parquet")


def convert_tpx3_file(
    tpx3_fpath: str | Path,
    extension: str = f_type.PARQUET,
    output_dir: str | Path | None = None,
    tw: float = DEFAULT_CLUSTER_TW,
    radius: int = DEFAULT_CLUSTER_RADIUS,
    energy_calib: np.ndarray | None = None,
    timewalk_correct: bool = False,
    trim_correct: bool = False,
    print_details: bool = False,
    overwrite: bool = True,
):
    """
    Convert a .tpx3 file into raw and centroided Pandas dataframes, which are stored in .h5 files.

    Parameters
    ----------
    tpx3_fpath : str | Path
        .tpx3 file path
    extension: str = ".parquet"
        type of file format (.h5 , .parquet) to export to. Can use internal f_type namespace to alias
    output_dir: str | Path | None = None
        Directory to save converted files to. Will save to same directory as file if None
    tw : float = DEFAULT_CLUSTER_TW_MICROSECONDS
        The time window, in Timepix timestamp units, to perform centroiding
    radius : int = DEFAULT_CLUSTER_RADIUS
        The radius, in pixels, to perform centroiding
    trim_correct : bool = False
        Whether to apply trim correction
    timewalk_correct : bool = False
        Whether to apply timewalk correction
    print_details : bool = False
        Boolean toggle about whether to print detailed data.
    overwrite : bool = True
        Boolean toggle about whether to overwrite pre-existing data.
    energy_calib: np.ndarray = None
        numpy array of dimension (514, 514, 4) and type float64 that contains the parameters to the E(ToT) function

    Raises
    ------
    FileNotFoundError
        If tpx3_fpath can't be found
    ValueError
        If the file doesn't have `.tpx3` suffix
    """

    if print_details:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    tpx3_fpath = Path(tpx3_fpath)

    if not tpx3_fpath.exists():
        raise FileNotFoundError(f"{tpx3_fpath} does not exist")
    if tpx3_fpath.suffix != ".tpx3":
        raise ValueError(f"{tpx3_fpath} is not a .tpx3 file")

    out_fpath = converted_path(tpx3_fpath, extension=extension, cent=False)
    cent_out_fpath = converted_path(tpx3_fpath, extension=extension, cent=True)

    if output_dir:
        output_dir = Path(output_dir)
        out_fpath = output_dir / out_fpath.name
        cent_out_fpath = output_dir / cent_out_fpath.name

    have_df = out_fpath.exists()  # Check if dfname exists
    have_dfc = cent_out_fpath.exists()  # Check if dfcname exists

    if have_df and have_dfc and not overwrite:
        print(f"-> {tpx3_fpath.name} already processed, skipping.")
        return False

    logger.info(f"-> Processing {tpx3_fpath.name}, size: {tpx3_fpath.stat().st_size / (1024 * 1024):.1f} MB")
    df = decode_tpx3_binary(raw_as_numpy(tpx3_fpath))
    num_events = df.shape[0]

    if num_events == 0:
        logger.info("No events found! Saving empty dataframes.")
        include_energy = isinstance(energy_calib, np.ndarray)
        save_df(empty_raw_df(include_energy=include_energy), out_fpath)
        save_df(empty_cent_df(include_energy=include_energy), cent_out_fpath)
        gc.collect()
        return True

    logger.info(f"Loading {tpx3_fpath.name} complete. {num_events} events found.")

    cdf = cluster_raw_df(
        df,
        tw,
        radius,
        energy_calib=energy_calib,
        timewalk_correct=timewalk_correct,
        trim_correct=trim_correct,
    )
    # maybe we should put this somewhere else...
    cdf.loc[cdf["xc"] >= 255.5, "xc"] += 2
    cdf.loc[cdf["yc"] >= 255.5, "yc"] += 2

    logger.info(f"Clustering and centroiding complete. Saving to {cent_out_fpath.name}...")

    save_df(cdf, cent_out_fpath)
    logger.info(f"Saving {cent_out_fpath.name} complete. Checking file existence...")

    if cent_out_fpath.exists():
        logger.info(f"Confirmed {cent_out_fpath.name} exists!")
        to_return = True
    else:
        logger.info(f"WARNING: {cent_out_fpath.name} doesn't exist but it should?!")
        to_return = False

    logger.info("Moving onto next file...")
    del df, cdf
    gc.collect()
    return to_return


def convert_tpx3_files(
    fpaths: list[str] | list[Path],
    extension: str = f_type.PARQUET,
    output_dir: str | Path | None = None,
    trim_correct: str | Path | None = None,
    print_details: bool = True,
    energy_calib: np.ndarray | str | Path | None = None,
    **kwargs,
):
    """
    Convert a list of .tpx3 files in a single process using convert_tpx3_file().

    Parameters
    ----------
    fpaths : Union[List[str], List[Path]]
        List of .tpx3 file paths to process.
    extension: str
        type of file format (.h5 , .parquet) to export to. Can use internal f_type namespace to alias
    trim_mask_fpath : str, optional
        Path to the trim correction mask. If None, no correction is applied.
    print_details : bool, optional
        Boolean toggle about whether to print detailed data. Default is True.
    **kwargs : dict
        Additional keyword arguments passed to `convert_tpx3_file()`.
    """
    # Load the mask once (only if provided)
    trim_mask = trim_corr_file(trim_correct)

    # Load energy estimation params
    if isinstance(energy_calib, (str, Path)):
        try:
            energy_calib = np.load(energy_calib)
        except Exception:
            print("Failed to load calibration: {e}")

    # Process files sequentially with tqdm progress bar
    for fpath in tqdm(fpaths, desc="Processing files"):
        try:
            convert_tpx3_file(
                fpath,
                extension=extension,
                output_dir=output_dir,
                trim_correct=trim_mask,
                print_details=print_details,
                energy_calib=energy_calib,
                **kwargs,
            )
        except Exception:  # noqa: PERF203
            logger.exception(f"Failed to process {fpath}")


def convert_tpx3_file_worker(fpath, **kwargs):
    """Worker function for convert_tpx3_files_parallel in order to catch potential errors"""
    try:
        convert_tpx3_file(fpath, **kwargs)
        return True
    except Exception:
        logger.exception(f"Failed to process {fpath}")
        return False


def convert_tpx3_files_parallel(
    fpaths: list[str] | list[Path],
    extension=f_type.PARQUET,
    output_dir: str | Path | None = None,
    num_workers: int | None = None,
    trim_correct: str | Path | None = None,
    energy_calib: np.ndarray | str | Path | None = None,
    **kwargs,
):
    """
    Convert a list of .tpx3 files in parallel using multiprocessing and convert_tpx3_file().

    Parameters
    ----------
    fpaths : Union[List[str], List[Path]]
        List of .tpx3 file paths to process.
    extension: str
        type of file format (.h5 , .parquet) to export to. Can use internal f_type namespace to alias
    num_workers : int, optional
        Number of worker processes to use. Defaults to (CPU count - 4) to leave room for other tasks.
    trim_mask_fpath : str, optional
        Path to the trim correction mask. If None, no correction is applied.
    energy_calib_fpath: np.ndarray = None
        fpath pointing to energy estimation parameters array saved as .npy file.
        if not specified then energy won't be estimated.
    **kwargs : dict
        Additional keyword arguments passed to `convert_tpx3_file()`.
    """
    if len(fpaths) > 0:
        if num_workers is None:
            max_workers = min(multiprocessing.cpu_count() - 4, len(fpaths))  # Leave 4 cores free
        else:
            max_workers = min(num_workers, len(fpaths))  # Don't use more workers than files

        max_workers = max(max_workers, 1)

        # Load the mask once
        trim_mask = trim_corr_file(trim_correct)

        # Load energy estimation params
        if isinstance(energy_calib, (str, Path)):
            try:
                energy_calib = np.load(energy_calib)
            except Exception as e:
                print(f"Failed to load calibration: {e}")

        # Pass the preloaded mask to all workers
        worker_func = partial(
            convert_tpx3_file_worker,
            extension=extension,
            output_dir=output_dir,
            trim_correct=trim_mask,
            energy_calib=energy_calib,
            **kwargs,
        )

        with multiprocessing.Pool(processes=max_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(worker_func, fpaths),
                    total=len(fpaths),
                    desc="Processing files",
                )
            )

        # Count successes
        num_true = sum(results)
    else:
        num_true = 0

    print(f"Successfully converted {num_true} out of {len(fpaths)}!")
