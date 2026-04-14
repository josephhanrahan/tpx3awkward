import gc
import logging
import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .cluster import cluster_decoded_df
from .config import Tpx3Config
from .corrections import estimate_energies
from .decoding import decode_tpx3_binary
from .files import converted_path, f_type, raw_as_numpy, save_df, trim_corr_file
from .schemas import empty_cent_df

logger = logging.getLogger(__name__)


def convert_tpx3_file(
    tpx3_fpath: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: Tpx3Config | None = None,
    **overrides,
):
    """
    Convert a .tpx3 file into raw and centroided Pandas dataframes, which are stored in .h5 files.

    Parameters
    ----------
    tpx3_fpath : str | Path
        .tpx3 file path
    output_dir : str | Path | None = None
        Directory to save converted files to. Will save to same directory as file if None
    config : Tpx3Config | None = None
        Defines the configurations for processing, considered the ground truth and overrides any kwargs

    Raises
    ------
    FileNotFoundError
        If tpx3_fpath can't be found
    ValueError
        If the file doesn't have `.tpx3` suffix
    """
    if config is not None and overrides:
        raise ValueError("Pass either `config` or keyword overrides, not both.")
    if config is None:
        config = Tpx3Config.from_defaults(**overrides)

    if config.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    tpx3_fpath = Path(tpx3_fpath)

    if not tpx3_fpath.exists():
        raise FileNotFoundError(f"{tpx3_fpath} does not exist")
    if tpx3_fpath.suffix != ".tpx3":
        raise ValueError(f"{tpx3_fpath} is not a .tpx3 file")

    out_fpath = converted_path(tpx3_fpath, extension=config.file_extension, cent=False)
    cent_out_fpath = converted_path(tpx3_fpath, extension=config.file_extension, cent=True)

    if output_dir:
        output_dir = Path(output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"{output_dir} exists but is not a directory.")
        output_dir.mkdir(
            parents=True, exist_ok=True
        )  # TODO should we be doing this? compare with what sophy and/or pymepix does
        out_fpath = output_dir / out_fpath.name
        cent_out_fpath = output_dir / cent_out_fpath.name

    have_df = out_fpath.exists()  # Check if dfname exists
    have_dfc = cent_out_fpath.exists()  # Check if dfcname exists

    if have_df and have_dfc and not config.overwrite:
        logger.info(f"-> {tpx3_fpath.name} already processed, skipping.")
        return False

    logger.info(f"-> Processing {tpx3_fpath.name}, size: {tpx3_fpath.stat().st_size / (1024 * 1024):.1f} MB")
    decoded_df = decode_tpx3_binary(raw_as_numpy(tpx3_fpath))
    num_events = decoded_df.shape[0]

    if num_events == 0:
        logger.info("No events found! Saving empty dataframes.")
        save_df(
            empty_cent_df(estimate_energy=config.estimate_energy, correct_timewalk=config.correct_timewalk), cent_out_fpath
        )
        gc.collect()
        return True

    logger.info(f"Loading {tpx3_fpath.name} complete. {num_events} events found.")

    if config.estimate_energy:
        decoded_df["e"] = estimate_energies(
            decoded_df["x"].to_numpy(),
            decoded_df["y"].to_numpy(),
            decoded_df["ToT"].to_numpy(),
            config.energy_estimation_parameters,
        )
    # if config.correct_timewalk:
    # decoded_df['t_corr'] = timewalk_corr(decoded_df['t'].to_numpy(),
    # decoded_df['ToT'].to_numpy(), b=config.timewalk_b, c=config.timewalk_c)

    if config.correct_trim:
        ...

    clustered_df = cluster_decoded_df(
        decoded_df,
        config.time_window,
        config.radius,
        energy_calib=config.energy_estimation_parameters,
        timewalk_correct=config.correct_timewalk,
        trim_correct=config.correct_trim,
    )
    # maybe we should put this somewhere else...
    clustered_df.loc[clustered_df["xc"] >= 255.5, "xc"] += 2
    clustered_df.loc[clustered_df["yc"] >= 255.5, "yc"] += 2

    logger.info(f"Clustering and centroiding complete. Saving to {cent_out_fpath.name}...")

    save_df(clustered_df, cent_out_fpath, config=config)
    logger.info(f"Saving {cent_out_fpath.name} complete. Checking file existence...")

    if cent_out_fpath.exists():
        logger.info(f"Confirmed {cent_out_fpath.name} exists!")
        to_return = True
    else:
        logger.info(f"WARNING: {cent_out_fpath.name} doesn't exist but it should?!")
        to_return = False

    del decoded_df, clustered_df
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
