"""
data_loading.py — Robust data loader for IEEE PHM 2012 bearing vibration data.

Reads acc_*.csv files per bearing folder, parses the 6-column format
(hour, min, sec, microsec, horiz_accel, vert_accel), reconstructs
monotonically increasing timestamps, and validates row counts.
"""

import os
import re
import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the central YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_acc_files(bearing_dir: str) -> List[str]:
    """
    Get sorted list of acc_*.csv files in a bearing directory.
    Returns absolute paths sorted by file number.
    """
    pattern = re.compile(r"acc_(\d+)\.csv$")
    files = []
    for fname in os.listdir(bearing_dir):
        match = pattern.match(fname)
        if match:
            files.append((int(match.group(1)), os.path.join(bearing_dir, fname)))
    files.sort(key=lambda x: x[0])
    return [f[1] for f in files]


def read_single_acc_file(filepath: str, expected_rows: int = 2560) -> Optional[pd.DataFrame]:
    """
    Read a single acc_*.csv file.

    Expected 6-column format (no header):
        hour, minute, second, microsecond, horiz_accel, vert_accel

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    expected_rows : int
        Expected number of rows (default 2560).

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns [hour, minute, second, microsecond, horiz_accel, vert_accel],
        or None if the file is corrupted.
    """
    columns = ["hour", "minute", "second", "microsecond", "horiz_accel", "vert_accel"]
    try:
        df = pd.read_csv(
            filepath,
            header=None,
            names=columns,
            dtype={
                "hour": np.int32,
                "minute": np.int32,
                "second": np.int32,
                "microsecond": np.int32,
                "horiz_accel": np.float64,
                "vert_accel": np.float64,
            },
        )

        if len(df) != expected_rows:
            logger.warning(
                f"File {filepath}: expected {expected_rows} rows, got {len(df)}. "
                f"Flagged as potentially corrupted."
            )

        return df

    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return None


def load_bearing_data(
    bearing_dir: str,
    expected_rows: int = 2560,
    recording_interval_s: float = 10.0,
) -> Dict:
    """
    Load all accelerometer data for a single bearing.

    Parameters
    ----------
    bearing_dir : str
        Path to the bearing directory (e.g., 'Learning_set/Bearing1_1').
    expected_rows : int
        Expected samples per file.
    recording_interval_s : float
        Time interval between recordings in seconds.

    Returns
    -------
    dict with keys:
        - 'bearing_name': str
        - 'num_files': int
        - 'timestamps_s': np.ndarray of shape (N,) — time in seconds for each file
        - 'horiz_accel': list of np.ndarray, each shape (samples_per_file,)
        - 'vert_accel': list of np.ndarray, each shape (samples_per_file,)
        - 'total_duration_s': float
        - 'corrupted_files': list of str — paths of files with unexpected row counts
    """
    bearing_name = os.path.basename(bearing_dir)
    acc_files = get_acc_files(bearing_dir)

    if not acc_files:
        raise FileNotFoundError(f"No acc_*.csv files found in {bearing_dir}")

    horiz_list = []
    vert_list = []
    timestamps = []
    corrupted = []

    for idx, fpath in enumerate(acc_files):
        df = read_single_acc_file(fpath, expected_rows)
        if df is None:
            corrupted.append(fpath)
            continue

        if len(df) != expected_rows:
            corrupted.append(fpath)

        horiz_list.append(df["horiz_accel"].values)
        vert_list.append(df["vert_accel"].values)
        timestamps.append(idx * recording_interval_s)

    timestamps = np.array(timestamps, dtype=np.float64)
    total_duration = timestamps[-1] + recording_interval_s if len(timestamps) > 0 else 0.0

    if corrupted:
        logger.warning(
            f"Bearing {bearing_name}: {len(corrupted)} corrupted/non-standard files "
            f"out of {len(acc_files)} total."
        )

    return {
        "bearing_name": bearing_name,
        "num_files": len(horiz_list),
        "timestamps_s": timestamps,
        "horiz_accel": horiz_list,
        "vert_accel": vert_list,
        "total_duration_s": total_duration,
        "corrupted_files": corrupted,
    }


def load_all_bearings(
    config: dict,
    dataset_type: str = "learning",
) -> Dict[str, Dict]:
    """
    Load data for all bearings of a given dataset type.

    Parameters
    ----------
    config : dict
        Configuration dictionary (from config.yaml).
    dataset_type : str
        'learning' or 'test'.

    Returns
    -------
    dict mapping bearing_name -> bearing data dict
    """
    if dataset_type == "learning":
        base_dir = config["paths"]["learning_set"]
        bearings = config["dataset"]["learning_bearings"]
    elif dataset_type == "test":
        base_dir = config["paths"]["test_set"]
        bearings = config["dataset"]["test_bearings"]
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    expected_rows = config["dataset"]["samples_per_file"]
    interval = config["dataset"]["recording_interval_s"]

    all_data = {}
    for bearing_name in bearings:
        bearing_dir = os.path.join(base_dir, bearing_name)
        logger.info(f"Loading {bearing_name} from {bearing_dir}...")

        data = load_bearing_data(
            bearing_dir,
            expected_rows=expected_rows,
            recording_interval_s=interval,
        )
        all_data[bearing_name] = data

        logger.info(
            f"  {bearing_name}: {data['num_files']} files, "
            f"{data['total_duration_s']:.0f}s ({data['total_duration_s']/3600:.1f}h), "
            f"{len(data['corrupted_files'])} corrupted"
        )

    return all_data


def compute_rul_labels(
    bearing_data: Dict,
    recording_interval_s: float = 10.0,
) -> np.ndarray:
    """
    Compute ground-truth RUL labels for each timestep of a learning-set bearing.

    RUL(t) = total_life_of_bearing - t (in seconds)

    Parameters
    ----------
    bearing_data : dict
        Output of load_bearing_data().
    recording_interval_s : float
        Time interval between recordings.

    Returns
    -------
    np.ndarray of shape (num_files,) — RUL in seconds for each recording.
    """
    n = bearing_data["num_files"]
    total_life = n * recording_interval_s
    rul = np.array([total_life - (i * recording_interval_s) for i in range(n)], dtype=np.float64)
    return rul


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config()

    print("=" * 60)
    print("Loading Learning Set")
    print("=" * 60)
    learning_data = load_all_bearings(config, "learning")

    print("\n" + "=" * 60)
    print("Loading Test Set")
    print("=" * 60)
    test_data = load_all_bearings(config, "test")

    # Show RUL labels for learning set
    print("\n" + "=" * 60)
    print("RUL Labels (Learning Set)")
    print("=" * 60)
    for name, data in learning_data.items():
        rul = compute_rul_labels(data, config["dataset"]["recording_interval_s"])
        print(f"  {name}: total life = {data['total_duration_s']:.0f}s, "
              f"RUL range = [{rul.min():.0f}, {rul.max():.0f}]s")
