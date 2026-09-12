"""
feature_extraction.py — Extract time-domain, frequency-domain, and envelope
features from raw vibration signals for bearing RUL prediction.

Each 10-second recording (2,560 samples at 25,600 Hz) produces a feature vector.
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Time-domain features
# ============================================================================

def time_domain_features(x: np.ndarray) -> dict:
    """
    Extract time-domain statistical features from a vibration signal.

    Parameters
    ----------
    x : np.ndarray
        1D vibration signal (e.g., 2560 samples).

    Returns
    -------
    dict of feature_name -> float
    """
    n = len(x)
    abs_x = np.abs(x)
    mean_val = np.mean(x)
    std_val = np.std(x, ddof=1) if n > 1 else 0.0
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(abs_x)
    peak_to_peak = np.max(x) - np.min(x)

    # Avoid division by zero
    eps = 1e-12

    # Kurtosis and skewness
    kurt = stats.kurtosis(x, fisher=True)  # excess kurtosis
    skew = stats.skew(x)

    # Crest factor = peak / RMS
    crest_factor = peak / (rms + eps)

    # Shape factor = RMS / mean(|x|)
    mean_abs = np.mean(abs_x)
    shape_factor = rms / (mean_abs + eps)

    # Impulse factor = peak / mean(|x|)
    impulse_factor = peak / (mean_abs + eps)

    # Clearance factor = peak / (mean(sqrt(|x|)))^2
    mean_sqrt_abs = np.mean(np.sqrt(abs_x))
    clearance_factor = peak / (mean_sqrt_abs ** 2 + eps)

    return {
        "rms": rms,
        "kurtosis": kurt,
        "skewness": skew,
        "crest_factor": crest_factor,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "std": std_val,
        "shape_factor": shape_factor,
        "impulse_factor": impulse_factor,
        "clearance_factor": clearance_factor,
    }


# ============================================================================
# Frequency-domain features
# ============================================================================

def frequency_domain_features(
    x: np.ndarray,
    sampling_rate: float = 25600.0,
    fft_bands: dict = None,
) -> dict:
    """
    Extract frequency-domain features using FFT.

    Parameters
    ----------
    x : np.ndarray
        1D vibration signal.
    sampling_rate : float
        Sampling rate in Hz.
    fft_bands : dict
        Dictionary of band_name -> [low_hz, high_hz] for band power computation.

    Returns
    -------
    dict of feature_name -> float
    """
    n = len(x)
    # Compute one-sided FFT
    yf = np.abs(fft(x))[:n // 2]
    xf = fftfreq(n, 1.0 / sampling_rate)[:n // 2]

    # Power spectrum (magnitude squared)
    power = yf ** 2
    total_power = np.sum(power)
    eps = 1e-12

    # Normalize to probability distribution for spectral moments
    if total_power > eps:
        p = power / total_power
    else:
        p = np.ones_like(power) / len(power)

    # Spectral centroid (mean frequency)
    spectral_centroid = np.sum(xf * p)

    # Spectral spread (std of frequency)
    spectral_spread = np.sqrt(np.sum(((xf - spectral_centroid) ** 2) * p))

    # Spectral kurtosis
    if spectral_spread > eps:
        spectral_kurtosis = (
            np.sum(((xf - spectral_centroid) ** 4) * p) / (spectral_spread ** 4)
        ) - 3.0
    else:
        spectral_kurtosis = 0.0

    features = {
        "spectral_centroid": spectral_centroid,
        "spectral_spread": spectral_spread,
        "spectral_kurtosis": spectral_kurtosis,
    }

    # Band power features
    if fft_bands:
        for band_name, (low, high) in fft_bands.items():
            mask = (xf >= low) & (xf <= high)
            band_power = np.sum(power[mask])
            # Relative band power
            features[f"band_power_{band_name}"] = band_power / (total_power + eps)

    return features


# ============================================================================
# Envelope analysis features
# ============================================================================

def envelope_features(
    x: np.ndarray,
    sampling_rate: float = 25600.0,
    bandpass_low: float = 500.0,
    bandpass_high: float = 5000.0,
) -> dict:
    """
    Extract envelope analysis features using Hilbert transform.

    Parameters
    ----------
    x : np.ndarray
        1D vibration signal.
    sampling_rate : float
        Sampling rate in Hz.
    bandpass_low : float
        Low cutoff for bandpass filter (Hz).
    bandpass_high : float
        High cutoff for bandpass filter (Hz).

    Returns
    -------
    dict of feature_name -> float
    """
    nyquist = sampling_rate / 2.0

    # Design bandpass filter
    low = bandpass_low / nyquist
    high = bandpass_high / nyquist

    # Clamp to valid range
    low = max(low, 0.001)
    high = min(high, 0.999)

    if low >= high:
        # Fallback: skip filtering
        filtered = x
    else:
        try:
            sos = signal.butter(4, [low, high], btype="band", output="sos")
            filtered = signal.sosfilt(sos, x)
        except Exception:
            filtered = x

    # Hilbert transform to get analytic signal
    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)

    # Envelope features
    env_rms = np.sqrt(np.mean(envelope ** 2))
    env_peak = np.max(envelope)
    env_kurtosis = stats.kurtosis(envelope, fisher=True)

    return {
        "envelope_rms": env_rms,
        "envelope_peak": env_peak,
        "envelope_kurtosis": env_kurtosis,
    }


# ============================================================================
# Combined feature extraction
# ============================================================================

# Canonical feature order (used throughout the pipeline)
FEATURE_NAMES = None  # Set dynamically on first call


def extract_features_single(
    horiz: np.ndarray,
    vert: np.ndarray,
    sampling_rate: float = 25600.0,
    fft_bands: dict = None,
    bandpass_low: float = 500.0,
    bandpass_high: float = 5000.0,
) -> dict:
    """
    Extract full feature vector from one 10-second recording (both channels).

    Parameters
    ----------
    horiz : np.ndarray
        Horizontal accelerometer signal (2560 samples).
    vert : np.ndarray
        Vertical accelerometer signal (2560 samples).
    sampling_rate : float
        Sampling rate in Hz.
    fft_bands : dict
        FFT band definitions for band-power features.
    bandpass_low, bandpass_high : float
        Envelope bandpass filter cutoffs.

    Returns
    -------
    dict of feature_name -> float
    """
    features = {}

    for channel_name, signal_data in [("h", horiz), ("v", vert)]:
        # Time domain
        td = time_domain_features(signal_data)
        for k, v in td.items():
            features[f"{channel_name}_{k}"] = v

        # Frequency domain
        fd = frequency_domain_features(signal_data, sampling_rate, fft_bands)
        for k, v in fd.items():
            features[f"{channel_name}_{k}"] = v

        # Envelope
        env = envelope_features(signal_data, sampling_rate, bandpass_low, bandpass_high)
        for k, v in env.items():
            features[f"{channel_name}_{k}"] = v

    return features


def extract_features_bearing(
    bearing_data: dict,
    config: dict,
) -> np.ndarray:
    """
    Extract features for all recordings of a single bearing.

    Parameters
    ----------
    bearing_data : dict
        Output of load_bearing_data().
    config : dict
        Configuration dictionary.

    Returns
    -------
    np.ndarray of shape (num_files, num_features)
    feature_names : list of str
    """
    global FEATURE_NAMES

    sampling_rate = config["dataset"]["sampling_rate_hz"]
    fft_bands = config["features"]["fft_bands"]
    bandpass_low = config["features"]["envelope_bandpass_low"]
    bandpass_high = config["features"]["envelope_bandpass_high"]

    feature_dicts = []
    n = bearing_data["num_files"]

    for i in range(n):
        horiz = bearing_data["horiz_accel"][i]
        vert = bearing_data["vert_accel"][i]

        feat = extract_features_single(
            horiz, vert, sampling_rate, fft_bands, bandpass_low, bandpass_high
        )
        feature_dicts.append(feat)

    # Convert to array
    if FEATURE_NAMES is None:
        FEATURE_NAMES = sorted(feature_dicts[0].keys())

    feature_matrix = np.array(
        [[fd[name] for name in FEATURE_NAMES] for fd in feature_dicts],
        dtype=np.float64,
    )

    return feature_matrix, FEATURE_NAMES


def get_feature_names() -> list:
    """Return the canonical feature name list (available after first extraction)."""
    return FEATURE_NAMES if FEATURE_NAMES is not None else []


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_loading import load_config, load_bearing_data

    logging.basicConfig(level=logging.INFO)
    config = load_config()

    # Test on first bearing
    bearing_dir = os.path.join(config["paths"]["learning_set"], "Bearing1_1")
    data = load_bearing_data(bearing_dir)

    print(f"Extracting features for {data['bearing_name']}...")
    features, names = extract_features_bearing(data, config)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature names ({len(names)}): {names[:5]}...")
    print(f"Sample feature vector: {features[0, :5]}")
