"""
health_index.py — Construct Health Index from engineered features.

Selects the most monotonic/trendable features and combines them into
a univariate Health Index (HI) normalized to [0, 1] per bearing.
"""

import numpy as np
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)


def monotonicity_score(feature_sequence: np.ndarray) -> float:
    """
    Compute monotonicity of a feature sequence.

    Monotonicity measures how consistently a feature increases or decreases
    over time. Calculated as the absolute Spearman correlation with time index.

    Parameters
    ----------
    feature_sequence : np.ndarray
        1D array of feature values over time.

    Returns
    -------
    float in [0, 1]
    """
    if len(feature_sequence) < 3:
        return 0.0

    time_idx = np.arange(len(feature_sequence))
    corr, _ = spearmanr(time_idx, feature_sequence)

    if np.isnan(corr):
        return 0.0

    return abs(corr)


def trendability_score(feature_sequence: np.ndarray) -> float:
    """
    Compute trendability of a feature sequence.

    Trendability measures the strength of the linear trend. Calculated as
    the absolute Pearson correlation coefficient with time index.

    Parameters
    ----------
    feature_sequence : np.ndarray
        1D array of feature values over time.

    Returns
    -------
    float in [0, 1]
    """
    if len(feature_sequence) < 3:
        return 0.0

    time_idx = np.arange(len(feature_sequence), dtype=np.float64)
    corr_matrix = np.corrcoef(time_idx, feature_sequence)
    corr = corr_matrix[0, 1]

    if np.isnan(corr):
        return 0.0

    return abs(corr)


def score_features(
    feature_matrix: np.ndarray,
    feature_names: list,
    weight_monotonicity: float = 0.5,
    weight_trendability: float = 0.5,
) -> list:
    """
    Score each feature by combined monotonicity + trendability.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Shape (num_timesteps, num_features).
    feature_names : list
        Feature names corresponding to columns.
    weight_monotonicity : float
        Weight for monotonicity in combined score.
    weight_trendability : float
        Weight for trendability in combined score.

    Returns
    -------
    list of (feature_name, combined_score, mono_score, trend_score)
        Sorted descending by combined score.
    """
    scores = []
    for i, name in enumerate(feature_names):
        seq = feature_matrix[:, i]

        # Handle NaN/Inf
        if not np.all(np.isfinite(seq)):
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        mono = monotonicity_score(seq)
        trend = trendability_score(seq)
        combined = weight_monotonicity * mono + weight_trendability * trend
        scores.append((name, combined, mono, trend))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def select_top_features(
    feature_matrices: dict,
    feature_names: list,
    top_k: int = 8,
) -> list:
    """
    Select top-K features across all learning-set bearings.

    Averages the feature scores across all bearings and selects the
    top K features by average combined score.

    Parameters
    ----------
    feature_matrices : dict
        Mapping bearing_name -> feature_matrix (num_timesteps, num_features).
    feature_names : list
        Feature names.
    top_k : int
        Number of top features to select.

    Returns
    -------
    list of str — selected feature names
    """
    # Accumulate scores across bearings
    avg_scores = {name: [] for name in feature_names}

    for bearing_name, matrix in feature_matrices.items():
        scores = score_features(matrix, feature_names)
        for name, combined, _, _ in scores:
            avg_scores[name].append(combined)

    # Average across bearings
    mean_scores = [
        (name, np.mean(scores_list))
        for name, scores_list in avg_scores.items()
    ]
    mean_scores.sort(key=lambda x: x[1], reverse=True)

    selected = [name for name, _ in mean_scores[:top_k]]

    logger.info(f"Top-{top_k} features selected:")
    for name, score in mean_scores[:top_k]:
        logger.info(f"  {name}: avg_score = {score:.4f}")

    return selected


def construct_health_index(
    feature_matrix: np.ndarray,
    feature_names: list,
    selected_features: list,
    smoothing_window: int = 5,
) -> np.ndarray:
    """
    Construct a univariate Health Index from selected features.

    The HI is a weighted average of z-scored selected features,
    normalized to [0, 1] where 0 = healthy and 1 = near-failure.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Shape (num_timesteps, num_features).
    feature_names : list
        All feature names.
    selected_features : list
        Names of selected top-K features.
    smoothing_window : int
        Moving average window for smoothing.

    Returns
    -------
    np.ndarray of shape (num_timesteps,) — HI values in [0, 1]
    """
    # Get indices of selected features
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    indices = [name_to_idx[name] for name in selected_features if name in name_to_idx]

    if not indices:
        logger.warning("No valid selected features found. Returning zeros.")
        return np.zeros(feature_matrix.shape[0])

    # Extract and z-score normalize each selected feature
    selected_data = feature_matrix[:, indices].copy()

    for j in range(selected_data.shape[1]):
        col = selected_data[:, j]
        col_std = np.std(col)
        if col_std > 1e-12:
            selected_data[:, j] = (col - np.mean(col)) / col_std
        else:
            selected_data[:, j] = 0.0

    # Ensure features are increasing (flip sign if they decrease over time)
    for j in range(selected_data.shape[1]):
        time_idx = np.arange(selected_data.shape[0], dtype=np.float64)
        corr = np.corrcoef(time_idx, selected_data[:, j])[0, 1]
        if corr < 0:
            selected_data[:, j] *= -1

    # Average across selected features
    hi_raw = np.mean(selected_data, axis=1)

    # Smooth with moving average
    if smoothing_window > 1 and len(hi_raw) >= smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        hi_smooth = np.convolve(hi_raw, kernel, mode="same")
    else:
        hi_smooth = hi_raw

    # Normalize to [0, 1]
    hi_min = np.min(hi_smooth)
    hi_max = np.max(hi_smooth)
    if hi_max - hi_min > 1e-12:
        hi_normalized = (hi_smooth - hi_min) / (hi_max - hi_min)
    else:
        hi_normalized = np.zeros_like(hi_smooth)

    return hi_normalized


def compute_all_health_indices(
    feature_matrices: dict,
    feature_names: list,
    selected_features: list,
    smoothing_window: int = 5,
) -> dict:
    """
    Compute HI for all bearings.

    Parameters
    ----------
    feature_matrices : dict
        Mapping bearing_name -> feature_matrix.
    feature_names : list
        All feature names.
    selected_features : list
        Selected top-K feature names.
    smoothing_window : int
        Smoothing window.

    Returns
    -------
    dict mapping bearing_name -> hi_array (shape: num_timesteps,)
    """
    hi_dict = {}
    for bearing_name, matrix in feature_matrices.items():
        hi = construct_health_index(matrix, feature_names, selected_features, smoothing_window)
        hi_dict[bearing_name] = hi
        logger.info(
            f"  {bearing_name}: HI range [{hi.min():.4f}, {hi.max():.4f}], "
            f"length = {len(hi)}"
        )
    return hi_dict
