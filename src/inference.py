"""
inference.py — Run RUL inference on the 11 truncated test bearings.

Loads the best fine-tuned PatchTST model and produces RUL predictions
for each test bearing, with optional rolling trajectory predictions.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import logging
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading import load_config
from src.dataset import prepare_test_windows, prepare_rolling_windows
from src.model import create_patchtst_model, load_model, LinearDegradationBaseline
from src.evaluate import (
    phm_challenge_score, print_results_table, save_metrics,
    generate_all_plots, plot_rul_trajectory,
)

logger = logging.getLogger(__name__)


def predict_rul_patchtst(
    model,
    feature_sequence: np.ndarray,
    window_size: int,
    denormalize_fn,
    device: torch.device,
    ensemble_n: int = 5,
) -> float:
    """
    Predict RUL for a single test bearing using the last N windows.

    Parameters
    ----------
    model : PatchTSTForRegression
    feature_sequence : np.ndarray
        Shape (T, num_channels).
    window_size : int
        Window size.
    denormalize_fn : callable
        Function to convert normalized RUL back to seconds.
    device : torch.device
    ensemble_n : int
        Number of last windows to average for final prediction.

    Returns
    -------
    float — predicted RUL in seconds.
    """
    model.eval()
    T = feature_sequence.shape[0]

    if T < window_size:
        pad_len = window_size - T
        feature_sequence = np.vstack([
            np.tile(feature_sequence[0:1], (pad_len, 1)),
            feature_sequence,
        ])
        T = len(feature_sequence)

    # Use last ensemble_n windows
    preds = []
    for offset in range(min(ensemble_n, T - window_size + 1)):
        start = T - window_size - offset
        if start < 0:
            break
        window = feature_sequence[start:start + window_size]
        window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(past_values=window_tensor)
            pred_normalized = outputs.regression_outputs.cpu().numpy().flatten()[0]

        pred_seconds = denormalize_fn(pred_normalized)
        preds.append(pred_seconds)

    # Average predictions
    avg_pred = float(np.mean(preds))
    return max(avg_pred, 0.0)


def predict_rul_trajectory(
    model,
    feature_sequence: np.ndarray,
    window_size: int,
    denormalize_fn,
    device: torch.device,
    stride: int = 1,
) -> np.ndarray:
    """
    Predict RUL at every timestep (rolling prediction mode).

    Returns array of predicted RUL values for trajectory plotting.
    """
    model.eval()
    rolling_windows = prepare_rolling_windows(feature_sequence, window_size, stride)

    preds = []
    batch_size = 64

    for i in range(0, len(rolling_windows), batch_size):
        batch = rolling_windows[i:i + batch_size].to(device)
        with torch.no_grad():
            outputs = model(past_values=batch)
            batch_preds = outputs.regression_outputs.cpu().numpy().flatten()

        for p in batch_preds:
            preds.append(max(denormalize_fn(p), 0.0))

    return np.array(preds)


def predict_rul_baseline_linear(
    hi_sequence: np.ndarray,
    window_size: int,
    recording_interval_s: float,
) -> float:
    """Predict RUL using linear degradation baseline."""
    baseline = LinearDegradationBaseline(window_size=window_size)
    rul_timesteps = baseline.predict_rul(hi_sequence)
    return rul_timesteps * recording_interval_s


def predict_rul_classical(
    model,
    feature_sequence: np.ndarray,
    window_size: int,
) -> float:
    """Predict RUL using classical ML baseline."""
    T = feature_sequence.shape[0]

    if T < window_size:
        pad_len = window_size - T
        feature_sequence = np.vstack([
            np.tile(feature_sequence[0:1], (pad_len, 1)),
            feature_sequence,
        ])

    # Use last window
    last_window = feature_sequence[-window_size:].flatten().reshape(1, -1)
    pred = model.predict(last_window)
    return max(float(pred[0]), 0.0)


def run_inference(config: dict, data: dict, training_results: dict) -> dict:
    """
    Run inference on all 11 test bearings with all trained models.

    Parameters
    ----------
    config : dict
        Configuration.
    data : dict
        Processed data from training pipeline.
    training_results : dict
        Results from training pipeline.

    Returns
    -------
    dict mapping model_name -> {bearing_name -> predicted_rul}
    """
    logger.info("\n" + "=" * 60)
    logger.info("Phase 6: Inference on Test Bearings")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actual_ruls = config["dataset"]["actual_rul"]
    test_bearings = config["dataset"]["test_bearings"]
    window_size = config["training"]["window_size"]
    interval = config["dataset"]["recording_interval_s"]
    ensemble_n = config["inference"]["ensemble_last_n"]

    all_feature_names = data["feature_names"]
    selected = data["selected_features"]
    selected_indices = [all_feature_names.index(f) for f in selected if f in all_feature_names]

    all_predictions = {}

    # 1. Linear Degradation Baseline
    logger.info("\n  Running Linear Degradation Baseline...")
    linear_preds = {}
    for bearing_name in test_bearings:
        hi = data["test_hi"][bearing_name]
        pred = predict_rul_baseline_linear(hi, window_size, interval)
        linear_preds[bearing_name] = pred
    all_predictions["Linear Degradation"] = linear_preds

    # 2. Classical ML Baselines
    if "classical" in training_results:
        for model_type in ["svr", "rf"]:
            if model_type in training_results["classical"]:
                logger.info(f"\n  Running {model_type.upper()} Baseline...")
                model_obj = training_results["classical"][model_type]["model"]
                preds = {}
                for bearing_name in test_bearings:
                    features = data["test_features"][bearing_name][:, selected_indices]
                    pred = predict_rul_classical(model_obj, features, window_size)
                    preds[bearing_name] = pred
                name = f"{'SVR' if model_type == 'svr' else 'Random Forest'}"
                all_predictions[name] = preds

    # 3. PatchTST Models
    for result_name, result in training_results.items():
        if result_name == "classical":
            continue
        if "strategy" not in result:
            continue

        strategy = result["strategy"]
        input_mode = result["input_mode"]
        model_name = result["model_name"]
        num_channels = result["num_channels"]
        max_rul = result["max_rul"]
        target_transform = config["training"]["target_transform"]

        # Create denormalize function
        log_max_rul = np.log1p(max_rul)
        if target_transform == "log":
            denormalize_fn = lambda x: np.expm1(x * log_max_rul)
        else:
            denormalize_fn = lambda x: x * max_rul

        logger.info(f"\n  Running {model_name}...")

        # Load best model
        try:
            run_config = config.copy()
            run_config["model"] = config["model"].copy()
            run_config["model"]["num_input_channels"] = num_channels

            model = load_model(run_config, config["paths"]["models"], model_name, strategy)
            model = model.to(device)
            model.eval()
        except Exception as e:
            logger.error(f"  Failed to load model {model_name}: {e}")
            continue

        preds = {}
        for bearing_name in test_bearings:
            if input_mode == "hi_only":
                features = data["test_hi"][bearing_name].reshape(-1, 1)
            else:
                features = data["test_features"][bearing_name][:, selected_indices]

            pred = predict_rul_patchtst(
                model, features, window_size, denormalize_fn, device, ensemble_n
            )
            preds[bearing_name] = pred

        # Determine display name
        strategy_name = {
            "frozen": "PatchTST Frozen",
            "lora": "PatchTST LoRA",
            "full": "PatchTST Full FT",
        }.get(strategy, strategy)
        input_name = {"hi_only": "HI", "multivariate": "Multi"}.get(input_mode, input_mode)
        display_name = f"{strategy_name} ({input_name})"
        all_predictions[display_name] = preds

        # Generate RUL trajectories for best models
        if config["inference"]["rolling_prediction"] and strategy == "full":
            figures_dir = config["paths"]["figures"]
            os.makedirs(figures_dir, exist_ok=True)

            for tb in test_bearings[:3]:  # First 3 test bearings
                if input_mode == "hi_only":
                    feat = data["test_hi"][tb].reshape(-1, 1)
                else:
                    feat = data["test_features"][tb][:, selected_indices]

                traj = predict_rul_trajectory(
                    model, feat, window_size, denormalize_fn, device, stride=5
                )

                plot_rul_trajectory(
                    tb, traj, actual_ruls[tb],
                    len(traj), interval, figures_dir
                )

    return all_predictions


def evaluate_all_predictions(config: dict, all_predictions: dict) -> dict:
    """
    Evaluate all model predictions using PHM scoring.

    Returns all_results dict.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Phase 7: Evaluation & Scoring")
    logger.info("=" * 60)

    actual_ruls = config["dataset"]["actual_rul"]
    all_results = {}

    for model_name, predictions in all_predictions.items():
        results = phm_challenge_score(actual_ruls, predictions)
        all_results[model_name] = results
        print_results_table(results, model_name)

    # Generate comparison plots
    figures_dir = config["paths"]["figures"]
    generate_all_plots(all_results, actual_ruls, figures_dir)

    # Save all metrics
    metrics_path = os.path.join(config["paths"]["reports"], "metrics.json")
    serializable_results = {}
    for name, r in all_results.items():
        serializable_results[name] = {
            "overall_score": r["overall_score"],
            "regression_metrics": r["regression_metrics"],
            "per_bearing": r["per_bearing"],
        }
    save_metrics(serializable_results, metrics_path)

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()
    logger.info("Run this via run_pipeline.py for the full pipeline.")
