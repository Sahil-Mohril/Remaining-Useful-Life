"""
evaluate.py — Metrics and PHM 2012 challenge scoring for bearing RUL prediction.

Implements:
- Standard regression metrics (RMSE, MAE, MAPE, R²)
- Official IEEE PHM 2012 asymmetric scoring function
- Model comparison table generation
"""

import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Standard Regression Metrics
# ============================================================================

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    mask = actual != 0
    if not np.any(mask):
        return float("inf")
    return float(100.0 * np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """R² (coefficient of determination)."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute all standard regression metrics."""
    return {
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "mape": mape(actual, predicted),
        "r2": r2_score(actual, predicted),
    }


# ============================================================================
# IEEE PHM 2012 Challenge Scoring Function
# ============================================================================

def phm_percent_error(actual_rul: float, predicted_rul: float) -> float:
    """
    Compute percentage error for a single test bearing.

    %Er_i = 100 * (ActualRUL_i - PredRUL_i) / ActualRUL_i

    Positive %Er means early prediction (safe).
    Negative %Er means late prediction (dangerous).
    """
    if actual_rul == 0:
        return 0.0
    return 100.0 * (actual_rul - predicted_rul) / actual_rul


def phm_score_single(percent_error: float) -> float:
    """
    Compute A_i score for a single bearing using the asymmetric PHM scoring function.

    A_i = exp(-ln(0.5) * (Er_i / 5))   if Er_i <= 0  (late prediction, penalized more)
    A_i = exp(+ln(0.5) * (Er_i / 20))  if Er_i > 0   (early prediction, less penalty)

    Perfect prediction (Er=0) gives A_i = 1.0.
    Late predictions are penalized more heavily than early ones.
    """
    ln05 = np.log(0.5)

    if percent_error <= 0:
        # Late prediction (dangerous) — steeper penalty
        return float(np.exp(-ln05 * (percent_error / 5.0)))
    else:
        # Early prediction (safe) — gentler penalty
        return float(np.exp(ln05 * (percent_error / 20.0)))


def phm_challenge_score(
    actual_ruls: dict,
    predicted_ruls: dict,
) -> dict:
    """
    Compute the official IEEE PHM 2012 challenge score across all test bearings.

    Parameters
    ----------
    actual_ruls : dict
        Mapping bearing_name -> actual RUL in seconds.
    predicted_ruls : dict
        Mapping bearing_name -> predicted RUL in seconds.

    Returns
    -------
    dict with:
        - 'per_bearing': list of {bearing, actual, predicted, percent_error, A_i}
        - 'overall_score': float (mean of A_i across all bearings)
        - 'regression_metrics': dict (RMSE, MAE, MAPE, R²)
    """
    per_bearing = []
    a_scores = []

    for bearing_name in sorted(actual_ruls.keys()):
        actual = actual_ruls[bearing_name]
        predicted = predicted_ruls.get(bearing_name, 0.0)

        pct_err = phm_percent_error(actual, predicted)
        a_i = phm_score_single(pct_err)

        per_bearing.append({
            "bearing": bearing_name,
            "actual_rul": actual,
            "predicted_rul": round(predicted, 1),
            "percent_error": round(pct_err, 2),
            "A_i": round(a_i, 4),
        })
        a_scores.append(a_i)

    overall_score = float(np.mean(a_scores))

    # Also compute regression metrics
    actuals = np.array([actual_ruls[b] for b in sorted(actual_ruls.keys())])
    preds = np.array([predicted_ruls.get(b, 0.0) for b in sorted(actual_ruls.keys())])
    reg_metrics = compute_regression_metrics(actuals, preds)

    return {
        "per_bearing": per_bearing,
        "overall_score": round(overall_score, 4),
        "regression_metrics": reg_metrics,
    }


def save_metrics(metrics: dict, save_path: str):
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {save_path}")


def print_results_table(results: dict, model_name: str = "Model"):
    """Print a formatted results table."""
    print(f"\n{'=' * 80}")
    print(f"Results: {model_name}")
    print(f"{'=' * 80}")

    print(f"\n{'Bearing':<15} {'Actual':>10} {'Predicted':>10} {'%Error':>10} {'A_i':>10}")
    print("-" * 55)

    for entry in results["per_bearing"]:
        print(
            f"{entry['bearing']:<15} "
            f"{entry['actual_rul']:>10.0f} "
            f"{entry['predicted_rul']:>10.1f} "
            f"{entry['percent_error']:>10.2f} "
            f"{entry['A_i']:>10.4f}"
        )

    print("-" * 55)
    print(f"{'Overall Score:':<37} {results['overall_score']:>10.4f}")
    print(f"{'RMSE (s):':<37} {results['regression_metrics']['rmse']:>10.1f}")
    print(f"{'MAE (s):':<37} {results['regression_metrics']['mae']:>10.1f}")
    print(f"{'MAPE (%):':<37} {results['regression_metrics']['mape']:>10.1f}")
    print(f"{'R²:':<37} {results['regression_metrics']['r2']:>10.4f}")


# ============================================================================
# Plotting
# ============================================================================

def generate_all_plots(
    all_results: dict,
    actual_ruls: dict,
    save_dir: str,
):
    """
    Generate all required plots and save to figures directory.

    Parameters
    ----------
    all_results : dict
        Mapping model_name -> results dict (from phm_challenge_score).
    actual_ruls : dict
        Official actual RULs.
    save_dir : str
        Directory to save figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Predicted vs Actual RUL scatter plot
    _plot_pred_vs_actual(all_results, save_dir)

    # 2. PHM scoring function shape with test points
    _plot_phm_scoring_function(all_results, save_dir)

    # 3. Bar chart comparing models
    _plot_model_comparison(all_results, save_dir)

    logger.info(f"All plots saved to {save_dir}")


def _plot_pred_vs_actual(all_results: dict, save_dir: str):
    """Predicted vs Actual RUL scatter plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (model_name, results), color in zip(all_results.items(), colors):
        actuals = [e["actual_rul"] for e in results["per_bearing"]]
        preds = [e["predicted_rul"] for e in results["per_bearing"]]
        ax.scatter(actuals, preds, label=model_name, s=50, alpha=0.8, color=color)

    # Reference line y = x
    max_val = max(max(e["actual_rul"] for e in r["per_bearing"]) for r in all_results.values())
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], "k--", alpha=0.5, label="Perfect")

    ax.set_xlabel("Actual RUL (seconds)", fontsize=12)
    ax.set_ylabel("Predicted RUL (seconds)", fontsize=12)
    ax.set_title("Predicted vs Actual RUL — Test Set", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pred_vs_actual.png"), dpi=150)
    plt.close()


def _plot_phm_scoring_function(all_results: dict, save_dir: str):
    """PHM scoring function shape with test bearing points overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the scoring function curve
    er_range = np.linspace(-50, 100, 500)
    scores = [phm_score_single(er) for er in er_range]
    ax.plot(er_range, scores, "k-", linewidth=2, label="PHM Score Function")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

    # Overlay test bearing points for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    for (model_name, results), color in zip(all_results.items(), colors):
        ers = [e["percent_error"] for e in results["per_bearing"]]
        ais = [e["A_i"] for e in results["per_bearing"]]
        ax.scatter(ers, ais, label=f"{model_name} (Score={results['overall_score']:.3f})",
                   s=50, alpha=0.8, color=color, zorder=5)

    ax.set_xlabel("%Error (Er_i)", fontsize=12)
    ax.set_ylabel("A_i Score", fontsize=12)
    ax.set_title("IEEE PHM 2012 Asymmetric Scoring Function", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-50, 100)
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "phm_scoring_function.png"), dpi=150)
    plt.close()


def _plot_model_comparison(all_results: dict, save_dir: str):
    """Bar chart comparing Score/RMSE across models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(all_results.keys())
    scores = [all_results[m]["overall_score"] for m in model_names]
    rmses = [all_results[m]["regression_metrics"]["rmse"] for m in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # PHM Score comparison
    bars1 = ax1.bar(range(len(model_names)), scores, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names))))
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("PHM Challenge Score", fontsize=12)
    ax1.set_title("Model Comparison — PHM Score (higher = better)", fontsize=13)
    ax1.set_ylim(0, 1.05)
    for bar, val in zip(bars1, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # RMSE comparison
    bars2 = ax2.bar(range(len(model_names)), rmses, color=plt.cm.plasma(np.linspace(0.3, 0.9, len(model_names))))
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("RMSE (seconds)", fontsize=12)
    ax2.set_title("Model Comparison — RMSE (lower = better)", fontsize=13)
    for bar, val in zip(bars2, rmses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=150)
    plt.close()


def plot_training_curves(train_losses: list, val_losses: list, save_dir: str, model_name: str):
    """Plot training/validation loss curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Training Curves — {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_curves_{model_name.lower().replace(' ', '_')}.png"), dpi=150)
    plt.close()


def plot_rul_trajectory(
    bearing_name: str,
    predicted_ruls: np.ndarray,
    actual_rul_at_end: float,
    num_timesteps: int,
    recording_interval_s: float,
    save_dir: str,
):
    """Plot predicted RUL trajectory for a single test bearing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    times = np.arange(len(predicted_ruls)) * recording_interval_s / 3600  # hours

    ax.plot(times, predicted_ruls, label="Predicted RUL", linewidth=2, color="blue")

    # Actual RUL line (linearly decreasing from start to end)
    # We don't know the total life, but we know the RUL at the end
    # So actual RUL at each point = actual_rul_at_end + (num_timesteps - i) * interval
    actual_ruls_traj = np.array([
        actual_rul_at_end + (len(predicted_ruls) - 1 - i) * recording_interval_s
        for i in range(len(predicted_ruls))
    ])
    ax.plot(times, actual_ruls_traj, label="Actual RUL", linewidth=2, color="red", linestyle="--")

    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("RUL (seconds)", fontsize=12)
    ax.set_title(f"RUL Trajectory — {bearing_name}", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"rul_trajectory_{bearing_name}.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    # Quick test of PHM scoring function
    print("Testing PHM scoring function:")
    print(f"  Er=0 (perfect):     A = {phm_score_single(0):.4f} (expected: 1.0)")
    print(f"  Er=-5 (5% late):    A = {phm_score_single(-5):.4f} (expected: 0.5)")
    print(f"  Er=+20 (20% early): A = {phm_score_single(20):.4f} (expected: 0.5)")
    print(f"  Er=-10 (10% late):  A = {phm_score_single(-10):.4f} (expected: 0.25)")
    print(f"  Er=+40 (40% early): A = {phm_score_single(40):.4f} (expected: 0.25)")
    print(f"  Er=+100 (100%):     A = {phm_score_single(100):.4f}")
