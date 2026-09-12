"""
train.py — Fine-tuning loop for PatchTST bearing RUL prediction.

Supports:
- Frozen backbone + linear head (linear probing)
- LoRA fine-tuning via peft
- Full fine-tuning
- Classical ML baselines (SVR, Random Forest)
- TensorBoard logging
- Early stopping on validation RMSE
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading import load_config, load_all_bearings, compute_rul_labels
from src.feature_extraction import extract_features_bearing, FEATURE_NAMES
from src.health_index import select_top_features, compute_all_health_indices
from src.dataset import BearingRULDataset, create_dataloaders
from src.model import (
    setup_model, save_model, create_patchtst_model,
    LinearDegradationBaseline, ClassicalMLBaseline,
)
from src.evaluate import (
    rmse, mae, compute_regression_metrics, plot_training_curves
)

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Data Preparation Pipeline
# ============================================================================

def prepare_data(config: dict) -> dict:
    """
    Full data preparation pipeline: load data, extract features, build HI.

    Returns
    -------
    dict with all processed data needed for training and evaluation.
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Loading raw data...")
    logger.info("=" * 60)

    learning_data = load_all_bearings(config, "learning")
    test_data = load_all_bearings(config, "test")

    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Extracting features...")
    logger.info("=" * 60)

    # Extract features for all bearings
    learning_features = {}
    learning_feature_names = None

    for bearing_name, data in learning_data.items():
        logger.info(f"  Extracting features: {bearing_name}...")
        features, names = extract_features_bearing(data, config)
        learning_features[bearing_name] = features
        if learning_feature_names is None:
            learning_feature_names = names
        logger.info(f"    Shape: {features.shape}")

    test_features = {}
    for bearing_name, data in test_data.items():
        logger.info(f"  Extracting features: {bearing_name}...")
        features, _ = extract_features_bearing(data, config)
        test_features[bearing_name] = features
        logger.info(f"    Shape: {features.shape}")

    # Select top features and build Health Index
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Building Health Index...")
    logger.info("=" * 60)

    top_k = config["features"]["top_k_features"]
    selected_features = select_top_features(learning_features, learning_feature_names, top_k=top_k)

    smoothing = config["features"]["hi_smoothing_window"]
    learning_hi = compute_all_health_indices(
        learning_features, learning_feature_names, selected_features, smoothing
    )
    test_hi = compute_all_health_indices(
        test_features, learning_feature_names, selected_features, smoothing
    )

    # Compute RUL labels for learning set
    learning_rul = {}
    interval = config["dataset"]["recording_interval_s"]
    for bearing_name, data in learning_data.items():
        rul = compute_rul_labels(data, interval)
        learning_rul[bearing_name] = rul

    # Save processed data
    processed_dir = config["paths"]["processed_data"]
    os.makedirs(processed_dir, exist_ok=True)

    np.savez(
        os.path.join(processed_dir, "learning_features.npz"),
        **{k: v for k, v in learning_features.items()},
    )
    np.savez(
        os.path.join(processed_dir, "test_features.npz"),
        **{k: v for k, v in test_features.items()},
    )
    np.savez(
        os.path.join(processed_dir, "learning_hi.npz"),
        **{k: v for k, v in learning_hi.items()},
    )
    np.savez(
        os.path.join(processed_dir, "test_hi.npz"),
        **{k: v for k, v in test_hi.items()},
    )
    np.savez(
        os.path.join(processed_dir, "learning_rul.npz"),
        **{k: v for k, v in learning_rul.items()},
    )

    # Save feature names and selected features
    with open(os.path.join(processed_dir, "feature_info.json"), "w") as f:
        json.dump({
            "all_features": learning_feature_names,
            "selected_features": selected_features,
        }, f, indent=2)

    logger.info(f"Processed data saved to {processed_dir}/")

    return {
        "learning_features": learning_features,
        "test_features": test_features,
        "learning_hi": learning_hi,
        "test_hi": test_hi,
        "learning_rul": learning_rul,
        "feature_names": learning_feature_names,
        "selected_features": selected_features,
    }


def load_processed_data(config: dict) -> dict:
    """Load previously processed data from disk."""
    processed_dir = config["paths"]["processed_data"]

    learning_features = dict(np.load(os.path.join(processed_dir, "learning_features.npz")))
    test_features = dict(np.load(os.path.join(processed_dir, "test_features.npz")))
    learning_hi = dict(np.load(os.path.join(processed_dir, "learning_hi.npz")))
    test_hi = dict(np.load(os.path.join(processed_dir, "test_hi.npz")))
    learning_rul = dict(np.load(os.path.join(processed_dir, "learning_rul.npz")))

    with open(os.path.join(processed_dir, "feature_info.json"), "r") as f:
        feature_info = json.load(f)

    return {
        "learning_features": learning_features,
        "test_features": test_features,
        "learning_hi": learning_hi,
        "test_hi": test_hi,
        "learning_rul": learning_rul,
        "feature_names": feature_info["all_features"],
        "selected_features": feature_info["selected_features"],
    }


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)  # (B, seq_len, channels)
        batch_y = batch_y.to(device)  # (B, 1)

        optimizer.zero_grad()

        outputs = model(past_values=batch_x, target_values=batch_y)
        loss = outputs.loss

        if loss is None:
            # Manual loss computation if model doesn't return loss
            preds = outputs.regression_outputs
            loss = criterion(preds, batch_y)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, val_loader, criterion, device, dataset=None):
    """Validate and return average loss and RMSE in seconds."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(past_values=batch_x, target_values=batch_y)
        loss = outputs.loss

        if loss is None:
            preds = outputs.regression_outputs
            loss = criterion(preds, batch_y)
        else:
            preds = outputs.regression_outputs

        total_loss += loss.item()
        num_batches += 1

        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)

    # Compute RMSE in original seconds
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    if dataset is not None:
        # Denormalize
        preds_seconds = np.array([dataset.denormalize_rul(p) for p in all_preds.flatten()])
        targets_seconds = np.array([dataset.denormalize_rul(t) for t in all_targets.flatten()])
        val_rmse = rmse(targets_seconds, preds_seconds)
    else:
        val_rmse = rmse(all_targets, all_preds)

    return avg_loss, val_rmse


def train_patchtst(
    config: dict,
    train_features: dict,
    train_rul: dict,
    val_features: dict,
    val_rul: dict,
    strategy: str = "frozen",
    input_mode: str = "hi_only",
    model_name: str = None,
) -> dict:
    """
    Full training pipeline for one PatchTST configuration.

    Parameters
    ----------
    config : dict
        Configuration.
    train_features : dict
        Mapping bearing_name -> feature array for training.
    train_rul : dict
        Mapping bearing_name -> RUL array for training.
    val_features : dict
        Same for validation.
    val_rul : dict
        Same for validation.
    strategy : str
        'frozen', 'lora', or 'full'.
    input_mode : str
        'hi_only' or 'multivariate'.
    model_name : str
        Name for saving checkpoints and logs.

    Returns
    -------
    dict with training results.
    """
    if model_name is None:
        model_name = f"patchtst_{strategy}_{input_mode}"

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training: {model_name}")
    logger.info(f"Strategy: {strategy}, Input: {input_mode}")
    logger.info(f"{'=' * 60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Determine number of input channels
    if input_mode == "hi_only":
        num_channels = 1
    else:
        first_key = list(train_features.keys())[0]
        num_channels = train_features[first_key].shape[1]

    # Update config for this run
    run_config = config.copy()
    run_config["model"] = config["model"].copy()
    run_config["model"]["num_input_channels"] = num_channels

    # Create model
    model, is_peft = setup_model(run_config, strategy)
    model = model.to(device)

    # Create dataloaders
    window_size = config["training"]["window_size"]
    stride = config["training"]["window_stride"]
    batch_size = config["training"]["batch_size"]
    target_transform = config["training"]["target_transform"]

    # Reshape features if HI-only
    if input_mode == "hi_only":
        train_feat = {k: v.reshape(-1, 1) for k, v in train_features.items()}
        val_feat = {k: v.reshape(-1, 1) for k, v in val_features.items()}
    else:
        train_feat = train_features
        val_feat = val_features

    all_train_ruls = np.concatenate(list(train_rul.values()))
    max_rul = float(np.max(all_train_ruls))

    train_dataset = BearingRULDataset(
        train_feat, train_rul,
        window_size=window_size, stride=stride,
        target_transform=target_transform, max_rul=max_rul,
    )
    val_dataset = BearingRULDataset(
        val_feat, val_rul,
        window_size=window_size, stride=stride,
        target_transform=target_transform, max_rul=max_rul,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Loss, optimizer, scheduler
    criterion = nn.HuberLoss(delta=config["training"]["huber_delta"])
    lr = config["training"]["learning_rate"]
    wd = config["training"]["weight_decay"]

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd,
    )

    num_epochs = config["training"]["num_epochs"]
    warmup_steps = max(1, int(num_epochs * config["training"]["warmup_ratio"]))

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # Training loop
    best_val_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0
    patience = config["training"]["patience"]
    min_delta = config["training"]["min_delta"]
    max_grad_norm = config["training"]["max_grad_norm"]

    train_losses = []
    val_losses = []

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"runs/{model_name}")
        use_tb = True
    except ImportError:
        use_tb = False

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm)
        val_loss, val_rmse_seconds = validate(model, val_loader, criterion, device, val_dataset)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if use_tb:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("RMSE/val_seconds", val_rmse_seconds, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{num_epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"val_rmse={val_rmse_seconds:.1f}s, "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        # Early stopping
        if val_rmse_seconds < best_val_rmse - min_delta:
            best_val_rmse = val_rmse_seconds
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            save_dir = config["paths"]["models"]
            save_model(model, save_dir, model_name, is_peft)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed:.1f}s. Best val RMSE: {best_val_rmse:.1f}s at epoch {best_epoch}")

    if use_tb:
        writer.close()

    # Save training curves plot
    figures_dir = config["paths"]["figures"]
    os.makedirs(figures_dir, exist_ok=True)
    plot_training_curves(train_losses, val_losses, figures_dir, model_name)

    return {
        "model_name": model_name,
        "strategy": strategy,
        "input_mode": input_mode,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "elapsed_seconds": elapsed,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "num_channels": num_channels,
        "max_rul": max_rul,
    }


# ============================================================================
# Classical Baseline Training
# ============================================================================

def train_classical_baselines(
    config: dict,
    train_features: dict,
    train_rul: dict,
    val_features: dict,
    val_rul: dict,
    input_mode: str = "multivariate",
) -> dict:
    """
    Train classical ML baselines (SVR, Random Forest).
    """
    logger.info("\n" + "=" * 60)
    logger.info("Training Classical Baselines")
    logger.info("=" * 60)

    window_size = config["training"]["window_size"]
    stride = config["training"]["window_stride"]

    # Prepare flattened training data
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for bearing_name, features in train_features.items():
        rul = train_rul[bearing_name]
        T = len(features)

        if input_mode == "hi_only":
            feat = features.reshape(-1, 1)
        else:
            feat = features

        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window = feat[start:end].flatten()
            X_train.append(window)
            y_train.append(rul[end - 1])

    for bearing_name, features in val_features.items():
        rul = val_rul[bearing_name]
        T = len(features)

        if input_mode == "hi_only":
            feat = features.reshape(-1, 1)
        else:
            feat = features

        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window = feat[start:end].flatten()
            X_val.append(window)
            y_val.append(rul[end - 1])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # Subsample for SVR if dataset is very large (SVR scales poorly)
    max_train_svr = 2000
    if len(X_train) > max_train_svr:
        indices = np.random.choice(len(X_train), max_train_svr, replace=False)
        X_train_svr = X_train[indices]
        y_train_svr = y_train[indices]
    else:
        X_train_svr = X_train
        y_train_svr = y_train

    results = {}

    for model_type in ["svr", "rf"]:
        logger.info(f"\n  Training {model_type.upper()}...")
        baseline = ClassicalMLBaseline(model_type)

        if model_type == "svr":
            baseline.fit(X_train_svr, y_train_svr)
        else:
            baseline.fit(X_train, y_train)

        # Validate
        preds = baseline.predict(X_val)
        val_rmse_val = rmse(y_val, preds)
        val_mae_val = mae(y_val, preds)

        logger.info(f"  {model_type.upper()}: Val RMSE = {val_rmse_val:.1f}s, Val MAE = {val_mae_val:.1f}s")

        results[model_type] = {
            "model": baseline,
            "val_rmse": val_rmse_val,
            "val_mae": val_mae_val,
        }

    return results


# ============================================================================
# Main Training Pipeline
# ============================================================================

def run_training_pipeline(config: dict, data: dict = None) -> dict:
    """
    Full training pipeline: data prep + all model training.

    Returns dict of all training results.
    """
    set_seed(config["seed"])

    # Prepare or load data
    if data is None:
        processed_dir = config["paths"]["processed_data"]
        if os.path.exists(os.path.join(processed_dir, "learning_features.npz")):
            logger.info("Loading previously processed data...")
            data = load_processed_data(config)
        else:
            data = prepare_data(config)

    # Split into train/val
    val_bearing = config["training"]["val_bearing"]
    train_bearings = [b for b in config["dataset"]["learning_bearings"] if b != val_bearing]

    logger.info(f"\nTrain bearings: {train_bearings}")
    logger.info(f"Val bearing: {val_bearing}")

    # Prepare HI-only and multivariate feature dicts
    train_hi = {b: data["learning_hi"][b] for b in train_bearings}
    val_hi = {val_bearing: data["learning_hi"][val_bearing]}
    train_rul = {b: data["learning_rul"][b] for b in train_bearings}
    val_rul = {val_bearing: data["learning_rul"][val_bearing]}

    # Get top-K feature columns for multivariate input
    all_feature_names = data["feature_names"]
    selected = data["selected_features"]
    selected_indices = [all_feature_names.index(f) for f in selected if f in all_feature_names]

    train_multi = {b: data["learning_features"][b][:, selected_indices] for b in train_bearings}
    val_multi = {val_bearing: data["learning_features"][val_bearing][:, selected_indices]}

    all_results = {}

    # 1. PatchTST training with different strategies
    strategies = config["training"]["ablations"]["strategies"]
    input_modes = config["training"]["ablations"]["input_modes"]

    for strategy in strategies:
        for input_mode in input_modes:
            model_name = f"patchtst_{strategy}_{input_mode}"

            if input_mode == "hi_only":
                tf, vf = train_hi, val_hi
            else:
                tf, vf = train_multi, val_multi

            try:
                result = train_patchtst(
                    config, tf, train_rul, vf, val_rul,
                    strategy=strategy, input_mode=input_mode,
                    model_name=model_name,
                )
                all_results[model_name] = result
            except Exception as e:
                logger.error(f"Training failed for {model_name}: {e}")
                import traceback
                traceback.print_exc()

    # 2. Classical baselines
    try:
        classical_results = train_classical_baselines(
            config, train_multi, train_rul, val_multi, val_rul,
            input_mode="multivariate",
        )
        all_results["classical"] = classical_results
    except Exception as e:
        logger.error(f"Classical baseline training failed: {e}")

    # Save training summary
    summary = {}
    for name, result in all_results.items():
        if name == "classical":
            for model_type, r in result.items():
                summary[f"classical_{model_type}"] = {
                    "val_rmse": r["val_rmse"],
                    "val_mae": r["val_mae"],
                }
        else:
            summary[name] = {
                "best_epoch": result.get("best_epoch"),
                "best_val_rmse": result.get("best_val_rmse"),
                "elapsed_seconds": result.get("elapsed_seconds"),
            }

    os.makedirs(config["paths"]["reports"], exist_ok=True)
    with open(os.path.join(config["paths"]["reports"], "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return all_results, data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()
    results, data = run_training_pipeline(config)

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    for name, result in results.items():
        if name == "classical":
            for mt, r in result.items():
                print(f"  {mt.upper()}: val_rmse={r['val_rmse']:.1f}s")
        else:
            print(f"  {name}: best_val_rmse={result.get('best_val_rmse', 'N/A')}s "
                  f"(epoch {result.get('best_epoch', 'N/A')})")
