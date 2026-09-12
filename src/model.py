"""
model.py — PatchTST-based model for bearing RUL regression.

Provides three model variants:
1. PatchTSTForRegression wrapper (frozen, LoRA, or full fine-tuning)
2. Classical ML baselines (Linear degradation, SVR, Random Forest)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import logging
from transformers import PatchTSTConfig, PatchTSTForRegression

logger = logging.getLogger(__name__)


# ============================================================================
# PatchTST-based RUL Model
# ============================================================================

def create_patchtst_model(config: dict) -> PatchTSTForRegression:
    """
    Create a PatchTSTForRegression model from configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary (model section).

    Returns
    -------
    PatchTSTForRegression model
    """
    model_cfg = config["model"]

    patchtst_config = PatchTSTConfig(
        num_input_channels=model_cfg["num_input_channels"],
        context_length=model_cfg["context_length"],
        patch_length=model_cfg["patch_length"],
        patch_stride=model_cfg["patch_stride"],
        d_model=model_cfg["d_model"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        dropout=model_cfg["dropout"],
        # Regression-specific
        num_targets=model_cfg["num_targets"],
        # Additional settings
        use_cls_token=False,
        channel_attention=False,
    )

    model = PatchTSTForRegression(patchtst_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"PatchTST model created: {total_params:,} total params, "
        f"{trainable_params:,} trainable params"
    )

    return model


def freeze_backbone(model: PatchTSTForRegression) -> PatchTSTForRegression:
    """
    Freeze all backbone parameters, leaving only the regression head trainable.

    Parameters
    ----------
    model : PatchTSTForRegression

    Returns
    -------
    Same model with backbone frozen.
    """
    # Freeze the PatchTST model backbone
    for name, param in model.named_parameters():
        if "head" not in name.lower() and "regression" not in name.lower():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Backbone frozen. Trainable: {trainable:,} / {total:,} params")

    return model


def unfreeze_all(model: PatchTSTForRegression) -> PatchTSTForRegression:
    """Unfreeze all model parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"All params unfrozen. Trainable: {trainable:,} params")

    return model


def apply_lora(model: PatchTSTForRegression, config: dict):
    """
    Apply LoRA adapters to the model using peft library.

    Parameters
    ----------
    model : PatchTSTForRegression
    config : dict
        Configuration dictionary.

    Returns
    -------
    peft model with LoRA adapters
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=config["model"]["lora_rank"],
            lora_alpha=config["model"]["lora_alpha"],
            lora_dropout=config["model"]["lora_dropout"],
            target_modules=["q_proj", "v_proj", "k_proj"],
            bias="none",
        )

        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"LoRA applied. Trainable: {trainable:,} / {total:,} params "
            f"({100 * trainable / total:.1f}%)"
        )

        return model

    except ImportError:
        logger.warning("peft not installed. Falling back to full fine-tuning.")
        return model
    except Exception as e:
        logger.warning(f"LoRA application failed: {e}. Falling back to full fine-tuning.")
        unfreeze_all(model)
        return model


def setup_model(config: dict, strategy: str = "frozen"):
    """
    Create and configure model based on training strategy.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    strategy : str
        'frozen' — freeze backbone, train head only
        'lora'  — apply LoRA adapters
        'full'  — unfreeze all parameters

    Returns
    -------
    model, is_peft (bool indicating if peft wrapper was applied)
    """
    model = create_patchtst_model(config)

    is_peft = False

    if strategy == "frozen":
        model = freeze_backbone(model)
    elif strategy == "lora":
        model = apply_lora(model, config)
        is_peft = True
    elif strategy == "full":
        model = unfreeze_all(model)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return model, is_peft


# ============================================================================
# Classical ML Baselines
# ============================================================================

class LinearDegradationBaseline:
    """
    Baseline: Linear degradation extrapolation of the Health Index.

    Fits a linear trend to the last N HI points and extrapolates to the
    failure threshold (HI = 1.0).
    """

    def __init__(self, window_size: int = 64):
        self.window_size = window_size

    def predict_rul(self, hi_sequence: np.ndarray) -> float:
        """
        Predict RUL by extrapolating HI linearly to threshold = 1.0.

        Parameters
        ----------
        hi_sequence : np.ndarray
            Health Index values (0 = healthy, 1 = failure).

        Returns
        -------
        float — predicted RUL in timesteps (multiply by 10 for seconds)
        """
        # Use last window_size points
        if len(hi_sequence) >= self.window_size:
            recent = hi_sequence[-self.window_size:]
        else:
            recent = hi_sequence

        # Fit linear trend
        t = np.arange(len(recent), dtype=np.float64)
        coeffs = np.polyfit(t, recent, deg=1)
        slope = coeffs[0]
        intercept = coeffs[1]

        current_hi = recent[-1]

        if slope <= 0:
            # HI is not increasing — estimate large RUL
            return 10000.0  # fallback: very large

        # Extrapolate to HI = 1.0
        # slope * t_fail + intercept = 1.0
        # t_fail = (1.0 - intercept) / slope
        # RUL = t_fail - len(recent) + 1
        t_fail = (1.0 - intercept) / slope
        rul_timesteps = t_fail - (len(recent) - 1)

        return max(rul_timesteps, 0.0)


class ClassicalMLBaseline:
    """
    Classical ML baseline using sklearn (SVR or Random Forest).
    """

    def __init__(self, model_type: str = "svr"):
        self.model_type = model_type
        self.model = None
        self.scaler_X = None
        self.scaler_y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model on flattened feature windows.

        Parameters
        ----------
        X : np.ndarray
            Shape (num_samples, window_size * num_features) — flattened windows.
        y : np.ndarray
            Shape (num_samples,) — RUL targets.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from sklearn.ensemble import RandomForestRegressor

        # Scale features and targets
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        if self.model_type == "svr":
            self.model = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
        elif self.model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        logger.info(f"Training {self.model_type.upper()} on {X.shape[0]} samples...")
        self.model.fit(X_scaled, y_scaled)
        logger.info(f"Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RUL.

        Parameters
        ----------
        X : np.ndarray
            Shape (num_samples, window_size * num_features).

        Returns
        -------
        np.ndarray of shape (num_samples,) — predicted RUL in seconds.
        """
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def save_model(model, save_dir: str, model_name: str, is_peft: bool = False):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.pt")

    if is_peft:
        # Save LoRA adapter weights
        model.save_pretrained(os.path.join(save_dir, f"{model_name}_lora"))
        logger.info(f"LoRA model saved to {save_dir}/{model_name}_lora/")
    else:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")


def load_model(config: dict, save_dir: str, model_name: str, strategy: str = "frozen"):
    """Load model checkpoint."""
    model = create_patchtst_model(config)

    if strategy == "lora":
        try:
            from peft import PeftModel
            adapter_path = os.path.join(save_dir, f"{model_name}_lora")
            model = PeftModel.from_pretrained(model, adapter_path)
            logger.info(f"LoRA model loaded from {adapter_path}")
        except Exception as e:
            logger.error(f"Failed to load LoRA model: {e}")
    else:
        save_path = os.path.join(save_dir, f"{model_name}.pt")
        state_dict = torch.load(save_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Model loaded from {save_path}")

    return model
