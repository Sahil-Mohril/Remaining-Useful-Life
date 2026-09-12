"""
dataset.py — PyTorch Dataset and DataLoader for bearing RUL prediction.

Creates sliding-window training examples from feature sequences
and RUL labels, compatible with PatchTST input format.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class BearingRULDataset(Dataset):
    """
    PyTorch Dataset that yields (input_window, normalized_rul_target) pairs
    from sliding windows over bearing feature sequences.

    Parameters
    ----------
    feature_sequences : dict
        Mapping bearing_name -> np.ndarray of shape (num_timesteps, num_channels).
    rul_sequences : dict
        Mapping bearing_name -> np.ndarray of shape (num_timesteps,) with RUL in seconds.
    window_size : int
        Number of timesteps per input window.
    stride : int
        Stride for sliding window.
    target_transform : str
        'log' for log(1 + RUL) normalization, 'linear' for min-max.
    max_rul : float
        Maximum RUL for linear normalization (ignored if target_transform='log').
    """

    def __init__(
        self,
        feature_sequences: dict,
        rul_sequences: dict,
        window_size: int = 64,
        stride: int = 1,
        target_transform: str = "log",
        max_rul: float = None,
    ):
        self.window_size = window_size
        self.target_transform = target_transform
        self.samples = []

        # Compute max_rul across all bearings for normalization
        if max_rul is None:
            all_ruls = np.concatenate(list(rul_sequences.values()))
            self.max_rul = float(np.max(all_ruls))
        else:
            self.max_rul = max_rul

        # Compute log max for log normalization
        self.log_max_rul = np.log1p(self.max_rul)

        # Create sliding window samples
        for bearing_name in feature_sequences:
            features = feature_sequences[bearing_name]  # (T, C)
            rul = rul_sequences[bearing_name]  # (T,)

            T = len(features)
            if T < window_size:
                logger.warning(
                    f"Bearing {bearing_name}: {T} timesteps < window_size {window_size}. "
                    f"Padding with first timestep."
                )
                # Pad at the beginning
                pad_len = window_size - T
                features = np.vstack([
                    np.tile(features[0:1], (pad_len, 1)),
                    features
                ])
                rul = np.concatenate([np.full(pad_len, rul[0]), rul])
                T = len(features)

            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                window = features[start:end]  # (window_size, num_channels)
                target_rul = rul[end - 1]  # RUL at the end of the window

                self.samples.append((window, target_rul, bearing_name))

        logger.info(
            f"Created dataset with {len(self.samples)} samples from "
            f"{len(feature_sequences)} bearings (window={window_size}, stride={stride})"
        )

    def normalize_rul(self, rul: float) -> float:
        """Normalize RUL target."""
        if self.target_transform == "log":
            return np.log1p(rul) / self.log_max_rul
        else:
            return rul / self.max_rul

    def denormalize_rul(self, normalized_rul: float) -> float:
        """Denormalize RUL target back to seconds."""
        if self.target_transform == "log":
            return np.expm1(normalized_rul * self.log_max_rul)
        else:
            return normalized_rul * self.max_rul

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, rul, bearing_name = self.samples[idx]

        # Normalize target
        normalized_rul = self.normalize_rul(rul)

        # Convert to tensors
        # PatchTST expects: (batch, seq_len, num_channels) — we return (seq_len, num_channels)
        window_tensor = torch.tensor(window, dtype=torch.float32)
        target_tensor = torch.tensor([normalized_rul], dtype=torch.float32)

        return window_tensor, target_tensor


def create_dataloaders(
    train_features: dict,
    train_rul: dict,
    val_features: dict,
    val_rul: dict,
    config: dict,
) -> tuple:
    """
    Create train and validation DataLoaders.

    Parameters
    ----------
    train_features : dict
        Mapping bearing_name -> feature_matrix for training bearings.
    train_rul : dict
        Mapping bearing_name -> rul_array for training bearings.
    val_features : dict
        Same for validation.
    val_rul : dict
        Same for validation.
    config : dict
        Configuration dictionary.

    Returns
    -------
    (train_loader, val_loader, train_dataset, val_dataset)
    """
    window_size = config["training"]["window_size"]
    stride = config["training"]["window_stride"]
    batch_size = config["training"]["batch_size"]
    target_transform = config["training"]["target_transform"]

    # Compute max_rul from training data only
    all_train_ruls = np.concatenate(list(train_rul.values()))
    max_rul = float(np.max(all_train_ruls))

    train_dataset = BearingRULDataset(
        train_features, train_rul,
        window_size=window_size,
        stride=stride,
        target_transform=target_transform,
        max_rul=max_rul,
    )

    val_dataset = BearingRULDataset(
        val_features, val_rul,
        window_size=window_size,
        stride=stride,
        target_transform=target_transform,
        max_rul=max_rul,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, val_loader, train_dataset, val_dataset


def prepare_test_windows(
    feature_sequence: np.ndarray,
    window_size: int = 64,
) -> torch.Tensor:
    """
    Prepare input windows for a test bearing (for inference).

    Returns the last window (and optionally all rolling windows).

    Parameters
    ----------
    feature_sequence : np.ndarray
        Shape (T, num_channels).
    window_size : int
        Window size.

    Returns
    -------
    torch.Tensor of shape (1, window_size, num_channels) — last window
    """
    T = feature_sequence.shape[0]

    if T < window_size:
        # Pad at the beginning
        pad_len = window_size - T
        padded = np.vstack([
            np.tile(feature_sequence[0:1], (pad_len, 1)),
            feature_sequence
        ])
        window = padded[-window_size:]
    else:
        window = feature_sequence[-window_size:]

    return torch.tensor(window, dtype=torch.float32).unsqueeze(0)


def prepare_rolling_windows(
    feature_sequence: np.ndarray,
    window_size: int = 64,
    stride: int = 1,
) -> torch.Tensor:
    """
    Prepare all rolling windows for trajectory prediction.

    Parameters
    ----------
    feature_sequence : np.ndarray
        Shape (T, num_channels).
    window_size : int
        Window size.
    stride : int
        Stride between windows.

    Returns
    -------
    torch.Tensor of shape (N, window_size, num_channels)
    """
    T = feature_sequence.shape[0]

    if T < window_size:
        pad_len = window_size - T
        feature_sequence = np.vstack([
            np.tile(feature_sequence[0:1], (pad_len, 1)),
            feature_sequence
        ])
        T = len(feature_sequence)

    windows = []
    for start in range(0, T - window_size + 1, stride):
        windows.append(feature_sequence[start:start + window_size])

    return torch.tensor(np.array(windows), dtype=torch.float32)
