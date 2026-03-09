"""
data_setup.py
=============
Dataset and DataLoader utilities for Ptychographic reconstruction with Neural Operators.

The dataset expects NumPy arrays (.npy) with shapes:
    - inputs (diffraction patterns): [N, 1, S, S]
    - intensity targets:             [N, 1, S, S]
    - phase targets:                 [N, 1, S, S]

The model receives inputs of shape [N, 1, S, S] and predicts
amplitude + phase stacked along the channel dimension: [N, 2, S, S].
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PtychoDataset(Dataset):
    """PyTorch Dataset for Ptychographic reconstruction with Neural Operators.

    Loads diffraction patterns as model inputs and stacks amplitude/phase
    arrays as the prediction target.

    Attributes:
        inputs  (Tensor): Diffraction patterns, shape [N, 1, S, S].
        targets (Tensor): Stacked amplitude and phase, shape [N, 2, S, S].
    """

    def __init__(
        self,
        x_path: str,
        intensity_target_path: str,
        phase_target_path: str,
    ) -> None:
        """Load and prepare the ptychography dataset from .npy files.

        Args:
            x_path: Path to .npy file containing diffraction patterns,
                shape [N, 1, S, S].
            intensity_target_path: Path to .npy file containing intensity
                (amplitude) targets, shape [N, 1, S, S].
            phase_target_path: Path to .npy file containing phase targets,
                shape [N, 1, S, S].
        """
        x = torch.from_numpy(np.load(x_path)).to(torch.float)                       # [N, 1, S, S]
        y_int = torch.from_numpy(np.load(intensity_target_path)).to(torch.float)    # [N, 1, S, S]
        y_phi = torch.from_numpy(np.load(phase_target_path)).to(torch.float)        # [N, 1, S, S]

        self.inputs = x                                                             # [N, 1, S, S]
        self.targets = torch.cat([y_int, y_phi], dim=1)                             # [N, 2, S, S]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample as a dict with keys 'x' (input) and 'y' (target).

        Args:
            idx: Sample index.

        Returns:
            Dict with:
                'x': Diffraction pattern tensor, shape [1, S, S].
                'y': Stacked amplitude and phase tensor, shape [2, S, S].
        """
        return {
            "x": self.inputs[idx],   # [1, S, S]
            "y": self.targets[idx],  # [2, S, S]
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for the ptychography dataset.

    Expects the following files inside *data_dir*:
        - X_train_final.npy, Y_I_train_final.npy, Y_P_train_final.npy
        - X_val_final.npy,   Y_I_val_final.npy,   Y_P_val_final.npy

    Args:
        data_dir: Directory that contains the .npy data files.
            Should end with a path separator (e.g. "/path/to/data/").
        batch_size: Number of samples per batch. Default: 64.
        num_workers: Number of worker processes for data loading. Default: 0.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if not data_dir.endswith("/"):
        data_dir = data_dir + "/"

    train_ds = PtychoDataset(
        x_path=data_dir + "X_train_final.npy",
        intensity_target_path=data_dir + "Y_I_train_final.npy",
        phase_target_path=data_dir + "Y_P_train_final.npy",
    )
    val_ds = PtychoDataset(
        x_path=data_dir + "X_val_final.npy",
        intensity_target_path=data_dir + "Y_I_val_final.npy",
        phase_target_path=data_dir + "Y_P_val_final.npy",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")

    return train_loader, val_loader

