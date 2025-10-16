"""Simple Temporal Convolutional Network (TCN) for 1D time-series regression.

Includes:
- TemporalConvNet model (causal dilated convolutions with residuals)
- SequenceDataset builder using a fixed-length rolling window over selected features
- Utilities for per-fold normalization and train/eval loops

Notes
- Designed to integrate with the project's walk-forward split and backtester.
- Expects a single row per unique `date_id`, sorted ascending.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Iterable

import numpy as np
import pandas as pd


try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:  # pragma: no cover - makes import error explicit during runtime
    raise RuntimeError(
        "PyTorch is required for the TCN module. Please install torch (CPU is fine): pip install torch"
    ) from e


class Chomp1d(nn.Module):
    """Remove padding on the right to maintain causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size]


class TemporalBlock(nn.Module):
    """A residual block with two causal Conv1d layers and dropout.

    Uses symmetric Conv1d padding and chomp to ensure output length equals input length,
    preserving the residual connection shape.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: Sequence[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_ch = num_inputs
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            )
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, L]
        return self.network(x)


class TCNRegressor(nn.Module):
    """TCN + pooling + linear head for regression."""

    def __init__(self, in_channels: int, channels: Sequence[int] = (32, 32, 32), kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.tcn = TemporalConvNet(in_channels, channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, x):
        # x: [B, C, L]
        h = self.tcn(x)
        # take the last time step features [B, C, L] -> [B, C]
        h_last = h[:, :, -1]
        out = self.head(h_last)
        return out.squeeze(-1)


def compute_channel_norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean/std over batch and time dims.

    x: [N, C, L] -> mean/std: [C]
    """
    # avoid circular import in type hints
    N, C, L = x.shape
    mean = x.permute(1, 0, 2).contiguous().view(C, -1).mean(dim=1)
    std = x.permute(1, 0, 2).contiguous().view(C, -1).std(dim=1)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return mean, std


class SequenceDataset(Dataset):
    """Build fixed-length sequences from a time-sorted DataFrame.

    Assumptions:
    - One row per unique time step (e.g., unique `date_id`).
    - DataFrame is sorted by `date_id` ascending before dataset construction.
    - `target_indices` are integer positional indices into the sorted DataFrame.
    - For each target index i, the sequence uses rows [i-seq_len, ..., i-1]. If not enough history, the sample is skipped.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        target_indices: Iterable[int],
        seq_len: int = 60,
        channel_mean: Optional[torch.Tensor] = None,
        channel_std: Optional[torch.Tensor] = None,
    ):
        self.df = df
        self.feature_cols = [c for c in feature_cols if c in df.columns]
        self.target_col = target_col
        self.seq_len = seq_len
        self.targets: List[int] = []  # original DataFrame indices that each sample predicts
        self.X_list: List[np.ndarray] = []
        self.y_list: List[float] = []

        # pre-extract as numpy for speed
        X_full = df[self.feature_cols].to_numpy(dtype=np.float32)
        y_full = df[target_col].to_numpy(dtype=np.float32)

        for i in target_indices:
            if i < seq_len:
                continue  # not enough history
            start = i - seq_len
            end = i  # exclusive
            x_seq = X_full[start:end]  # [L, C]
            if np.any(np.isnan(x_seq)):
                # basic guard: skip sequences with NaNs
                continue
            self.targets.append(i)
            # reshape to [C, L]
            self.X_list.append(x_seq.T)
            self.y_list.append(float(y_full[i]))

        self.X = torch.from_numpy(np.stack(self.X_list)) if self.X_list else torch.empty(0)
        self.y = torch.from_numpy(np.array(self.y_list, dtype=np.float32)) if self.y_list else torch.empty(0)

        # Apply normalization if provided
        if self.X.numel() > 0 and channel_mean is not None and channel_std is not None:
            self.X = (self.X - channel_mean[None, :, None]) / channel_std[None, :, None]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def used_indices(self) -> List[int]:
        """Return the original row indices in df that this dataset predicts for.

        Useful to align predictions back to the test partition without guessing.
        """
        return self.targets


@dataclass
class TCNTrainConfig:
    seq_len: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50
    patience: int = 5
    channels: Tuple[int, ...] = (32, 32, 32)
    kernel_size: int = 3
    dropout: float = 0.1
    device: str = "cpu"


def train_tcn(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    config: TCNTrainConfig,
) -> Tuple[TCNRegressor, dict]:
    model = TCNRegressor(in_channels=X_tr.shape[1], channels=config.channels, kernel_size=config.kernel_size, dropout=config.dropout)
    device = torch.device(config.device)
    model.to(device)

    train_ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(config.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * xb.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)

        tr_loss /= max(1, len(train_ds))
        val_loss /= max(1, len(val_ds))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {"best_val_mse": float(best_val)}
    return model, metrics


def build_fold_tensors(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    train_idx: Iterable[int],
    test_idx: Iterable[int],
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
    """Return (X_tr, y_tr, X_te, y_te, mean, std, train_used_idx, test_used_idx) with per-channel normalization fitted on train."""
    # Assumes df is already in chronological order consistent with provided indices.

    # datasets
    ds_tr_no_norm = SequenceDataset(df, feature_cols, target_col, train_idx, seq_len=seq_len)
    if len(ds_tr_no_norm) == 0:
        raise RuntimeError("No training sequences constructed; check seq_len and indices.")
    # compute normalization on train
    mean, std = compute_channel_norm(ds_tr_no_norm.X)
    ds_tr = SequenceDataset(df, feature_cols, target_col, train_idx, seq_len=seq_len, channel_mean=mean, channel_std=std)
    ds_te = SequenceDataset(df, feature_cols, target_col, test_idx, seq_len=seq_len, channel_mean=mean, channel_std=std)

    # to tensors
    return ds_tr.X, ds_tr.y, ds_te.X, ds_te.y, mean, std, ds_tr.used_indices, ds_te.used_indices


def predict(model: nn.Module, X: torch.Tensor, device: str = "cpu") -> np.ndarray:
    model.eval()
    preds = []
    device_t = torch.device(device)
    with torch.no_grad():
        for i in range(0, X.shape[0], 1024):
            xb = X[i : i + 1024].to(device_t)
            pb = model(xb).cpu().numpy()
            preds.append(pb)
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.float32)
