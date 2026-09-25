"""MLP position classifier in PyTorch.

PyTorch concepts beyond micrograd, in one place:
  * Tensor    — like micrograd's Value, but a whole array at once and able to live on the GPU.
  * nn.Module — a class that registers its parameters automatically (no manual parameter list).
  * DataLoader— hands out shuffled mini-batches instead of looping over the data by hand.
  * Optimizer — does the parameter update for every parameter, with momentum/adaptive rates.
  * .to(device) — moves tensors/modules to the GPU; data and model must be on the same device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PositionMLP(nn.Module):
    """104 features -> hidden layers -> one score ("logit") per position class."""

    def __init__(self, num_features: int, num_classes: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.BatchNorm1d(hidden),   # keeps activations in a stable range -> faster, steadier training
            nn.ReLU(),
            nn.Dropout(dropout),      # randomly zeroes activations while training -> less overfitting
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_step(model: nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor,
               loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """One optimisation step on a single mini-batch. Returns the loss as a plain float.

    This is exactly micrograd's loop, with PyTorch names:
      forward pass -> loss -> zero the old gradients -> backward pass -> update the parameters.

    Args:
        model:     the network; `model(batch_x)` runs the forward pass and returns logits.
        batch_x:   (B, num_features) input batch.
        batch_y:   (B,) integer class labels.
        loss_fn:   cross entropy; call it as `loss_fn(logits, batch_y)`.
        optimizer: holds the parameters; `optimizer.zero_grad()` and `optimizer.step()`.

    Careful: gradients in PyTorch *accumulate*. Forgetting to clear them is the single most
    common bug in a training loop, and it does not raise an error — the model just learns badly.
    """
    # TODO(human): implement the five lines of the training step (owner task B).
    raise NotImplementedError("owner task B")


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 512
    dropout: float = 0.2
    seed: int = 0


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: str) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32, device=device),
                            torch.as_tensor(y, dtype=torch.long, device=device))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()   # no gradients needed for evaluation -> less memory, faster
def predict_proba(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    model.eval()   # switches BatchNorm/Dropout into inference behaviour
    out = []
    for start in range(0, len(X), batch_size):
        batch = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        out.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def fit(X_train: np.ndarray, y_train: np.ndarray, num_classes: int, config: TrainConfig,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
        device: str | None = None, verbose: bool = True):
    """Train the MLP. Returns (model, history) with per-epoch loss and validation accuracy."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    model = PositionMLP(X_train.shape[1], num_classes, config.hidden, config.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    loader = make_loader(X_train, y_train, config.batch_size, shuffle=True, device=device)

    history = []
    for epoch in range(config.epochs):
        model.train()
        losses = [train_step(model, xb, yb, loss_fn, optimizer) for xb, yb in loader]
        scheduler.step()
        entry = {"epoch": epoch, "loss": float(np.mean(losses))}
        if X_val is not None:
            entry["val_accuracy"] = float((predict_proba(model, X_val, device).argmax(1) == y_val).mean())
        history.append(entry)
        if verbose:
            extra = f" val_acc={entry['val_accuracy']:.3f}" if "val_accuracy" in entry else ""
            print(f"  epoch {epoch + 1:>3}/{config.epochs} loss={entry['loss']:.4f}{extra}")
    return model, history
