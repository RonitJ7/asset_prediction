#!/usr/bin/env python3
"""
mlp.py
------
Simple feedforward MLP for binary return-direction classification (+1 / −1).

Architecture:
    Input(F) → [Linear → (LayerNorm?) → Activation → Dropout] × N_layers
             → Linear(H_last, 1)   (logit output)

Loss: BCEWithLogitsLoss with automatic pos_weight from training labels.

Includes:
    - ReturnClassifierMLP : nn.Module
    - train_mlp_fold      : function  (full training loop for one fold)
"""

import math
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from data_preparation import FoldData


# ---------------------------------------------------------------------------
# Activation helper (shared with GNN_Model, duplicated to keep mlp standalone)
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


def _get_activation_module(name: str) -> nn.Module:
    cls = _ACTIVATIONS.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}")
    return cls()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ReturnClassifierMLP(nn.Module):
    """
    Per-stock feedforward classifier.

    Parameters
    ----------
    input_dim  : number of features per stock (F_price + F_fund)
    layers     : sequence of hidden sizes, e.g. [64, 64]
    dropout    : dropout probability between hidden layers
    activation : activation name (elu, relu, leaky_relu, gelu)
    use_layernorm : whether to include LayerNorm between hidden layers
    """

    def __init__(
        self,
        input_dim: int,
        layers: Sequence[int] = (64, 64),
        dropout: float = 0.1,
        activation: str = "elu",
        use_layernorm: bool = False,
    ):
        super().__init__()

        blocks: List[nn.Module] = []
        in_dim = input_dim

        for hidden_dim in layers:
            blocks.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                blocks.append(nn.LayerNorm(hidden_dim))
            blocks.append(_get_activation_module(activation))
            blocks.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [*, F]  arbitrary leading dims, last dim = features

        Returns
        -------
        logits : [*]  one logit per stock (use sigmoid for probability)
        """
        h = self.backbone(x)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mlp_fold(
    model: ReturnClassifierMLP,
    fold_data: FoldData,
    cfg: DictConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Train the MLP classifier for one fold.

    Targets are binarised on the fly:  y_cls = (y > 0).float()
    pos_weight is computed from training labels to handle class imbalance.

    Parameters
    ----------
    model     : ReturnClassifierMLP (already on device)
    fold_data : FoldData for this fold
    cfg       : Hydra DictConfig with cfg.model.mlp.* and cfg.training.mlp.*
    device    : torch device

    Returns
    -------
    dict with keys: epoch_losses, epoch_accs
    """
    tcfg = cfg.training.mlp   # MLP-specific training params
    mcfg = cfg.model.mlp      # MLP architecture params (includes epochs)

    lr = float(tcfg.lr)
    weight_decay = float(tcfg.weight_decay)
    batch_size = int(tcfg.batch_size)
    num_epochs = int(mcfg.epochs)
    max_lr = float(tcfg.max_lr)
    grad_clip = float(tcfg.grad_clip)

    # --- Binarise targets ---
    y_train_cls = (fold_data.y_train_tensor > 0).float()  # [S, N]
    y_test_cls = (fold_data.y_test_tensor > 0).float()

    # --- Compute pos_weight for class imbalance ---
    n_pos = y_train_cls.sum()
    n_neg = y_train_cls.numel() - n_pos
    pos_weight = (n_neg / n_pos.clamp(min=1.0)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimiser ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=tuple(tcfg.betas),
    )

    steps_per_epoch = max(1, math.ceil(fold_data.X_train_tensor.shape[0] / batch_size))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=float(tcfg.warmup_pct),
        anneal_strategy="cos",
    )

    # --- History ---
    history: Dict[str, List[float]] = {
        "epoch_losses": [],
        "epoch_accs": [],
    }

    # --- Epoch loop ---
    S_train = fold_data.X_train_tensor.shape[0]

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0

        for batch_start in range(0, S_train, batch_size):
            batch_end = min(batch_start + batch_size, S_train)

            x_batch = fold_data.X_train_tensor[batch_start:batch_end]  # [B, N, F]
            y_batch = y_train_cls[batch_start:batch_end]               # [B, N]

            B, N, F_dim = x_batch.shape
            logits = model(x_batch.reshape(B * N, F_dim))  # [B*N]
            logits = logits.view(B, N)

            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

            # --- Accuracy ---
            with torch.no_grad():
                preds = (logits > 0).float()
                epoch_correct += (preds == y_batch).sum().item()
                epoch_total += y_batch.numel()

            epoch_loss += loss.item()
            num_batches += 1

        denom = max(1, num_batches)
        acc = epoch_correct / max(1, epoch_total)
        history["epoch_losses"].append(epoch_loss / denom)
        history["epoch_accs"].append(acc)

        if (epoch + 1) % 20 == 0:
            print(
                f"    Fold {fold_data.fold_idx} | Epoch {epoch+1:>3d}/{num_epochs} | "
                f"Loss {epoch_loss/denom:.6f} | Acc {acc:.4f}"
            )

    return history
