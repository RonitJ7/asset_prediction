#!/usr/bin/env python3
"""
GNN_Model.py
------------
Multi-relational GNN for asset return prediction.

Architecture (from Architecture.md):
    PATH 1: GCN(F → H//2, edge_corr)    → LayerNorm → activation
    PATH 2: GCN(2F → H//2, edge_vendor) → LayerNorm → activation
    MERGE:  concat → [N, H] → Dropout
    REFINE: GCN(H → H, edge_combined)   → LayerNorm → activation → Dropout
    OUTPUT: Linear(H → 1)

Includes:
    - MultiRelGNN         : nn.Module (the model)
    - ICLoss              : nn.Module (IC + turnover loss)
    - train_one_fold      : function  (full training loop for one fold)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from omegaconf import DictConfig

from data_preparation import FoldData


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "elu": F.elu,
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "gelu": F.gelu,
}


def _get_activation(name: str):
    """Return an activation function by name."""
    fn = _ACTIVATIONS.get(name.lower())
    if fn is None:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}")
    return fn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiRelGNN(nn.Module):
    """
    Lightweight multi-relational GCN whose outputs are meant to be used as
    features for a downstream model (MLP, etc.).
    """

    def __init__(
        self,
        num_price_features: int,
        num_fundamental_features: int,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        lag_steps: int = 2,
        activation: str = "elu",
    ):
        super().__init__()

        self.input_dim = num_price_features + num_fundamental_features
        self.hidden_dim = hidden_dim
        self.lag_steps = lag_steps
        self.act_fn = _get_activation(activation)

        # -- PATH 1: correlation edges (peer effects, same-time) --
        self.conv_corr = GCNConv(self.input_dim, hidden_dim // 2)
        self.ln_corr = nn.LayerNorm(hidden_dim // 2)

        # -- PATH 2: vendor edges (supply-chain, lagged) --
        # 2× input because we concatenate [x_current, x_lagged]
        self.conv_vendor = GCNConv(self.input_dim * 2, hidden_dim // 2)
        self.ln_vendor = nn.LayerNorm(hidden_dim // 2)

        # -- MERGE dropout --
        self.dropout = nn.Dropout(dropout)

        # -- REFINE: shared GCN on combined edges --
        self.conv_refine = GCNConv(hidden_dim, hidden_dim)
        self.ln_refine = nn.LayerNorm(hidden_dim)

        # -- OUTPUT: linear projection --
        self.fc_out = nn.Linear(hidden_dim, 1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index_corr: torch.Tensor,
        edge_index_vendor: torch.Tensor,
        x_lagged: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x              : [N, F]   current node features
        edge_index_corr: [2, E1]  correlation edges
        edge_index_vendor: [2, E2] vendor edges
        x_lagged       : [N, F]   lagged node features (optional)

        Returns
        -------
        out : [N]  per-stock scalar prediction
        """
        # --- PATH 1: correlation ---
        h_corr = self.act_fn(self.ln_corr(self.conv_corr(x, edge_index_corr)))

        # --- PATH 2: vendor (with lag enrichment) ---
        if x_lagged is not None:
            x_vendor = torch.cat([x, x_lagged], dim=1)  # [N, 2F]
        else:
            x_vendor = torch.cat([x, x], dim=1)          # fallback

        h_vendor = self.act_fn(self.ln_vendor(self.conv_vendor(x_vendor, edge_index_vendor)))

        # --- MERGE ---
        h = torch.cat([h_corr, h_vendor], dim=1)  # [N, H]
        h = self.dropout(h)

        # --- REFINE ---
        edge_combined = torch.cat([edge_index_corr, edge_index_vendor], dim=1)
        h = self.act_fn(self.ln_refine(self.conv_refine(h, edge_combined)))
        h = self.dropout(h)

        # --- OUTPUT ---
        out = self.fc_out(h).squeeze(-1)  # [N]
        return out


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class ICLoss(nn.Module):
    """IC + turnover penalty, batched."""

    def __init__(self, turnover_weight: float = 0.0):
        super().__init__()
        self.turnover_weight = turnover_weight
        self.prev_pred: Optional[torch.Tensor] = None

    def reset_state(self):
        """Call at the start of each epoch to clear inter-batch state."""
        self.prev_pred = None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Ensure batch dim
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        # --- IC ---
        p_c = pred - pred.mean(dim=1, keepdim=True)
        t_c = target - target.mean(dim=1, keepdim=True)
        cov = (p_c * t_c).sum(dim=1)
        p_std = torch.sqrt((p_c ** 2).sum(dim=1) + 1e-8)
        t_std = torch.sqrt((t_c ** 2).sum(dim=1) + 1e-8)
        ic = cov / (p_std * t_std + 1e-8)
        ic_loss = 1.0 - ic

        # --- Turnover penalty ---
        turnover_penalty = torch.zeros_like(ic_loss)
        if self.turnover_weight > 0:
            if pred.size(0) > 1:
                turnover_penalty[1:] = self.turnover_weight * F.mse_loss(
                    pred[1:], pred[:-1].detach(), reduction="none"
                ).mean(dim=1)
            if self.prev_pred is not None:
                turnover_penalty[0] = self.turnover_weight * F.mse_loss(
                    pred[0], self.prev_pred.detach(), reduction="mean"
                )
            self.prev_pred = pred[-1].detach().clone()

        total = (ic_loss + turnover_penalty).mean()

        stats = {
            "ic": float(ic.mean().detach().cpu().item()),
            "ic_loss": float(ic_loss.mean().detach().cpu().item()),
            "turnover_penalty": float(turnover_penalty.mean().detach().cpu().item()),
        }
        return total, stats


# ---------------------------------------------------------------------------
# Batched edge-index helper
# ---------------------------------------------------------------------------

def _batch_edge_index(
    edge_index: torch.Tensor, batch_size: int, num_nodes: int
) -> torch.Tensor:
    """Replicate edge_index for a batch with per-graph node offsets."""
    E = edge_index.shape[1]
    offsets = (
        torch.arange(batch_size, device=edge_index.device)
        .repeat_interleave(E)
        * num_nodes
    )
    return edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_fold(
    model: MultiRelGNN,
    fold_data: FoldData,
    lag_lookup: np.ndarray,
    edge_index_corr: torch.Tensor,
    edge_index_vendor: torch.Tensor,
    cfg: DictConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Train for one fold and return per-epoch loss history.

    Parameters
    ----------
    model            : MultiRelGNN (already on device)
    fold_data        : FoldData for this fold
    lag_lookup       : [S, max_lag+1] array
    edge_index_corr  : [2, E1] on device
    edge_index_vendor: [2, E2] on device
    cfg              : Hydra DictConfig (must have cfg.model.* and cfg.training.*)
    device           : torch device

    Returns
    -------
    dict with keys: epoch_losses, epoch_ics
    """
    tcfg = cfg.training
    mcfg = cfg.model

    lag_steps = int(mcfg.lag_steps)
    lr = float(tcfg.lr)
    weight_decay = float(tcfg.weight_decay)
    batch_size = int(tcfg.batch_size)
    num_epochs = int(tcfg.epochs)
    max_lr = float(tcfg.max_lr)
    turnover_weight = float(tcfg.turnover_weight)
    grad_clip = float(tcfg.grad_clip)

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

    criterion = ICLoss(turnover_weight=turnover_weight)

    # --- Pre-compute lag indices for the train set ---
    train_lag_idx = torch.tensor(
        lag_lookup[fold_data.train_idx, lag_steps],
        device=device,
        dtype=torch.long,
    )
    global_to_train = {int(g): int(i) for i, g in enumerate(fold_data.train_idx)}

    # --- History ---
    history: Dict[str, List[float]] = {
        "epoch_losses": [],
        "epoch_ics": [],
    }

    # --- Epoch loop ---
    for epoch in range(num_epochs):
        model.train()
        criterion.reset_state()

        epoch_loss = 0.0
        epoch_ic = 0.0
        epoch_turn = 0.0
        num_batches = 0

        for batch_start in range(0, fold_data.X_train_tensor.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, fold_data.X_train_tensor.shape[0])

            x_batch = fold_data.X_train_tensor[batch_start:batch_end]   # [B, N, F]
            y_batch = fold_data.y_train_tensor[batch_start:batch_end]   # [B, N]
            lag_idx_batch = train_lag_idx[batch_start:batch_end]         # [B]

            # --- Resolve lagged features ---
            if lag_steps > 0:
                x_lagged_batch = x_batch.clone()
                valid_mask = lag_idx_batch != -1
                if valid_mask.any():
                    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                    lag_globals = lag_idx_batch[valid_mask].tolist()
                    lag_locals = [global_to_train.get(int(g), -1) for g in lag_globals]
                    lag_locals_t = torch.tensor(lag_locals, device=device, dtype=torch.long)
                    still_valid = lag_locals_t != -1
                    if still_valid.any():
                        src = lag_locals_t[still_valid]
                        dst = valid_idx[still_valid]
                        x_lagged_batch[dst] = fold_data.X_train_tensor[src]
            else:
                x_lagged_batch = x_batch

            # --- Flatten batch for GCNConv ---
            B, N, Feat = x_batch.shape
            x_flat = x_batch.reshape(B * N, Feat)
            x_lag_flat = x_lagged_batch.reshape(B * N, Feat)

            ec_b = _batch_edge_index(edge_index_corr, B, N)
            ev_b = _batch_edge_index(edge_index_vendor, B, N)

            # --- Forward ---
            out_flat = model(x_flat, ec_b, ev_b, x_lag_flat)  # [B*N]
            out = out_flat.view(B, N)

            batch_loss, stats = criterion(out, y_batch)

            # --- Backward ---
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += batch_loss.item()
            epoch_ic += stats["ic_loss"]
            epoch_turn += stats["turnover_penalty"]
            num_batches += 1

        denom = max(1, num_batches)
        history["epoch_losses"].append(epoch_loss / denom)
        history["epoch_ics"].append(epoch_ic / denom)

        if (epoch + 1) % 20 == 0:
            print(
                f"    Fold {fold_data.fold_idx} | Epoch {epoch+1:>3d}/{num_epochs} | "
                f"Loss {epoch_loss/denom:.6f} | "
                f"IC {epoch_ic/denom:.6f} | "
                f"Turn {epoch_turn/denom:.6f}"
            )

    return history
