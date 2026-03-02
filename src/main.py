#!/usr/bin/env python3
"""
main.py
-------
Entry point for the asset-prediction pipeline.
All parameters are driven by configs/ via Hydra.

Usage
-----
# default run
python main.py

# override individual params from CLI
python main.py data.feature_window=30 training.lr=0.0005

# swap to a different model config file (configs/model/large.yaml)
python main.py model=large

# grid search with multirun
python main.py --multirun model.gnn_hidden_dim=64,96,128 training.lr=0.001,0.0005
"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from asset_prediction.src.data_loader import prepare_all_data
from asset_prediction.src.data_preparation import seed_everything, get_best_device


def _resolve_device(device_str: str) -> torch.device:
    """Convert config device string to torch.device."""
    if device_str == "auto":
        return get_best_device()
    return torch.device(device_str)


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    # Print the full resolved config at the start of every run
    print("=" * 80)
    print("CONFIG")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Seed & device
    # -------------------------------------------------------------------------
    seed_everything(cfg.training.seed)
    device = _resolve_device(cfg.training.device)

    # -------------------------------------------------------------------------
    # Data preparation
    # Every parameter comes from cfg — nothing is hardcoded below this line.
    # -------------------------------------------------------------------------
    data = prepare_all_data(
        data_dir=cfg.data.data_dir,
        device=device,
        n_splits=cfg.data.n_splits,
        embargo=cfg.data.embargo,
        seed=cfg.training.seed,
        # feature params
        feature_window=cfg.data.feature_window,
        target_horizon=cfg.data.target_horizon,
        lookbacks=tuple(cfg.data.lookbacks),
        vol_window=cfg.data.vol_window,
        # lag params
        lag_configs=list(cfg.data.lag_configs),
        # data-source params
        target_returns_filename=cfg.data.target_returns_filename,
    )

    fold_data_list           = data["fold_data_list"]
    lag_lookup               = data["lag_lookup"]
    edge_index_vendor        = data["edge_index_vendor"]
    returns                  = data["returns"]
    stocks                   = data["stocks"]
    sample_time_indices      = data["sample_time_indices"]
    num_price_features       = data["num_price_features"]
    num_fundamental_features = data["num_fundamental_features"]

    print(f"\nPrice features  : {num_price_features}")
    print(f"Fund. features  : {num_fundamental_features}")
    print(f"Stocks          : {len(stocks)}")
    print(f"Folds           : {len(fold_data_list)}")

    # -------------------------------------------------------------------------
    # Model, training and evaluation go here — cfg.model / cfg.training
    # will be passed to whatever you build in model.py / trainer.py
    # -------------------------------------------------------------------------
    # TODO: build model
    # TODO: train per fold
    # TODO: evaluate


if __name__ == "__main__":
    main()
