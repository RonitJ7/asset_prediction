#!/usr/bin/env python3
"""
main.py
-------
Entry point for the asset-prediction pipeline.
All parameters are driven by configs/ via Hydra.

Pipeline per fold:
    1. Build correlation edge index (per-fold, to avoid leakage)
    2. Train GNN (regression: ICLoss)
    3. Generate GNN scores on train + test sets
    4. Augment features: [X, gnn_score] → train MLP (classification: BCE)
    5. Evaluate MLP via portfolio construction → metrics

Usage
-----
python main.py
python main.py data.feature_window=30 data.n_splits=3
python main.py seed=0 device=cpu
"""

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from data_loader import prepare_all_data
from data_preparation import seed_everything, get_best_device, build_corr_edge_index
from gnn_model import MultiRelGNN, train_one_fold
from mlp import build_mlp_pipeline, train_and_predict
from backtester import evaluate_predictions_per_fold, get_sharpe_ratio


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return get_best_device()
    return torch.device(device_str)


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("CONFIG")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # -----------------------------------------------------------------
    # Seed & device
    # -----------------------------------------------------------------
    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)

    # -----------------------------------------------------------------
    # Data preparation
    # -----------------------------------------------------------------
    data = prepare_all_data(
        data_dir=cfg.data.data_dir,
        device=device,
        n_splits=cfg.data.n_splits,
        embargo=cfg.data.embargo,
        seed=cfg.seed,
        feature_window=cfg.data.feature_window,
        target_horizon=cfg.data.target_horizon,
        lookbacks=tuple(cfg.data.lookbacks),
        vol_window=cfg.data.vol_window,
        lag_configs=list(cfg.data.lag_configs),
        target_returns_filename=cfg.data.target_returns_filename,
        use_fundamental_data=cfg.data.use_fundamental_data,
    )

    fold_data_list           = data["fold_data_list"]
    lag_lookup               = data["lag_lookup"]
    edge_index_vendor        = data["edge_index_vendor"]
    returns                  = data["returns"]
    stocks                   = data["stocks"]
    sample_time_indices      = data["sample_time_indices"]
    num_price_features       = data["num_price_features"]
    num_fundamental_features = data["num_fundamental_features"]

    total_input_features = num_price_features + num_fundamental_features

    print(f"\nPrice features  : {num_price_features}")
    print(f"Fund. features  : {num_fundamental_features}")
    print(f"Stocks          : {len(stocks)}")
    print(f"Folds           : {len(fold_data_list)}")

    # -----------------------------------------------------------------
    # Per-fold training & evaluation
    # -----------------------------------------------------------------
    all_fold_results = []

    for fold_data in fold_data_list:
        fold_idx = fold_data.fold_idx
        train_dates = returns.index[sample_time_indices[fold_data.train_idx[0]]]
        train_end   = returns.index[sample_time_indices[fold_data.train_idx[-1]]]
        test_dates  = returns.index[sample_time_indices[fold_data.test_idx[0]]]
        test_end    = returns.index[sample_time_indices[fold_data.test_idx[-1]]]

        print("\n" + "=" * 80)
        print(f"FOLD {fold_idx}")
        print(f"  Train: {train_dates.date()} → {train_end.date()}  ({len(fold_data.train_idx)} samples)")
        print(f"  Test:  {test_dates.date()} → {test_end.date()}  ({len(fold_data.test_idx)} samples)")
        print("=" * 80)

        # -------------------------------------------------------------
        # Step 1: Build correlation edges (per-fold, using train only)
        # -------------------------------------------------------------
        train_time_idx = np.array([sample_time_indices[i] for i in fold_data.train_idx])
        edge_index_corr = build_corr_edge_index(
            returns, stocks, device,
            corr_threshold=cfg.data.corr_threshold,
            time_index=train_time_idx,
        )
        print(f"  Correlation edges: {edge_index_corr.shape[1]}")

        # -------------------------------------------------------------
        # Step 2: Train GNN
        # -------------------------------------------------------------
        print(f"\n  [GNN] Training ({cfg.training.epochs} epochs)...")
        seed_everything(cfg.seed)
        gnn_model = MultiRelGNN(
            num_price_features=num_price_features,
            num_fundamental_features=num_fundamental_features,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
            lag_steps=cfg.model.lag_steps,
            activation=cfg.model.activation,
        ).to(device)

        gnn_history = train_one_fold(
            model=gnn_model,
            fold_data=fold_data,
            lag_lookup=lag_lookup,
            edge_index_corr=edge_index_corr,
            edge_index_vendor=edge_index_vendor,
            cfg=cfg,
            device=device,
        )

        # -------------------------------------------------------------
        # Step 3: Generate GNN scores → augmented features
        # -------------------------------------------------------------
        print(f"\n  [GNN] Generating scores for train + test sets...")
        lag_steps = int(cfg.model.lag_steps)
        global_to_train = {int(g): int(i) for i, g in enumerate(fold_data.train_idx)}
        # Also index test samples for lag resolution during test scoring
        global_to_test = {int(g): int(i) for i, g in enumerate(fold_data.test_idx)}

        def _score_set(X_tensor, indices, allow_test=False):
            scores = []
            with torch.no_grad():
                for i in range(X_tensor.shape[0]):
                    x = X_tensor[i]  # [N, F]
                    lag_global = int(lag_lookup[int(indices[i]), lag_steps]) if lag_steps > 0 else -1

                    if lag_global != -1:
                        local = global_to_train.get(lag_global)
                        if local is not None:
                            x_lagged = fold_data.X_train_tensor[local]
                        elif allow_test and lag_global in global_to_test:
                            x_lagged = fold_data.X_test_tensor[global_to_test[lag_global]]
                        else:
                            x_lagged = x
                    else:
                        x_lagged = x

                    scores.append(gnn_model(x, edge_index_corr, edge_index_vendor, x_lagged).unsqueeze(0))
            return torch.cat(scores, dim=0).unsqueeze(-1)  # [S, N, 1]

        gnn_model.eval()
        gnn_train_scores = _score_set(fold_data.X_train_tensor, fold_data.train_idx)
        gnn_test_scores = _score_set(fold_data.X_test_tensor, fold_data.test_idx, allow_test=True)

        # Augment fold tensors with GNN score as extra feature
        X_train_aug = torch.cat([fold_data.X_train_tensor, gnn_train_scores], dim=2)
        X_test_aug = torch.cat([fold_data.X_test_tensor, gnn_test_scores], dim=2)

        # Convert to numpy for sklearn: [S, N, F+1]
        X_train_np = X_train_aug.cpu().numpy()
        X_test_np = X_test_aug.cpu().numpy()

        # -------------------------------------------------------------
        # Step 4: Train sklearn MLP on augmented features
        # -------------------------------------------------------------
        F_aug = X_train_np.shape[2]
        mlp_cfg = cfg.model.mlp
        print(f"\n  [MLP] Training sklearn MLPClassifier (input_dim={F_aug}, "
              f"hidden={tuple(mlp_cfg.layers)}, max_iter={mlp_cfg.max_iter})...")

        mlp_pipeline = build_mlp_pipeline(cfg, seed=cfg.seed)
        test_preds = train_and_predict(mlp_pipeline, X_train_np, fold_data.y_train, X_test_np)

        # -------------------------------------------------------------
        # Step 5: Evaluate
        # -------------------------------------------------------------
        print(f"\n  [EVAL] Portfolio evaluation:")
        fold_results = evaluate_predictions_per_fold(
            predictions=test_preds,
            fold_data=fold_data,
            top_k=cfg.backtester.top_k,
            softmax_temp=cfg.backtester.softmax_temp,
            transaction_cost_bps=cfg.backtester.transaction_cost_bps,
        )

        fold_result = {
            'fold': fold_idx,
            'sharpe': fold_results['sharpe_ratio'],
            'avg_return': fold_results['avg_return'],
            'avg_ic': fold_results['avg_spearman_ic'],
            'avg_hit_rate': fold_results['avg_hit_rate'],
            'avg_turnover': fold_results['avg_turnover'],
            'max_drawdown': fold_results['max_drawdown_pct']/100.0,
            'fold_returns': fold_results['fold_returns'],
        }
        all_fold_results.append(fold_result)

    # -----------------------------------------------------------------
    # Aggregate results across folds
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    all_returns = []
    for r in all_fold_results:
        all_returns.extend(r['fold_returns'])

    avg_sharpe = np.mean([r['sharpe'] for r in all_fold_results])
    avg_ic = np.nanmean([r['avg_ic'] for r in all_fold_results])
    avg_hit = np.mean([r['avg_hit_rate'] for r in all_fold_results])
    avg_ret = np.mean([r['avg_return'] for r in all_fold_results])
    avg_dd = np.mean([r['max_drawdown'] for r in all_fold_results])
    avg_sharpe = np.mean([r['sharpe'] for r in all_fold_results])
    overall_sharpe = get_sharpe_ratio(all_returns, 252)

    print(f"Overall Sharpe      : {overall_sharpe:.4f}")
    print(f"Avg Sharpe (per-fold) : {avg_sharpe:.4f}")
    print(f"Avg IC              : {avg_ic:.4f}")
    print(f"Avg Hit Rate        : {avg_hit:.4f}")
    print(f"Avg Daily Return    : {avg_ret:.6f}")
    print(f"Avg Max Drawdown    : {avg_dd*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
