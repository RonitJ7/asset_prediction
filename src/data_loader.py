#!/usr/bin/env python3
"""
data_loader.py
--------------
High-level dataset construction:  feature creation, fold building, and the
one-call `prepare_all_data` orchestrator.

Lower-level helpers (loaders, graph builders, scalers, etc.) live in
data_preparation.py and are imported here.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

from asset_prediction.src.data_preparation import (
    # constants
    BASE_SEED,
    EMBARGO,
    EPS,
    FEATURE_WINDOW,
    LAG_CONFIGS_TO_TEST,
    LOOKBACKS,
    N_SPLITS,
    TARGET_HORIZON,
    # dataclass
    FoldData,
    # helpers
    _ewma_last_vectorized,
    _rsi_from_returns_window,
    build_fundamental_tensor,
    build_lag_lookup,
    build_vendor_edge_index,
    fill_nan_with_medians,
    get_best_device,
    load_fundamental_data,
    load_returns,
    load_vendor_relations,
    log_tscv_fold_date_ranges,
    seed_everything,
)


# ---------------------------------------------------------------------------
# 1. Feature creation
# ---------------------------------------------------------------------------

def create_features(
    returns: pd.DataFrame,
    target_returns: pd.DataFrame,
    fundamental_tensor: Optional[np.ndarray] = None,
    feature_window: int = FEATURE_WINDOW,
    target_horizon: int = TARGET_HORIZON,
    lookbacks: Tuple[int, ...] = LOOKBACKS,
    vol_window: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], Dict[int, int]]:
    """
    Build the sample-level dataset from raw returns.

    Parameters
    ----------
    returns          : daily returns DataFrame [T, N]
    target_returns   : forward-return target DataFrame [T, N]
    fundamental_tensor : optional [T, N, F_fund] array
    feature_window   : number of past days used as raw lag features
    target_horizon   : how many days ahead the target is (used to skip the last N rows)
    lookbacks        : sub-windows over which rolling stats are computed
    vol_window       : rolling window for realised-volatility calculation

    Returns
    -------
    X                  : np.ndarray  [S, N, F_price]
    y                  : np.ndarray  [S, N]
    vol_arr            : np.ndarray  [S, N]
    fundamentals_array : np.ndarray  [S, N, F_fund]  (empty last dim if unused)
    sample_time_indices: list[int]   (maps sample → original time index t)
    time_to_sample_idx : dict[int, int]
    """
    volatilities_df = returns.rolling(window=vol_window).std()
    volatilities_df = volatilities_df.fillna(volatilities_df.mean())

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    vol_list: List[np.ndarray] = []
    fund_list: List[np.ndarray] = []
    sample_time_indices: List[int] = []
    time_to_sample_idx: Dict[int, int] = {}

    for t in range(feature_window, len(returns) - target_horizon):
        # --- Price features ---
        window = returns.iloc[t - feature_window : t].values.astype(np.float32)  # [W, N]

        # (A) raw lag features: [N, W]
        lag_feats = window.T

        # (B) engineered rolling stats over sub-windows
        engineered_parts: List[np.ndarray] = []
        for L in lookbacks:
            if L > feature_window:
                continue
            wL = window[-L:, :]
            mu  = wL.mean(axis=0).astype(np.float32)
            sd  = wL.std(axis=0).astype(np.float32)
            ew  = _ewma_last_vectorized(wL, span=L)
            rsi = _rsi_from_returns_window(wL, eps=EPS)
            engineered_parts.extend([mu[:, None], sd[:, None], ew[:, None], rsi[:, None]])

        engineered_feats = (
            np.concatenate(engineered_parts, axis=1).astype(np.float32)
            if engineered_parts
            else np.zeros((lag_feats.shape[0], 0), dtype=np.float32)
        )

        price_features = np.concatenate([lag_feats, engineered_feats], axis=1).astype(np.float32)

        # --- Targets & volatilities ---
        targets = target_returns.iloc[t].values
        vols    = volatilities_df.iloc[t].values

        # --- Fundamentals snapshot ---
        fund_snapshot = fundamental_tensor[t] if fundamental_tensor is not None else None

        # --- Quality gate: skip samples with missing targets or vols ---
        if np.isnan(targets).any() or np.isnan(vols).any():
            continue

        X_list.append(price_features)
        y_list.append(targets)
        vol_list.append(vols)
        if fund_snapshot is not None:
            fund_list.append(fund_snapshot)

        sample_idx = len(X_list) - 1
        sample_time_indices.append(t)
        time_to_sample_idx[t] = sample_idx

    X       = np.array(X_list)
    y       = np.array(y_list)
    vol_arr = np.array(vol_list)

    if fundamental_tensor is not None and fund_list:
        fundamentals_array = np.array(fund_list, dtype=np.float32)
    else:
        n_stocks = X.shape[1] if X.ndim == 3 else 0
        fundamentals_array = np.zeros((len(X_list), n_stocks, 0), dtype=np.float32)

    return X, y, vol_arr, fundamentals_array, sample_time_indices, time_to_sample_idx


# ---------------------------------------------------------------------------
# 2. Fold construction
# ---------------------------------------------------------------------------

def build_folds(
    X: np.ndarray,
    y: np.ndarray,
    fundamentals_array: np.ndarray,
    stocks: List[str],
    device: torch.device,
    vol_arr: np.ndarray,
    n_splits: int = N_SPLITS,
    embargo: int = EMBARGO,
) -> List[FoldData]:
    """
    Build TimeSeriesSplit folds with embargo, RobustScaler, and tensor
    conversion.

    Returns
    -------
    fold_data_list : list[FoldData]
    """
    num_price_features       = X.shape[2]
    num_fundamental_features = fundamentals_array.shape[2]

    tscv            = TimeSeriesSplit(n_splits=n_splits)
    fold_data_list: List[FoldData] = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        train_idx = np.array(train_idx)[:-embargo]   # apply embargo
        test_idx  = np.array(test_idx)

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        F_train_raw, F_test_raw = fundamentals_array[train_idx], fundamentals_array[test_idx]
        y_train, y_test         = y[train_idx], y[test_idx]

        # --- Scale price features (fit on train only) ---
        scaler_price    = RobustScaler()
        X_train_scaled  = scaler_price.fit_transform(
            X_train_raw.reshape(-1, num_price_features)
        ).reshape(X_train_raw.shape)
        X_test_scaled   = scaler_price.transform(
            X_test_raw.reshape(-1, num_price_features)
        ).reshape(X_test_raw.shape)

        # --- Scale fundamental features (if any) ---
        if num_fundamental_features > 0:
            fund_train_flat = F_train_raw.reshape(-1, num_fundamental_features)
            fund_test_flat  = F_test_raw.reshape(-1, num_fundamental_features)
            train_medians   = np.nanmedian(fund_train_flat, axis=0)
            train_medians   = np.where(np.isnan(train_medians), 0.0, train_medians)
            fund_train_fill = fill_nan_with_medians(fund_train_flat, train_medians)
            fund_test_fill  = fill_nan_with_medians(fund_test_flat,  train_medians)
            fund_scaler       = RobustScaler()
            fund_train_scaled = fund_scaler.fit_transform(fund_train_fill).reshape(F_train_raw.shape)
            fund_test_scaled  = fund_scaler.transform(fund_test_fill).reshape(F_test_raw.shape)
            X_train_full = np.concatenate([X_train_scaled, fund_train_scaled], axis=2)
            X_test_full  = np.concatenate([X_test_scaled,  fund_test_scaled],  axis=2)
        else:
            X_train_full = X_train_scaled
            X_test_full  = X_test_scaled

        # --- Convert to tensors ---
        X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32, device=device)
        X_test_tensor  = torch.tensor(X_test_full,  dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train,       dtype=torch.float32, device=device)
        y_test_tensor  = torch.tensor(y_test,        dtype=torch.float32, device=device)

        # --- Volatility slices ---
        vol_train = vol_arr[train_idx]
        vol_test  = vol_arr[test_idx]

        # --- Global index lookup (sample_idx → set + local idx) ---
        global_lookup: Dict[int, Tuple[str, int]] = {}
        for local_idx, g_idx in enumerate(train_idx):
            global_lookup[int(g_idx)] = ("train", local_idx)
        for local_idx, g_idx in enumerate(test_idx):
            global_lookup[int(g_idx)] = ("test", local_idx)

        fold_data_list.append(
            FoldData(
                fold_idx=fold_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                X_train_tensor=X_train_tensor,
                X_test_tensor=X_test_tensor,
                y_train_tensor=y_train_tensor,
                y_test_tensor=y_test_tensor,
                y_train=y_train,
                y_test=y_test,
                vol_train=vol_train,
                vol_test=vol_test,
                global_lookup=global_lookup,
                symbols=list(stocks),
            )
        )

    return fold_data_list


# ---------------------------------------------------------------------------
# 3. Top-level orchestrator
# ---------------------------------------------------------------------------

def prepare_all_data(
    data_dir: str = "processed_data",
    device: Optional[torch.device] = None,
    n_splits: int = N_SPLITS,
    embargo: int = EMBARGO,
    seed: int = BASE_SEED,
    # feature params
    feature_window: int = FEATURE_WINDOW,
    target_horizon: int = TARGET_HORIZON,
    lookbacks: Tuple[int, ...] = LOOKBACKS,
    vol_window: int = 20,
    # lag params
    lag_configs: List[int] = None,
    # data-source params
    target_returns_filename: str = "future_1day_returns.csv",
) -> dict:
    """
    One-call data preparation.  Loads everything, builds features / graphs /
    folds, and returns a dict ready for training.

    Parameters
    ----------
    data_dir                : directory with processed CSV / JSON files
    device                  : torch device (auto-detected if None)
    n_splits                : number of TimeSeriesSplit folds
    embargo                 : samples to drop from end of each train set
    seed                    : global RNG seed
    feature_window          : past days used as raw lag features
    target_horizon          : forward-return horizon (rows to skip at end)
    lookbacks               : sub-windows for rolling-stat features
    vol_window              : rolling window for realised-volatility
    lag_configs             : list of lag steps to pre-compute in lag_lookup
    target_returns_filename : CSV filename for forward returns

    Returns
    -------
    dict with keys:
        fold_data_list           : list[FoldData]
        lag_lookup               : np.ndarray
        edge_index_vendor        : torch.Tensor
        returns                  : pd.DataFrame
        stocks                   : list[str]
        sample_time_indices      : list[int]
        num_price_features       : int
        num_fundamental_features : int
        device                   : torch.device
        volatilities             : np.ndarray  [S, N]
    """
    if lag_configs is None:
        lag_configs = LAG_CONFIGS_TO_TEST

    seed_everything(seed)

    if device is None:
        device = get_best_device()
    print(f"Device: {device}")

    # --- Load raw data ---
    print("\nLoading data...")
    returns, target_returns = load_returns(data_dir, target_returns_filename)
    vendor_json             = load_vendor_relations(data_dir)
    fundamental_df, fund_source = load_fundamental_data(data_dir)
    print(f"Fundamental source: {fund_source}")

    stocks       = returns.columns.tolist()
    stock_to_idx = {s: i for i, s in enumerate(stocks)}

    # --- Fundamental tensor ---
    fundamental_tensor, fund_features = build_fundamental_tensor(
        fundamental_df, returns, stocks
    )
    print(f"Stocks: {len(stocks)}, Time steps: {len(returns)}")
    if fundamental_tensor is not None:
        print(f"Fundamental features ({len(fund_features)}): {fund_features}")
    else:
        print("Fundamental features disabled.")

    # --- Graph edges ---
    edge_index_vendor = build_vendor_edge_index(vendor_json, stock_to_idx, device)
    print(f"Vendor + Competition edges: {edge_index_vendor.shape[1]}")

    # --- Feature engineering ---
    print("\nCreating features...")
    X, y, vol_arr, fundamentals_array, sample_time_indices, time_to_sample_idx = (
        create_features(
            returns,
            target_returns,
            fundamental_tensor,
            feature_window=feature_window,
            target_horizon=target_horizon,
            lookbacks=lookbacks,
            vol_window=vol_window,
        )
    )
    print(f"Dataset: {X.shape[0]} samples, {X.shape[2]} price features")

    num_price_features       = X.shape[2]
    num_fundamental_features = fundamentals_array.shape[2]

    # --- Lag lookup ---
    lag_lookup = build_lag_lookup(
        len(X), sample_time_indices, time_to_sample_idx, lag_configs=lag_configs
    )

    # --- Log fold date ranges ---
    tscv = TimeSeriesSplit(n_splits=n_splits)
    print("\n" + "=" * 80)
    print("FOLD DATE RANGES (embargo applied later)")
    print("=" * 80)
    log_tscv_fold_date_ranges(tscv, returns.index, sample_time_indices, X)

    # --- Build folds ---
    print("\nBuilding folds...")
    fold_data_list = build_folds(
        X, y, fundamentals_array, stocks, device,
        vol_arr=vol_arr, n_splits=n_splits, embargo=embargo,
    )
    print(f"Built {len(fold_data_list)} folds.")

    return {
        "fold_data_list"          : fold_data_list,
        "lag_lookup"              : lag_lookup,
        "edge_index_vendor"       : edge_index_vendor,
        "returns"                 : returns,
        "stocks"                  : stocks,
        "sample_time_indices"     : sample_time_indices,
        "num_price_features"      : num_price_features,
        "num_fundamental_features": num_fundamental_features,
        "device"                  : device,
        "volatilities"            : vol_arr,
    }
