#!/usr/bin/env python3
"""
data_preparation.py
-------------------
Data loading, feature engineering, graph construction, and fold building
for the asset prediction pipeline.

All public functions are designed to be called from main.py.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_SEED = 42
FEATURE_WINDOW = 20       # past days used as features
TARGET_HORIZON = 1       # days ahead to predict
LOOKBACKS = (5, 10)       # sub-windows for engineered features
EPS = 1e-12
LAG_CONFIGS_TO_TEST = [1, 2, 5, 10]
EMBARGO = 15              # gap between train/test to avoid leakage
N_SPLITS = 5              # TimeSeriesSplit folds

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = BASE_SEED, deterministic: bool = False) -> None:
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dataclass shared across the pipeline
# ---------------------------------------------------------------------------

@dataclass
class FoldData:
    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    X_train_tensor: torch.Tensor
    X_test_tensor: torch.Tensor
    y_train_tensor: torch.Tensor
    y_test_tensor: torch.Tensor         
    y_train: np.ndarray
    y_test: np.ndarray
    vol_train: np.ndarray               
    vol_test: np.ndarray
    global_lookup: Dict[int, Tuple[str, int]]
    symbols: List[str]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_best_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fill_nan_with_medians(data: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 2-D array with provided per-column medians."""
    if not np.isnan(data).any():
        return data
    filled = data.copy()
    nan_rows, nan_cols = np.where(np.isnan(filled))
    filled[nan_rows, nan_cols] = medians[nan_cols]
    return filled


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _ewma_last_vectorized(window: np.ndarray, span: int) -> np.ndarray:
    """
    Vectorized EWMA-last across stocks.
    window: [L, N] (time × stocks)
    returns: [N]
    """
    span = int(max(1, span))
    alpha = 2.0 / (span + 1.0)
    v = window[0].astype(np.float32)
    for i in range(1, window.shape[0]):
        v = alpha * window[i].astype(np.float32) + (1.0 - alpha) * v
    return v


def _rsi_from_returns_window(window: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    RSI from returns for each stock, scaled to [0, 1].
    window: [L, N]
    returns: [N]
    """
    gains = np.maximum(window, 0.0)
    losses = np.maximum(-window, 0.0)
    avg_gain = gains.mean(axis=0)
    avg_loss = losses.mean(axis=0)
    rs = avg_gain / (avg_loss + eps)
    rsi_0_1 = 1.0 - (1.0 / (1.0 + rs))
    return rsi_0_1.astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_returns(
    data_dir: str = "processed_data",
    target_returns_filename: str = "future_1day_returns.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load daily returns and forward returns, fill NaNs with 0.

    Parameters
    ----------
    data_dir : str
        Directory containing the CSV files.
    target_returns_filename : str
        Filename for the forward-return target CSV (e.g. future_1day_returns.csv).

    Returns
    -------
    returns : pd.DataFrame   [T, N]
    target_returns : pd.DataFrame   [T, N]
    """
    returns = pd.read_csv(
        Path(data_dir) / "daily_returns.csv", index_col=0, parse_dates=True
    )
    target_returns = pd.read_csv(
        Path(data_dir) / target_returns_filename, index_col=0, parse_dates=True
    )
    returns = returns.fillna(0)
    target_returns = target_returns.fillna(0)
    return returns, target_returns


def load_vendor_relations(
    data_dir: str = "processed_data",
    filename: str = "nifty100_vendor_relations_cleaned.json",
) -> list:
    """Load vendor/supplier/competitor relations from JSON."""
    path = Path(data_dir) / filename
    with open(path, "r") as f:
        vendor_json = json.load(f)
    if not isinstance(vendor_json, list):
        raise ValueError("Vendor relations JSON must be a list of records.")
    return vendor_json


def load_fundamental_data(
    data_dir: str = "processed_data",
    candidates: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Try loading fundamental data from candidate CSV files.

    Returns
    -------
    fundamental_df : pd.DataFrame
    source_name    : str  (filename that was loaded)

    Raises
    ------
    FileNotFoundError if none of the candidates exist.
    """
    if candidates is None:
        candidates = ["fundamental_history.csv", "fundamental_data.csv"]

    for name in candidates:
        path = Path(data_dir) / name
        if path.exists():
            return pd.read_csv(path), name

    raise FileNotFoundError(
        f"No fundamental data found. Checked: {candidates} in {data_dir}"
    )


# ---------------------------------------------------------------------------
# 2. Fundamental tensor construction
# ---------------------------------------------------------------------------

def build_fundamental_tensor(
    fundamental_df: pd.DataFrame,
    returns: pd.DataFrame,
    stocks: List[str],
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Align fundamental data to the returns time axis and produce a 3-D tensor
    [T, N, F_fund] with forward-filled values.

    Returns
    -------
    fundamental_tensor : np.ndarray | None
    fundamental_features : list[str]  (column names)
    """
    df = fundamental_df.copy()

    # --- Harmonise column names ---
    df.columns = [str(c).strip() for c in df.columns]
    if "symbol" not in df.columns:
        for alias in ("stock", "ticker"):
            if alias in df.columns:
                df = df.rename(columns={alias: "symbol"})
                break
    if "symbol" not in df.columns:
        df = df.reset_index().rename(columns={"index": "symbol"})

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df[df["symbol"].isin(stocks)].copy()

    if "timestamp" not in df.columns:
        raise ValueError(
            "Fundamental data must include a 'timestamp' column for time-aware alignment"
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != "timestamp"
    ]
    if not numeric_cols:
        raise ValueError("No numeric fundamental columns available.")

    df = df[["symbol", "timestamp"] + numeric_cols].copy()

    tensor = np.full(
        (len(returns), len(stocks), len(numeric_cols)), np.nan, dtype=np.float32
    )

    for stock_idx, stock in enumerate(stocks):
        stock_vals = (
            df[df["symbol"] == stock].sort_values("timestamp")
        )
        if stock_vals.empty:
            continue
        series = stock_vals.set_index("timestamp")[numeric_cols]
        series = series.apply(pd.to_numeric, errors="coerce")
        series = series[~series.index.duplicated(keep="last")]
        series = series.reindex(columns=numeric_cols)
        aligned = series.reindex(returns.index, method="ffill")
        tensor[:, stock_idx, :] = aligned.values.astype(np.float32)

    has_data = np.isfinite(tensor).any()
    # NOTE: original code force-disables fundamentals;
    # flip the flag below when you want them back.
    has_data = False

    if not has_data:
        return None, []

    return tensor, numeric_cols


# ---------------------------------------------------------------------------
# 3. Graph / edge-index construction
# ---------------------------------------------------------------------------

def build_vendor_edge_index(
    vendor_json: list,
    stock_to_idx: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Parse vendor JSON and build a directed edge_index [2, E] tensor.
    Includes supplier→company, company→client, competitor→company edges.
    """
    edge_set: set = set()

    for rec in vendor_json:
        if not isinstance(rec, dict):
            continue
        company = str(rec.get("ticker", "")).strip()
        if not company or company not in stock_to_idx:
            continue

        rel = rec.get("relationships", {}) or {}
        if not isinstance(rel, dict):
            rel = {}

        company_idx = stock_to_idx[company]

        for s in (rel.get("suppliers", []) or []):
            supplier = str(s).strip()
            if supplier in stock_to_idx:
                edge_set.add((stock_to_idx[supplier], company_idx))

        for c in (rel.get("clients", []) or []):
            client = str(c).strip()
            if client in stock_to_idx:
                edge_set.add((company_idx, stock_to_idx[client]))

        for comp in (rel.get("competitors", []) or []):
            competitor = str(comp).strip()
            if competitor in stock_to_idx:
                edge_set.add((stock_to_idx[competitor], company_idx))

    edges = [list(e) for e in edge_set]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return edge_index.to(device)


def build_corr_edge_index(
    returns: pd.DataFrame,
    stocks: List[str],
    device: torch.device,
    corr_threshold: float,
    time_index: np.ndarray,
) -> torch.Tensor:
    """
    Build correlation edge_index using lag-1 returns to avoid leakage.
    """
    tidx = np.asarray(time_index, dtype=int)
    tidx = tidx[(tidx >= 0) & (tidx < len(returns))]
    returns_lag1 = returns.iloc[tidx].shift(1).fillna(0)
    corr_matrix = returns_lag1.corr()

    corr_edges: List[List[int]] = []
    for i, stock1 in enumerate(stocks):
        for j in range(i + 1, len(stocks)):
            stock2 = stocks[j]
            if abs(float(corr_matrix.loc[stock1, stock2])) > float(corr_threshold):
                corr_edges.append([i, j])
                corr_edges.append([j, i])

    edge_index = (
        torch.tensor(corr_edges, dtype=torch.long).t().contiguous()
        if corr_edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return edge_index.to(device)


# ---------------------------------------------------------------------------
# 4. Lag lookup table
# ---------------------------------------------------------------------------

def build_lag_lookup(
    num_samples: int,
    sample_time_indices: List[int],
    time_to_sample_idx: Dict[int, int],
    lag_configs: List[int] = LAG_CONFIGS_TO_TEST,
) -> np.ndarray:
    """
    Build lag lookup table of shape [num_samples, max_lag + 1].
    Value -1 means "no sample available at that lag".
    """
    max_lag = max(lag_configs)
    lag_lookup = np.full((num_samples, max_lag + 1), -1, dtype=int)

    for sample_idx, time_idx in enumerate(sample_time_indices):
        for lag in range(1, max_lag + 1):
            prev_time = time_idx - lag
            prev_sample = time_to_sample_idx.get(prev_time)
            if prev_sample is not None:
                lag_lookup[sample_idx, lag] = prev_sample

    return lag_lookup


# ---------------------------------------------------------------------------
# 5. Fold construction (log utility)
# ---------------------------------------------------------------------------

def log_tscv_fold_date_ranges(
    tscv: TimeSeriesSplit,
    returns_index: pd.DatetimeIndex,
    sample_time_indices: List[int],
    X: np.ndarray,
    prefix: str = "[FOLDS]",
) -> None:
    """Log train/test date ranges for each fold."""
    n_samples = int(X.shape[0])
    if len(sample_time_indices) != n_samples:
        print(
            f"{prefix} len(sample_time_indices)={len(sample_time_indices)} "
            f"!= n_samples={n_samples}"
        )
        return

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        tr0, tr1 = int(train_idx[0]), int(train_idx[-1])
        te0, te1 = int(test_idx[0]), int(test_idx[-1])

        tr_t0 = int(sample_time_indices[tr0])
        tr_t1 = int(sample_time_indices[tr1])
        te_t0 = int(sample_time_indices[te0])
        te_t1 = int(sample_time_indices[te1])

        print(
            f"{prefix} Fold {fold_idx}: "
            f"train[{tr0}:{tr1}] dates "
            f"{returns_index[tr_t0].date()}→{returns_index[tr_t1].date()} "
            f"(n={len(train_idx)}) | "
            f"test[{te0}:{te1}] dates "
            f"{returns_index[te_t0].date()}→{returns_index[te_t1].date()} "
            f"(n={len(test_idx)})"
        )



