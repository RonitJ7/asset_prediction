#!/usr/bin/env python3
"""
xgb.py
------
Sklearn-based XGBoost classifier for binary return-direction prediction.

Wraps XGBClassifier in a Pipeline.
Handles flattening [S, N, F] → [S*N, F] for training and
reshaping predictions back to [S_test, N].
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig
from xgboost import XGBClassifier


def build_xgb_pipeline(cfg: DictConfig, seed: int = 42) -> Pipeline:
    """
    Create a StandardScaler + XGBClassifier pipeline from config.

    Reads cfg.model.xgb.* for model params.
    """
    xgb_cfg = cfg.model.xgb
    xgb = XGBClassifier(
        n_estimators=int(xgb_cfg.n_estimators),
        learning_rate=float(xgb_cfg.learning_rate),
        max_depth=int(xgb_cfg.max_depth),
        subsample=float(xgb_cfg.subsample),
        colsample_bytree=float(xgb_cfg.colsample_bytree),
        reg_lambda=float(xgb_cfg.reg_lambda),
        min_child_weight=float(xgb_cfg.min_child_weight),
        gamma=float(xgb_cfg.gamma),
        reg_alpha=float(xgb_cfg.reg_alpha),
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
        random_state=seed,
    )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", xgb),
    ])


def train_and_predict(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Train the pipeline and return test predictions.

    Parameters
    ----------
    pipeline : sklearn Pipeline (from build_xgb_pipeline)
    X_train  : [S_train, N, F]
    y_train  : [S_train, N]  (raw returns — binarised internally)
    X_test   : [S_test, N, F]

    Returns
    -------
    test_preds : [S_test, N]  P(up) probabilities
    """
    S_train, N, F = X_train.shape
    S_test = X_test.shape[0]

    X_train_flat = X_train.reshape(S_train * N, F)
    y_train_flat = (y_train > 0).astype(np.int32).reshape(S_train * N)
    X_test_flat = X_test.reshape(S_test * N, F)

    pipeline.fit(X_train_flat, y_train_flat)

    proba_flat = pipeline.predict_proba(X_test_flat)[:, 1]
    return proba_flat.reshape(S_test, N)
