#!/usr/bin/env python3
"""
rf.py
-----
Sklearn-based Random Forest classifier for binary return-direction prediction.

Wraps RandomForestClassifier in a Pipeline.
Handles flattening [S, N, F] → [S*N, F] for training and
reshaping predictions back to [S_test, N].
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from omegaconf import DictConfig


def build_rf_pipeline(cfg: DictConfig, seed: int = 42) -> Pipeline:
    """
    Create a RandomForestClassifier pipeline from config.

    Reads cfg.model.rf.* for model params.
    """
    rf_cfg = cfg.model.rf
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=int(rf_cfg.n_estimators),
            max_depth=None if rf_cfg.max_depth is None else int(rf_cfg.max_depth),
            min_samples_split=int(rf_cfg.min_samples_split),
            min_samples_leaf=int(rf_cfg.min_samples_leaf),
            max_features=rf_cfg.max_features,
            n_jobs=int(rf_cfg.n_jobs),
            random_state=seed,
        )),
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
    pipeline : sklearn Pipeline (from build_rf_pipeline)
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
