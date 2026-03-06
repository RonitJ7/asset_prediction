#!/usr/bin/env python3
"""
mlp.py
------
Sklearn-based MLP classifier for binary return-direction prediction.

Wraps MLPClassifier in a Pipeline with StandardScaler.
Handles flattening [S, N, F] → [S*N, F] for training and
reshaping predictions back to [S_test, N].
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig


def build_mlp_pipeline(cfg: DictConfig, seed: int = 42) -> Pipeline:
    """
    Create a StandardScaler + MLPClassifier pipeline from config.

    Reads cfg.model.mlp.* for architecture params.
    """
    mlp_cfg = cfg.model.mlp
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=tuple(mlp_cfg.layers),
            activation=mlp_cfg.activation,
            solver="adam",
            alpha=float(mlp_cfg.alpha),
            learning_rate_init=float(mlp_cfg.learning_rate_init),
            max_iter=int(mlp_cfg.max_iter),
            early_stopping=True,
            n_iter_no_change=5,
            validation_fraction=0.1,
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
    pipeline : sklearn Pipeline (from build_mlp_pipeline)
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
