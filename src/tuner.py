#!/usr/bin/env python3
"""
tuner.py
--------
Unified Optuna tuner for the asset_prediction pipeline.

What it does:
- Chooses model type: mlp | rf | xgb
- Samples model-specific hyperparameters
- Optionally samples shared pipeline params (e.g., use_gnn, feature windows)
- Runs `src/main.py` as a subprocess with Hydra overrides
- Maximizes `Overall Sharpe` parsed from stdout

Usage examples
--------------
python3 src/tuner.py --n-trials 30
python3 src/tuner.py --n-trials 80 --study-name all_models --storage sqlite:///outputs/optuna_all.db
python3 src/tuner.py --n-trials 40 --n-splits 3 --transaction-cost-bps 25 --enable-gnn-search
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import optuna
from omegaconf import OmegaConf

SHARPE_REGEX = re.compile(r"Overall Sharpe\s*:\s*([-+]?\d*\.?\d+)")


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _sample_trial_params(trial: optuna.Trial, enable_gnn_search: bool) -> Dict[str, object]:
    params: Dict[str, object] = {}

    params["model"] = trial.suggest_categorical("model", ["mlp", "rf", "xgb"])

    if enable_gnn_search:
        params["use_gnn"] = trial.suggest_categorical("use_gnn", [False, True])
        params["corr_threshold"] = trial.suggest_float("corr_threshold", 0.20, 0.60, step=0.05)
        params["lag_steps"] = trial.suggest_int("lag_steps", 1, 3)
        params["gnn_hidden_dim"] = trial.suggest_categorical("gnn_hidden_dim", [64, 96, 128, 160])
        params["gnn_dropout"] = trial.suggest_float("gnn_dropout", 0.0, 0.4, step=0.05)
    else:
        params["use_gnn"] = False

    params["feature_window"] = trial.suggest_categorical("feature_window", [10, 15, 20, 30])
    params["lookbacks"] = trial.suggest_categorical(
        "lookbacks",
        ["[5,10]", "[5,10,20]", "[10,20]"],
    )
    params["top_k"] = trial.suggest_categorical("top_k", [20, 30, 40, 50])

    model = params["model"]
    if model == "mlp":
        params["mlp_layers"] = trial.suggest_categorical(
            "mlp_layers", ["[32,32]", "[64,64]", "[128,64]", "[128,128]"]
        )
        params["mlp_alpha"] = trial.suggest_float("mlp_alpha", 1e-6, 1e-2, log=True)
        params["mlp_lr"] = trial.suggest_float("mlp_lr", 1e-4, 5e-3, log=True)
        params["mlp_max_iter"] = trial.suggest_categorical("mlp_max_iter", [100, 150, 200, 300])

    elif model == "rf":
        params["rf_n_estimators"] = trial.suggest_int("rf_n_estimators", 200, 1000, step=100)
        params["rf_max_depth"] = trial.suggest_categorical("rf_max_depth", [3, 4, 5, 6, 8, 10, 12])
        params["rf_min_samples_split"] = trial.suggest_categorical("rf_min_samples_split", [2, 5, 10, 20])
        params["rf_min_samples_leaf"] = trial.suggest_categorical("rf_min_samples_leaf", [1, 2, 5, 10])
        params["rf_max_features"] = trial.suggest_categorical("rf_max_features", ["sqrt", "log2"])

    else:
        params["xgb_n_estimators"] = trial.suggest_int("xgb_n_estimators", 200, 1200, step=100)
        params["xgb_learning_rate"] = trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True)
        params["xgb_max_depth"] = trial.suggest_int("xgb_max_depth", 2, 8)
        params["xgb_subsample"] = trial.suggest_float("xgb_subsample", 0.5, 1.0)
        params["xgb_colsample_bytree"] = trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0)
        params["xgb_reg_lambda"] = trial.suggest_float("xgb_reg_lambda", 1e-4, 50.0, log=True)
        params["xgb_min_child_weight"] = trial.suggest_float("xgb_min_child_weight", 1.0, 30.0, log=True)
        params["xgb_gamma"] = trial.suggest_float("xgb_gamma", 0.0, 5.0)
        params["xgb_reg_alpha"] = trial.suggest_float("xgb_reg_alpha", 1e-6, 5.0, log=True)

    return params


def _build_overrides(
    params: Dict[str, object],
    seed: int,
    n_splits: int,
    transaction_cost_bps: float,
) -> List[str]:
    model = str(params["model"])

    overrides = [
        f"seed={seed}",
        f"model.selected_model={model}",
        f"use_gnn={_bool_str(bool(params['use_gnn']))}",
        f"data.n_splits={int(n_splits)}",
        f"data.feature_window={int(params['feature_window'])}",
        f"data.lookbacks={params['lookbacks']}",
        f"backtester.top_k={int(params['top_k'])}",
        f"backtester.transaction_cost_bps={float(transaction_cost_bps)}",
    ]

    if bool(params["use_gnn"]):
        overrides.extend(
            [
                f"data.corr_threshold={float(params['corr_threshold'])}",
                f"model.lag_steps={int(params['lag_steps'])}",
                f"model.hidden_dim={int(params['gnn_hidden_dim'])}",
                f"model.dropout={float(params['gnn_dropout'])}",
            ]
        )

    if model == "mlp":
        overrides.extend(
            [
                f"model.mlp.layers={params['mlp_layers']}",
                f"model.mlp.alpha={float(params['mlp_alpha'])}",
                f"model.mlp.learning_rate_init={float(params['mlp_lr'])}",
                f"model.mlp.max_iter={int(params['mlp_max_iter'])}",
            ]
        )
    elif model == "rf":
        max_features = params["rf_max_features"]
        max_features_value = "null" if max_features is None else str(max_features)
        overrides.extend(
            [
                f"model.rf.n_estimators={int(params['rf_n_estimators'])}",
                f"model.rf.max_depth={int(params['rf_max_depth'])}",
                f"model.rf.min_samples_split={int(params['rf_min_samples_split'])}",
                f"model.rf.min_samples_leaf={int(params['rf_min_samples_leaf'])}",
                f"model.rf.max_features={max_features_value}",
            ]
        )
    else:
        overrides.extend(
            [
                f"model.xgb.n_estimators={int(params['xgb_n_estimators'])}",
                f"model.xgb.learning_rate={float(params['xgb_learning_rate'])}",
                f"model.xgb.max_depth={int(params['xgb_max_depth'])}",
                f"model.xgb.subsample={float(params['xgb_subsample'])}",
                f"model.xgb.colsample_bytree={float(params['xgb_colsample_bytree'])}",
                f"model.xgb.reg_lambda={float(params['xgb_reg_lambda'])}",
                f"model.xgb.min_child_weight={float(params['xgb_min_child_weight'])}",
                f"model.xgb.gamma={float(params['xgb_gamma'])}",
                f"model.xgb.reg_alpha={float(params['xgb_reg_alpha'])}",
            ]
        )

    return overrides


def _run_trial_command(
    repo_root: Path,
    overrides: List[str],
    timeout_sec: int,
    stream_logs: bool = False,
) -> Tuple[float, str]:
    cmd = [sys.executable, "src/main.py", *overrides]
    if stream_logs:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines: List[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            output_lines.append(line)
        proc.wait(timeout=timeout_sec)
        combined = "".join(output_lines)
        if proc.returncode != 0:
            raise RuntimeError(
                f"main.py failed with code {proc.returncode}\n"
                f"full_output:\n{combined}"
            )
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )

        combined = f"{proc.stdout}\n{proc.stderr}"
        if proc.returncode != 0:
            raise RuntimeError(
                f"main.py failed with code {proc.returncode}\n"
                f"full_output:\n{combined}"
            )

    match = SHARPE_REGEX.search(combined)
    if not match:
        raise ValueError("Could not parse 'Overall Sharpe' from main.py output")

    return float(match.group(1)), combined


def _load_tuner_config(config_path: str) -> Dict[str, object]:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = repo_root / cfg_path

    if not cfg_path.exists():
        return {}

    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if not isinstance(cfg, dict):
        return {}
    return cfg


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="configs/tuner/default.yaml",
        help="Path to tuner yaml config",
    )
    pre_args, _ = pre_parser.parse_known_args()
    cfg = _load_tuner_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Unified Optuna tuner for mlp/rf/xgb")
    parser.add_argument(
        "--config",
        type=str,
        default=pre_args.config,
        help="Path to tuner yaml config",
    )
    parser.add_argument("--n-trials", type=int, default=int(cfg.get("n_trials", 50)), help="Number of Optuna trials")
    parser.add_argument("--timeout-sec", type=int, default=int(cfg.get("timeout_sec", 0)), help="Study timeout in seconds (0 disables)")
    parser.add_argument("--trial-timeout-sec", type=int, default=int(cfg.get("trial_timeout_sec", 3600)), help="Timeout per trial subprocess")
    parser.add_argument("--study-name", type=str, default=str(cfg.get("study_name", "all_models_tuning")), help="Optuna study name")
    parser.add_argument(
        "--storage",
        type=str,
        default=str(cfg.get("storage", "sqlite:///outputs/optuna_tuning.db")),
        help="Optuna storage URL",
    )
    parser.add_argument("--seed", type=int, default=int(cfg.get("seed", 42)), help="Global seed passed into main.py")
    parser.add_argument("--n-splits", type=int, default=int(cfg.get("n_splits", 3)), help="CV splits for tuning runs")
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=float(cfg.get("transaction_cost_bps", 25.0)),
        help="Backtest transaction cost used while tuning",
    )
    parser.add_argument(
        "--enable-gnn-search",
        action="store_true",
        default=bool(cfg.get("enable_gnn_search", False)),
        help="If set, also tune GNN on/off + related GNN hyperparameters",
    )
    parser.add_argument(
        "--show-main-logs",
        action="store_true",
        default=bool(cfg.get("show_main_logs", False)),
        help="Print full main.py output for each trial",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    (repo_root / "outputs").mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        sampled = _sample_trial_params(trial, enable_gnn_search=bool(args.enable_gnn_search))
        overrides = _build_overrides(
            sampled,
            seed=int(args.seed),
            n_splits=int(args.n_splits),
            transaction_cost_bps=float(args.transaction_cost_bps),
        )

        print(f"\n[Trial {trial.number}] starting...")
        print(f"[Trial {trial.number}] Overrides: {overrides}")

        try:
            sharpe, output = _run_trial_command(
                repo_root=repo_root,
                overrides=overrides,
                timeout_sec=int(args.trial_timeout_sec),
                stream_logs=bool(args.show_main_logs),
            )
            trial.set_user_attr("overrides", overrides)
            if args.show_main_logs and not output.strip().startswith("================================================================================"):
                print("\n" + "-" * 80)
                print(f"[Trial {trial.number}] Overrides: {overrides}")
                print(output)
                print("-" * 80 + "\n")
            print(
                f"[Trial {trial.number}] sharpe={sharpe:.4f} model={sampled['model']} "
                f"use_gnn={sampled['use_gnn']}"
            )
            return float(sharpe)
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            print(f"[Trial {trial.number}] failed: {exc}")
            return -1e9

    print("=" * 80)
    print("Starting unified tuning")
    print(f"Study name   : {args.study_name}")
    print(f"Storage      : {args.storage}")
    print(f"Trials       : {args.n_trials}")
    print(f"CV splits    : {args.n_splits}")
    print(f"Tune GNN     : {args.enable_gnn_search}")
    print("=" * 80)

    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=None if int(args.timeout_sec) <= 0 else int(args.timeout_sec),
        show_progress_bar=True,
    )

    print("\n" + "=" * 80)
    print("BEST RESULT")
    print("=" * 80)
    print(f"Best Sharpe : {study.best_value:.6f}")
    print(f"Best Trial  : {study.best_trial.number}")
    print("Best Params:")
    print(json.dumps(study.best_trial.params, indent=2, sort_keys=True))

    best_overrides = study.best_trial.user_attrs.get("overrides", [])
    if best_overrides:
        print("\nHydra command:")
        print(" ".join(["python3 src/main.py", *best_overrides]))


if __name__ == "__main__":
    main()
