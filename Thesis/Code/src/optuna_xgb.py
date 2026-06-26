from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


def tune_xgb_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_trials: int,
    k_folds: int,
    out_path: Path,
) -> dict:
    """Tune XGBRegressor via Optuna with K-Fold CV on trainval data."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    sampler = optuna.samplers.TPESampler(seed=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2500),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "tree_method": "hist",
            "random_state": seed,
            "eval_metric": "mae",
        }

        maes = []
        for tr_idx, va_idx in kf.split(X):
            model = XGBRegressor(**params)
            model.fit(X[tr_idx], y[tr_idx], verbose=False)
            maes.append(mean_absolute_error(y[va_idx], model.predict(X[va_idx])))
        return float(np.mean(maes))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_trial.params
    best_params.update({"tree_method": "hist", "random_state": seed, "eval_metric": "mae"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"best_value": float(study.best_value), "best_params": best_params}, f, indent=2)

    return best_params
