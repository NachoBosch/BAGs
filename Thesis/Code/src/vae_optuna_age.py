from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import gc
import numpy as np
import optuna
import tensorflow as tf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .vae_train import build_vae
from .vae_callbacks import BetaScheduler


@dataclass(frozen=True)
class OptunaSearchSpace:
    latent_dim_choices: tuple[int, ...] = (16, 32, 64, 128, 224)
    hidden_dims_keys: tuple[str, ...] = (
        "256", "512", "512_256", "1024_512", "1024_512_256",
    )
    beta_min: float = 1e-4
    beta_max: float = 2e-1
    lr_min: float = 1e-4
    lr_max: float = 3e-3
    drop_min: float = 0.0
    drop_max: float = 0.20
    warmup_min: int = 10
    warmup_max: int = 100
    embedding_kinds: tuple[str, ...] = ("mu", "z")


_HIDDEN_DIMS_MAP: dict[str, list[int]] = {
    "256": [256],
    "512": [512],
    "512_256": [512, 256],
    "1024_512": [1024, 512],
    "1024_512_256": [1024, 512, 256],
}


def _dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _build_feats(
    Z: np.ndarray | None,
    T1: np.ndarray | None,
    sex: np.ndarray | None,
    diag: np.ndarray | None,
    *,
    use_Z: bool,
    use_T1: bool,
    use_sex: bool,
    use_diag: bool,
) -> np.ndarray:
    """Assemble feature matrix from optional blocks."""
    parts: list[np.ndarray] = []
    for arr, use, is_1d in [
        (T1, use_T1, False), (Z, use_Z, False),
        (sex, use_sex, True), (diag, use_diag, True),
    ]:
        if use:
            assert arr is not None, f"Feature block enabled but array is None"
            a = np.asarray(arr, dtype=np.float32)
            parts.append(a.reshape(-1, 1) if is_1d else a)

    if not parts:
        raise ValueError("At least one feature block must be enabled.")
    X = np.concatenate(parts, axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def folds_ids_to_indices(
    trainval_ids: list[str],
    folds: list[dict],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert fold ID lists to integer index arrays for array slicing."""
    id_to_idx = {rid: i for i, rid in enumerate(trainval_ids)}
    return [
        (
            np.array([id_to_idx[r] for r in f["train_ids"]], dtype=int),
            np.array([id_to_idx[r] for r in f["val_ids"]], dtype=int),
        )
        for f in folds
    ]


def run_vae_optuna_for_age(
    *,
    X_trainval_fc: np.ndarray,
    y_trainval: np.ndarray,
    T1_trainval: np.ndarray,
    sex_trainval: np.ndarray | None,
    diag_trainval: np.ndarray | None,
    folds_idx: list[tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    seed: int,
    n_trials: int,
    vae_fixed: dict,
    xgb_fixed_params: dict,
    use_sex: bool = False,
    use_diag: bool = False,
    use_t1w: bool = True,
    search_space: OptunaSearchSpace = OptunaSearchSpace(),
    study_name: str = "vae_pipe",
    storage_path: Path | None = None,
) -> dict:
    """Tune VAE hyperparameters to minimize downstream brain-age MAE via CV.

    The VAE is trained on FC only; embeddings are evaluated jointly with T1w
    features through a frozen XGBoost evaluator.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    xgb_fixed_params = dict(xgb_fixed_params)
    xgb_fixed_params["random_state"] = int(seed)
    xgb_fixed_params.setdefault("eval_metric", "mae")
    xgb_fixed_params.setdefault("tree_method", "hist")
    xgb_fixed_params.setdefault("verbosity", 0)
    xgb_fixed_params.setdefault("n_jobs", 1)

    input_dim = int(X_trainval_fc.shape[1])

    if storage_path is None:
        storage_path = out_dir / f"{study_name}.db"
    storage_url = f"sqlite:///{storage_path}"

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    def _cleanup_tf() -> None:
        tf.keras.backend.clear_session()
        gc.collect()

    def objective(trial: optuna.Trial) -> float:
        latent_dim = int(trial.suggest_categorical("latent_dim", search_space.latent_dim_choices))
        hd_key = str(trial.suggest_categorical("hidden_dims", search_space.hidden_dims_keys))
        hidden_dims = _HIDDEN_DIMS_MAP[hd_key]
        beta_target = float(trial.suggest_float("beta_target", search_space.beta_min, search_space.beta_max, log=True))
        lr = float(trial.suggest_float("lr", search_space.lr_min, search_space.lr_max, log=True))
        drop_rate = float(trial.suggest_float("drop_rate", search_space.drop_min, search_space.drop_max))
        warmup_ep = int(trial.suggest_int("warmup_ep", search_space.warmup_min, search_space.warmup_max))
        embedding_kind = str(trial.suggest_categorical("embedding_kind", search_space.embedding_kinds))

        fold_maes: list[float] = []

        for fold_i, (tr_idx, va_idx) in enumerate(folds_idx):
            _cleanup_tf()

            X_tr_fc = X_trainval_fc[tr_idx]
            X_va_fc = X_trainval_fc[va_idx]

            vae = build_vae(
                input_dim,
                hidden_dims=hidden_dims, latent_dim=latent_dim,
                l2_reg=float(vae_fixed["l2_reg"]), drop_rate=drop_rate,
                activation=str(vae_fixed["activation"]),
                norm_kind=str(vae_fixed["norm_kind"]),
                beta_target=beta_target, recon_kind=str(vae_fixed["recon_kind"]),
                lr=lr, clipnorm=float(vae_fixed["clipnorm"]),
            )

            cbs = [
                BetaScheduler(target_beta=beta_target, warmup_epochs=warmup_ep),
                EarlyStopping(monitor="val_recon_loss", mode="min",
                              patience=int(vae_fixed["patience"]),
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_recon_loss", mode="min",
                                  factor=0.5, patience=10, min_lr=1e-5, verbose=0),
            ]

            vae.fit(
                X_tr_fc, epochs=int(vae_fixed["max_epochs"]),
                batch_size=int(vae_fixed["batch_size"]), shuffle=True,
                validation_data=(X_va_fc, None), callbacks=cbs, verbose=0,
            )

            mu_tr, _, z_tr = vae.encoder.predict(X_tr_fc, verbose=0)
            mu_va, _, z_va = vae.encoder.predict(X_va_fc, verbose=0)

            Z_tr = (mu_tr if embedding_kind == "mu" else z_tr).astype(np.float32)
            Z_va = (mu_va if embedding_kind == "mu" else z_va).astype(np.float32)

            T1_tr = T1_trainval[tr_idx]
            T1_va = T1_trainval[va_idx]
            sex_tr = sex_trainval[tr_idx] if sex_trainval is not None else None
            sex_va = sex_trainval[va_idx] if sex_trainval is not None else None
            diag_tr = diag_trainval[tr_idx] if diag_trainval is not None else None
            diag_va = diag_trainval[va_idx] if diag_trainval is not None else None

            Xtr = _build_feats(Z_tr, T1_tr, sex_tr, diag_tr,
                               use_Z=True, use_T1=use_t1w, use_sex=use_sex, use_diag=use_diag)
            Xva = _build_feats(Z_va, T1_va, sex_va, diag_va,
                               use_Z=True, use_T1=use_t1w, use_sex=use_sex, use_diag=use_diag)

            model = XGBRegressor(**xgb_fixed_params)
            model.fit(Xtr, y_trainval[tr_idx].astype(float), verbose=False)
            mae = float(mean_absolute_error(y_trainval[va_idx].astype(float), model.predict(Xva)))
            fold_maes.append(mae)

            trial.report(float(np.mean(fold_maes)), step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

            del model, vae, mu_tr, mu_va, z_tr, z_va, Z_tr, Z_va, Xtr, Xva

        return float(np.mean(fold_maes))

    def _save_best_so_far(study_: optuna.Study, trial_: optuna.trial.FrozenTrial) -> None:
        if study_.best_trial is None:
            return
        _dump_json(out_dir / "vae_optuna_age_best_so_far.json", {
            "best_value_mae": float(study_.best_value),
            "best_params": dict(study_.best_trial.params),
            "trial_number": int(study_.best_trial.number),
            "n_complete": len([t for t in study_.trials
                               if t.state == optuna.trial.TrialState.COMPLETE]),
        })

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        objective, n_trials=int(n_trials), show_progress_bar=True,
        callbacks=[_save_best_so_far], n_jobs=1,
    )

    best = {
        "best_value_mae": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials_target": int(n_trials),
        "n_trials_total": int(len(study.trials)),
        "seed": int(seed),
        "use_t1w": bool(use_t1w),
        "use_sex": bool(use_sex),
        "use_diag": bool(use_diag),
        "study_name": study_name,
        "storage": str(storage_path),
        "xgb_frozen_params": dict(xgb_fixed_params),
        "vae_fixed": dict(vae_fixed),
    }

    _dump_json(out_dir / "vae_optuna_age_best.json", best)
    _dump_json(out_dir / "vae_optuna_age_study_trials.json", {
        "trials": [
            {"number": t.number, "value": t.value, "state": str(t.state), "params": t.params}
            for t in study.trials
            if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        ]
    })

    return best
