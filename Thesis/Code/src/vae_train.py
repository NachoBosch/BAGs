from __future__ import annotations

from pathlib import Path
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .vae_model import make_encoder, make_decoder, VAE
from .vae_callbacks import BetaScheduler, ValidationCorrelations

logger = logging.getLogger(__name__)


def build_vae(
    input_dim: int,
    *,
    hidden_dims: list[int],
    latent_dim: int,
    l2_reg: float,
    drop_rate: float,
    activation: str,
    norm_kind: str,
    beta_target: float,
    recon_kind: str,
    lr: float,
    clipnorm: float,
) -> VAE:
    """Build, compile and initialize a VAE."""
    encoder = make_encoder(input_dim, hidden_dims, latent_dim,
                           l2_reg, drop_rate, activation, norm_kind)
    decoder = make_decoder(input_dim, hidden_dims, latent_dim,
                           l2_reg, drop_rate, activation, norm_kind)
    vae = VAE(encoder, decoder, beta=beta_target, recon=recon_kind, name="VAE")
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm))
    _ = vae(tf.zeros((1, input_dim)))
    return vae


def _dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _save_vae_artifacts(vae: VAE, out_dir: Path, *, hparams: dict,
                        history: dict | None = None) -> None:
    """Persist weights, hyperparameters, and optionally training history."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vae.save_weights(out_dir / "vae.weights.h5")
    _dump_json(out_dir / "hparams.json", hparams)
    if history is not None:
        _dump_json(out_dir / "history.json", history)


def load_vae_from_dir(model_dir: Path) -> VAE:
    """Rebuild architecture from hparams.json and load saved weights."""
    hparams_path = model_dir / "hparams.json"
    weights_path = model_dir / "vae.weights.h5"
    if not hparams_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing VAE artifacts in {model_dir}")

    with open(hparams_path, "r", encoding="utf-8") as f:
        hp = json.load(f)

    vae = build_vae(
        int(hp["input_dim"]),
        hidden_dims=list(hp["hidden_dims"]),
        latent_dim=int(hp["latent_dim"]),
        l2_reg=float(hp["l2_reg"]),
        drop_rate=float(hp["drop_rate"]),
        activation=str(hp["activation"]),
        norm_kind=str(hp["norm_kind"]),
        beta_target=float(hp["beta_target"]),
        recon_kind=str(hp["recon_kind"]),
        lr=float(hp["lr"]),
        clipnorm=float(hp["clipnorm"]),
    )
    vae.load_weights(weights_path)
    return vae


def _build_hparams_dict(input_dim, hidden_dims, latent_dim, beta_target, warmup_ep,
                        l2_reg, lr, recon_kind, drop_rate, activation, norm_kind,
                        max_epochs, batch_size, clipnorm, seed, **extra) -> dict:
    """Canonical hparams dictionary for serialization."""
    hp = dict(
        input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim,
        beta_target=beta_target, warmup_ep=warmup_ep, l2_reg=l2_reg, lr=lr,
        recon_kind=recon_kind, drop_rate=drop_rate, activation=activation,
        norm_kind=norm_kind, max_epochs=max_epochs, batch_size=batch_size,
        clipnorm=clipnorm, seed=seed,
    )
    hp.update(extra)
    return hp


def train_vae_kfold(
    X_trainval: np.ndarray,
    out_dir: Path,
    seed: int,
    k: int,
    *,
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
    hidden_dims: list[int],
    latent_dim: int,
    beta_target: float,
    warmup_ep: int,
    l2_reg: float,
    lr: float,
    recon_kind: str,
    drop_rate: float,
    activation: str,
    norm_kind: str,
    max_epochs: int,
    batch_size: int,
    clipnorm: float,
) -> dict:
    """Train VAE with K-Fold cross-validation.

    Parameters
    ----------
    fold_indices : pre-computed (train_idx, val_idx) arrays.
        When provided, these folds are used directly for consistency with
        the global split definitions. Falls back to an internal KFold if None.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if fold_indices is not None:
        folds_iter = list(enumerate(fold_indices))
        actual_k = len(fold_indices)
        if actual_k != k:
            logger.warning("fold_indices has %d folds but k=%d; using %d.", actual_k, k, actual_k)
            k = actual_k
    else:
        from sklearn.model_selection import KFold
        logger.warning("No fold_indices provided; creating internal KFold.")
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds_iter = list(enumerate(kf.split(X_trainval)))

    fold_summaries = []
    input_dim = int(X_trainval.shape[1])

    for fold, (train_idx, val_idx) in folds_iter:
        print(f"\n--- VAE Fold {fold+1}/{k} ---")
        X_tr = X_trainval[train_idx]
        X_va = X_trainval[val_idx]

        vae = build_vae(
            input_dim,
            hidden_dims=hidden_dims, latent_dim=latent_dim, l2_reg=l2_reg,
            drop_rate=drop_rate, activation=activation, norm_kind=norm_kind,
            beta_target=beta_target, recon_kind=recon_kind, lr=lr, clipnorm=clipnorm,
        )

        cbs = [
            BetaScheduler(target_beta=beta_target, warmup_epochs=warmup_ep),
            EarlyStopping(monitor="val_recon_loss", mode="min", patience=20,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_recon_loss", mode="min", factor=0.5,
                              patience=10, min_lr=1e-5, verbose=1),
            ValidationCorrelations(X_va, batch_size=512),
        ]

        hist = vae.fit(
            X_tr, epochs=max_epochs, batch_size=batch_size,
            shuffle=True, validation_data=(X_va, None), callbacks=cbs, verbose=1,
        )

        val_recon = hist.history.get("val_recon_loss", [])
        val_pear = hist.history.get("val_pearson", [])
        val_cos = hist.history.get("val_cosine", [])

        best_recon = float(np.min(val_recon)) if val_recon else float("nan")
        best_pear = float(np.max(val_pear)) if val_pear else float("nan")
        best_cos = float(np.max(val_cos)) if val_cos else float("nan")
        best_epoch = int(np.argmin(val_recon) + 1) if val_recon else len(hist.history.get("loss", []))

        fold_dir = out_dir / f"fold_{fold}"
        hparams = _build_hparams_dict(
            input_dim, hidden_dims, latent_dim, beta_target, warmup_ep,
            l2_reg, lr, recon_kind, drop_rate, activation, norm_kind,
            max_epochs, batch_size, clipnorm, seed, k_folds=k, fold=fold,
        )
        _save_vae_artifacts(vae, fold_dir, hparams=hparams, history=hist.history)

        fold_summaries.append({
            "fold": fold,
            "best_epoch": best_epoch,
            "best_val_recon_loss": best_recon,
            "best_val_pearson": best_pear,
            "best_val_cosine": best_cos,
        })

    summary = {
        "k": k,
        "seed": seed,
        "folds": fold_summaries,
        "mean_best_val_recon_loss": float(np.nanmean([f["best_val_recon_loss"] for f in fold_summaries])),
        "mean_best_val_pearson": float(np.nanmean([f["best_val_pearson"] for f in fold_summaries])),
        "mean_best_val_cosine": float(np.nanmean([f["best_val_cosine"] for f in fold_summaries])),
        "suggested_epochs_for_final": int(np.round(np.nanmedian([f["best_epoch"] for f in fold_summaries]))),
    }
    _dump_json(out_dir / "kfold_summary.json", summary)
    return summary


def train_vae_final(
    X_trainval: np.ndarray,
    out_dir: Path,
    *,
    hidden_dims: list[int],
    latent_dim: int,
    beta_target: float,
    warmup_ep: int,
    l2_reg: float,
    lr: float,
    recon_kind: str,
    drop_rate: float,
    activation: str,
    norm_kind: str,
    epochs: int,
    batch_size: int,
    clipnorm: float,
    seed: int,
) -> tuple[VAE, dict]:
    """Train final VAE on full trainval set. No validation split is used."""
    input_dim = int(X_trainval.shape[1])
    vae = build_vae(
        input_dim,
        hidden_dims=hidden_dims, latent_dim=latent_dim, l2_reg=l2_reg,
        drop_rate=drop_rate, activation=activation, norm_kind=norm_kind,
        beta_target=beta_target, recon_kind=recon_kind, lr=lr, clipnorm=clipnorm,
    )

    cbs = [
        BetaScheduler(target_beta=beta_target, warmup_epochs=warmup_ep),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10, min_lr=1e-5, verbose=1),
    ]

    hist = vae.fit(
        X_trainval, epochs=int(epochs), batch_size=batch_size,
        shuffle=True, callbacks=cbs, verbose=1,
    )

    hparams = _build_hparams_dict(
        input_dim, hidden_dims, latent_dim, beta_target, warmup_ep,
        l2_reg, lr, recon_kind, drop_rate, activation, norm_kind,
        int(epochs), batch_size, clipnorm, seed,
    )
    _save_vae_artifacts(vae, out_dir, hparams=hparams, history=hist.history)
    return vae, hist.history


def load_kfold_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_history(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
