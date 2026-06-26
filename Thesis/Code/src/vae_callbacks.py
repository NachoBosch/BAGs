from __future__ import annotations

import numpy as np
from tensorflow.keras.callbacks import Callback


def pearsonr_per_sample(true_vecs: np.ndarray, pred_vecs: np.ndarray) -> np.ndarray:
    """Pearson correlation per sample between paired row vectors."""
    true = true_vecs - true_vecs.mean(axis=1, keepdims=True)
    pred = pred_vecs - pred_vecs.mean(axis=1, keepdims=True)
    num = np.sum(true * pred, axis=1)
    denom = np.sqrt(np.sum(true ** 2, axis=1) * np.sum(pred ** 2, axis=1))
    return num / np.maximum(denom, 1e-8)


def cosine_similarity_per_sample(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity per sample between paired row vectors."""
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / np.maximum(norm_a * norm_b, 1e-8)


class BetaScheduler(Callback):
    """Linear warmup of beta from 0 to target over warmup_epochs."""

    def __init__(self, target_beta: float, warmup_epochs: int):
        super().__init__()
        self.target_beta = float(target_beta)
        self.warmup_epochs = int(max(1, warmup_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        t = min(1.0, epoch / self.warmup_epochs)
        self.model.beta.assign(t * self.target_beta)


class ValidationCorrelations(Callback):
    """Log per-sample Pearson and cosine similarity on a held-out validation set."""

    def __init__(self, X_val: np.ndarray, batch_size: int = 512,
                 name_pearson: str = "val_pearson", name_cosine: str = "val_cosine"):
        super().__init__()
        self.X_val = X_val
        self.bs = int(batch_size)
        self.name_pearson = name_pearson
        self.name_cosine = name_cosine
        self.best_pearson = -np.inf
        self.best_cosine = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        preds = self.model(self.X_val, training=False).numpy()

        r = pearsonr_per_sample(self.X_val, preds)
        r_mean = float(np.nanmean(r))
        logs[self.name_pearson] = r_mean
        self.best_pearson = max(self.best_pearson, r_mean)

        cos = cosine_similarity_per_sample(self.X_val, preds)
        cos_mean = float(np.nanmean(cos))
        logs[self.name_cosine] = cos_mean
        self.best_cosine = max(self.best_cosine, cos_mean)

        print(
            f"\nEpoch {epoch+1}: Pearson={r_mean:.4f} (best={self.best_pearson:.4f}), "
            f"Cosine={cos_mean:.4f} (best={self.best_cosine:.4f})"
        )
