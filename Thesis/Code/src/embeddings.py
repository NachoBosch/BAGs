from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def encode_mu(encoder, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Extract deterministic (mu) embeddings from a trained encoder."""
    mu, _, _ = encoder.predict(X, batch_size=batch_size, verbose=0)
    return mu.astype(np.float32)


def save_embeddings(path: Path, ids: list[str], Z: np.ndarray) -> None:
    """Persist embeddings as .npy with an accompanying .json metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), Z)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "shape": list(Z.shape)}, f, indent=2)


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    """Load embeddings and their associated ID list."""
    Z = np.load(path.with_suffix(".npy"))
    with open(path.with_suffix(".json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["ids"], Z
