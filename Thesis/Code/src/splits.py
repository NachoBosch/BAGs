from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


def make_holdout_split(
    cohort_df: pd.DataFrame,
    seed: int,
    test_size: float,
) -> dict:
    """Stratified holdout split returning ID lists (stratifies by sex x diagnosis)."""
    ids = cohort_df["record_id"].to_numpy()
    strat = (cohort_df["sex"].astype(str) + "__" + cohort_df["diagnosis"].astype(str)).to_numpy()
    train_ids, test_ids = train_test_split(
        ids, test_size=test_size, random_state=seed, stratify=strat,
    )
    return {
        "trainval_ids": sorted(train_ids.tolist()),
        "test_ids": sorted(test_ids.tolist()),
    }


def make_kfold_splits(trainval_ids: list[str], seed: int, k: int) -> list[dict]:
    """K-Fold splits over trainval IDs."""
    ids = np.array(trainval_ids)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [
        {"fold": i, "train_ids": ids[tr].tolist(), "val_ids": ids[va].tolist()}
        for i, (tr, va) in enumerate(kf.split(ids))
    ]


def save_splits(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_splits(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
