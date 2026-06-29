"""Train/test splits for coma cohort (sex×dx or dx-only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_holdout_split_coma(
    cohort_df: pd.DataFrame,
    seed: int,
    test_size: float,
) -> dict:
    ids = cohort_df["record_id"].to_numpy()
    has_sex = "sex" in cohort_df.columns and cohort_df["sex"].notna().all()
    if has_sex:
        strat = (
            cohort_df["sex"].astype(str) + "__" + cohort_df["diagnosis"].astype(str)
        ).to_numpy()
    else:
        strat = cohort_df["diagnosis"].astype(str).to_numpy()

    train_ids, test_ids = train_test_split(
        ids, test_size=test_size, random_state=seed, stratify=strat,
    )
    return {
        "trainval_ids": sorted(train_ids.tolist()),
        "test_ids": sorted(test_ids.tolist()),
    }
