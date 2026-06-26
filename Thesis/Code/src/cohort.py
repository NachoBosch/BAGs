from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd

from .data_io import (
    read_metadata,
    read_t1w_csv,
    list_fc_files,
    extract_fc_record_id_from_filename,
)

logger = logging.getLogger(__name__)


def _collapse_t1w_duplicates(t1w: pd.DataFrame) -> pd.DataFrame:
    """Average numeric columns across technical replicates of the same subject."""
    if not t1w["record_id"].duplicated().any():
        return t1w

    dup_ids = t1w.loc[t1w["record_id"].duplicated(keep=False), "record_id"].unique()
    logger.info(
        "Found %d subjects with multiple T1w entries; collapsing by mean.",
        len(dup_ids),
    )

    t1_cols = [c for c in t1w.columns if c != "record_id"]
    mask_dup = t1w["record_id"].isin(dup_ids)
    unique_rows = t1w.loc[~mask_dup].copy()

    # .copy() defragments the slice before groupby to avoid PerformanceWarning
    collapsed = (
        t1w.loc[mask_dup].copy()
        .groupby("record_id", as_index=False)[t1_cols]
        .mean(numeric_only=True)
    )

    return pd.concat([unique_rows, collapsed], ignore_index=True)


def build_final_cohort_df(
    excel_path: Path,
    fc_folder: Path,
    t1w_csv_path: Path,
    diagnoses_to_use: tuple[str, ...],
) -> pd.DataFrame:
    """Build final cohort as the intersection of FC, metadata and T1w sources.

    Filters by diagnosis and non-null age. Returns one row per subject.
    """
    meta = read_metadata(excel_path)
    t1w = read_t1w_csv(t1w_csv_path)

    if meta["record_id"].duplicated().any():
        dups = meta.loc[meta["record_id"].duplicated(), "record_id"].head(5).tolist()
        raise ValueError(f"Metadata has duplicated record_id (e.g. {dups}).")

    t1w = _collapse_t1w_duplicates(t1w)

    fc_ids = set()
    for p in list_fc_files(fc_folder):
        rid = extract_fc_record_id_from_filename(p)
        if rid:
            fc_ids.add(rid)

    meta = meta[meta["diagnosis"].astype(str).isin(diagnoses_to_use)].copy()
    meta = meta[meta["age"].notna()].copy()

    cohort = meta.merge(t1w, on="record_id", how="inner", validate="one_to_one")
    cohort = cohort[cohort["record_id"].isin(fc_ids)].copy()
    cohort = cohort.sort_values("record_id").reset_index(drop=True)

    logger.info(
        "Final cohort: %s (%d subjects, diagnoses: %s)",
        cohort.shape, len(cohort), cohort["diagnosis"].value_counts().to_dict(),
    )
    return cohort
