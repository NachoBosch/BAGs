"""Load FC matrices from inflamacion/coma folder layout."""

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pandas as pd

from .data_io import (
    _load_fc_matrix,
    fisher_z_transform,
    matrix_to_vector_upper,
    read_t1w_csv,
)
from .utils_ids import normalize_record_id

logger = logging.getLogger(__name__)

DEFAULT_GROUP_MAP: dict[str, str] = {
    "controls": "CTRL",
    "control": "CTRL",
    "anoxia": "ANOX",
    "traumatic": "TRAU",
    "trauma": "TRAU",
}

TOPO_COLUMNS = [
    "local_efficiency",
    "global_efficiency",
    "clustering_coeff",
    "fp_local_efficiency",
    "fp_betweenness",
    "fp_degree",
]

FRONTOPARIETAL_IDX = [
    2, 3, 6, 7, 32, 33, 60, 61, 62, 63, 66, 67, 80, 81,
]


def list_inflamacion_mats(
    fc_root: Path,
    group_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    fc_root = Path(fc_root)
    if not fc_root.is_dir():
        raise FileNotFoundError(f"FC root not found: {fc_root}")

    gmap = {k.lower(): v for k, v in (group_map or DEFAULT_GROUP_MAP).items()}
    rows: list[dict] = []

    for group_dir in sorted(fc_root.iterdir()):
        if not group_dir.is_dir():
            continue
        dx = gmap.get(group_dir.name.lower())
        if dx is None:
            logger.warning("Skipping unknown group folder: %s", group_dir.name)
            continue
        for mat_path in sorted(group_dir.glob("*.mat")):
            rows.append({
                "record_id": normalize_record_id(mat_path.stem),
                "diagnosis": dx,
                "group_folder": group_dir.name,
                "mat_path": str(mat_path.resolve()),
            })

    if not rows:
        raise ValueError(f"No .mat files found under {fc_root}")

    df = pd.DataFrame(rows)
    if df["record_id"].duplicated().any():
        dup = df.loc[df["record_id"].duplicated(), "record_id"].unique()
        logger.warning("Duplicate record_id (keeping last): %s", dup[:5].tolist())
        df = df.drop_duplicates("record_id", keep="last")

    return df.sort_values("record_id").reset_index(drop=True)


def load_metadata_optional(path: Path | None) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None

    path = Path(path)
    df = pd.read_excel(path, sheet_name=0) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)

    col_map = {
        "subject_id": "record_id", "id": "record_id", "MRI_ID": "record_id",
        "demo_age": "age", "Age": "age", "demo_sex": "sex", "Sex": "sex",
        "clinical_diagnosis": "diagnosis", "group": "diagnosis",
        "years_education": "education", "cog_ed": "education", "site": "site",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "record_id" not in df.columns:
        raise ValueError(f"Metadata at {path} must contain a subject id column")

    df["record_id"] = df["record_id"].astype(str).map(normalize_record_id)
    if "sex" in df.columns:
        sex_map = {1: "Male", 2: "Female", "1": "Male", "2": "Female",
                   "M": "Male", "F": "Female", "male": "Male", "female": "Female"}
        df["sex"] = df["sex"].map(sex_map).fillna(df["sex"])
    return df


def merge_cohort_metadata(
    fc_index: pd.DataFrame,
    meta: pd.DataFrame | None,
    t1w_path: Path | None = None,
) -> pd.DataFrame:
    cohort = fc_index.copy()
    if meta is not None:
        cohort = cohort.merge(meta, on="record_id", how="left", suffixes=("", "_meta"))
        if "diagnosis_meta" in cohort.columns:
            cohort["diagnosis"] = cohort["diagnosis_meta"].fillna(cohort["diagnosis"])
            cohort = cohort.drop(columns=["diagnosis_meta"])
    if t1w_path is not None and Path(t1w_path).exists():
        cohort = cohort.merge(read_t1w_csv(Path(t1w_path)), on="record_id", how="left")
    return cohort.sort_values("record_id").reset_index(drop=True)


def load_fc_vectors_from_cohort(
    cohort: pd.DataFrame,
    ids: list[str],
    *,
    apply_fisher_z: bool = False,
) -> np.ndarray:
    id_to_path = cohort.set_index("record_id")["mat_path"].to_dict()
    missing = [i for i in ids if i not in id_to_path]
    if missing:
        raise ValueError(f"Missing FC for {len(missing)} ids (e.g. {missing[:3]})")

    vecs = [matrix_to_vector_upper(_load_fc_matrix(Path(id_to_path[rid]))) for rid in ids]
    X = np.stack(vecs, axis=0).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if apply_fisher_z:
        X = fisher_z_transform(X)
    return X


def discover_metadata_path(fc_root: Path) -> Path | None:
    for p in [
        fc_root / "metadata.csv", fc_root / "subjects.csv",
        fc_root / "metadata.xlsx", fc_root / "clinical.csv",
        fc_root.parent / "metadata.csv",
    ]:
        if p.exists():
            return p
    return None
