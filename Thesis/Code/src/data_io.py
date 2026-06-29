from __future__ import annotations

from pathlib import Path
import os
import re
import logging
import numpy as np
import pandas as pd
from scipy.io import loadmat

from .utils_ids import normalize_record_id

logger = logging.getLogger(__name__)

_FC_ID_RE = re.compile(r"sub-([A-Za-z0-9]+)_timeseries", re.IGNORECASE)


def read_full_metadata(excel_path: Path) -> pd.DataFrame:
    """Load metadata Excel keeping all columns. Normalizes record_id."""
    df = pd.read_excel(excel_path, sheet_name=0)
    df["record_id"] = df["record_id"].astype(str).map(normalize_record_id)
    return df


def read_metadata(excel_path: Path) -> pd.DataFrame:
    """Load metadata Excel and standardize column names."""
    df = pd.read_excel(excel_path, sheet_name=0)
    required = ["record_id", "demo_age", "demo_sex", "clinical_diagnosis"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in metadata: {missing}")

    out = df[required].copy()
    out = out.rename(columns={
        "demo_age": "age",
        "demo_sex": "sex",
        "clinical_diagnosis": "diagnosis",
    })

    sex_map = {1: "Male", 2: "Female", "1": "Male", "2": "Female",
               "Male": "Male", "Female": "Female"}
    out["sex"] = out["sex"].map(sex_map).fillna(out["sex"])
    out["record_id"] = out["record_id"].astype(str).map(normalize_record_id)
    return out


def read_t1w_csv(csv_path: Path) -> pd.DataFrame:
    """Load T1w CSV (headerless) and normalize record_id from the first column."""
    df = pd.read_csv(csv_path, header=None)
    df = df.rename(columns={0: "record_id"})
    df["record_id"] = df["record_id"].astype(str).map(normalize_record_id)

    feat_cols = [f"t1_{i:04d}" for i in range(df.shape[1] - 1)]
    df.columns = ["record_id"] + feat_cols
    return df


def list_fc_files(fc_folder: Path) -> list[Path]:
    """Return sorted list of .mat files in fc_folder."""
    return sorted(Path(fc_folder) / f for f in os.listdir(fc_folder)
                  if f.lower().endswith(".mat"))


def extract_fc_record_id_from_filename(path: Path) -> str | None:
    """Extract and normalize the subject ID embedded in an FC filename."""
    m = _FC_ID_RE.search(path.name)
    if not m:
        return None
    return normalize_record_id(m.group(1))


def _load_fc_matrix(mat_path: Path) -> np.ndarray:
    """Load the first square matrix from a .mat file, enforcing diagonal = 1."""
    md = loadmat(mat_path)
    for v in md.values():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == v.shape[1]:
            m = np.asarray(v, dtype=np.float32)
            np.fill_diagonal(m, 1.0)
            return m
    raise ValueError(f"No square matrix found in {mat_path}")


def matrix_to_vector_upper(m: np.ndarray) -> np.ndarray:
    """Flatten symmetric matrix to upper-triangle vector (k=1, excludes diagonal)."""
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"Expected square matrix, got {m.shape}")
    return m[np.triu_indices(m.shape[0], k=1)].astype(np.float32)


def vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    """Reconstruct symmetric matrix from an upper-triangle vector."""
    L = int(vector.shape[0])
    N = int((1 + np.sqrt(1 + 8 * L)) // 2)
    m = np.zeros((N, N), dtype=vector.dtype)
    idx = np.triu_indices(N, k=1)
    m[idx] = vector
    m[(idx[1], idx[0])] = vector
    np.fill_diagonal(m, 1.0)
    return m


def fisher_z_transform(X: np.ndarray) -> np.ndarray:
    """Apply Fisher z-transform (arctanh) to correlation vectors.

    Values are clamped to (-0.9999, 0.9999) to avoid infinities at +/-1.
    """
    return np.arctanh(np.clip(X, -0.9999, 0.9999)).astype(np.float32)


def load_fc_vectors_for_ids(
    fc_folder: Path,
    ids: list[str],
    *,
    apply_fisher_z: bool = False,
) -> np.ndarray:
    """Load FC upper-triangle vectors aligned to the given ID list.

    Raises ValueError if any ID is missing from disk.
    """
    id_set = set(ids)
    id_to_path: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}

    for p in list_fc_files(fc_folder):
        rid = extract_fc_record_id_from_filename(p)
        if rid and rid in id_set:
            if rid in id_to_path:
                duplicates.setdefault(rid, [str(id_to_path[rid])]).append(str(p))
            id_to_path[rid] = p

    if duplicates:
        logger.warning(
            "Found %d subjects with multiple .mat files (using last alphabetically): %s",
            len(duplicates), list(duplicates.keys())[:5],
        )

    missing = [rid for rid in ids if rid not in id_to_path]
    if missing:
        raise ValueError(f"Missing FC files for {len(missing)} ids (e.g. {missing[:5]})")

    vecs = [matrix_to_vector_upper(_load_fc_matrix(id_to_path[rid])) for rid in ids]
    X = np.stack(vecs, axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if apply_fisher_z:
        X = fisher_z_transform(X)
        logger.info("Applied Fisher z-transform to %d vectors.", X.shape[0])

    return X
