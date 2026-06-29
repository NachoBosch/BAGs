"""Graph metrics and small-world helpers for coma pipeline."""

from __future__ import annotations

from pathlib import Path

import bct
import networkx as nx
import numpy as np
import pandas as pd

from .coma_data_io import FRONTOPARIETAL_IDX, TOPO_COLUMNS
from .data_io import _load_fc_matrix, fisher_z_transform


def threshold_fixed(fc_z: np.ndarray, thr: float = 0.20) -> np.ndarray:
    mat = fc_z.copy()
    np.fill_diagonal(mat, 0)
    B = (mat >= thr).astype(float)
    return np.maximum(B, B.T)


def threshold_proportional(fc_z: np.ndarray, prop: float) -> np.ndarray:
    n = fc_z.shape[0]
    triu = fc_z[np.triu_indices(n, k=1)]
    thr = np.quantile(triu, 1.0 - prop)
    B = (fc_z >= thr).astype(float)
    np.fill_diagonal(B, 0)
    return np.maximum(B, B.T)


def compute_graph_metrics(binary_matrix: np.ndarray, fp_idx: list[int] | None = None) -> dict:
    fp_idx = fp_idx or FRONTOPARIETAL_IDX
    n = binary_matrix.shape[0]
    B = binary_matrix.astype(float)
    local_eff_nodes = bct.efficiency_bin(B, local=True)
    betweenness = bct.betweenness_bin(B)
    betweenness_norm = betweenness / max((n - 1) * (n - 2), 1)
    degree_norm = np.sum(B, axis=0) / max(n - 1, 1)
    return {
        "local_efficiency": float(np.mean(local_eff_nodes)),
        "global_efficiency": float(bct.efficiency_bin(B, local=False)),
        "clustering_coeff": float(np.mean(bct.clustering_coef_bu(B))),
        "degree_mean": float(np.mean(degree_norm)),
        "betweenness_mean": float(np.mean(betweenness_norm)),
        "fp_local_efficiency": float(np.mean(local_eff_nodes[fp_idx])),
        "fp_betweenness": float(np.mean(betweenness_norm[fp_idx])),
        "fp_degree": float(np.mean(degree_norm[fp_idx])),
    }


def compute_topo_table(
    cohort: pd.DataFrame,
    *,
    threshold: float = 0.20,
    apply_fisher_z: bool = True,
) -> pd.DataFrame:
    rows = []
    for _, row in cohort.iterrows():
        mat = _load_fc_matrix(Path(row["mat_path"]))
        if apply_fisher_z:
            mat = np.arctanh(np.clip(mat, -0.9999, 0.9999)).astype(np.float32)
        B = threshold_fixed(mat, threshold)
        m = compute_graph_metrics(B)
        m["record_id"] = row["record_id"]
        m["diagnosis"] = row["diagnosis"]
        rows.append(m)
    return pd.DataFrame(rows)
