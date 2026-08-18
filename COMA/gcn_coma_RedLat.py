"""
GCN para BAG: CTRL(COMA + CN ReDLaT) vs pacientes(COMA).

Igual que gcn_coma.py, reforzando controles con CN ReDLaT.
Tablas con sufijo _RedLat. LOO se omite si n_CTRL es grande (hold-out 20%).

Qué sí se puede replicar
------------------------
Pasos 1–5: grafo FC por sujeto. La GCN aprende edad **solo en CTRL**
(COMA + ReDLaT); luego BAG en controles y pacientes.

Qué no coincide 1:1 con el paper
--------------------------------
- Atlas: AAL-116, no 82 regiones.
- Pearson reusa FC precomputada; O-info opcional via copula gaussiana.
- Contraste: CTRL vs pacientes (ANOX+TRAU). diagnosis_orig = ANOX/TRAU/CTRL/CN.
- Paso 6: GBR clínico (sin vars de país).

O-info (Organizational Information)
-----------------------------------
Ω = TC − DTC. Positivo ≈ redundancia; negativo ≈ sinergia (Rosas et al.).
En n=2, Ω=0, así que una matriz de aristas no puede ser el Ω del par aislado.
Aquí, si CONNECTIVITY="oinfo", la arista i–j es la información mutua gaussiana
entre el par {i,j} y el resto del cerebro (cópula gaussiana / entropía diferencial).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Thesis" / "Code"))

from src.coma_data_io import DEFAULT_GROUP_MAP, list_inflamacion_mats  # noqa: E402
from src.config import DEFAULT_PATHS  # noqa: E402
from src.data_io import (  # noqa: E402
    _load_fc_matrix,
    extract_fc_record_id_from_filename,
    list_fc_files,
    read_metadata,
)

# --- config ---
FC_ROOT = Path("/home/usuario/disco1/proyectos/2024-autoencoders/databases/fc/inflamacion")
DEMO_XLSX = ROOT / "demographics.xlsx"
IDMAP_PATH = ROOT / "id_map_coma.csv"
TABLE_DIR = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "coma"
SUFFIX = "_RedLat"
GCN_DIR = OUT / f"gcn_coma{SUFFIX}"
LOO_MAX_N = 40  # por encima: hold-out 20% en vez de LOO

SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5
HIDDEN = 64
LATENT = 32
DROPOUT = 0.30
BATCH_SIZE = 4
EDGE_THRESH = 0.0  # 0 = grafo completo (salvo diagonal)
GRID_LR = (1e-3, 5e-4, 1e-4)
GRID_EPOCHS = (80, 150)
PATIENCE = 25
PATIENT_LABEL = "pacientes"
PATIENT_SOURCES = {"ANOX", "TRAU"}

SUB_RE = re.compile(r"(ANOX|TC|CONTROLES)\d{3}", re.I)


def _need_torch():
    try:
        import torch
        from torch_geometric.data import Data  # noqa: F401
        from torch_geometric.loader import DataLoader  # noqa: F401
        from torch_geometric.nn import GCNConv, global_mean_pool  # noqa: F401
        from torch_geometric.utils import dense_to_sparse  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Faltan PyTorch / PyTorch Geometric.\n"
            "  pip install torch\n"
            "  pip install torch-geometric\n"
            f"Detalle: {e}"
        ) from e
    return torch


# ---------------------------------------------------------------------------
# Demografía (mismo join que train_coma_bag.py)
# ---------------------------------------------------------------------------

def mat_sub_code(path) -> str | None:
    m = SUB_RE.search(Path(path).name)
    return m.group(0).upper() if m else None


def load_demographics_xlsx(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_pat = pd.read_excel(path, sheet_name="Coma3D patient", header=None)
    demo_pat = raw_pat.iloc[2:].copy()
    demo_pat.columns = range(demo_pat.shape[1])
    demo_pat = demo_pat.rename(columns={
        0: "patient", 1: "initials_demo", 2: "sex", 3: "age",
        7: "etiology", 8: "gcs", 12: "outcome_crsr_raw", 13: "deceased", 14: "binary_outcome",
    })
    demo_pat = demo_pat[[
        "patient", "initials_demo", "sex", "age", "etiology", "gcs",
        "outcome_crsr_raw", "deceased", "binary_outcome",
    ]].copy()
    demo_pat["patient"] = pd.to_numeric(demo_pat["patient"], errors="coerce")
    demo_pat = demo_pat.dropna(subset=["patient"]).copy()
    demo_pat["patient"] = demo_pat["patient"].astype(int)
    demo_pat["age"] = pd.to_numeric(demo_pat["age"], errors="coerce")
    demo_pat["sheet"] = "patient"

    raw_ctl = pd.read_excel(path, sheet_name="Coma3D controle", header=None)
    demo_ctl = raw_ctl.iloc[2:].copy()
    demo_ctl.columns = range(demo_ctl.shape[1])
    demo_ctl = demo_ctl.rename(columns={
        0: "patient", 1: "initials_demo", 2: "sex", 4: "age", 5: "etiology",
    })
    demo_ctl = demo_ctl[["patient", "initials_demo", "sex", "age", "etiology"]].copy()
    demo_ctl["patient"] = pd.to_numeric(demo_ctl["patient"], errors="coerce")
    demo_ctl = demo_ctl.dropna(subset=["patient"]).copy()
    demo_ctl["patient"] = demo_ctl["patient"].astype(int)
    demo_ctl["age"] = pd.to_numeric(demo_ctl["age"], errors="coerce")
    demo_ctl["gcs"] = np.nan
    demo_ctl["outcome_crsr_raw"] = np.nan
    demo_ctl["deceased"] = np.nan
    demo_ctl["binary_outcome"] = np.nan
    demo_ctl["sheet"] = "controle"

    for df in (demo_pat, demo_ctl):
        df["etiology"] = df["etiology"].astype(str).str.strip()
        df["sex"] = normalize_sex_col(df["sex"])
    return demo_pat, demo_ctl


def normalize_sex_col(s: pd.Series) -> pd.Series:
    return (
        s.astype(str).str.strip()
        .replace({
            "H": "M", "Male": "M", "MALE": "M", "male": "M",
            "Female": "F", "FEMALE": "F", "female": "F",
        })
    )


def build_redlat_cn() -> pd.DataFrame:
    excel = DEFAULT_PATHS.excel_path
    fc_folder = DEFAULT_PATHS.fc_folder
    if not excel.exists():
        raise FileNotFoundError(f"Falta metadata ReDLaT: {excel}")
    if not fc_folder.exists():
        raise FileNotFoundError(f"Falta FC ReDLaT: {fc_folder}")

    meta = read_metadata(excel)
    meta = meta[meta["diagnosis"].astype(str) == "CN"].copy()
    meta = meta[meta["age"].notna()].copy()

    id_to_path: dict[str, Path] = {}
    for p in list_fc_files(fc_folder):
        rid = extract_fc_record_id_from_filename(p)
        if rid:
            id_to_path[rid] = p
    meta = meta[meta["record_id"].isin(id_to_path)].copy()
    if meta.empty:
        raise ValueError("No hay CN ReDLaT con FC disponible")

    rows = []
    for _, r in meta.iterrows():
        rid = r["record_id"]
        rows.append({
            "record_id": rid,
            "diagnosis": "CTRL",
            "diagnosis_orig": "CN",
            "group_folder": "redlat_CN",
            "mat_path": str(id_to_path[rid].resolve()),
            "sub_code": rid,
            "patient": np.nan,
            "initials": np.nan,
            "nombre_original": np.nan,
            "group": "CN",
            "initials_demo": np.nan,
            "sex": r["sex"],
            "age": float(r["age"]),
            "etiology": "CN",
            "gcs": np.nan,
            "outcome_crsr_raw": np.nan,
            "deceased": np.nan,
            "binary_outcome": np.nan,
            "sheet": "redlat",
            "source": "redlat",
        })
    out = pd.DataFrame(rows)
    out["sex"] = normalize_sex_col(out["sex"])
    return out.sort_values("record_id").reset_index(drop=True)


def build_cohort() -> pd.DataFrame:
    coma = list_inflamacion_mats(FC_ROOT, group_map=DEFAULT_GROUP_MAP)
    coma = link_demographics(coma)
    coma = merge_patients(coma)
    coma["source"] = "coma"
    coma["sex"] = normalize_sex_col(coma["sex"])
    redlat = build_redlat_cn()
    cols = sorted(set(coma.columns) | set(redlat.columns))
    for c in cols:
        if c not in coma.columns:
            coma[c] = np.nan
        if c not in redlat.columns:
            redlat[c] = np.nan
    cohort = pd.concat([coma[cols], redlat[cols]], ignore_index=True)
    if cohort["record_id"].duplicated().any():
        dups = cohort.loc[cohort["record_id"].duplicated(), "record_id"].unique()[:5]
        raise ValueError(f"record_id duplicados COMA/ReDLaT: {list(dups)}")
    return cohort.sort_values("record_id").reset_index(drop=True)


def link_demographics(cohort: pd.DataFrame) -> pd.DataFrame:
    id_map = pd.read_csv(IDMAP_PATH)
    id_map["patient"] = pd.to_numeric(id_map["patient"], errors="coerce")
    demo_pat, demo_ctl = load_demographics_xlsx(DEMO_XLSX)

    cohort = cohort.copy()
    cohort["sub_code"] = cohort["mat_path"].map(mat_sub_code)
    cohort = cohort.merge(
        id_map[["sub_code", "patient", "initials", "nombre_original", "group"]],
        on="sub_code", how="left",
    )
    meta_cols = [
        "patient", "initials_demo", "sex", "age", "etiology", "gcs",
        "outcome_crsr_raw", "deceased", "binary_outcome", "sheet",
    ]
    is_ctrl = cohort["diagnosis"] == "CTRL"
    part_pat = cohort.loc[~is_ctrl].merge(demo_pat[meta_cols], on="patient", how="left")
    part_ctl = cohort.loc[is_ctrl].merge(demo_ctl[meta_cols], on="patient", how="left")
    return pd.concat([part_pat, part_ctl], ignore_index=True).sort_values("record_id").reset_index(drop=True)


def merge_patients(cohort: pd.DataFrame) -> pd.DataFrame:
    """ANOX + TRAU → pacientes; conserva diagnosis_orig."""
    out = cohort.copy()
    out["diagnosis_orig"] = out["diagnosis"]
    out.loc[out["diagnosis"].isin(PATIENT_SOURCES), "diagnosis"] = PATIENT_LABEL
    return out


# ---------------------------------------------------------------------------
# Paso 1: conectividad
# ---------------------------------------------------------------------------

def _symmetrize_clip(m: np.ndarray) -> np.ndarray:
    m = 0.5 * (m + m.T)
    m = np.clip(m, -0.9999, 0.9999).astype(np.float64)
    np.fill_diagonal(m, 1.0)
    return m


def pearson_from_fc(mat: np.ndarray) -> np.ndarray:
    """Los .mat ya son Pearson ROI×ROI; se simetriza y se recorta a correlaciones válidas."""
    return _symmetrize_clip(np.asarray(mat, dtype=np.float64))


def oinfo_pair_vs_rest(corr: np.ndarray) -> np.ndarray:
    """I({i,j}; resto) gaussiana: 0.5 * [log det(R_ij) + log det(P_ij)].

    P = R^{-1}. No requiere series temporales; asume copula gaussiana sobre Pearson.
    """
    r = _symmetrize_clip(corr)
    n = r.shape[0]
    jitter = 1e-4
    r = r + np.eye(n) * jitter
    try:
        p = np.linalg.inv(r)
    except np.linalg.LinAlgError:
        p = np.linalg.pinv(r)

    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            r2 = r[np.ix_([i, j], [i, j])]
            p2 = p[np.ix_([i, j], [i, j])]
            s_r, logdet_r = np.linalg.slogdet(r2)
            s_p, logdet_p = np.linalg.slogdet(p2)
            if s_r <= 0 or s_p <= 0:
                val = 0.0
            else:
                val = 0.5 * (logdet_r + logdet_p)
            out[i, j] = out[j, i] = float(val)
    np.fill_diagonal(out, 0.0)
    return out


def load_connectivity(cohort: pd.DataFrame, kind: str) -> tuple[np.ndarray, pd.DataFrame]:
    mats, keep = [], []
    for _, row in cohort.iterrows():
        m = _load_fc_matrix(Path(row["mat_path"]))
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            continue
        if not np.isfinite(m).all():
            print("NaN/Inf, se descarta:", row["record_id"])
            continue
        if kind == "oinfo":
            c = oinfo_pair_vs_rest(m)
        else:
            c = pearson_from_fc(m)
        if not np.isfinite(c).all():
            print("conectividad no finita, se descarta:", row["record_id"])
            continue
        mats.append(c.astype(np.float32))
        keep.append(row)
    if not mats:
        raise ValueError("ninguna matriz FC usable")
    shapes = {m.shape for m in mats}
    if len(shapes) != 1:
        raise ValueError(f"tamaños de FC mixtos: {shapes}")
    X = np.stack(mats, axis=0)
    meta = pd.DataFrame(keep).reset_index(drop=True)
    print(f"Paso 1: {kind}  X={X.shape}  (N, n_roi, n_roi)")
    if X.shape[1] != 82:
        print(f"  aviso: atlas {X.shape[1]}×{X.shape[1]} (paper: 82). Se usa el tamaño real.")
    return X, meta


# ---------------------------------------------------------------------------
# Paso 2: grafos PyG
# ---------------------------------------------------------------------------

def matrices_to_data_list(X: np.ndarray, ages: np.ndarray, ids: list[str]):
    torch = _need_torch()
    from torch_geometric.data import Data
    from torch_geometric.utils import dense_to_sparse

    graphs = []
    kept_ids = []
    for i in range(len(X)):
        a = np.asarray(X[i], dtype=np.float32)
        w = np.abs(a).copy()
        np.fill_diagonal(w, 0.0)
        w[w <= EDGE_THRESH] = 0.0
        w_t = torch.tensor(w, dtype=torch.float32)
        edge_index, edge_attr = dense_to_sparse(w_t)
        if edge_index.numel() == 0:
            print("grafo sin aristas, se descarta:", ids[i])
            continue
        data = Data(
            x=torch.tensor(a, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([float(ages[i])], dtype=torch.float32),
            sid=torch.tensor([len(kept_ids)], dtype=torch.long),
        )
        graphs.append(data)
        kept_ids.append(ids[i])
    n_e = int(graphs[0].edge_index.size(1)) if graphs else 0
    print(f"Paso 2: {len(graphs)} grafos  nodos={X.shape[1]}  aristas={n_e}")
    return graphs, kept_ids


# ---------------------------------------------------------------------------
# Paso 3: interpolación de matrices (solo train) — DESACTIVADO
# Sobreajustaba (más sintéticas que sujetos reales). Se deja comentado.
# ---------------------------------------------------------------------------
#
# def augment_age_gaps(X: np.ndarray, ages: np.ndarray, ids: list[str]):
#     """Rellena años enteros faltantes: M_t = (1-α) M1 + α M2."""
#     ages = np.asarray(ages, dtype=float)
#     years = np.round(ages).astype(int)
#     ymin, ymax = int(years.min()), int(years.max())
#     have = set(years.tolist())
#     missing = [y for y in range(ymin, ymax + 1) if y not in have]
#     if not missing:
#         print("Paso 3: sin huecos de edad entera; no se interpola.")
#         return X, ages, ids, 0
#
#     Xs, ys, new_ids = [X], [ages], list(ids)
#     n_syn = 0
#     for t in missing:
#         lo = np.where(ages < t)[0]
#         hi = np.where(ages > t)[0]
#         if lo.size == 0 or hi.size == 0:
#             continue
#         i1 = lo[np.argmin(np.abs(ages[lo] - t))]
#         i2 = hi[np.argmin(np.abs(ages[hi] - t))]
#         a1, a2 = ages[i1], ages[i2]
#         if abs(a2 - a1) < 1e-6:
#             continue
#         alpha = (t - a1) / (a2 - a1)
#         m = (1.0 - alpha) * X[i1] + alpha * X[i2]
#         Xs.append(m[None, ...])
#         ys.append(np.array([float(t)]))
#         new_ids.append(f"SYN_{t}_{ids[i1]}_{ids[i2]}")
#         n_syn += 1
#
#     X_out = np.concatenate(Xs, axis=0)
#     y_out = np.concatenate(ys, axis=0)
#     print(f"Paso 3: +{n_syn} matrices sintéticas (huecos {len(missing)} años).")
#     return X_out, y_out, new_ids, n_syn


# ---------------------------------------------------------------------------
# Paso 4: GCN
# ---------------------------------------------------------------------------

def build_model(in_dim: int):
    torch = _need_torch()
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool

    class BrainGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_dim, HIDDEN)
            self.conv2 = GCNConv(HIDDEN, LATENT)
            self.lin = nn.Linear(LATENT, 1)
            self.dropout = DROPOUT

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            ew = data.edge_attr.view(-1) if data.edge_attr is not None else None
            x = F.relu(self.conv1(x, edge_index, edge_weight=ew))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv2(x, edge_index, edge_weight=ew))
            x = global_mean_pool(x, batch)
            return self.lin(x).view(-1)

    return BrainGCN()


def _set_seed(seed: int):
    torch = _need_torch()
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device():
    torch = _need_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(graphs, shuffle: bool):
    from torch_geometric.loader import DataLoader
    return DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_one(model, loader, opt, device):
    torch = _need_torch()
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        pred = model(batch)
        y = batch.y.view(-1)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        opt.step()
        total += float(loss.item()) * y.numel()
        n += int(y.numel())
    return total / max(n, 1)


def _eval_preds(model, loader, device):
    torch = _need_torch()
    model.eval()
    preds, ys, sids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).detach().cpu().numpy()
            y = batch.y.view(-1).detach().cpu().numpy()
            sid = batch.sid.view(-1).detach().cpu().numpy()
            preds.append(pred)
            ys.append(y)
            sids.append(sid)
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(sids).astype(int)


def fit_gcn(graphs, lr: float, epochs: int, device, seed: int = SEED):
    torch = _need_torch()
    _set_seed(seed)
    in_dim = int(graphs[0].x.size(1))
    model = build_model(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = _loader(graphs, shuffle=True)
    best_state, best_loss = None, float("inf")
    stale = 0
    for ep in range(1, epochs + 1):
        loss = train_one(model, loader, opt, device)
        if loss < best_loss - 1e-6:
            best_loss = loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, dict(train_mse=best_loss, epochs_run=min(ep, epochs))


def grid_cv(train_graphs, ages_train: np.ndarray, device):
    n = len(train_graphs)
    n_splits = min(N_FOLDS, max(2, n // 4))
    print(f"Paso 4: grid search  folds={n_splits}  lr={GRID_LR}  epochs={GRID_EPOCHS}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for lr in GRID_LR:
        for ep in GRID_EPOCHS:
            rmses = []
            for tr, va in kf.split(np.arange(n)):
                g_tr = [train_graphs[i] for i in tr]
                g_va = [train_graphs[i] for i in va]
                model, _ = fit_gcn(g_tr, lr=lr, epochs=ep, device=device)
                pred, y, _ = _eval_preds(model, _loader(g_va, False), device)
                rmses.append(float(np.sqrt(mean_squared_error(y, pred))))
            mean_rmse = float(np.mean(rmses))
            rows.append({"lr": lr, "epochs": ep, "cv_rmse": mean_rmse})
            print(f"  lr={lr:.1e}  ep={ep}  CV-RMSE={mean_rmse:.2f}")
    best = min(rows, key=lambda r: r["cv_rmse"])
    print("mejor:", best)
    return best, rows


def loo_preds(graphs, lr: float, epochs: int, device):
    """Edad predicha leave-one-out (solo para evaluar CTRL)."""
    n = len(graphs)
    preds = np.zeros(n, dtype=np.float64)
    ys = np.zeros(n, dtype=np.float64)
    print(f"LOO CTRL: {n} modelos…")
    for i in range(n):
        g_tr = [graphs[j] for j in range(n) if j != i]
        model, _ = fit_gcn(g_tr, lr=lr, epochs=epochs, device=device)
        pred, y, _ = _eval_preds(model, _loader([graphs[i]], False), device)
        preds[i] = float(pred[0])
        ys[i] = float(y[0])
        print(f"  LOO {i + 1}/{n}  pred={preds[i]:.1f}  age={ys[i]:.1f}")
    return preds, ys


def age_strata(ages: np.ndarray, n_bins: int = 4) -> np.ndarray:
    ages = np.asarray(ages, dtype=float)
    n_bins = min(n_bins, max(2, len(ages) // 5))
    try:
        bins = pd.qcut(ages, q=n_bins, labels=False, duplicates="drop")
        return np.asarray(bins, dtype=int)
    except ValueError:
        return np.zeros(len(ages), dtype=int)


# ---------------------------------------------------------------------------
# Paso 5: métricas + BAG
# ---------------------------------------------------------------------------

def mde(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.sign(y_pred - y_true)
    d[np.isclose(y_pred, y_true)] = 0.0
    return float(np.mean(d))


def cohens_f2(r2: float) -> float:
    if r2 >= 1.0:
        return float("inf")
    return float(r2 / max(1.0 - r2, 1e-12))


def eval_holdout(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if len(y_true) >= 3:
        r, p = stats.pearsonr(y_true, y_pred)
    else:
        r, p = np.nan, np.nan
    slope, intercept, r_lin, p_lin, _ = stats.linregress(y_true, y_pred) if len(y_true) >= 3 else (
        np.nan, np.nan, np.nan, np.nan, np.nan)
    return {
        "n_test": int(len(y_true)),
        "r2": r2,
        "pearson_r": float(r),
        "pearson_p": float(p),
        "linreg_slope": float(slope),
        "linreg_p": float(p_lin),
        "rmse": rmse,
        "mde": mde(y_true, y_pred),
        "cohens_f2": cohens_f2(r2),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
    }


# ---------------------------------------------------------------------------
# Paso 6: GBR sobre BAG (covariables clínicas disponibles)
# ---------------------------------------------------------------------------

def factor_analysis(bag_df: pd.DataFrame, table_dir: Path) -> dict | None:
    work = bag_df.copy()
    y = pd.to_numeric(work["BAG"], errors="coerce")
    parts = []
    names = []

    if "sex" in work.columns:
        sex = work["sex"].astype(str).str.upper().str.strip().replace({"H": "M", "MALE": "M", "FEMALE": "F"})
        parts.append((sex == "M").astype(float).to_numpy().reshape(-1, 1))
        names.append("sex_M")

    for col, prefix in (("diagnosis", "dx"), ("etiology", "etio")):
        if col not in work.columns:
            continue
        enc = OneHotEncoder(handle_unknown="ignore")
        try:
            enc.set_params(sparse_output=False)
        except ValueError:
            enc.set_params(sparse=False)
        arr = enc.fit_transform(work[[col]].astype(str))
        parts.append(arr)
        names.extend([f"{prefix}_{c}" for c in enc.categories_[0]])

    for col in ("gcs", "binary_outcome"):
        if col in work.columns:
            v = pd.to_numeric(work[col], errors="coerce")
            if v.notna().sum() >= 8:
                med = float(v.median()) if v.notna().any() else 0.0
                parts.append(v.fillna(med).to_numpy().reshape(-1, 1))
                names.append(col)

    if not parts:
        print("Paso 6: sin predictores clínicos; se omite GBR.")
        return None

    X = np.hstack(parts)
    mask = np.isfinite(y.to_numpy(float))
    X, yv = X[mask], y.to_numpy(float)[mask]
    if len(yv) < 10:
        print("Paso 6: n demasiado chico para GBR.")
        return None

    gbr = GradientBoostingRegressor(random_state=SEED, n_estimators=80, max_depth=2)
    gbr.fit(X, yv)
    mdi = gbr.feature_importances_
    perm = permutation_importance(gbr, X, yv, n_repeats=20, random_state=SEED, scoring="r2")

    shap_mean = None
    try:
        import shap
        expl = shap.TreeExplainer(gbr)
        sv = expl.shap_values(X)
        shap_mean = np.abs(sv).mean(axis=0)
    except Exception as e:
        print("Paso 6: SHAP no disponible (", e, "). Sigue MDI + permutation.")

    imp = pd.DataFrame({
        "feature": names,
        "mdi": mdi,
        "permutation_mean": perm.importances_mean,
        "permutation_std": perm.importances_std,
    })
    if shap_mean is not None:
        imp["shap_abs_mean"] = shap_mean
        zcols = ["mdi", "permutation_mean", "shap_abs_mean"]
    else:
        zcols = ["mdi", "permutation_mean"]
    z = (imp[zcols] - imp[zcols].mean()) / imp[zcols].std().replace(0, 1)
    imp["combined_z"] = z.mean(axis=1)
    imp = imp.sort_values("combined_z", ascending=False)
    path = table_dir / f"gcn_bag_factor_importance{SUFFIX}.csv"
    imp.to_csv(path, index=False)
    print("Paso 6: GBR sobre BAG (clínica COMA; no hay vars de país).")
    print(imp.head(8).to_string(index=False))
    print("guardado:", path)
    return {
        "n": int(len(yv)),
        "gbr_r2_insample": float(gbr.score(X, yv)),
        "features": names,
        "note": "sin contaminación/Gini/carga-país; predictores individuales de COMA",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GCN BAG en COMA+ReDLaT")
    p.add_argument("--connectivity", choices=("pearson", "oinfo"), default="pearson")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-factors", action="store_true")
    p.add_argument("--no-grid", action="store_true", help="salta CV; usa lr=1e-3 epochs=150")
    return p.parse_args()


def main():
    args = parse_args()
    torch = _need_torch()
    device = _device()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    GCN_DIR.mkdir(parents=True, exist_ok=True)
    _set_seed(SEED)
    print("device:", device)
    print("connectivity:", args.connectivity)
    print("suffix:", SUFFIX)

    if not FC_ROOT.exists():
        raise FileNotFoundError(f"FC_ROOT no existe: {FC_ROOT}")
    if not DEMO_XLSX.exists():
        raise FileNotFoundError(f"Falta {DEMO_XLSX}")
    if not IDMAP_PATH.exists():
        raise FileNotFoundError(f"Falta {IDMAP_PATH}")

    cohort = build_cohort()
    cohort = cohort.dropna(subset=["age"]).copy()
    print("contraste: CTRL(COMA+ReDLaT CN) vs", PATIENT_LABEL)
    print("cohorte con edad:", len(cohort), cohort.groupby("diagnosis").size().to_dict())
    if "source" in cohort.columns:
        print("CTRL por source:",
              cohort.loc[cohort["diagnosis"] == "CTRL"].groupby("source").size().to_dict())
    if "diagnosis_orig" in cohort.columns:
        print("orig:", cohort.groupby("diagnosis_orig").size().to_dict())

    X, meta = load_connectivity(cohort, args.connectivity)
    ages = meta["age"].to_numpy(float)
    ids = meta["record_id"].astype(str).tolist()
    is_ctrl = meta["diagnosis"].to_numpy() == "CTRL"
    n_ctrl = int(is_ctrl.sum())
    n_pat = int((~is_ctrl).sum())
    print(f"entrenamiento: SOLO CTRL COMA+ReDLaT (n={n_ctrl}); aplicación: {PATIENT_LABEL} (n={n_pat})")
    if n_ctrl < 8:
        raise ValueError(f"pocos CTRL para entrenar GCN: n={n_ctrl}")

    idx_ctrl = np.where(is_ctrl)[0]
    X_ctrl, y_ctrl, id_ctrl = X[idx_ctrl], ages[idx_ctrl], [ids[i] for i in idx_ctrl]
    n_syn = 0
    print("Paso 3: omitido (sin matrices interpoladas).")

    g_ctrl, id_ctrl_kept = matrices_to_data_list(X_ctrl, y_ctrl, id_ctrl)
    in_dim = int(g_ctrl[0].x.size(1))

    if args.no_grid:
        best = {"lr": 1e-3, "epochs": 150, "cv_rmse": None}
        cv_rows = []
    else:
        best, cv_rows = grid_cv(g_ctrl, y_ctrl, device)

    print("modelo final: todos los CTRL…")
    model, fit_info = fit_gcn(g_ctrl, lr=best["lr"], epochs=int(best["epochs"]), device=device)
    torch.save(model.state_dict(), GCN_DIR / f"gcn_coma{SUFFIX}.pt")

    pred_ctrl_fit, y_ctrl_fit, sid_ctrl = _eval_preds(model, _loader(g_ctrl, False), device)
    metrics_ctrl_fit = eval_holdout(y_ctrl_fit, pred_ctrl_fit)
    print("CTRL (fit, optimista):",
          {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics_ctrl_fit.items()})

    metrics_ctrl_loo = None
    metrics_ctrl_holdout = None
    holdout_ids: set[str] = set()
    if n_ctrl <= LOO_MAX_N:
        pred_ctrl_loo, y_ctrl_loo = loo_preds(
            g_ctrl, lr=best["lr"], epochs=int(best["epochs"]), device=device,
        )
        metrics_ctrl_loo = eval_holdout(y_ctrl_loo, pred_ctrl_loo)
        print("CTRL (LOO, honesto):",
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics_ctrl_loo.items()})
        loo_map = {id_ctrl_kept[i]: pred_ctrl_loo[i] for i in range(len(id_ctrl_kept))}
    else:
        print(f"n_CTRL={n_ctrl} > {LOO_MAX_N}: se usa hold-out {TEST_SIZE:.0%} (sin LOO).")
        strata = age_strata(y_ctrl)
        try:
            tr_h, te_h = next(iter(StratifiedShuffleSplit(
                n_splits=1, test_size=TEST_SIZE, random_state=SEED,
            ).split(np.arange(n_ctrl), strata)))
        except ValueError:
            tr_h, te_h = train_test_split(np.arange(n_ctrl), test_size=TEST_SIZE, random_state=SEED)
        g_tr_h = [g_ctrl[i] for i in tr_h]
        g_te_h = [g_ctrl[i] for i in te_h]
        model_h, _ = fit_gcn(g_tr_h, lr=best["lr"], epochs=int(best["epochs"]), device=device)
        pred_h, y_h, sid_h = _eval_preds(model_h, _loader(g_te_h, False), device)
        metrics_ctrl_holdout = eval_holdout(y_h, pred_h)
        print("CTRL (hold-out, honesto):",
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics_ctrl_holdout.items()})
        holdout_ids = {id_ctrl_kept[i] for i in te_h}
        loo_map = {id_ctrl_kept[int(s)]: float(pred_h[j]) for j, s in enumerate(sid_h)}

    g_all, id_all_kept = matrices_to_data_list(X, ages, ids)
    pred_all, _, sid_all = _eval_preds(model, _loader(g_all, False), device)
    id_all_ord = [id_all_kept[i] for i in sid_all]
    bag_all = meta.set_index("record_id").loc[id_all_ord].reset_index()
    bag_all["predicted_age"] = pred_all
    bag_all["BAG"] = pred_all - bag_all["age"].to_numpy(float)
    bag_all["abs_err"] = np.abs(bag_all["BAG"])
    bag_all["predicted_age_loo"] = np.nan
    bag_all["BAG_loo"] = np.nan
    bag_all["abs_err_loo"] = np.nan

    for i, rid in enumerate(bag_all["record_id"].astype(str)):
        if rid in loo_map:
            p_loo = float(loo_map[rid])
            bag_all.loc[i, "predicted_age_loo"] = p_loo
            bag_all.loc[i, "BAG_loo"] = p_loo - float(bag_all.loc[i, "age"])
            bag_all.loc[i, "abs_err_loo"] = abs(p_loo - float(bag_all.loc[i, "age"]))

    split = np.where(bag_all["diagnosis"] == "CTRL", "train_ctrl", "apply_pacientes")
    if holdout_ids:
        split = np.where(
            bag_all["record_id"].astype(str).isin(holdout_ids),
            "holdout_ctrl",
            split,
        )
    bag_all["split"] = split
    bag_all["region"] = bag_all["source"].fillna("COMA") if "source" in bag_all.columns else "COMA"

    pat = bag_all[bag_all["diagnosis"] == PATIENT_LABEL]
    metrics_pat = eval_holdout(pat["age"].to_numpy(float), pat["predicted_age"].to_numpy(float))
    print("pacientes (modelo CTRL, nunca vistos):",
          {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics_pat.items()})
    if bag_all["BAG_loo"].notna().any():
        print("BAG medio CTRL (LOO/hold-out):",
              round(float(bag_all.loc[bag_all["BAG_loo"].notna(), "BAG_loo"].mean()), 2))
    print("BAG medio pacientes:", round(float(pat["BAG"].mean()), 2))

    cols = [
        "record_id", "diagnosis", "diagnosis_orig", "source", "age", "predicted_age", "BAG", "abs_err",
        "predicted_age_loo", "BAG_loo", "abs_err_loo",
        "sex", "region", "etiology", "gcs", "binary_outcome", "sub_code", "split",
    ]
    cols = [c for c in cols if c in bag_all.columns]
    bag_path = TABLE_DIR / f"bag_gcn_coma{SUFFIX}.csv"
    bag_all[cols].to_csv(bag_path, index=False)
    bag_all_path = TABLE_DIR / f"bag_gcn_coma_all{SUFFIX}.csv"
    bag_all[cols].to_csv(bag_all_path, index=False)

    bag_for_gbr = bag_all.copy()
    if bag_for_gbr["BAG_loo"].notna().any():
        bag_for_gbr["BAG"] = bag_for_gbr["BAG_loo"].fillna(bag_for_gbr["BAG"])

    factor_info = None
    if not args.no_factors:
        factor_info = factor_analysis(bag_for_gbr, TABLE_DIR)

    payload = {
        "contrast": "CTRL(COMA+ReDLaT_CN)_vs_pacientes",
        "train": "CTRL_only_coma_plus_redlat",
        "patient_sources": sorted(PATIENT_SOURCES),
        "patient_label": PATIENT_LABEL,
        "suffix": SUFFIX,
        "connectivity": args.connectivity,
        "n_roi": in_dim,
        "n_subjects": int(len(ids)),
        "n_train_ctrl": n_ctrl,
        "n_ctrl_coma": int(((meta["diagnosis"] == "CTRL") & (meta["source"] == "coma")).sum())
        if "source" in meta.columns else None,
        "n_ctrl_redlat": int(((meta["diagnosis"] == "CTRL") & (meta["source"] == "redlat")).sum())
        if "source" in meta.columns else None,
        "n_apply_pacientes": n_pat,
        "n_synthetic_train": int(n_syn),
        "augment": False,
        "architecture": {"in": in_dim, "gcn1": HIDDEN, "gcn2": LATENT, "out": 1, "dropout": DROPOUT},
        "best_hp": best,
        "cv": cv_rows,
        "fit": fit_info,
        "ctrl_fit": metrics_ctrl_fit,
        "ctrl_loo": metrics_ctrl_loo,
        "ctrl_holdout": metrics_ctrl_holdout,
        "pacientes": metrics_pat,
        "factor_analysis": factor_info,
        "edge_thresh": EDGE_THRESH,
        "seed": SEED,
        "notes": {
            "atlas": "AAL (tamaño real de las matrices, no 82)",
            "pearson": "FC ya precomputada en .mat; no hay BOLD crudo",
            "oinfo": "I(par; resto) gaussiana sobre la FC, no Ω de series",
            "bag": "edad entrenada solo en CTRL (COMA+ReDLaT); BAG en CTRL y pacientes",
            "loo": f"LOO solo si n_CTRL<={LOO_MAX_N}; si no, hold-out {TEST_SIZE}",
            "step6": "sin vars de país; GBR clínico opcional",
        },
    }
    met_path = TABLE_DIR / f"metrics_gcn_coma{SUFFIX}.json"
    with open(met_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("guardado:", bag_path)
    print("guardado:", bag_all_path)
    print("guardado:", met_path)
    print("pesos:", GCN_DIR / f"gcn_coma{SUFFIX}.pt")
    print("listo")


if __name__ == "__main__":
    main()
