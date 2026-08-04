"""
Pipeline CTRL(COMA+ReDLaT CN) vs pacientes(COMA ANOX+TRAU).

Mismo flujo que train_coma_bag_ctrl_pacientes.py, reforzando controles con CN ReDLaT.
Tablas en COMA/ con sufijo _RedLat; VAE/embeddings en outputs/coma/.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Thesis" / "Code"))

from src.coma_data_io import (  # noqa: E402
    DEFAULT_GROUP_MAP,
    TOPO_COLUMNS,
    list_inflamacion_mats,
    load_fc_vectors_from_cohort,
)
from src.coma_graph import compute_topo_table  # noqa: E402
from src.config import DEFAULT_PATHS  # noqa: E402
from src.data_io import (  # noqa: E402
    extract_fc_record_id_from_filename,
    list_fc_files,
    read_metadata,
)
from src.embeddings import encode_mu, save_embeddings  # noqa: E402
from src.utils_seed import set_global_seed  # noqa: E402
from src.vae_train import load_vae_from_dir, train_vae_final  # noqa: E402

# --- config ---
FC_ROOT = Path("/home/usuario/disco1/proyectos/2024-autoencoders/databases/fc/inflamacion")
DEMO_XLSX = ROOT / "demographics.xlsx"
IDMAP_PATH = ROOT / "id_map_coma.csv"
TABLE_DIR = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "coma"
VAE_DIR = OUT / "vae_coma_ctrl_pacientes_RedLat"
SUFFIX = "_RedLat"

PATIENT_LABEL = "pacientes"
PATIENT_SOURCES = {"ANOX", "TRAU"}
CTRL_LABEL = "CTRL"

SEED = 42
FISHER_Z = True
THRESHOLD_TOPO = 0.20
RIDGE_ALPHA = 267.7
REUSE_VAE = True
USE_SEX = True

VAE_HP = dict(
    hidden_dims=[512],
    latent_dim=64,
    beta_target=0.056663247229966504,
    warmup_ep=73,
    l2_reg=2.897389671945472e-07,
    lr=0.001892443497356961,
    recon_kind="mae",
    drop_rate=0.036861053246000725,
    activation="elu",
    norm_kind="layernorm",
    batch_size=8,
    clipnorm=1.0,
    epochs=96,
)

SUB_RE = re.compile(r"(ANOX|TC|CONTROLES)\d{3}", re.I)


def encode_sex(sex: np.ndarray) -> np.ndarray:
    """M=1, F=0; NaN → 0.5 (neutro)."""
    out = np.full(len(sex), 0.5, dtype=np.float32)
    s = pd.Series(sex).astype(str).str.upper().str.strip()
    out[s.isin(["M", "MALE", "H"]).to_numpy()] = 1.0
    out[s.isin(["F", "FEMALE"]).to_numpy()] = 0.0
    return out


def build_features(Z: np.ndarray, topo: np.ndarray, sex: np.ndarray | None = None) -> np.ndarray:
    parts = [np.asarray(Z, np.float32), np.asarray(topo, np.float32)]
    if sex is not None:
        parts.append(encode_sex(sex).reshape(-1, 1))
    return np.hstack(parts)


def mat_sub_code(path) -> str | None:
    m = SUB_RE.search(Path(path).name)
    return m.group(0).upper() if m else None


def normalize_sex_col(s: pd.Series) -> pd.Series:
    return (
        s.astype(str).str.strip()
        .replace({
            "H": "M", "Male": "M", "MALE": "M", "male": "M",
            "Female": "F", "FEMALE": "F", "female": "F",
        })
    )


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


def link_demographics_coma(cohort: pd.DataFrame) -> pd.DataFrame:
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
    out = pd.concat([part_pat, part_ctl], ignore_index=True)
    return out.sort_values("record_id").reset_index(drop=True)


def build_coma_part() -> pd.DataFrame:
    cohort = list_inflamacion_mats(FC_ROOT, group_map=DEFAULT_GROUP_MAP)
    cohort = link_demographics_coma(cohort)
    cohort = cohort.copy()
    cohort["diagnosis_orig"] = cohort["diagnosis"]
    cohort.loc[cohort["diagnosis"].isin(PATIENT_SOURCES), "diagnosis"] = PATIENT_LABEL
    cohort["source"] = "coma"
    cohort["sex"] = normalize_sex_col(cohort["sex"])
    return cohort


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
            "diagnosis": CTRL_LABEL,
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


def step_cohort() -> pd.DataFrame:
    if not FC_ROOT.exists():
        raise FileNotFoundError(f"FC_ROOT no existe: {FC_ROOT}")
    if not DEMO_XLSX.exists():
        raise FileNotFoundError(f"Falta {DEMO_XLSX}")
    if not IDMAP_PATH.exists():
        raise FileNotFoundError(f"Falta {IDMAP_PATH}")

    coma = build_coma_part()
    redlat_cn = build_redlat_cn()

    # columnas alineadas
    cols = sorted(set(coma.columns) | set(redlat_cn.columns))
    for c in cols:
        if c not in coma.columns:
            coma[c] = np.nan
        if c not in redlat_cn.columns:
            redlat_cn[c] = np.nan

    cohort = pd.concat([coma[cols], redlat_cn[cols]], ignore_index=True)
    if cohort["record_id"].duplicated().any():
        dups = cohort.loc[cohort["record_id"].duplicated(), "record_id"].unique()[:5]
        raise ValueError(f"record_id duplicados entre COMA y ReDLaT: {list(dups)}")

    cohort = cohort.sort_values("record_id").reset_index(drop=True)
    path = TABLE_DIR / f"cohort_coma_ctrl_pacientes{SUFFIX}.csv"
    cohort.to_csv(path, index=False)

    print("cohorte:", len(cohort))
    print("diagnosis:", cohort.groupby("diagnosis").size().to_dict())
    print("source:", cohort.groupby("source").size().to_dict())
    print("CTRL por source:",
          cohort.loc[cohort["diagnosis"] == CTRL_LABEL].groupby("source").size().to_dict())
    print("orig:", cohort.groupby("diagnosis_orig").size().to_dict())
    print("con age:", int(cohort["age"].notna().sum()))
    print("con sex:", int(cohort["sex"].notna().sum()))
    return cohort


def step_topo(cohort: pd.DataFrame) -> pd.DataFrame:
    topo = compute_topo_table(cohort, threshold=THRESHOLD_TOPO, apply_fisher_z=FISHER_Z)
    topo.to_csv(TABLE_DIR / f"graph_metrics_coma_ctrl_pacientes{SUFFIX}.csv", index=False)
    print("TOPO:", topo.shape)
    return topo


def step_vae(cohort: pd.DataFrame) -> tuple[object, np.ndarray, list[str]]:
    ids = cohort["record_id"].tolist()
    X = load_fc_vectors_from_cohort(cohort, ids, apply_fisher_z=FISHER_Z)
    print("FC:", X.shape)

    if REUSE_VAE and (VAE_DIR / "vae.weights.h5").exists():
        print("reusando VAE:", VAE_DIR)
        vae = load_vae_from_dir(VAE_DIR)
    else:
        print("entrenando VAE CTRL(+ReDLaT) vs pacientes…")
        vae, hist = train_vae_final(X, VAE_DIR, seed=SEED, **VAE_HP)
        with open(OUT / f"vae_history_summary_ctrl_pacientes{SUFFIX}.json", "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in hist.items()}, f)

    Z = encode_mu(vae.encoder, X)
    save_embeddings(OUT / f"embeddings_mu_ctrl_pacientes{SUFFIX}", ids, Z)
    print("Z:", Z.shape)
    return vae, Z, ids


def step_ridge_bag(
    cohort: pd.DataFrame,
    topo: pd.DataFrame,
    Z: np.ndarray,
    ids: list[str],
) -> pd.DataFrame:
    meta = cohort.set_index("record_id").loc[ids]
    ages = meta["age"].to_numpy(float)
    dx = meta["diagnosis"].values
    sex = meta["sex"].values
    topo_mat = topo.set_index("record_id").loc[ids, TOPO_COLUMNS].to_numpy(np.float32)
    X = build_features(Z, topo_mat, sex=sex if USE_SEX else None)
    feat_desc = "Z+TOPO+sex" if USE_SEX else "Z+TOPO"
    print(f"features: {feat_desc}  X={X.shape}")

    mask = np.isfinite(ages)
    if mask.sum() < 3:
        raise ValueError("pocos sujetos con edad para Ridge")

    X_ok, y_ok, dx_ok = X[mask], ages[mask], dx[mask]

    loo = LeaveOneOut()
    y_hat_cv = np.zeros_like(y_ok)
    for tr, te in loo.split(X_ok):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_ok[tr])
        Xte = sc.transform(X_ok[te])
        ridge = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
        ridge.fit(Xtr, y_ok[tr])
        y_hat_cv[te] = ridge.predict(Xte)

    mae_cv = mean_absolute_error(y_ok, y_hat_cv)
    rmse_cv = float(np.sqrt(mean_squared_error(y_ok, y_hat_cv)))
    r2_cv = r2_score(y_ok, y_hat_cv)

    mae_by_dx = {}
    for g in sorted(set(dx_ok)):
        m = dx_ok == g
        if m.sum() == 0:
            continue
        mae_by_dx[g] = float(mean_absolute_error(y_ok[m], y_hat_cv[m]))

    print(f"LOO MAE={mae_cv:.2f}  RMSE={rmse_cv:.2f}  R2={r2_cv:.3f}  n={len(y_ok)}")
    print("MAE LOO por dx:", {k: round(v, 2) for k, v in mae_by_dx.items()})

    sc = StandardScaler()
    ridge = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
    ridge.fit(sc.fit_transform(X_ok), y_ok)
    pred_all = ridge.predict(sc.transform(X))

    bag_df = pd.DataFrame({
        "record_id": ids,
        "diagnosis": dx,
        "age": ages,
        "predicted_age": pred_all,
        "predicted_age_loo": np.nan,
        "BAG": pred_all - ages,
        "abs_err": np.abs(pred_all - ages),
        "BAG_loo": np.nan,
        "abs_err_loo": np.nan,
    })
    bag_df.loc[mask, "predicted_age_loo"] = y_hat_cv
    bag_df.loc[mask, "BAG_loo"] = y_hat_cv - y_ok
    bag_df.loc[mask, "abs_err_loo"] = np.abs(y_hat_cv - y_ok)

    extra_cols = [
        "sub_code", "patient", "sex", "gcs", "binary_outcome",
        "etiology", "source", "diagnosis_orig",
    ]
    extra_cols = [c for c in extra_cols if c in cohort.columns]
    extra = cohort.set_index("record_id")[extra_cols]
    bag_df = bag_df.join(extra, on="record_id")

    bag_path = TABLE_DIR / f"bag_coma_ctrl_pacientes{SUFFIX}.csv"
    bag_df.to_csv(bag_path, index=False)

    metrics = {
        "contrast": "CTRL(COMA+ReDLaT_CN)_vs_pacientes",
        "patient_sources": sorted(PATIENT_SOURCES),
        "patient_label": PATIENT_LABEL,
        "ctrl_sources": ["coma", "redlat"],
        "features": feat_desc,
        "use_sex": bool(USE_SEX),
        "n_with_age": int(mask.sum()),
        "n_total": int(len(ids)),
        "n_ctrl_coma": int(((cohort["diagnosis"] == CTRL_LABEL) & (cohort["source"] == "coma")).sum()),
        "n_ctrl_redlat": int(((cohort["diagnosis"] == CTRL_LABEL) & (cohort["source"] == "redlat")).sum()),
        "n_pacientes": int((cohort["diagnosis"] == PATIENT_LABEL).sum()),
        "mae_loo": float(mae_cv),
        "rmse_loo": float(rmse_cv),
        "r2_loo": float(r2_cv),
        "mae_loo_by_dx": mae_by_dx,
        "mae_fit_all": float(mean_absolute_error(y_ok, ridge.predict(sc.transform(X_ok)))),
        "ridge_alpha": RIDGE_ALPHA,
        "threshold_topo": THRESHOLD_TOPO,
        "vae": {k: (list(v) if isinstance(v, list) else v) for k, v in VAE_HP.items()},
        "seed": SEED,
        "suffix": SUFFIX,
    }
    with open(TABLE_DIR / f"metrics_coma_ctrl_pacientes{SUFFIX}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("guardado:", bag_path)
    print(bag_df.groupby("diagnosis")[["age", "predicted_age", "BAG", "abs_err_loo"]]
          .mean().round(2))
    return bag_df


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    VAE_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)
    print("TABLE_DIR:", TABLE_DIR)
    print("OUT:", OUT)
    print(f"contraste: CTRL(COMA+ReDLaT CN) vs {PATIENT_LABEL}  suffix={SUFFIX}")

    cohort = step_cohort()
    topo = step_topo(cohort)
    _, Z, ids = step_vae(cohort)
    step_ridge_bag(cohort, topo, Z, ids)
    print("listo")


if __name__ == "__main__":
    main()
