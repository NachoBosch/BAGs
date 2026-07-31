"""
Pipeline COMA-only (equivalente nb09 §1–3b, sin ReDLaT).

1. Inventario FC + demografía
2. TOPO (umbral 0.20)
3. β-VAE entrenado en FC de coma
4. Ridge(Z+TOPO → edad) en coma + BAG/MAE
5. CSV en outputs/coma/
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
from src.embeddings import encode_mu, save_embeddings  # noqa: E402
from src.utils_seed import set_global_seed  # noqa: E402
from src.vae_train import load_vae_from_dir, train_vae_final  # noqa: E402

# --- config ---
FC_ROOT = Path("/home/usuario/disco1/proyectos/2024-autoencoders/databases/fc/inflamacion")
DEMO_XLSX = ROOT / "demographics.xlsx"
IDMAP_PATH = ROOT / "id_map_coma.csv"
OUT = ROOT / "outputs" / "coma"
VAE_DIR = OUT / "vae_coma"

SEED = 42
FISHER_Z = True
THRESHOLD_TOPO = 0.20
RIDGE_ALPHA = 267.7
REUSE_VAE = True

# mismos hiperparámetros que nb09; batch_size bajado por n≈42
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


def build_z_topo(Z: np.ndarray, topo: np.ndarray) -> np.ndarray:
    return np.hstack([np.asarray(Z, np.float32), np.asarray(topo, np.float32)])


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
        df["sex"] = df["sex"].replace({"H": "M", "F": "F"})

    return demo_pat, demo_ctl


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
    out = pd.concat([part_pat, part_ctl], ignore_index=True)
    return out.sort_values("record_id").reset_index(drop=True)


def step_cohort() -> pd.DataFrame:
    if not FC_ROOT.exists():
        raise FileNotFoundError(f"FC_ROOT no existe: {FC_ROOT}")
    if not DEMO_XLSX.exists():
        raise FileNotFoundError(f"Falta {DEMO_XLSX}")
    if not IDMAP_PATH.exists():
        raise FileNotFoundError(f"Falta {IDMAP_PATH}")

    cohort = list_inflamacion_mats(FC_ROOT, group_map=DEFAULT_GROUP_MAP)
    cohort = link_demographics(cohort)
    cohort.to_csv(OUT / "cohort_coma_with_demographics.csv", index=False)
    print("cohorte:", len(cohort))
    print(cohort.groupby("diagnosis").size())
    print("con age:", int(cohort["age"].notna().sum()))
    return cohort


def step_topo(cohort: pd.DataFrame) -> pd.DataFrame:
    topo = compute_topo_table(cohort, threshold=THRESHOLD_TOPO, apply_fisher_z=FISHER_Z)
    topo.to_csv(OUT / "graph_metrics_coma.csv", index=False)
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
        print("entrenando VAE COMA…")
        vae, hist = train_vae_final(X, VAE_DIR, seed=SEED, **VAE_HP)
        with open(OUT / "vae_history_summary.json", "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in hist.items()}, f)

    Z = encode_mu(vae.encoder, X)
    save_embeddings(OUT / "embeddings_mu", ids, Z)
    print("Z:", Z.shape)
    return vae, Z, ids


def step_ridge_bag(
    cohort: pd.DataFrame,
    topo: pd.DataFrame,
    Z: np.ndarray,
    ids: list[str],
) -> pd.DataFrame:
    ages = cohort.set_index("record_id").loc[ids, "age"].to_numpy(float)
    dx = cohort.set_index("record_id").loc[ids, "diagnosis"].values
    topo_mat = topo.set_index("record_id").loc[ids, TOPO_COLUMNS].to_numpy(np.float32)
    X = build_z_topo(Z, topo_mat)

    mask = np.isfinite(ages)
    if mask.sum() < 3:
        raise ValueError("pocos sujetos con edad para Ridge")

    X_ok, y_ok, ids_ok, dx_ok = X[mask], ages[mask], np.array(ids)[mask], dx[mask]

    # MAE por leave-one-out (evaluación honesta en n chico)
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

    # modelo final en todos (para BAG almacenado)
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

    # extras demográficos
    extra = cohort.set_index("record_id")[
        ["sub_code", "patient", "sex", "gcs", "binary_outcome", "etiology"]
    ]
    bag_df = bag_df.join(extra, on="record_id")

    bag_df.to_csv(OUT / "bag_coma_trained_on_coma.csv", index=False)

    metrics = {
        "n_with_age": int(mask.sum()),
        "n_total": int(len(ids)),
        "mae_loo": float(mae_cv),
        "rmse_loo": float(rmse_cv),
        "r2_loo": float(r2_cv),
        "mae_loo_by_dx": mae_by_dx,
        "mae_fit_all": float(mean_absolute_error(y_ok, ridge.predict(sc.transform(X_ok)))),
        "ridge_alpha": RIDGE_ALPHA,
        "threshold_topo": THRESHOLD_TOPO,
        "vae": {k: (list(v) if isinstance(v, list) else v) for k, v in VAE_HP.items()},
        "seed": SEED,
    }
    with open(OUT / "metrics_coma_bag.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("guardado:", OUT / "bag_coma_trained_on_coma.csv")
    print(bag_df.groupby("diagnosis")[["age", "predicted_age", "BAG", "abs_err_loo"]]
          .mean().round(2))
    return bag_df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VAE_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)
    print("OUT:", OUT)

    cohort = step_cohort()
    topo = step_topo(cohort)
    _, Z, ids = step_vae(cohort)
    step_ridge_bag(cohort, topo, Z, ids)
    print("listo")


if __name__ == "__main__":
    main()
