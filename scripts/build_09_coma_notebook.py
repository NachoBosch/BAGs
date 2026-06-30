#!/usr/bin/env python3
"""Generate 09_coma.ipynb — coma + transfer Z+TOPO + full SW/criticality."""

import json
from pathlib import Path


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t: str) -> dict:
    return {
        "cell_type": "code", "metadata": {},
        "source": t.splitlines(keepends=True),
        "outputs": [], "execution_count": None,
    }


cells = [
    md(
        "# 09 — Coma / inflamación + edad cerebral transferida (Z + TOPO)\n\n"
        "Pipeline coma + comparación con data-iipsi (CN, AD, FTD). "
        "Mundo pequeño y criticalidad alineados con nb08, con **más sujetos por grupo** "
        "y **N_NULL = 50**."
    ),

    code(
        """# === Configuración ===
import sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, "Thesis/Code")

from src.config import DEFAULT_PATHS, ExperimentConfig
from src.utils_seed import set_global_seed
from src.cohort import build_final_cohort_df
from src.splits import make_holdout_split
from src.data_io import load_fc_vectors_for_ids, vector_to_matrix
from src.vae_train import train_vae_final, load_vae_from_dir
from src.embeddings import encode_mu
from src.metrics import regression_metrics
from src.coma_data_io import (
    list_inflamacion_mats, load_fc_vectors_from_cohort,
    TOPO_COLUMNS, DEFAULT_GROUP_MAP,
)
from src.coma_graph import compute_topo_table, threshold_fixed, threshold_proportional
from src.utils_ids import normalize_record_id

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

FC_ROOT = Path("/home/usuario/disco1/proyectos/2024-autoencoders/databases/fc/inflamacion")

OUT = Path("outputs/nb09_coma")
FIG = Path("figs")
VAE_DIR = OUT / "vae_redlat"
SW_OUT = OUT / "sw_metrics"
CRIT_OUT = OUT / "criticality"
for p in [OUT, FIG, VAE_DIR, SW_OUT, CRIT_OUT]:
    p.mkdir(parents=True, exist_ok=True)

paths = DEFAULT_PATHS
cfg = ExperimentConfig(seed=42, fisher_z=True, use_optuna=False)
set_global_seed(cfg.seed)
FISHER_Z = cfg.fisher_z
THRESHOLD_TOPO = 0.20

VAE_NB08 = Path("outputs/nb08_thesis/vae_final")
REUSE_VAE = True
RETRAIN_IF_MISSING = True
RIDGE_ALPHA = 267.7

# --- Mundo pequeño (más robusto que nb08: 10/dx, N_NULL=20) ---
N_PER_DX = 50
N_NULL = 50
N_REWIRE = 10
N_JOBS = 2
QUICK_MODE = True   # False → todos los sujetos coma en SW

# --- Criticalidad (más robusto: 30/dx en nb08 → 50/dx aquí) ---
PROP_CRIT = 0.10
CRIT_N_PER_DX = 50
CRIT_N_JOBS = 2
N_AVALANCHAS = 50_000
N_BOOT_BRANCH = 1000
PROP_SENS = np.arange(0.02, 0.21, 0.02)  # 2%–20% sensibilidad λ₁

DIAG_COMA = ["CTRL", "ANOX", "TRAU"]
DIAG_REDLAT = ["CN", "AD", "FTD"]
PALETTE = {
    "CN": "#2166ac", "AD": "#d6604d", "FTD": "#4dac26",
    "CTRL": "#542788", "ANOX": "#807dba", "TRAU": "#9970ab",
}
COLORS_CRIT = {"CTRL": "#542788", "ANOX": "#807dba", "TRAU": "#9970ab"}

def build_z_topo(Z, topo):
    return np.hstack([np.asarray(Z, np.float32), np.asarray(topo, np.float32)])

def stratified_sample(df, n_per_dx, seed):
    parts = []
    for _, g in df.groupby("diagnosis"):
        n = min(n_per_dx, len(g))
        parts.append(g.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)

print("FC_ROOT:", FC_ROOT.exists())"""
    ),

    md("## 1. Cohorte coma — TOPO"),
    code(
        """cohort_coma = list_inflamacion_mats(FC_ROOT, group_map=DEFAULT_GROUP_MAP)
cohort_coma.to_csv(OUT / "cohort_coma_index.csv", index=False)
print(cohort_coma.groupby("diagnosis").size())

topo_coma = compute_topo_table(cohort_coma, threshold=THRESHOLD_TOPO, apply_fisher_z=FISHER_Z)
topo_coma.to_csv(OUT / "graph_metrics_coma.csv", index=False)"""
    ),

    md("## 2. data-iipsi — TOPO + VAE compartido"),
    code(
        """cohort_rl = build_final_cohort_df(
    paths.excel_path, paths.fc_folder, paths.t1w_csv_path,
    diagnoses_to_use=("CN", "AD", "FTD"),
)
graph_rl = pd.read_csv("outputs/graph_metrics_table.csv")
graph_rl["record_id"] = graph_rl["MRI_ID"].map(normalize_record_id)
graph_rl = graph_rl.drop_duplicates("record_id", keep="last")
topo_rl = graph_rl.set_index("record_id").loc[cohort_rl["record_id"], TOPO_COLUMNS].reset_index()
topo_rl["diagnosis"] = cohort_rl["diagnosis"].values
topo_rl["age"] = cohort_rl["age"].values

split = make_holdout_split(cohort_rl, seed=cfg.seed, test_size=cfg.test_size)
trainval_ids, test_ids = split["trainval_ids"], split["test_ids"]

X_tv = load_fc_vectors_for_ids(paths.fc_folder, trainval_ids, apply_fisher_z=FISHER_Z)
X_rl_all = load_fc_vectors_for_ids(
    paths.fc_folder, cohort_rl["record_id"].tolist(), apply_fisher_z=FISHER_Z,
)

if REUSE_VAE and (VAE_NB08 / "vae.weights.h5").exists():
    import shutil
    if not (VAE_DIR / "vae.weights.h5").exists():
        shutil.copytree(VAE_NB08, VAE_DIR, dirs_exist_ok=True)
    vae = load_vae_from_dir(VAE_DIR)
elif (VAE_DIR / "vae.weights.h5").exists():
    vae = load_vae_from_dir(VAE_DIR)
elif RETRAIN_IF_MISSING:
    VAE_HP = dict(
        hidden_dims=[512], latent_dim=64, beta_target=0.056663247229966504,
        warmup_ep=73, l2_reg=2.897389671945472e-07, lr=0.001892443497356961,
        recon_kind="mae", drop_rate=0.036861053246000725, activation="elu",
        norm_kind="layernorm", batch_size=64, clipnorm=1.0, epochs=96,
    )
    vae, _ = train_vae_final(X_tv, VAE_DIR, seed=cfg.seed, **VAE_HP)
else:
    raise FileNotFoundError("No hay VAE disponible")

rl_ids = cohort_rl["record_id"].tolist()
Z_rl = encode_mu(vae.encoder, X_rl_all)
ids_coma = cohort_coma["record_id"].tolist()
Z_coma = encode_mu(vae.encoder, load_fc_vectors_from_cohort(cohort_coma, ids_coma, apply_fisher_z=FISHER_Z))
topo_coma_mat = topo_coma.set_index("record_id").loc[ids_coma, TOPO_COLUMNS].to_numpy(np.float32)
topo_rl_mat = topo_rl.set_index("record_id").loc[rl_ids, TOPO_COLUMNS].to_numpy(np.float32)
print("Z RedLaT", Z_rl.shape, "| Z coma", Z_coma.shape)"""
    ),

    md("## 3. Ridge(Z + TOPO → edad) — CN data-iipsi + predicción coma"),
    code(
        """cdf = cohort_rl.set_index("record_id")
z_map = dict(zip(rl_ids, Z_rl))
topo_map = dict(zip(rl_ids, topo_rl_mat))

def pack_rl(ids):
    Z = np.stack([z_map[i] for i in ids])
    T = np.stack([topo_map[i] for i in ids])
    y = cdf.loc[ids, "age"].to_numpy(float)
    dx = cdf.loc[ids, "diagnosis"].values
    return build_z_topo(Z, T), y, dx

X_tv, y_tv, dx_tv = pack_rl(trainval_ids)
X_te, y_te, dx_te = pack_rl(test_ids)
cn_mask = dx_tv == "CN"
sc = StandardScaler()
ridge = Ridge(alpha=RIDGE_ALPHA, random_state=cfg.seed)
ridge.fit(sc.fit_transform(X_tv[cn_mask]), y_tv[cn_mask])

mae_cn = mean_absolute_error(y_te[dx_te == "CN"], ridge.predict(sc.transform(X_te[dx_te == "CN"])))
print(f"MAE CN test (Z+TOPO): {mae_cn:.2f} años")

X_rl, y_rl, dx_rl = pack_rl(rl_ids)
pred_rl = ridge.predict(sc.transform(X_rl))
pred_df = pd.DataFrame({
    "record_id": rl_ids, "cohort": "redlat", "diagnosis": dx_rl,
    "age": y_rl, "predicted_age": pred_rl, "BAG_raw": pred_rl - y_rl,
})
pred_coma = ridge.predict(sc.transform(build_z_topo(Z_coma, topo_coma_mat)))
ref_cn_mean = pred_df.loc[pred_df.diagnosis == "CN", "predicted_age"].mean()
pred_coma_df = pd.DataFrame({
    "record_id": ids_coma, "cohort": "coma", "diagnosis": cohort_coma["diagnosis"].values,
    "age": np.nan, "predicted_age": pred_coma,
    "BAG_vs_CN_redlat": pred_coma - ref_cn_mean,
})
pred_all = pd.concat([pred_df, pred_coma_df], ignore_index=True)
pred_all.to_csv(OUT / "predicted_age_all_cohorts.csv", index=False)
print(pred_coma_df.groupby("diagnosis")["predicted_age"].mean().round(2))"""
    ),

    code(
        """fig, axes = plt.subplots(1, 2, figsize=(14, 5))
order = ["CN", "AD", "FTD", "CTRL", "ANOX", "TRAU"]
sns.boxplot(data=pred_all, x="diagnosis", y="predicted_age", order=order,
            hue="cohort", dodge=False, ax=axes[0], palette=PALETTE)
sub_bag = pred_all.copy()
sub_bag["bag_plot"] = sub_bag["BAG_raw"]
sub_bag.loc[sub_bag.cohort == "coma", "bag_plot"] = sub_bag.loc[sub_bag.cohort == "coma", "BAG_vs_CN_redlat"]
sns.boxplot(data=sub_bag, x="diagnosis", y="bag_plot", order=order,
            hue="cohort", dodge=False, ax=axes[1], palette=PALETTE)
axes[1].axhline(0, color="k", ls="--", lw=0.8)
plt.tight_layout()
plt.savefig(FIG / "09_predicted_age_coma_vs_redlat.png", dpi=150, bbox_inches="tight")
plt.show()"""
    ),

    md("## 4. Mundo pequeño (σ, ω) — N_PER_DX=50, N_NULL=50"),
    code(
        """import bct, networkx as nx

def graph_metrics_sw(B):
    C = float(np.mean(bct.clustering_coef_bu(B)))
    D = bct.distance_bin(B)
    L = float(bct.charpath(D, include_diagonal=False, include_infinite=False)[0])
    return C, L

def sw_one(row, fc_z, threshold, n_null, n_rewire, seed):
    B = threshold_fixed(fc_z, threshold)
    G = nx.from_numpy_array(B)
    p = float(B.sum()) / (B.shape[0] * (B.shape[0] - 1))
    if not nx.is_connected(G) or p == 0:
        return None
    C, L = graph_metrics_sw(B)
    n = B.shape[0]
    rng = np.random.default_rng(seed + hash(row["record_id"]) % 10000)
    Cs_er, Ls_er, C_ws, L_ws = [], [], [], []
    for _ in range(n_null):
        G_er = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(1e6)))
        B_er = nx.to_numpy_array(G_er, dtype=float)
        np.fill_diagonal(B_er, 0)
        if nx.is_connected(G_er):
            c = float(np.mean(bct.clustering_coef_bu(B_er)))
            d = bct.distance_bin(B_er)
            l = float(bct.charpath(d, include_diagonal=False, include_infinite=False)[0])
            Cs_er.append(c); Ls_er.append(l)
        B_r = (bct.randmio_und(B, n_rewire)[0] > 0).astype(float)
        np.fill_diagonal(B_r, 0)
        if nx.is_connected(nx.from_numpy_array(B_r)):
            c, l = graph_metrics_sw(B_r)
            C_ws.append(c); L_ws.append(l)
    B_latt = (bct.latmio_und(B, n_rewire)[0] > 0).astype(float)
    np.fill_diagonal(B_latt, 0)
    C_latt = float(np.mean(bct.clustering_coef_bu(B_latt)))
    C_er, L_er = np.nanmean(Cs_er), np.nanmean(Ls_er)
    C_ws_m, L_ws_m = np.nanmean(C_ws), np.nanmean(L_ws)
    if not (C_ws_m and L and C_er and C_latt):
        return None
    return {
        "record_id": row["record_id"], "diagnosis": row["diagnosis"],
        "sigma_ws": (C / C_ws_m) / (L / L_ws_m),
        "sigma_er": (C / C_er) / (L / L_er),
        "omega_ws": (L_ws_m / L) - (C / C_latt),
        "C": C, "L": L,
    }

sw_sub = stratified_sample(cohort_coma[["record_id", "diagnosis"]], N_PER_DX, cfg.seed) if QUICK_MODE else cohort_coma
print(f"SW: n={len(sw_sub)} ({N_PER_DX}/dx, N_NULL={N_NULL})")
fc_z_map = {rid: vector_to_matrix(load_fc_vectors_from_cohort(cohort_coma, [rid], apply_fisher_z=FISHER_Z)[0])
            for rid in sw_sub["record_id"]}
rows_sw = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(sw_one)(r.to_dict(), fc_z_map[r["record_id"]], THRESHOLD_TOPO, N_NULL, N_REWIRE, cfg.seed)
    for _, r in sw_sub.iterrows()
)
sw_df = pd.DataFrame([x for x in rows_sw if x])
sw_df.to_csv(SW_OUT / "sw_metrics.csv", index=False)
print(sw_df.groupby("diagnosis")[["sigma_ws", "sigma_er", "omega_ws"]].mean().round(3))"""
    ),

    code(
        """for col in ["sigma_ws", "sigma_er", "omega_ws"]:
    groups = [sw_df.loc[sw_df.diagnosis == g, col].dropna().values for g in DIAG_COMA]
    H, p = kruskal(*groups)
    print(f"{col}: Kruskal-Wallis p={p:.4g}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["sigma_ws", "sigma_er", "omega_ws"]):
    sw_df.boxplot(column=col, by="diagnosis", ax=ax)
    if "sigma" in col:
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_title(col)
plt.suptitle(f"Mundo pequeño coma (n={len(sw_df)}, N_NULL={N_NULL})")
plt.tight_layout()
plt.savefig(FIG / "09_sw_by_diagnosis.png", dpi=150, bbox_inches="tight")
plt.show()"""
    ),

    md("## 5. Criticalidad — λ₁, rango dinámico, referencia ER (cohorte completa coma)"),
    code(
        """import networkx as nx

N_ROI = 116
CRIT_SEED = cfg.seed
coma_idx = cohort_coma.set_index("record_id")

ids_c = cohort_coma["record_id"].tolist()
fc_z_crit = {rid: vector_to_matrix(v) for rid, v in zip(
    ids_c, load_fc_vectors_from_cohort(cohort_coma, ids_c, apply_fisher_z=FISHER_Z))}

# λ₁ individual (PROP=10%)
lambda1_records = []
for rid in ids_c:
    B = threshold_proportional(fc_z_crit[rid], PROP_CRIT)
    lambda1_records.append({
        "record_id": rid, "diagnosis": coma_idx.loc[rid, "diagnosis"],
        "lambda1": float(np.linalg.eigvalsh(B)[-1]),
    })
lambda1_df = pd.DataFrame(lambda1_records)
print(lambda1_df.groupby("diagnosis")["lambda1"].describe().round(2))

# matrices promedio grupales
fc_mean = {}
for g in DIAG_COMA:
    mats = [fc_z_crit[r] for r in ids_c if coma_idx.loc[r, "diagnosis"] == g]
    fc_mean[g] = np.mean(mats, axis=0)
adj_mean = {g: threshold_proportional(fc_mean[g], PROP_CRIT) for g in DIAG_COMA}
lambda1_mean = {g: float(np.linalg.eigvalsh(adj_mean[g])[-1]) for g in DIAG_COMA}

def er_lambda1_ref(density, n=N_ROI, n_ref=100, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_ref):
        G = nx.erdos_renyi_graph(n, density, seed=int(rng.integers(1e6)))
        A = nx.to_numpy_array(G, dtype=float)
        np.fill_diagonal(A, 0)
        out.append(np.linalg.eigvalsh(A)[-1])
    return float(np.mean(out))

er_ref = {}
for g in DIAG_COMA:
    dens = adj_mean[g].sum() / 2 / (N_ROI * (N_ROI - 1) / 2)
    mu_er = er_lambda1_ref(dens)
    er_ref[g] = mu_er
    print(f"{g}: λ₁={lambda1_mean[g]:.2f}  λ₁_ER={mu_er:.2f}  ratio={lambda1_mean[g]/mu_er:.2f}")

lambda1_df["lambda1_norm"] = lambda1_df.apply(
    lambda r: r["lambda1"] / er_ref[r["diagnosis"]], axis=1,
)

# rango dinámico Δ sobre matrices promedio
def simulate_response(A, h_vals, n_steps=500):
    n = A.shape[0]
    responses = np.zeros(len(h_vals))
    for i, h in enumerate(h_vals):
        F = np.zeros(n)
        for _ in range(n_steps):
            F_new = 1 - np.exp(-h - A @ F)
            if np.max(np.abs(F_new - F)) < 1e-9:
                break
            F = F_new
        responses[i] = F.mean()
    return responses

def dynamic_range(h_vals, F):
    F_norm = (F - F.min()) / (F.max() - F.min() + 1e-12)
    i01 = np.searchsorted(F_norm, 0.1)
    i09 = np.searchsorted(F_norm, 0.9)
    if i01 >= len(h_vals) or i09 >= len(h_vals) or h_vals[i01] <= 0:
        return np.nan
    return 10 * np.log10(h_vals[i09] / h_vals[i01])

h_vals = np.logspace(-5, 1, 300)
delta_vals = {g: dynamic_range(h_vals, simulate_response(adj_mean[g], h_vals)) for g in DIAG_COMA}
for g in DIAG_COMA:
    print(f"  {g}: Δ={delta_vals[g]:.2f} dB")

lambda1_df.to_csv(CRIT_OUT / "metricas_criticalidad.csv", index=False)"""
    ),

    code(
        """fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.violinplot(data=lambda1_df, x="diagnosis", y="lambda1", order=DIAG_COMA,
               hue="diagnosis", palette=COLORS_CRIT, legend=False, ax=axes[0])
axes[0].set_title("λ₁ individual")
sns.violinplot(data=lambda1_df, x="diagnosis", y="lambda1_norm", order=DIAG_COMA,
               hue="diagnosis", palette=COLORS_CRIT, legend=False, ax=axes[1])
axes[1].set_title("λ₁ normalizado (ER)")
for g in DIAG_COMA:
    axes[2].scatter(lambda1_mean[g], delta_vals[g], s=200, color=COLORS_CRIT[g], label=g)
    axes[2].annotate(g, (lambda1_mean[g], delta_vals[g]), xytext=(6, 4), textcoords="offset points")
axes[2].set_xlabel("λ₁ matriz promedio"); axes[2].set_ylabel("Δ (dB)")
axes[2].set_title("Rango dinámico")
plt.tight_layout()
plt.savefig(FIG / "09_crit_lambda1.png", dpi=150, bbox_inches="tight")
plt.show()"""
    ),

    md("## 6. Avalanchas (matrices promedio, N=50 000)"),
    code(
        """import powerlaw

np.random.seed(CRIT_SEED)
p_crit = 1.0 / lambda1_mean["CTRL"]
print(f"p = 1/λ₁_CTRL = {p_crit:.5f}")

def simulate_avalanches(A, p_activation, n_avalanchas):
    n = A.shape[0]
    nbrs = [np.where(A[i] > 0)[0] for i in range(n)]
    sizes, durations, generations = [], [], []
    for _ in range(n_avalanchas):
        active, visited = {np.random.randint(n)}, set()
        visited |= active
        gen_list = [1]
        while active:
            nxt = set()
            for node in active:
                for nb in nbrs[node]:
                    if nb not in visited and np.random.random() < p_activation:
                        nxt.add(nb); visited.add(nb)
            active = nxt
            if active:
                gen_list.append(len(active))
        sizes.append(len(visited))
        durations.append(len(gen_list))
        generations.append(gen_list)
    return np.array(sizes), np.array(durations), generations

avalanche_data, pl_results = {}, {}
for g in DIAG_COMA:
    print(f"Avalanchas {g}…")
    s, T, gens = simulate_avalanches(adj_mean[g], p_crit, N_AVALANCHAS)
    avalanche_data[g] = {"sizes": s, "durations": T, "generations": gens}
    fit_s = powerlaw.Fit(s[s > 1], discrete=True, verbose=False)
    fit_T = powerlaw.Fit(T[T > 1], discrete=True, verbose=False)
    pl_results[g] = {"alpha_s": fit_s.power_law.alpha, "alpha_T": fit_T.power_law.alpha}
    print(f"  ⟨s⟩={s.mean():.2f}  α(s)={pl_results[g]['alpha_s']:.3f}")

with open(CRIT_OUT / "avalanchas_pl.json", "w") as f:
    json.dump({g: {"mean_s": float(avalanche_data[g]["sizes"].mean()),
                   "alpha_s": float(pl_results[g]["alpha_s"]),
                   "alpha_T": float(pl_results[g]["alpha_T"])} for g in DIAG_COMA}, f, indent=2)"""
    ),

    code(
        """fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for g in DIAG_COMA:
    s = avalanche_data[g]["sizes"]
    bins = np.logspace(0, np.log10(max(s)), 20)
    hist, edges = np.histogram(s, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = hist > 0
    axes[0].loglog(centers[mask], hist[mask] / hist[mask].sum(), "o-", color=COLORS_CRIT[g], label=g, ms=3)
axes[0].set_title("P(s)"); axes[0].legend()
alpha_vals = [pl_results[g]["alpha_s"] for g in DIAG_COMA]
axes[1].bar(DIAG_COMA, alpha_vals, color=[COLORS_CRIT[g] for g in DIAG_COMA], edgecolor="k")
axes[1].axhline(2.0, color="gray", ls="--", label="α=2")
axes[1].set_title("α(s)"); axes[1].legend()
for g in DIAG_COMA:
    s, T = avalanche_data[g]["sizes"], avalanche_data[g]["durations"]
    axes[2].loglog(T, s, "o", color=COLORS_CRIT[g], alpha=0.15, ms=2, label=g)
axes[2].set_title("s vs T"); axes[2].legend()
plt.suptitle(f"Avalanchas (N={N_AVALANCHAS}, p=1/λ₁_CTRL)")
plt.tight_layout()
plt.savefig(FIG / "09_crit_avalanchas.png", dpi=150, bbox_inches="tight")
plt.show()"""
    ),

    md("## 7. Percolación (CRIT_N_PER_DX=50)"),
    code(
        """PROPS_FINE = np.arange(0.002, 0.032, 0.002)
PROPS_COARSE = np.arange(0.040, 0.210, 0.010)
PROPS_PERC = np.round(np.concatenate([PROPS_FINE, PROPS_COARSE]), 3)

crit_sub = stratified_sample(cohort_coma[["record_id", "diagnosis"]], CRIT_N_PER_DX, CRIT_SEED)
print(f"Percolación: {len(crit_sub)} sujetos × {len(PROPS_PERC)} props")

def perc_one_subject(row, fc_z, props):
    rows = []
    for prop in props:
        B = threshold_proportional(fc_z, prop)
        comps = sorted(nx.connected_components(nx.from_numpy_array(B)), key=len, reverse=True)
        S1 = len(comps[0]) / N_ROI if comps else 0.0
        S2 = len(comps[1]) / N_ROI if len(comps) >= 2 else 0.0
        rows.append({"record_id": row["record_id"], "diagnosis": row["diagnosis"],
                     "prop": prop, "S1": S1, "S2": S2, "chi": S2**2 / N_ROI})
    return rows

perc_nested = Parallel(n_jobs=CRIT_N_JOBS, verbose=1)(
    delayed(perc_one_subject)(row.to_dict(), fc_z_crit[row["record_id"]], PROPS_PERC)
    for _, row in crit_sub.iterrows()
)
perc_df = pd.DataFrame([r for sub in perc_nested for r in sub])
perc_df.to_csv(CRIT_OUT / "percolacion.csv", index=False)

perc_agg = perc_df.groupby(["diagnosis", "prop"])[["S1", "S2", "chi"]].agg(["mean", "sem"]).reset_index()
perc_agg.columns = ["diagnosis", "prop", "S1_mean", "S1_sem", "S2_mean", "S2_sem", "chi_mean", "chi_sem"]
chi_peak = {}
for g in DIAG_COMA:
    sub = perc_agg[perc_agg.diagnosis == g]
    idx = sub["chi_mean"].idxmax()
    chi_peak[g] = float(sub.loc[idx, "prop"])
    print(f"  {g}: χ peak @ {chi_peak[g]*100:.1f}%")"""
    ),

    code(
        """fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for g in DIAG_COMA:
    sub = perc_agg[perc_agg.diagnosis == g]
    for ax, met in zip(axes, ["S1", "S2", "chi"]):
        ax.plot(sub["prop"] * 100, sub[f"{met}_mean"], color=COLORS_CRIT[g], lw=2, label=g)
        ax.fill_between(sub["prop"] * 100,
                        sub[f"{met}_mean"] - sub[f"{met}_sem"],
                        sub[f"{met}_mean"] + sub[f"{met}_sem"],
                        color=COLORS_CRIT[g], alpha=0.2)
        ax.axvline(chi_peak[g] * 100, color=COLORS_CRIT[g], ls=":", lw=1.2)
for ax, t in zip(axes, ["S1", "S2", "χ"]):
    ax.set_xlabel("% conexiones"); ax.set_ylabel(t); ax.legend(fontsize=8)
plt.suptitle(f"Percolación (n={len(crit_sub)})")
plt.tight_layout()
plt.savefig(FIG / "09_crit_percolacion.png", dpi=150, bbox_inches="tight")
plt.show()"""
    ),

    md("## 8. Razón de ramificación σ_br"),
    code(
        """def compute_branching_ratio(generations):
    num, den = 0, 0
    for gen_list in generations:
        for t in range(len(gen_list) - 1):
            num += gen_list[t + 1]
            den += gen_list[t]
    return num / den if den > 0 else np.nan

branching_results = {}
for g in DIAG_COMA:
    gens = avalanche_data[g]["generations"]
    sigma_obs = compute_branching_ratio(gens)
    boot = []
    for _ in range(N_BOOT_BRANCH):
        idx = np.random.randint(0, len(gens), len(gens))
        boot.append(compute_branching_ratio([gens[i] for i in idx]))
    branching_results[g] = {
        "sigma": sigma_obs,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }
    print(f"{g}: σ_br={sigma_obs:.4f}  IC95%[{branching_results[g]['ci_low']:.3f}, {branching_results[g]['ci_high']:.3f}]")

with open(CRIT_OUT / "branching_ratio.json", "w") as f:
    json.dump(branching_results, f, indent=2)"""
    ),

    code(
        """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sigmas = [branching_results[g]["sigma"] for g in DIAG_COMA]
yerr = [[s - branching_results[g]["ci_low"] for s, g in zip(sigmas, DIAG_COMA)],
        [branching_results[g]["ci_high"] - s for s, g in zip(sigmas, DIAG_COMA)]]
axes[0].bar(DIAG_COMA, sigmas, yerr=yerr, capsize=4,
            color=[COLORS_CRIT[g] for g in DIAG_COMA], edgecolor="k")
axes[0].axhline(1.0, color="gray", ls="--")
axes[0].set_title("σ_br (avalanchas)")
for g in DIAG_COMA:
    axes[1].scatter(lambda1_mean[g], branching_results[g]["sigma"],
                    color=COLORS_CRIT[g], s=200, label=g)
axes[1].axhline(1.0, color="gray", ls="--")
axes[1].set_xlabel("λ₁ promedio"); axes[1].set_ylabel("σ_br")
axes[1].legend()
plt.tight_layout()
plt.savefig(FIG / "09_crit_branching.png", dpi=150, bbox_inches="tight")
plt.show()"""
    ),

    md("## 9. Sensibilidad de λ₁ (2%–20%, cohorte completa)"),
    code(
        """sens_rows = []
for prop in PROP_SENS:
    for rid in ids_c:
        B = threshold_proportional(fc_z_crit[rid], float(prop))
        sens_rows.append({
            "record_id": rid, "diagnosis": coma_idx.loc[rid, "diagnosis"],
            "prop": float(prop),
            "lambda1": float(np.linalg.eigvalsh(B)[-1]),
        })
sens_df = pd.DataFrame(sens_rows)
sens_agg = sens_df.groupby(["diagnosis", "prop"])["lambda1"].agg(["mean", "std"]).reset_index()
sens_agg.columns = ["diagnosis", "prop", "lambda1_mean", "lambda1_std"]
sens_df.to_csv(CRIT_OUT / "sensibilidad_lambda1.csv", index=False)
print(sens_agg.pivot(index="prop", columns="diagnosis", values="lambda1_mean").round(2).to_string())"""
    ),

    code(
        """fig, ax = plt.subplots(figsize=(9, 5))
for g in DIAG_COMA:
    sub = sens_agg[sens_agg.diagnosis == g]
    ax.plot(sub["prop"] * 100, sub["lambda1_mean"], color=COLORS_CRIT[g], lw=2.5,
            marker="o", ms=5, label=g)
    ax.fill_between(sub["prop"] * 100,
                    sub["lambda1_mean"] - sub["lambda1_std"],
                    sub["lambda1_mean"] + sub["lambda1_std"],
                    color=COLORS_CRIT[g], alpha=0.15)
ax.axvline(PROP_CRIT * 100, color="k", ls=":", lw=1, label=f"PROP={PROP_CRIT*100:.0f}%")
ax.set_xlabel("% conexiones retenidas"); ax.set_ylabel("λ₁ medio")
ax.set_title("Sensibilidad λ₁ (cohorte coma completa)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "09_crit_sensibilidad.png", dpi=150, bbox_inches="tight")
plt.show()"""
    ),

    md("## 10. Estadística grupal + metadatos"),
    code(
        """from scipy.stats import kruskal

# λ₁
H, p = kruskal(*[lambda1_df.loc[lambda1_df.diagnosis == g, "lambda1"] for g in DIAG_COMA])
print(f"λ₁ Kruskal-Wallis: H={H:.3f} p={p:.4g}")

stats_rows = []
for metric, df, col in [
    ("lambda1", lambda1_df, "lambda1"),
    ("lambda1_norm", lambda1_df, "lambda1_norm"),
]:
    H, p = kruskal(*[df.loc[df.diagnosis == g, col] for g in DIAG_COMA])
    stats_rows.append({"metric": metric, "H": H, "p": p})

for g1, g2 in [("CTRL", "ANOX"), ("CTRL", "TRAU"), ("ANOX", "TRAU")]:
    for col in ["lambda1", "lambda1_norm"]:
        a = lambda1_df.loc[lambda1_df.diagnosis == g1, col]
        b = lambda1_df.loc[lambda1_df.diagnosis == g2, col]
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        stats_rows.append({"metric": f"{col}_{g1}_vs_{g2}", "H": np.nan, "p": p})

stats_df = pd.DataFrame(stats_rows)
stats_df["p_fdr"] = multipletests(stats_df["p"].fillna(1), method="fdr_bh")[1]
stats_df.to_csv(CRIT_OUT / "estadisticas_criticalidad.csv", index=False)
print(stats_df.round(4).to_string())

run_meta = {
    "n_coma": int(len(cohort_coma)),
    "n_redlat": int(len(cohort_rl)),
    "mae_cn_test": float(mae_cn),
    "N_PER_DX": N_PER_DX,
    "N_NULL": N_NULL,
    "CRIT_N_PER_DX": CRIT_N_PER_DX,
    "N_AVALANCHAS": N_AVALANCHAS,
    "PROP_CRIT": PROP_CRIT,
    "seed": cfg.seed,
}
with open(CRIT_OUT / "run_meta.json", "w") as f:
    json.dump(run_meta, f, indent=2)
with open(OUT / "transfer_meta.json", "w") as f:
    json.dump({**run_meta, "features": "Z+TOPO"}, f, indent=2)
print("Listo →", OUT)"""
    ),
]

Path("09_coma.ipynb").write_text(
    json.dumps({
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": cells,
    }, indent=1, ensure_ascii=False),
    encoding="utf-8",
)
print("Wrote 09_coma.ipynb,", len(cells), "cells")
