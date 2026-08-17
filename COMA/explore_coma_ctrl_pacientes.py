"""Exploración COMA: CTRL vs pacientes (tablas *_ctrl_pacientes en COMA/). Guarda figs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path(__file__).resolve().parent
FIG_DIR = DATA / "figs_ctrl_pacientes"

DIAG_ORDER = ["CTRL", "pacientes"]
PALETTE = {"CTRL": "#0072B2", "pacientes": "#D55E00"}
KEEP = set(DIAG_ORDER)
TITLE = "COMA CTRL vs pacientes"

sns.set_theme(style="whitegrid", context="notebook")


def save_fig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("fig:", path)
    plt.show()


def load_coma():
    cohort = pd.read_csv(DATA / "cohort_coma_ctrl_pacientes.csv")
    bag = pd.read_csv(DATA / "bag_coma_ctrl_pacientes.csv")
    topo = pd.read_csv(DATA / "graph_metrics_coma_ctrl_pacientes.csv")

    cohort = cohort[cohort["diagnosis"].isin(KEEP)].copy()
    bag = bag[bag["diagnosis"].isin(KEEP)].copy()
    topo = topo[topo["diagnosis"].isin(KEEP)].copy()

    cohort["sex"] = cohort["sex"].replace({"H": "M", "F": "F"})
    if "binary_outcome" in cohort.columns:
        cohort["binary_outcome"] = pd.to_numeric(cohort["binary_outcome"], errors="coerce")
    if "gcs" in cohort.columns:
        cohort["gcs"] = pd.to_numeric(cohort["gcs"], errors="coerce")
    if "deceased" in cohort.columns:
        cohort["deceased"] = pd.to_numeric(cohort["deceased"], errors="coerce")

    return cohort, bag, topo


def _order(df, col="diagnosis"):
    present = [d for d in DIAG_ORDER if d in set(df[col])]
    return present


def plot_cohort_overview(cohort):
    order = _order(cohort)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    counts = cohort["diagnosis"].value_counts().reindex(order)
    axes[0].bar(counts.index, counts.values, color=[PALETTE[d] for d in counts.index])
    axes[0].set_title("n por grupo")
    axes[0].set_ylabel("sujetos")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 0.2, str(int(v)), ha="center")

    sex_ct = pd.crosstab(cohort["diagnosis"], cohort["sex"]).reindex(order)
    sex_ct.plot(kind="bar", ax=axes[1], rot=0, color=["#e41a1c", "#377eb8"])
    axes[1].set_title("sexo × grupo")
    axes[1].set_xlabel("")

    if cohort["age"].notna().any():
        sns.boxplot(data=cohort, x="diagnosis", y="age", order=order,
                    palette=PALETTE, ax=axes[2])
        axes[2].set_title("edad cronológica")
    else:
        axes[2].set_visible(False)

    fig.suptitle(f"{TITLE} — composición", y=1.02)
    plt.tight_layout()
    save_fig("01_composicion")

    print(cohort.groupby("diagnosis")["age"].describe().round(2).reindex(order))
    if "diagnosis_orig" in cohort.columns:
        print("\norig × grupo:")
        print(pd.crosstab(cohort["diagnosis"], cohort["diagnosis_orig"]))


def plot_age_distributions(cohort):
    order = _order(cohort)
    sub = cohort.dropna(subset=["age"])
    if sub.empty:
        print("sin edades")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for dx in order:
        s = sub.loc[sub.diagnosis == dx, "age"]
        sns.kdeplot(s, ax=axes[0], fill=True, label=dx, color=PALETTE[dx],
                    alpha=0.25, linewidth=2)
        axes[0].scatter(s, np.full(len(s), -0.002), marker="|", s=80,
                        color=PALETTE[dx], alpha=0.9)
    axes[0].set_xlabel("edad")
    axes[0].set_ylim(bottom=-0.005)
    axes[0].legend()
    axes[0].set_title("densidad de edad")

    sns.histplot(data=sub, x="age", hue="diagnosis", hue_order=order,
                 palette=PALETTE, multiple="dodge", shrink=0.85,
                 bins=10, ax=axes[1], edgecolor="white", linewidth=0.5)
    axes[1].set_title("histograma edad (barras lado a lado)")
    fig.suptitle(f"{TITLE} — distribuciones de edad", y=1.02)
    plt.tight_layout()
    save_fig("02_distribuciones_edad")


def plot_clinical(cohort):
    order = _order(cohort)
    has_gcs = cohort["gcs"].notna().any() if "gcs" in cohort.columns else False
    has_out = cohort["binary_outcome"].notna().any() if "binary_outcome" in cohort.columns else False
    if not has_gcs and not has_out:
        print("sin GCS / outcome")
        return

    n = int(has_gcs) + int(has_out)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(4.5 * max(n, 1), 4))
    if n == 1:
        axes = [axes]
    i = 0
    if has_gcs:
        sns.boxplot(data=cohort.dropna(subset=["gcs"]), x="diagnosis", y="gcs",
                    order=order, palette=PALETTE, ax=axes[i])
        axes[i].set_title("GCS")
        i += 1
    if has_out:
        ct = pd.crosstab(cohort["diagnosis"], cohort["binary_outcome"]).reindex(order)
        ct.plot(kind="bar", ax=axes[i], rot=0, color=["#999999", "#4daf4a"])
        axes[i].set_title("outcome binario (0 malo / 1 bueno)")
        axes[i].set_xlabel("")
    fig.suptitle(f"{TITLE} — variables clínicas", y=1.02)
    plt.tight_layout()
    save_fig("03_clinica")


def plot_bag_mae(bag):
    order = _order(bag)
    if "predicted_age_loo" in bag.columns and bag["predicted_age_loo"].notna().any():
        bag = bag.copy()
        bag["predicted_age"] = bag["predicted_age_loo"].fillna(bag["predicted_age"])
        bag["BAG"] = bag.get("BAG_loo", bag["BAG"]).fillna(bag["BAG"])
        if "abs_err_loo" in bag.columns:
            bag["abs_err"] = bag["abs_err_loo"].fillna(bag["abs_err"])
    bag = bag.dropna(subset=["age", "predicted_age"]).copy()
    if bag.empty:
        print("sin BAG con edad")
        return

    mae_all = mean_absolute_error(bag["age"], bag["predicted_age"])
    rmse_all = float(np.sqrt(mean_squared_error(bag["age"], bag["predicted_age"])))
    r2_all = r2_score(bag["age"], bag["predicted_age"])
    mae_dx = {dx: mean_absolute_error(g["age"], g["predicted_age"])
              for dx, g in bag.groupby("diagnosis")}

    print(f"MAE={mae_all:.2f}  RMSE={rmse_all:.2f}  R2={r2_all:.3f}  n={len(bag)}")
    print("MAE por dx:", {k: round(v, 2) for k, v in mae_dx.items()})
    print(bag.groupby("diagnosis")[["age", "predicted_age", "BAG", "abs_err"]]
          .agg(["mean", "std"]).round(2).reindex(order))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for dx in order:
        sub = bag[bag.diagnosis == dx]
        axes[0].scatter(sub["age"], sub["predicted_age"], c=PALETTE[dx],
                        label=f"{dx} MAE={mae_dx[dx]:.1f}", s=45, alpha=0.85)
    lims = [
        min(bag["age"].min(), bag["predicted_age"].min()) - 3,
        max(bag["age"].max(), bag["predicted_age"].max()) + 3,
    ]
    axes[0].plot(lims, lims, "k--", lw=0.9)
    axes[0].set_xlabel("edad real")
    axes[0].set_ylabel("edad predicha")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"pred vs real  MAE={mae_all:.2f}")

    sns.boxplot(data=bag, x="diagnosis", y="BAG", order=order, palette=PALETTE, ax=axes[1])
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].set_title("BAG = pred − age")

    sns.boxplot(data=bag, x="diagnosis", y="abs_err", order=order, palette=PALETTE, ax=axes[2])
    axes[2].set_title("|error| (años)")
    axes[2].set_ylabel("abs_err")

    fig.suptitle(f"{TITLE} — brain age / BAG / error", y=1.02)
    plt.tight_layout()
    save_fig("04_bag_mae")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=bag, x="age", y="BAG", hue="diagnosis",
                    hue_order=order, palette=PALETTE, ax=ax)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_title(f"{TITLE} — BAG vs edad cronológica")
    plt.tight_layout()
    save_fig("05_bag_vs_edad")


def plot_topo(topo):
    order = _order(topo)
    metrics = [
        "local_efficiency", "global_efficiency",
        "clustering_coeff", "degree_mean",
    ]
    metrics = [m for m in metrics if m in topo.columns]
    n = len(metrics)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, metrics):
        sns.boxplot(data=topo, x="diagnosis", y=m, order=order, palette=PALETTE, ax=ax)
        ax.set_title(m)
        ax.set_xlabel("")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{TITLE} — métricas topológicas (TOPO)", y=1.01)
    plt.tight_layout()
    save_fig("06_topo")


def plot_bag_vs_clinical(bag, cohort):
    need = [c for c in ["gcs", "binary_outcome", "sex"] if c not in bag.columns]
    m = bag.merge(cohort[["record_id"] + need], on="record_id", how="left") if need else bag.copy()
    order = _order(m)
    has_gcs = m["gcs"].notna().any()
    has_out = m["binary_outcome"].notna().any()
    if not has_gcs and not has_out:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if has_gcs:
        sns.scatterplot(data=m.dropna(subset=["gcs"]), x="gcs", y="BAG",
                        hue="diagnosis", hue_order=order, palette=PALETTE, ax=axes[0])
        axes[0].axhline(0, color="k", ls="--", lw=0.8)
        axes[0].set_title("BAG vs GCS")
    else:
        axes[0].set_visible(False)

    if has_out:
        sns.boxplot(data=m.dropna(subset=["binary_outcome"]),
                    x="binary_outcome", y="BAG", hue="diagnosis",
                    hue_order=order, palette=PALETTE, ax=axes[1])
        axes[1].axhline(0, color="k", ls="--", lw=0.8)
        axes[1].set_title("BAG vs outcome")
        axes[1].set_xlabel("binary_outcome")
    else:
        axes[1].set_visible(False)

    fig.suptitle(f"{TITLE} — BAG vs clínica", y=1.02)
    plt.tight_layout()
    save_fig("07_bag_vs_clinica")


def plot_summary_table(cohort, bag, topo):
    order = _order(cohort)
    rows = []
    for dx in order:
        c = cohort[cohort.diagnosis == dx]
        b = bag[bag.diagnosis == dx] if not bag.empty else pd.DataFrame()
        t = topo[topo.diagnosis == dx]
        row = {
            "diagnosis": dx,
            "n": len(c),
            "age_mean": c["age"].mean(),
            "age_std": c["age"].std(),
            "pct_female": 100 * (c["sex"] == "F").mean() if "sex" in c else np.nan,
            "gcs_mean": c["gcs"].mean() if "gcs" in c else np.nan,
            "BAG_mean": b["BAG"].mean() if len(b) else np.nan,
            "MAE": mean_absolute_error(b["age"], b["predicted_age"]) if len(b) else np.nan,
            "GE_mean": t["global_efficiency"].mean() if "global_efficiency" in t else np.nan,
        }
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("diagnosis").round(2)
    print(f"\n=== Resumen {TITLE} ===")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary.values,
        rowLabels=summary.index.tolist(),
        colLabels=summary.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 1.4)
    ax.set_title(f"{TITLE} — tabla resumen", pad=12)
    plt.tight_layout()
    save_fig("08_resumen")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("datos:", DATA)
    print("figs:", FIG_DIR)
    print("grupos:", DIAG_ORDER)
    cohort, bag, topo = load_coma()
    print(f"cohort={len(cohort)}  bag={len(bag)}  topo={len(topo)}")

    plot_cohort_overview(cohort)
    plot_age_distributions(cohort)
    plot_clinical(cohort)
    plot_bag_mae(bag)
    plot_bag_vs_clinical(bag, cohort)
    plot_topo(topo)
    plot_summary_table(cohort, bag, topo)


if __name__ == "__main__":
    main()
