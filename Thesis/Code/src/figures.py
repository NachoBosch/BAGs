from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .vae_callbacks import pearsonr_per_sample

# Standard AAL-116 atlas region names in canonical order (indices 0–115).
AAL116_REGION_NAMES: tuple[str, ...] = (
    "Precentral_L", "Precentral_R",
    "Frontal_Sup_L", "Frontal_Sup_R",
    "Frontal_Sup_Orb_L", "Frontal_Sup_Orb_R",
    "Frontal_Mid_L", "Frontal_Mid_R",
    "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_R",
    "Frontal_Inf_Oper_L", "Frontal_Inf_Oper_R",
    "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R",
    "Frontal_Inf_Orb_L", "Frontal_Inf_Orb_R",
    "Rolandic_Oper_L", "Rolandic_Oper_R",
    "Supp_Motor_Area_L", "Supp_Motor_Area_R",
    "Olfactory_L", "Olfactory_R",
    "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R",
    "Frontal_Med_Orb_L", "Frontal_Med_Orb_R",
    "Rectus_L", "Rectus_R",
    "Insula_L", "Insula_R",
    "Cingulum_Ant_L", "Cingulum_Ant_R",
    "Cingulum_Mid_L", "Cingulum_Mid_R",
    "Cingulum_Post_L", "Cingulum_Post_R",
    "Hippocampus_L", "Hippocampus_R",
    "ParaHippocampal_L", "ParaHippocampal_R",
    "Amygdala_L", "Amygdala_R",
    "Calcarine_L", "Calcarine_R",
    "Cuneus_L", "Cuneus_R",
    "Lingual_L", "Lingual_R",
    "Occipital_Sup_L", "Occipital_Sup_R",
    "Occipital_Mid_L", "Occipital_Mid_R",
    "Occipital_Inf_L", "Occipital_Inf_R",
    "Fusiform_L", "Fusiform_R",
    "Postcentral_L", "Postcentral_R",
    "Parietal_Sup_L", "Parietal_Sup_R",
    "Parietal_Inf_L", "Parietal_Inf_R",
    "SupraMarginal_L", "SupraMarginal_R",
    "Angular_L", "Angular_R",
    "Precuneus_L", "Precuneus_R",
    "Paracentral_Lobule_L", "Paracentral_Lobule_R",
    "Caudate_L", "Caudate_R",
    "Putamen_L", "Putamen_R",
    "Pallidum_L", "Pallidum_R",
    "Thalamus_L", "Thalamus_R",
    "Heschl_L", "Heschl_R",
    "Temporal_Sup_L", "Temporal_Sup_R",
    "Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R",
    "Temporal_Mid_L", "Temporal_Mid_R",
    "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R",
    "Temporal_Inf_L", "Temporal_Inf_R",
    "Cerebelum_Crus1_L", "Cerebelum_Crus1_R",
    "Cerebelum_Crus2_L", "Cerebelum_Crus2_R",
    "Cerebelum_3_L", "Cerebelum_3_R",
    "Cerebelum_4_5_L", "Cerebelum_4_5_R",
    "Cerebelum_6_L", "Cerebelum_6_R",
    "Cerebelum_7b_L", "Cerebelum_7b_R",
    "Cerebelum_8_L", "Cerebelum_8_R",
    "Cerebelum_9_L", "Cerebelum_9_R",
    "Cerebelum_10_L", "Cerebelum_10_R",
    "Vermis_1_2", "Vermis_3",
    "Vermis_4_5", "Vermis_6",
    "Vermis_7", "Vermis_8",
    "Vermis_9", "Vermis_10",
)


def readable_feature_name(name: str) -> str:
    """Map an encoded feature name to a human-readable label.

    - ``t1_XXXX`` → AAL-116 region name (e.g. ``Hippocampus_L``)
    - ``z_mu_XXX`` or ``z_XXX`` → ``VAE_dim_XX`` (integer, no zero-padding)
    - ``sex``, ``diag``, ``education``, ``site_*`` → kept as-is
    """
    m = re.match(r"t1_(\d+)", name)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(AAL116_REGION_NAMES):
            return AAL116_REGION_NAMES[idx]
        return name
    m = re.match(r"z_(?:mu_)?(\d+)", name)
    if m:
        return f"VAE_dim_{int(m.group(1))}"
    return name


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_vae_history(history: dict, out_path: Path) -> None:
    """Plot VAE training curves (losses and validation correlations)."""
    keys = ["loss", "recon_loss", "kl_loss", "val_recon_loss", "val_pearson", "val_cosine"]
    present = [k for k in keys if k in history]

    plt.figure(figsize=(7, 4))
    for k in present:
        plt.plot(history[k], label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    _savefig(out_path)


def plot_reconstructions(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    vec_to_mat_fn,
    out_dir: Path,
    n: int = 5,
) -> None:
    """Plot original vs. reconstructed FC matrices for n evenly-spaced samples."""
    out_dir.mkdir(parents=True, exist_ok=True)
    r = pearsonr_per_sample(X_true, X_pred)
    idxs = np.linspace(0, len(X_true) - 1, num=min(n, len(X_true)), dtype=int)

    for j, i in enumerate(idxs):
        m_true = vec_to_mat_fn(X_true[i])
        m_pred = vec_to_mat_fn(X_pred[i])

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].imshow(m_true)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(m_pred)
        axes[1].set_title(f"Reconstruction (r={r[i]:.3f})")
        axes[1].axis("off")

        _savefig(out_dir / f"recon_{j:02d}_idx{i}.png")


def save_recon_panels_for_animation(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    vec_to_mat_fn,
    out_dir: Path,
    index: int | None = None,
    *,
    dpi: int = 200,
    seed: int = 42,
) -> tuple[float, int]:
    """Guarda panel_original.png y panel_recon.png: solo el heatmap, sin títulos ni bordes.

    Pensado para la animación Manim. Si *index* es None, elige un sujeto con buena
    reconstrucción (r en percentil 75 superior). Escribe también best_r.txt con el r usado.

    Devuelve (pearson_r, index_usado).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = pearsonr_per_sample(X_true, X_pred)
    if index is None:
        q75 = np.percentile(r, 75)
        candidates = np.where(r >= q75)[0]
        rng = np.random.default_rng(seed)
        index = int(rng.choice(candidates))
    r_val = float(r[index])
    m_true = vec_to_mat_fn(X_true[index])
    m_pred = vec_to_mat_fn(X_pred[index])

    for name, mat in [("panel_original.png", m_true), ("panel_recon.png", m_pred)]:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(mat, aspect="equal")
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(out_dir / name, bbox_inches="tight", pad_inches=0, dpi=dpi)
        plt.close()

    (out_dir / "best_r.txt").write_text(f"{r_val:.3f}", encoding="utf-8")
    return r_val, index


def plot_pred_scatter(y_true, y_pred, out_path: Path, title: str) -> None:
    """Scatter plot of true vs. predicted age with identity line."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=15, alpha=0.75)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "--", color="gray")
    plt.xlabel("Chronological age")
    plt.ylabel("Predicted age")
    plt.title(title)
    plt.grid(True)
    _savefig(out_path)


def plot_feature_importance(model, feature_names: list[str], out_path: Path,
                            top_k: int = 20, *,
                            use_readable_names: bool = True) -> None:
    """Horizontal bar chart of top-k XGBoost feature importances.

    When *use_readable_names* is True (the default), feature names are
    mapped through :func:`readable_feature_name` so that ``t1_XXXX``
    indices become AAL-116 region names and ``z_mu_XXX`` indices become
    ``VAE_dim_XX`` labels.
    """
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]

    if use_readable_names:
        names = [readable_feature_name(feature_names[i]) for i in idx][::-1]
    else:
        names = [feature_names[i] for i in idx][::-1]

    vals = importances[idx][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(idx)), vals)
    plt.yticks(range(len(idx)), names, fontsize=8)
    plt.xlabel("Gain")
    plt.title(f"Top-{top_k} feature importances")
    _savefig(out_path)
