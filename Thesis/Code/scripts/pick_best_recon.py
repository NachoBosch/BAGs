#!/usr/bin/env python3
"""
Genera panel_original.png y panel_recon.png desde código: solo el heatmap de la
matriz FC y su reconstrucción, sin títulos ni bordes blancos, para la animación Manim.

Elige un sujeto con buena reconstrucción (r en percentil 75 superior). Requiere
haber ejecutado main.ipynb hasta tener el VAE entrenado y los splits.

  python Code/scripts/pick_best_recon.py   # desde raíz del repo
  python scripts/pick_best_recon.py         # desde Code/
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolver raíz del proyecto
_script = Path(__file__).resolve()
_code = _script.parent.parent  # Code/
_repo = _code.parent            # Thesis-comp-sci/
if str(_code) not in sys.path:
    sys.path.insert(0, str(_code))

from src.config import Paths, ExperimentConfig
from src.data_io import load_fc_vectors_for_ids, vector_to_matrix
from src.figures import save_recon_panels_for_animation
from src.splits import load_splits
from src.vae_train import load_vae_from_dir


def main() -> None:
    paths = Paths(
        excel_path=_repo / "Data" / "datos-redlat.xlsx",
        fc_folder=_repo / "Data" / "fc_mats",
        t1w_csv_path=_repo / "Data" / "Redlat_VGM_AAL_.csv",
        out_dir=_repo / "Outputs",
    )
    cfg = ExperimentConfig(
        seed=42,
        diagnoses_to_use=("CN", "AD", "FTD"),
        test_size=0.10,
        k_folds=5,
        fisher_z=True,
        reuse_artifacts=True,
        use_optuna=True,
        optuna_xgb_trials=100,
        optuna_vae_trials=0,
    )

    splits_path = paths.out_dir / "splits" / f"splits_seed{cfg.seed}_test{cfg.test_size}.json"
    if not splits_path.exists():
        print("No splits found. Run main.ipynb first.")
        sys.exit(1)

    splits = load_splits(splits_path)
    test_ids = splits["holdout"]["test_ids"]
    X_test = load_fc_vectors_for_ids(paths.fc_folder, test_ids, apply_fisher_z=cfg.fisher_z)

    final_vae_dir = paths.out_dir / "vae" / (
        "vae_final_trainval_optuna" if cfg.use_optuna else "vae_final_trainval"
    )
    if not (final_vae_dir / "vae.weights.h5").exists():
        print("VAE weights not found. Run main.ipynb to train/load VAE.")
        sys.exit(1)

    final_vae = load_vae_from_dir(final_vae_dir)
    _, _, z_test = final_vae.encoder.predict(X_test, verbose=0)
    X_test_pred = final_vae.decoder.predict(z_test, verbose=0)

    recon_dir = paths.out_dir / "figures" / "vae_recons"
    r_val, index = save_recon_panels_for_animation(
        X_test, X_test_pred, vector_to_matrix, recon_dir, index=None, seed=42
    )

    print(f"Pearson r = {r_val:.3f} (index {index}).")
    print("Panels saved: panel_original.png, panel_recon.png (solo heatmap, sin bordes ni títulos).")


if __name__ == "__main__":
    main()
