#!/usr/bin/env python3
"""
Re-run Experiment 2 (CN-Only Training) with two fixes:

1. DATA-LEAKAGE FIX: Use CN patients from the MAIN trainval split (not an
   independent split) so that no test patients leak into training.

2. EVALUATION FIX: Since the model is trained *exclusively* on CN patients,
   ALL AD and ALL FTD patients are unseen → we can evaluate the brain-age gap
   on the full clinical cohort, not just the small main-pipeline test subset.
   CN evaluation still uses only the 53 held-out CN test patients.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # Code/
REPO = ROOT.parent                                     # Thesis-comp-sci/

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Paths, ExperimentConfig
from src.utils_seed import set_global_seed
from src.cohort import build_final_cohort_df
from src.splits import make_kfold_splits, save_splits
from src.data_io import load_fc_vectors_for_ids
from src.vae_train import train_vae_final
from src.embeddings import encode_mu, save_embeddings
from src.xgb_train import build_feats, clean_xy, train_xgb
from src.metrics import regression_metrics

paths = Paths(
    excel_path=REPO / "Data" / "datos-redlat.xlsx",
    fc_folder=REPO / "Data" / "fc_mats",
    t1w_csv_path=REPO / "Data" / "Redlat_VGM_AAL_.csv",
    out_dir=REPO / "Outputs",
)

cfg = ExperimentConfig(
    seed=42,
    diagnoses_to_use=("CN", "AD", "FTD"),
    test_size=0.10,
    k_folds=5,
    fisher_z=True,
    reuse_artifacts=True,
    use_optuna=False,
)

set_global_seed(cfg.seed)

EXP_DIR = paths.out_dir / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Load cohort & main splits ---
print("=" * 70)
print("Loading cohort and main-pipeline splits ...")
print("=" * 70)

cohort_df = build_final_cohort_df(
    paths.excel_path,
    paths.fc_folder,
    paths.t1w_csv_path,
    diagnoses_to_use=cfg.diagnoses_to_use,
)
print(f"Total cohort: N={len(cohort_df)}")
print(f"Diagnoses: {cohort_df['diagnosis'].value_counts().to_dict()}")

splits_file = paths.out_dir / "splits" / f"splits_seed{cfg.seed}_test{cfg.test_size}.json"
with open(splits_file) as f:
    splits_data = json.load(f)

trainval_ids = splits_data["holdout"]["trainval_ids"]
test_ids     = splits_data["holdout"]["test_ids"]
print(f"Main split  → trainval: {len(trainval_ids)}, test: {len(test_ids)}")

# --- 2. FIX: CN trainval from main split ---
cn_all_ids      = set(cohort_df.loc[cohort_df["diagnosis"] == "CN", "record_id"])
cn_trainval_ids = sorted([rid for rid in trainval_ids if rid in cn_all_ids])
cn_test_ids     = sorted([rid for rid in test_ids     if rid in cn_all_ids])

print(f"CN trainval (from main split): {len(cn_trainval_ids)}")
print(f"CN test     (from main split): {len(cn_test_ids)}")

cn_kfold = make_kfold_splits(cn_trainval_ids, seed=cfg.seed, k=cfg.k_folds)
print(f"CN KFold: {len(cn_kfold)} folds")

cn_splits_dir = EXP_DIR / "exp2_cn_only" / "splits"
cn_splits_dir.mkdir(parents=True, exist_ok=True)
save_splits(cn_splits_dir / "cn_splits.json", {
    "cn_trainval_ids": cn_trainval_ids,
    "cn_test_ids": cn_test_ids,
    "folds": cn_kfold,
})

# --- 3. Load FC data ---
print("\nLoading FC vectors ...")
X_all = load_fc_vectors_for_ids(
    paths.fc_folder,
    cohort_df["record_id"].tolist(),
    apply_fisher_z=cfg.fisher_z,
)
print(f"FC matrix: {X_all.shape}")

id_to_idx = {rec_id: i for i, rec_id in enumerate(cohort_df["record_id"])}
cn_trainval_idx = [id_to_idx[rid] for rid in cn_trainval_ids]

X_cn_trainval = X_all[cn_trainval_idx]
print(f"X_cn_trainval: {X_cn_trainval.shape}")

# Build the expanded evaluation set:
#   CN  → only the 53 held-out test patients (the other 473 CN were used for training)
#   AD  → ALL 422 AD patients (none were used for training)
#   FTD → ALL 297 FTD patients (none were used for training)
ad_all_ids  = sorted(cohort_df.loc[cohort_df["diagnosis"] == "AD",  "record_id"].tolist())
ftd_all_ids = sorted(cohort_df.loc[cohort_df["diagnosis"] == "FTD", "record_id"].tolist())
eval_ids    = cn_test_ids + ad_all_ids + ftd_all_ids

eval_idx    = [id_to_idx[rid] for rid in eval_ids]
X_eval      = X_all[eval_idx]
print(f"Evaluation set: {len(eval_ids)} subjects "
      f"(CN={len(cn_test_ids)}, AD={len(ad_all_ids)}, FTD={len(ftd_all_ids)})")
print(f"X_eval: {X_eval.shape}")

# --- 4. Load best hyper-parameters from main pipeline ---
vae_params_path = paths.out_dir / "vae" / "vae_final_trainval_optuna" / "hparams.json"
with open(vae_params_path) as f:
    best_vae_params = json.load(f)
print(f"\nBest VAE params loaded ({vae_params_path})")

xgb_params_path = paths.out_dir / "optuna" / "xgb_best_cv.json"
with open(xgb_params_path) as f:
    best_xgb_params = json.load(f)["best_params"]
print(f"Best XGBoost params loaded ({xgb_params_path})")

# --- 5. Train VAE on CN trainval ---
print("\n" + "=" * 70)
print("Training VAE on CN subjects ...")
print("=" * 70)

cn_vae_dir = EXP_DIR / "exp2_cn_only" / "vae"
cn_vae_dir.mkdir(parents=True, exist_ok=True)

cn_vae, cn_vae_history = train_vae_final(
    X_cn_trainval,
    out_dir=cn_vae_dir,
    epochs=96,
    seed=cfg.seed,
    **{k: best_vae_params[k] for k in [
        "hidden_dims", "latent_dim", "beta_target", "warmup_ep", "l2_reg", "lr",
        "recon_kind", "drop_rate", "activation", "norm_kind", "batch_size", "clipnorm",
    ]},
)
print(f"\nCN VAE training complete. Final loss: {cn_vae_history['loss'][-1]:.2f}")

# --- 6. Extract embeddings ---
print("\nExtracting embeddings ...")
Z_cn_tr = encode_mu(cn_vae.encoder, X_cn_trainval)
Z_eval  = encode_mu(cn_vae.encoder, X_eval)
print(f"CN trainval embeddings: {Z_cn_tr.shape}")
print(f"Eval embeddings:        {Z_eval.shape}")

cn_emb_dir = EXP_DIR / "exp2_cn_only" / "embeddings"
cn_emb_dir.mkdir(parents=True, exist_ok=True)
save_embeddings(cn_emb_dir / "Z_cn_trainval", cn_trainval_ids, Z_cn_tr)
save_embeddings(cn_emb_dir / "Z_eval", eval_ids, Z_eval)

# --- 7. Helper to extract y / T1w ---
def get_data_for_ids(ids):
    df_sub = (cohort_df[cohort_df["record_id"].isin(ids)]
              .set_index("record_id").loc[ids].reset_index())
    y   = df_sub["age"].values
    sex = (df_sub["sex"] == "M").astype(float).values
    diag = df_sub["diagnosis"].map({"CN": 0, "AD": 1, "FTD": 2}).values
    T1   = df_sub[[c for c in df_sub.columns if c.startswith("t1_")]].values
    return y, sex, diag, T1

y_cn_tr, _, _, T1_cn_tr       = get_data_for_ids(cn_trainval_ids)
y_eval, _, diag_eval, T1_eval = get_data_for_ids(eval_ids)

# --- 8. Train XGBoost on CN trainval ---
print("\n" + "=" * 70)
print("Training XGBoost on CN subjects ...")
print("=" * 70)

X_cn_xgb_tr = build_feats(Z=Z_cn_tr, T1=T1_cn_tr)
X_cn_xgb_tr_clean, y_cn_tr_clean = clean_xy(X_cn_xgb_tr, y_cn_tr)
model_cn = train_xgb(X_cn_xgb_tr_clean, y_cn_tr_clean, best_xgb_params)
print(f"XGBoost trained on {len(y_cn_tr_clean)} CN subjects")

# --- 9. Evaluate: CN test (53) + ALL AD (422) + ALL FTD (297) ---
print("\n" + "=" * 70)
print("EXPERIMENT 2: CN-Only Model — Full Evaluation")
print("=" * 70)

eval_df = (cohort_df[cohort_df["record_id"].isin(eval_ids)]
           .set_index("record_id").loc[eval_ids].reset_index())

X_eval_xgb = build_feats(Z=Z_eval, T1=T1_eval)
X_eval_clean, y_eval_clean = clean_xy(X_eval_xgb, y_eval)

y_pred_eval = model_cn.predict(X_eval_clean)

eval_df["y_true"]        = y_eval_clean
eval_df["y_pred"]        = y_pred_eval
eval_df["brain_age_gap"] = eval_df["y_pred"] - eval_df["y_true"]

exp2_results = {}
for diag in ["CN", "AD", "FTD"]:
    sub_df = eval_df[eval_df["diagnosis"] == diag]
    if len(sub_df) == 0:
        continue
    metrics_diag = regression_metrics(sub_df["y_true"].values,
                                      sub_df["y_pred"].values)
    gap_mean = sub_df["brain_age_gap"].mean()
    gap_std  = sub_df["brain_age_gap"].std()

    print(f"\n{diag} (N={len(sub_df)}):")
    print(f"  MAE:  {metrics_diag['MAE']:.2f} years")
    print(f"  RMSE: {metrics_diag['RMSE']:.2f} years")
    print(f"  R²:   {metrics_diag['R2']:.3f}")
    print(f"  Pearson: {metrics_diag['Pearson']:.3f}")
    print(f"  Brain Age Gap: {gap_mean:+.2f} ± {gap_std:.2f} years")

    exp2_results[diag] = {
        "n": len(sub_df),
        "mae": float(metrics_diag["MAE"]),
        "rmse": float(metrics_diag["RMSE"]),
        "r2": float(metrics_diag["R2"]),
        "pearson": float(metrics_diag["Pearson"]),
        "brain_age_gap_mean": float(gap_mean),
        "brain_age_gap_std":  float(gap_std),
    }

print("\n" + "=" * 70)

# --- 10. Save results ---
exp2_path = EXP_DIR / "exp2_cn_only_results.json"
with open(exp2_path, "w") as f:
    json.dump(exp2_results, f, indent=2)
print(f"\nResults saved: {exp2_path}")

pred_csv = EXP_DIR / "exp2_cn_only_predictions.csv"
eval_df[["record_id", "diagnosis", "age", "y_true", "y_pred", "brain_age_gap"]].to_csv(
    pred_csv, index=False
)
print(f"Predictions saved: {pred_csv}")

print("\nDone!")
