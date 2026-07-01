#!/usr/bin/env python3
"""Generate 11_anox_augment.ipynb — VAE synthesis for ANOX FC."""

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
    md("# 11 — Sintesis ANOX con VAE\n\nObjetivo: generar matrices FC sinteticas para anoxia a partir de los .mat reales.\nCon n=9 se usa VAE (no GAN). Los sinteticos son para exploracion; no reemplazan sujetos reales en inferencia."),

    code("""# pip install torch scikit-learn  # si falta
import sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, "Thesis/Code")

from src.coma_data_io import list_inflamacion_mats, load_fc_vectors_from_cohort, DEFAULT_GROUP_MAP
from src.data_io import vector_to_matrix, fisher_z_transform
from scipy.io import savemat

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FC_ROOT = Path("/home/usuario/disco1/proyectos/2024-autoencoders/databases/fc/inflamacion")
OUT = Path("outputs/nb11_anox_synth")
FIG = Path("figs")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEED = 42
N_ROI = 116
N_SYNTH = 20
PCA_DIM = 6
LATENT_DIM = 4
HIDDEN = 32
EPOCHS = 800
BATCH_SIZE = 9
LR = 1e-3
BETA = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)
print("device:", DEVICE)"""),

    md("## 1. Cargar ANOX"),
    code("""cohort = list_inflamacion_mats(FC_ROOT, group_map=DEFAULT_GROUP_MAP)
anox = cohort[cohort.diagnosis == "ANOX"].copy().reset_index(drop=True)
print(anox.shape[0], "sujetos ANOX")
print(anox[["record_id", "group_folder"]].to_string(index=False))

ids = anox["record_id"].tolist()
X_raw = load_fc_vectors_from_cohort(anox, ids, apply_fisher_z=False)
X_z = fisher_z_transform(X_raw)
print("vectores:", X_z.shape)"""),

    md("## 2. PCA + estandarizacion\n\n6670 dimensiones con 9 sujetos: reducimos antes del VAE."),
    code("""scaler = StandardScaler()
X_sc = scaler.fit_transform(X_z)

n_comp = min(PCA_DIM, X_sc.shape[0] - 1, X_sc.shape[1])
pca = PCA(n_components=n_comp, random_state=SEED)
X_pca = pca.fit_transform(X_sc)
print("PCA:", n_comp, "componentes, var explicada:", round(pca.explained_variance_ratio_.sum(), 3))

X_t = torch.tensor(X_pca, dtype=torch.float32)
loader = DataLoader(TensorDataset(X_t), batch_size=BATCH_SIZE, shuffle=True)"""),

    md("## 3. VAE (PyTorch)"),
    code("""class Encoder(nn.Module):
    def __init__(self, d_in, d_hidden, d_latent):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU())
        self.mu = nn.Linear(d_hidden, d_latent)
        self.logvar = nn.Linear(d_hidden, d_latent)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, d_latent, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, z):
        return self.net(z)


def reparam(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


encoder = Encoder(n_comp, HIDDEN, LATENT_DIM).to(DEVICE)
decoder = Decoder(LATENT_DIM, HIDDEN, n_comp).to(DEVICE)
opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

def vae_loss(x, x_hat, mu, logvar):
    recon = nn.functional.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + BETA * kl, recon.item(), kl.item()

print(encoder)"""),

    md("## 4. Entrenamiento"),
    code("""history = {"loss": [], "recon": [], "kl": []}

for ep in range(1, EPOCHS + 1):
    encoder.train()
    decoder.train()
    ep_loss, ep_rec, ep_kl, n = 0.0, 0.0, 0.0, 0
    for (xb,) in loader:
        xb = xb.to(DEVICE)
        mu, logvar = encoder(xb)
        z = reparam(mu, logvar)
        x_hat = decoder(z)
        loss, rec, kl = vae_loss(xb, x_hat, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ep_loss += loss.item()
        ep_rec += rec
        ep_kl += kl
        n += 1
    history["loss"].append(ep_loss / n)
    history["recon"].append(ep_rec / n)
    history["kl"].append(ep_kl / n)
    if ep % 100 == 0 or ep == 1:
        print(f"ep {ep:4d}  loss={history['loss'][-1]:.4f}  recon={history['recon'][-1]:.4f}  kl={history['kl'][-1]:.4f}")

plt.figure(figsize=(8, 3))
plt.plot(history["loss"], label="loss")
plt.plot(history["recon"], label="recon")
plt.plot(history["kl"], label="kl")
plt.legend()
plt.xlabel("epoch")
plt.tight_layout()
plt.savefig(FIG / "11_anox_vae_training.png", dpi=150)
plt.show()"""),

    md("## 5. Reconstruccion en datos reales"),
    code("""encoder.eval()
decoder.eval()
with torch.no_grad():
    mu, logvar = encoder(X_t.to(DEVICE))
    X_rec_pca = decoder(mu).cpu().numpy()

X_rec_z = scaler.inverse_transform(pca.inverse_transform(X_rec_pca))
errs = np.mean((X_rec_z - X_z) ** 2, axis=1)
print("MSE recon por sujeto:", np.round(errs, 4))

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
rid = ids[0]
M_real = vector_to_matrix(X_z[0])
M_rec = vector_to_matrix(X_rec_z[0])
v = max(abs(M_real).max(), abs(M_rec).max())
axes[0].imshow(M_real, cmap="RdBu_r", vmin=-v, vmax=v)
axes[0].set_title(f"real {rid}")
axes[1].imshow(M_rec, cmap="RdBu_r", vmin=-v, vmax=v)
axes[1].set_title("recon")
axes[2].imshow(M_rec - M_real, cmap="RdBu_r")
axes[2].set_title("diff")
plt.tight_layout()
plt.savefig(FIG / "11_anox_vae_recon.png", dpi=150)
plt.show()"""),

    md("## 6. Muestreo latente y sintesis"),
    code("""with torch.no_grad():
    mu, logvar = encoder(X_t.to(DEVICE))
    z_train = mu.cpu().numpy()

z_mean = z_train.mean(axis=0)
z_std = z_train.std(axis=0) + 1e-6

synth_rows = []
synth_z = []
rng = np.random.default_rng(SEED)

for i in range(N_SYNTH):
    z = z_mean + rng.normal(0, 1, size=LATENT_DIM) * z_std
    z = z_mean + 0.8 * (z - z_mean)
    z_t = torch.tensor(z, dtype=torch.float32, device=DEVICE)
    pca_vec = decoder(z_t).cpu().numpy()
    x_z = scaler.inverse_transform(pca.inverse_transform(pca_vec.reshape(1, -1)))[0]
    x_z = np.clip(x_z, -3.5, 3.5)
    synth_z.append(x_z)
    synth_rows.append({
        "record_id": f"ANOX_SYN_{i+1:03d}",
        "diagnosis": "ANOX",
        "synthetic": True,
        "source": "vae_pca",
    })

synth_z = np.stack(synth_z)
synth_df = pd.DataFrame(synth_rows)
print(synth_df.head())"""),

    md("## 7. Control de calidad"),
    code("""def offdiag(mat):
    m = mat.copy()
    np.fill_diagonal(m, 0)
    return m[np.triu_indices(N_ROI, 1)]

real_flat = np.concatenate([offdiag(vector_to_matrix(x)) for x in X_z])
syn_flat = np.concatenate([offdiag(vector_to_matrix(x)) for x in synth_z])

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].hist(real_flat, bins=40, density=True, alpha=0.6, label="real")
axes[0].hist(syn_flat, bins=40, density=True, alpha=0.6, label="synth")
axes[0].legend()
axes[0].set_title("distribucion Fisher-z")

def mean_eigval(x):
    return np.linalg.eigvalsh(vector_to_matrix(x))[-1]

eig_real = [mean_eigval(x) for x in X_z]
eig_syn = [mean_eigval(x) for x in synth_z]
axes[1].boxplot([eig_real, eig_syn], tick_labels=["real", "synth"])
axes[1].set_title("lambda1 max (sin umbral)")
plt.tight_layout()
plt.savefig(FIG / "11_anox_qc.png", dpi=150)
plt.show()

print("lambda1 real:", np.round(eig_real, 2))
print("lambda1 synth:", np.round(eig_syn, 2))"""),

    md("## 8. Guardar .mat y indice"),
    code("""mat_dir = OUT / "synthetic_mats"
mat_dir.mkdir(exist_ok=True)

for i, row in synth_df.iterrows():
    M = vector_to_matrix(synth_z[i])
    np.fill_diagonal(M, 1.0)
    savemat(mat_dir / f"{row['record_id']}.mat", {"fc": M})

real_idx = anox.assign(synthetic=False, source="measured")[["record_id", "diagnosis", "synthetic", "source", "mat_path"]]
syn_idx = synth_df.copy()
syn_idx["mat_path"] = syn_idx["record_id"].apply(lambda r: str((mat_dir / f"{r}.mat").resolve()))
aug_idx = pd.concat([real_idx, syn_idx], ignore_index=True)
aug_idx.to_csv(OUT / "anox_augmented_index.csv", index=False)

np.save(OUT / "synth_vectors_fisherz.npy", synth_z)
torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
            "pca": pca, "scaler": scaler, "n_comp": n_comp, "latent_dim": LATENT_DIM}, OUT / "vae_anox.pt")

meta = {"n_real": len(anox), "n_synth": N_SYNTH, "pca_dim": n_comp, "latent_dim": LATENT_DIM,
        "epochs": EPOCHS, "seed": SEED}
with open(OUT / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("real:", len(real_idx), "| synth:", len(syn_idx), "| total:", len(aug_idx))
print("salida:", OUT)"""),

    md("## 9. Uso sugerido\n\nMezclar sinteticos con reales solo en analisis exploratorios. Para p-values usar bootstrap sobre los 9 reales. Comparar si el signo del efecto se mantiene al agregar sinteticos como sensibilidad, no como cohorte ampliada."),
]

Path("11_anox_augment.ipynb").write_text(
    json.dumps({
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": cells,
    }, indent=1, ensure_ascii=False),
    encoding="utf-8",
)
print("Wrote 11_anox_augment.ipynb")
