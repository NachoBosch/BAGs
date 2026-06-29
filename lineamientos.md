# Lineamientos del Proyecto: Reorganización Topológica de Redes Cerebrales y Envejecimiento Acelerado en Población Latinoamericana (ReDLaT / BrainLat)

> **Objetivo central:** Determinar si cerebros con BAG positivo exhiben reducción en eficiencia local y centralidad de hubs frontoparietales, mediante análisis de grafos sobre matrices de conectividad funcional (fMRI en reposo), en la cohorte multicéntrica BrainLat, controlando por edad cronológica, nivel educativo y sitio de adquisición.

---

## Índice

1. [Estructura y auditoría de datos](#1-estructura-y-auditoría-de-datos)
2. [Exploración descriptiva](#2-exploración-descriptiva)
3. [Preprocesamiento de neuroimagen](#3-preprocesamiento-de-neuroimagen)
4. [Extracción de matrices de conectividad funcional](#4-extracción-de-matrices-de-conectividad-funcional)
5. [Umbralización de la matriz de conectividad y construcción de grafos](#5-umbralización-de-la-matriz-de-conectividad-y-construcción-de-grafos)
6. [Extracción de métricas topológicas por ROI](#6-extracción-de-métricas-topológicas-por-roi)
7. [Estimación del Brain Age Gap (BAG)](#7-estimación-del-brain-age-gap-bag)
8. [Comparación entre grupos de edad y diagnóstico](#8-comparación-entre-grupos-de-edad-y-diagnóstico)
9. [Regresión y análisis estadístico del objetivo principal](#9-regresión-y-análisis-estadístico-del-objetivo-principal)
10. [Clasificación con Machine Learning clásico (Random Forest)](#10-clasificación-con-machine-learning-clásico-random-forest)
11. [Cruce de información multimodal](#11-cruce-de-información-multimodal)
12. [Stack tecnológico resumido](#12-stack-tecnológico-resumido)

---

## 1. Estructura y auditoría de datos

### 1.1 Archivos disponibles

| Archivo | Contenido |
|---|---|
| `BrainLat_Demographic_MRI.csv` | MRI_ID, EEG_ID, diagnóstico, sexo, edad, años de educación, lateralidad |
| `BrainLat_Cognition_MRI.csv` | Scores MoCA (total + subescalas), IFS (total + subescalas), mini-SEA (reconocimiento facial, ToM), emotion recog |
| `BrainLat_records_MRI.csv` | Modalidades disponibles por sujeto: T1, Rest-fMRI, DWI, campo magnético (1.5T / 3T), EEG |
| `brainlat/sub-*_scans.tsv` | Rutas relativas a los archivos NIfTI por sujeto |
| `brainlat/sub-*_rs-HEP_channels.tsv` | Canales EEG para la tarea rs-HEP |
| `brainlat/sub-*_dwi.bval` | Valores b para difusión (DWI) |

### 1.2 Pasos de auditoría

```python
import pandas as pd

demo  = pd.read_csv("BrainLat_Demographic_MRI.csv")
cog   = pd.read_csv("BrainLat_Cognition_MRI.csv")
rec   = pd.read_csv("BrainLat_records_MRI.csv")

# Merge maestro por MRI_ID
df = demo.merge(cog[["MRI_ID", "moca_total", "ifs_total_score",
                      "mini_sea_fer", "mini_sea_tom"]],
                on="MRI_ID", how="left")
df = df.merge(rec[["MRI_ID", "T1", "Rest", "DWI", "MF", "eeg"]],
              on="MRI_ID", how="left")
```

- Extraer el **prefijo de sitio** del MRI_ID (CLB, PSL, COA, MXA, MXB, CLA, PE) como variable `site`.
- Verificar N por diagnóstico (CN, AD, FTD) y por sitio.
- Identificar sujetos con `Rest=1` (únicos elegibles para el análisis de conectividad funcional).
- Chequear datos faltantes en covariables clave: edad, educación, sitio, MoCA.

### 1.3 Criterios de inclusión / exclusión

- **Incluir:** T1=1 + Rest=1 (fMRI en reposo disponible)
- **Excluir:** movimiento excesivo en fMRI (framewise displacement medio > 0.5 mm o > 20% de volúmenes censurados)
- **Subcohorte DWI:** sujetos con DWI=1 para análisis de conectividad estructural como validación convergente

---

## 2. Exploración descriptiva

### 2.1 Distribuciones demográficas

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Edad por diagnóstico y sitio
sns.violinplot(data=df, x="diagnosis", y="Age", hue="site")

# Distribución de scores cognitivos por grupo
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["moca_total", "ifs_total_score", "mini_sea_fer"]):
    sns.boxplot(data=df, x="diagnosis", y=col, ax=ax)
```

### 2.2 Correlaciones entre variables clínicas y cognitivas

```python
import pingouin as pg

# Correlación edad vs. cognición por diagnóstico
corr_matrix = df[["Age", "years_education", "moca_total",
                   "ifs_total_score", "mini_sea_fer", "mini_sea_tom"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
```

- Reportar Pearson/Spearman según distribución (test de normalidad Shapiro-Wilk).
- Examinar si el efecto de sitio es confundidor: ANOVA de un factor sobre edad y educación con `site` como factor.

### 2.3 Caracterización del efecto de sitio (pre-harmonización)

```python
import statsmodels.formula.api as smf

model = smf.ols("Age ~ C(site) + C(diagnosis)", data=df).fit()
print(model.summary())
```

Esto justificará el uso posterior de ComBat.

---

## 3. Preprocesamiento de neuroimagen

### 3.1 Pipeline fMRIPrep

Correr **fMRIPrep** (≥ 23.x) en modo BIDS sobre cada sujeto con Rest=1:

```bash
fmriprep /data/bids /data/derivatives/fmriprep \
    participant \
    --participant-label SUB_ID \
    --fs-no-reconall \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --use-aroma \
    --fd-spike-threshold 0.5 \
    --dummy-scans 5 \
    --nthreads 8 --omp-nthreads 4 \
    --work-dir /scratch/fmriprep_wf
```

Salidas clave por sujeto:
- `*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz` — señal BOLD preprocesada
- `*_desc-confounds_timeseries.tsv` — regresores de movimiento, WM, CSF, AROMA

### 3.2 Volumetría estructural (VBM)

Con las imágenes T1 preprocesadas por fMRIPrep, usar **FSL-VBM** o **SPM12 VBM**:
- Segmentación de materia gris → mapas de probabilidad de GM
- Modulación por Jacobiano (VBM modulated)
- Suavizado 8 mm FWHM
- Extracción de volúmenes regionales por atlas AAL-116 → vector de features estructurales por sujeto

---

## 4. Extracción de matrices de conectividad funcional

### 4.1 Regresión de señales de ruido y extracción de series temporales por ROI

```python
from nilearn import image, signal
from nilearn.datasets import fetch_atlas_aal
from nilearn.input_data import NiftiLabelsMasker
import numpy as np

# Cargar atlas AAL-116
aal = fetch_atlas_aal(version="SPM12")
masker = NiftiLabelsMasker(
    labels_img=aal.maps,
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0,            # ajustar según el TR real del protocolo
    memory="nilearn_cache"
)

# Para cada sujeto con Rest=1
bold_img = "derivatives/fmriprep/sub-XXX/.../preproc_bold.nii.gz"
confounds_df = pd.read_csv("..._desc-confounds_timeseries.tsv", sep="\t")

# Seleccionar regresores: 6 mov params, WM, CSF, sus derivadas
regs = ["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z",
        "white_matter","csf",
        "trans_x_derivative1","trans_y_derivative1","trans_z_derivative1",
        "rot_x_derivative1","rot_y_derivative1","rot_z_derivative1"]
confounds = confounds_df[regs].fillna(0).values

time_series = masker.fit_transform(bold_img, confounds=confounds)
# shape: (n_timepoints, 116)
```

### 4.2 Cómputo de la matriz de correlación (FC matrix)

```python
from nilearn.connectome import ConnectivityMeasure

conn_measure = ConnectivityMeasure(kind="correlation")
fc_matrix = conn_measure.fit_transform([time_series])[0]
# shape: (116, 116) — correlaciones de Pearson entre todas las ROIs del AAL
```

- Guardar la FC matrix por sujeto en formato `.npy` o HDF5.
- Aplicar la transformación **Fisher z** antes de cualquier análisis estadístico sobre las correlaciones:

```python
fc_z = np.arctanh(np.clip(fc_matrix, -0.999, 0.999))
np.fill_diagonal(fc_z, 0)
```

---

## 5. Umbralización de la matriz de conectividad y construcción de grafos

Este es el paso central del análisis. La "umbralización" convierte la matriz de correlación continua en un grafo binario o ponderado que puede analizarse con métricas de teoría de grafos.

### 5.1 Estrategia de umbral de proporcionalidad (metodología Coronel-Oliveros et al. 2025)

En lugar de usar un umbral fijo de correlación (que produce grafos de densidades muy distintas entre sujetos), se usa un **umbral de proporcionalidad**: conservar solo el K% de las conexiones más fuertes. Esto iguala la densidad del grafo entre sujetos.

```python
import numpy as np
import networkx as nx

def threshold_proportional(fc_matrix, density):
    """
    Retorna una matriz binaria manteniendo el `density` % de conexiones más fuertes.
    Solo se consideran conexiones positivas (correlaciones positivas).
    """
    n = fc_matrix.shape[0]
    mat = fc_matrix.copy()
    np.fill_diagonal(mat, 0)
    mat[mat < 0] = 0                          # solo conexiones positivas

    k_total = n * (n - 1) / 2                 # conexiones posibles (triangular sup)
    k_keep = int(np.round(density * k_total))

    triu_idx = np.triu_indices(n, k=1)
    values = mat[triu_idx]

    threshold = np.sort(values)[::-1][k_keep]
    binary = (mat >= threshold).astype(float)
    return binary

# Rango de densidades según el paper de referencia
densities = np.arange(0.02, 0.21, 0.01)       # 0.02 a 0.20 en pasos de 0.01
```

### 5.2 Por qué múltiples umbrales

- Evita que los resultados dependan de la elección arbitraria de un único umbral.
- Permite visualizar la estabilidad de las métricas topológicas a través de densidades.
- Se reporta la **media del área bajo la curva** de la métrica vs. densidad como valor resumen por sujeto.

```python
def auc_metric(metrics_per_density):
    """Área bajo la curva (trapecio) de una métrica sobre el rango de densidades."""
    return np.trapz(metrics_per_density, dx=1)
```

### 5.3 Construcción del grafo con NetworkX

```python
def build_graph(binary_matrix):
    G = nx.from_numpy_array(binary_matrix)
    return G
```

---

## 6. Extracción de métricas topológicas por ROI

### 6.1 Métricas globales del grafo

```python
import bct  # Brain Connectivity Toolbox (bctpy)

def compute_graph_metrics(binary_matrix):
    n = binary_matrix.shape[0]
    G = build_graph(binary_matrix)

    metrics = {}

    # Eficiencia local: segregación modular
    # bct.efficiency_bin devuelve eficiencia local por nodo cuando local=True
    metrics["local_efficiency"]  = np.mean(bct.efficiency_bin(binary_matrix, local=True))

    # Eficiencia global: integración
    metrics["global_efficiency"] = bct.efficiency_bin(binary_matrix, local=False)

    # Coeficiente de clustering promedio
    metrics["clustering_coeff"]  = np.mean(bct.clustering_coef_bu(binary_matrix))

    # Degree centrality (grado normalizado)
    degree = np.array(list(dict(G.degree()).values())) / (n - 1)
    metrics["mean_degree"]       = np.mean(degree)

    # Betweenness centrality
    bc = bct.betweenness_bin(binary_matrix.astype(float))
    metrics["betweenness_mean"]  = np.mean(bc)
    metrics["betweenness_nodes"] = bc   # vector completo (116,) para análisis por ROI

    return metrics
```

### 6.2 Métricas por ROI individual (hubs frontoparietales)

El objetivo específico es la **centralidad de hubs frontoparietales**. Identificar las ROIs del atlas AAL-116 correspondientes a:

- Córtex prefrontal dorsolateral (DLPFC): Superior Frontal Gyrus (AAL: 3, 4, 7, 8)
- Corteza cingulada anterior (ACC): AAL 33, 34
- Córtex parietal inferior: AAL 61, 62, 63, 64
- Precuña: AAL 67, 68
- Córtex temporal superior: AAL 81, 82

```python
# Índices AAL para ROIs frontoparietales (0-indexed)
frontoparietal_idx = [2, 3, 6, 7,   # superior frontal
                      32, 33,        # ACC
                      60, 61, 62, 63, # parietal inferior
                      66, 67,        # precuneus
                      80, 81]        # superior temporal

def frontoparietal_hub_centrality(binary_matrix, fp_idx):
    bc_all = bct.betweenness_bin(binary_matrix.astype(float))
    dc_all = np.sum(binary_matrix, axis=0) / (binary_matrix.shape[0] - 1)
    return {
        "fp_betweenness": np.mean(bc_all[fp_idx]),
        "fp_degree":      np.mean(dc_all[fp_idx])
    }
```

### 6.3 Bucle completo sobre múltiples umbrales por sujeto

```python
def subject_metrics_auc(fc_matrix, densities, fp_idx):
    local_eff_d  = []
    global_eff_d = []
    clust_d      = []
    fp_bc_d      = []
    fp_dc_d      = []

    for d in densities:
        bmat = threshold_proportional(fc_matrix, d)
        m    = compute_graph_metrics(bmat)
        fp   = frontoparietal_hub_centrality(bmat, fp_idx)

        local_eff_d.append(m["local_efficiency"])
        global_eff_d.append(m["global_efficiency"])
        clust_d.append(m["clustering_coeff"])
        fp_bc_d.append(fp["fp_betweenness"])
        fp_dc_d.append(fp["fp_degree"])

    return {
        "local_efficiency_auc":  np.trapz(local_eff_d, densities),
        "global_efficiency_auc": np.trapz(global_eff_d, densities),
        "clustering_auc":        np.trapz(clust_d, densities),
        "fp_betweenness_auc":    np.trapz(fp_bc_d, densities),
        "fp_degree_auc":         np.trapz(fp_dc_d, densities),
    }
```

Construir una tabla `graph_metrics_df` con una fila por sujeto y columnas para cada métrica AUC + covariables (MRI_ID, diagnosis, Age, site, etc.).

---

## 7. Estimación del Brain Age Gap (BAG)

### 7.1 Features para el modelo de edad cerebral

```python
# Features: FC matrix (triangular superior aplanada) + volumetría regional VBM
n_rois = 116
triu_idx = np.triu_indices(n_rois, k=1)  # 116*115/2 = 6670 features de FC
fc_features = fc_z[triu_idx]

vbm_features = [...]  # vector de 116 volúmenes de GM por sujeto

X_subject = np.concatenate([fc_features, vbm_features])
```

### 7.2 Estrategia de entrenamiento

- **Sujetos de entrenamiento:** solo CN (controles cognitivamente normales), ya que el modelo debe aprender la relación "edad normal → señal neuroimagen" sin la distorsión de la patología.
- **Variable objetivo:** edad cronológica.
- **Validación:** Leave-One-Site-Out cross-validation (LOSO-CV) para evitar sobreajuste al sitio.

```python
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

X_cn = graph_metrics_df[graph_metrics_df["diagnosis"] == "CN"][feature_cols].values
y_cn = graph_metrics_df[graph_metrics_df["diagnosis"] == "CN"]["Age"].values
sites_cn = graph_metrics_df[graph_metrics_df["diagnosis"] == "CN"]["site"].values

logo = LeaveOneGroupOut()
predicted_ages = np.zeros(len(y_cn))

for train_idx, test_idx in logo.split(X_cn, y_cn, sites_cn):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
    ])
    pipe.fit(X_cn[train_idx], y_cn[train_idx])
    predicted_ages[test_idx] = pipe.predict(X_cn[test_idx])

# Entrenar modelo final con todos los CN para predecir en AD y FTD
pipe_final = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
])
pipe_final.fit(X_cn, y_cn)
```

### 7.3 Corrección del sesgo de edad en el BAG

El BAG "crudo" tiene una correlación negativa artefactual con la edad (los modelos tienden a sobrepredecir en jóvenes y subpredecir en mayores). Corrección por regresión:

```python
from scipy import stats

# Ajustar en el set CN
slope, intercept, _, _, _ = stats.linregress(y_cn, predicted_ages - y_cn)
bag_corrected = (predicted_ages - y_cn) - (slope * y_cn + intercept)
```

### 7.4 Aplicar a toda la cohorte

```python
X_all = graph_metrics_df[feature_cols].values
y_all = graph_metrics_df["Age"].values

pred_age_all = pipe_final.predict(X_all)
bag_raw      = pred_age_all - y_all
slope_all, intercept_all, _, _, _ = stats.linregress(y_cn, predicted_ages - y_cn)
bag_corrected_all = bag_raw - (slope_all * y_all + intercept_all)

graph_metrics_df["predicted_age"] = pred_age_all
graph_metrics_df["BAG_corrected"] = bag_corrected_all
graph_metrics_df["BAG_group"]     = (bag_corrected_all > 0).map({True: "BAG+", False: "BAG-"})
```

---

## 8. Comparación entre grupos de edad y diagnóstico

### 8.1 Grupos etarios

Segmentar la cohorte en bandas de edad para examinar si el efecto del BAG en la topología varía con la edad:

```python
bins   = [40, 55, 65, 75, 90]
labels = ["40-54", "55-64", "65-74", "75+"]
graph_metrics_df["age_group"] = pd.cut(graph_metrics_df["Age"], bins=bins, labels=labels)
```

### 8.2 Comparación BAG+ vs BAG- en métricas topológicas

```python
import pingouin as pg

topological_metrics = ["local_efficiency_auc", "global_efficiency_auc",
                        "clustering_auc", "fp_betweenness_auc", "fp_degree_auc"]

results = []
for metric in topological_metrics:
    bag_pos = graph_metrics_df[graph_metrics_df["BAG_group"] == "BAG+"][metric].dropna()
    bag_neg = graph_metrics_df[graph_metrics_df["BAG_group"] == "BAG-"][metric].dropna()

    # Test de normalidad
    _, p_norm = stats.shapiro(np.concatenate([bag_pos, bag_neg]))

    if p_norm > 0.05:
        stat = pg.ttest(bag_pos, bag_neg)
        test_used = "t-test"
    else:
        stat = pg.mwu(bag_pos, bag_neg)
        test_used = "Mann-Whitney"

    results.append({
        "metric": metric,
        "test": test_used,
        "statistic": stat["T"].values[0] if "T" in stat else stat["U-val"].values[0],
        "p_val": stat["p-val"].values[0],
        "cohen_d": stat.get("cohen-d", [None])[0] if "cohen-d" in stat.columns else None
    })

results_df = pd.DataFrame(results)

# Corrección FDR (Benjamini-Hochberg)
from statsmodels.stats.multitest import multipletests
_, results_df["p_fdr"], _, _ = multipletests(results_df["p_val"], method="fdr_bh")
```

### 8.3 Comparación entre diagnósticos (CN vs AD vs FTD)

```python
for metric in topological_metrics:
    aov = pg.kruskal(data=graph_metrics_df, dv=metric, between="diagnosis")
    print(f"{metric}: H={aov['H'].values[0]:.3f}, p={aov['p-unc'].values[0]:.4f}")

    # Post-hoc Dunn con corrección Bonferroni
    posthoc = pg.pairwise_tests(data=graph_metrics_df, dv=metric,
                                between="diagnosis", padjust="bonf")
```

### 8.4 Interacción diagnóstico × grupo etario

```python
import statsmodels.formula.api as smf

for metric in topological_metrics:
    formula = f"{metric} ~ C(diagnosis) * C(age_group) + years_education + C(site)"
    model = smf.ols(formula, data=graph_metrics_df).fit()
    print(model.summary())
```

---

## 9. Regresión y análisis estadístico del objetivo principal

### 9.1 Armonización de sitio con ComBat

Antes de la regresión principal, harmonizar las métricas topológicas para remover el efecto del sitio de adquisición:

```python
import neuroCombat as nc

# Matriz de features topológicos: shape (n_sujetos, n_metricas)
data_for_combat = graph_metrics_df[topological_metrics].T.values  # (n_metricas, n_sujetos)
batch = graph_metrics_df["site"].values

# Covariables a preservar (no eliminar su varianza)
covars = pd.DataFrame({
    "batch": batch,
    "Age": graph_metrics_df["Age"].values,
    "diagnosis": pd.Categorical(graph_metrics_df["diagnosis"]).codes,
    "years_education": graph_metrics_df["years_education"].values
})

data_harmonized = nc.neuroCombat(
    dat=data_for_combat,
    covars=covars,
    batch_col="batch",
    continuous_cols=["Age", "years_education"],
    categorical_cols=["diagnosis"]
)["data"]

# Actualizar dataframe
for i, metric in enumerate(topological_metrics):
    graph_metrics_df[metric + "_harmonized"] = data_harmonized[i, :]
```

### 9.2 Regresión múltiple: métricas topológicas ~ BAG + covariables

```python
harmonized_metrics = [m + "_harmonized" for m in topological_metrics]
regression_results = []

for metric in harmonized_metrics:
    formula = f"{metric} ~ BAG_corrected + Age + years_education + C(site) + C(sex)"
    model = smf.ols(formula, data=graph_metrics_df).fit()
    coef_bag = model.params["BAG_corrected"]
    p_bag    = model.pvalues["BAG_corrected"]
    ci_bag   = model.conf_int().loc["BAG_corrected"]

    regression_results.append({
        "metric": metric,
        "coef_BAG": coef_bag,
        "p_val":    p_bag,
        "CI_low":   ci_bag[0],
        "CI_high":  ci_bag[1],
        "R2_adj":   model.rsquared_adj
    })

reg_df = pd.DataFrame(regression_results)
_, reg_df["p_fdr"], _, _ = multipletests(reg_df["p_val"], method="fdr_bh")
```

### 9.3 Análisis de mediación (exploratorio)

Si la adversidad socioeconómica estuviera disponible como variable (GINI, índice socioeconómico por sitio/país), se podría testear si la reorganización topológica media la relación `SEI → BAG`:

```python
# Usando pingouin.mediation_analysis cuando haya datos de adversidad social
# mediation_analysis(data=df, x="SEI_index", m="local_efficiency_auc_harmonized",
#                    y="BAG_corrected", covar=["Age", "years_education", "site"])
```

---

## 10. Clasificación con Machine Learning clásico (Random Forest)

El objetivo de esta sección es doble: (a) evaluar si las métricas topológicas permiten clasificar grupos clínicos y (b) contrastar con el enfoque de grafos para identificar qué representación de los datos (raw FC vs. métricas topológicas vs. combinación) es más discriminativa.

### 10.1 Diseño experimental

| Tarea | Variable objetivo | Features |
|---|---|---|
| Clasificación diagnóstica | CN vs. AD vs. FTD | Métricas topológicas AUC |
| Predicción BAG group | BAG+ vs. BAG- | Métricas topológicas AUC |
| Regresión de edad cerebral | Edad cronológica (continua) | FC matrix + VBM |
| Comparación de representaciones | BAG group | Raw FC (6670) vs. métricas topológicas (5) vs. combinado |

### 10.2 Pipeline de clasificación con Random Forest

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

feature_cols_topo = [m + "_harmonized" for m in topological_metrics]

X = graph_metrics_df[feature_cols_topo].values
y = graph_metrics_df["diagnosis"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

cv_results = cross_validate(
    pipe_rf, X, y_enc, cv=cv,
    scoring=["accuracy", "f1_macro", "roc_auc_ovr"],
    return_train_score=True
)

print(f"Accuracy: {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
print(f"F1 macro: {cv_results['test_f1_macro'].mean():.3f} ± {cv_results['test_f1_macro'].std():.3f}")
```

### 10.3 Importancia de features e identificación de ROIs relevantes

```python
pipe_rf.fit(X, y_enc)
rf_model = pipe_rf.named_steps["clf"]

importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature": feature_cols_topo,
    "importance": importances
}).sort_values("importance", ascending=False)

# Visualizar
sns.barplot(data=feat_imp_df, x="importance", y="feature", orient="h")
plt.title("Random Forest — Importancia de métricas topológicas")
```

### 10.4 Comparación de representaciones de datos

```python
from sklearn.metrics import make_scorer, f1_score

scorer = make_scorer(f1_score, average="macro")

# Representación 1: raw FC (6670 features)
X_fc = np.array([...])  # FC matrices aplanadas por sujeto

# Representación 2: métricas topológicas (5 features)
X_topo = graph_metrics_df[feature_cols_topo].values

# Representación 3: combinada
X_combined = np.hstack([X_fc, X_topo])

results_comparison = {}
for name, X_repr in [("raw_FC", X_fc), ("topological", X_topo), ("combined", X_combined)]:
    scores = cross_validate(pipe_rf, X_repr, y_enc, cv=cv, scoring=scorer)
    results_comparison[name] = scores["test_score"]

# Comparar con Friedman test o Wilcoxon pareado
from scipy.stats import wilcoxon
stat, p = wilcoxon(results_comparison["topological"], results_comparison["raw_FC"])
print(f"Topological vs Raw FC: W={stat}, p={p:.4f}")
```

### 10.5 Tuning de hiperparámetros (opcional)

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "clf__n_estimators": [100, 200, 500],
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__max_features": ["sqrt", "log2", 0.3]
}

search = RandomizedSearchCV(
    pipe_rf, param_dist, n_iter=30, cv=cv,
    scoring="f1_macro", random_state=42, n_jobs=-1
)
search.fit(X, y_enc)
print(search.best_params_)
```

---

## 11. Cruce de información multimodal

### 11.1 FC funcional + VBM estructural

Evaluar si la conectividad funcional y la volumetría de materia gris son predictores independientes del BAG:

```python
formula = ("BAG_corrected ~ local_efficiency_auc_harmonized + "
           "fp_betweenness_auc_harmonized + gm_volume_prefrontal + "
           "Age + years_education + C(site)")
model_multimodal = smf.ols(formula, data=graph_metrics_df).fit()
```

### 11.2 Cognición como variable de resultado

Correlacionar las métricas topológicas armonizadas con los scores cognitivos disponibles:

```python
cog_scores = ["moca_total", "ifs_total_score", "mini_sea_fer", "mini_sea_tom"]

cross_corr_results = []
for topo in harmonized_metrics:
    for cog in cog_scores:
        subset = graph_metrics_df[[topo, cog, "Age", "years_education"]].dropna()
        # Correlación parcial controlando por edad y educación
        result = pg.partial_corr(data=subset, x=topo, y=cog,
                                  covar=["Age", "years_education"])
        cross_corr_results.append({
            "topological": topo, "cognitive": cog,
            "r": result["r"].values[0], "p": result["p-val"].values[0]
        })

cross_corr_df = pd.DataFrame(cross_corr_results)
_, cross_corr_df["p_fdr"], _, _ = multipletests(cross_corr_df["p"], method="fdr_bh")
```

### 11.3 Heatmap de correlaciones cruzadas topológicas × cognitivas

```python
pivot = cross_corr_df.pivot(index="topological", columns="cognitive", values="r")
sig_mask = cross_corr_df.pivot(index="topological", columns="cognitive", values="p_fdr") > 0.05

sns.heatmap(pivot, mask=sig_mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
            cbar_kws={"label": "Correlación parcial (r)"})
plt.title("Correlaciones topología × cognición (celdas significativas FDR<0.05)")
```

### 11.4 Cruce BAG + cognición + topología

Perfil integrado por sujeto para análisis de clúster:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

profile_cols = harmonized_metrics + ["BAG_corrected", "moca_total", "ifs_total_score"]
profile_data = graph_metrics_df[profile_cols].dropna()
profile_scaled = StandardScaler().fit_transform(profile_data)

# k-means con k=2,3,4 y selección por silhouette
from sklearn.metrics import silhouette_score

for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(profile_scaled)
    sil = silhouette_score(profile_scaled, labels)
    print(f"k={k}: silhouette={sil:.3f}")
```

---

## 12. Stack tecnológico resumido

| Tarea | Herramienta |
|---|---|
| Preprocesamiento fMRI | fMRIPrep ≥ 23.x |
| VBM estructural | FSL-VBM / SPM12 |
| Series temporales ROI | Nilearn + atlas AAL-116 |
| Matrices FC | Nilearn `ConnectivityMeasure` |
| Umbralización y grafos | bctpy + NetworkX |
| Análisis estadístico | Pingouin + Statsmodels |
| Armonización de sitio | neuroCombat (Python) |
| Estimación BAG | XGBoost / SVR + scikit-learn |
| ML clásico | RandomForestClassifier / Regressor |
| Corrección múltiple | FDR Benjamini-Hochberg (statsmodels) |
| Visualización | Matplotlib + Seaborn + Nilearn plotting |
| Gestión de datos | Pandas + NumPy + HDF5 |

---

## Notas metodológicas finales

1. **Umbralización proporcional vs. peso absoluto:** para análisis secundarios con grafos ponderados (weighted), usar la correlación de Pearson directamente como peso (sin umbral) con `bct.efficiency_wei`. Comparar resultados con los binarios para evaluar robustez.

2. **Corrección por edad en métricas topológicas:** las métricas de grafo correlacionan intrínsecamente con la edad. Antes de toda comparación de grupos, regresionar la edad fuera de cada métrica en el set CN para obtener residuales age-corrected.

3. **Leave-One-Site-Out:** en todos los modelos de ML, usar LOSO-CV como estrategia principal para garantizar que el modelo generaliza a nuevos sitios (lo que equivale a nuevos escáneres, operadores y protocolos).

4. **Reporte de tamaño de efecto:** siempre reportar Cohen's d (diferencias de medias) o ω² / η² (ANOVA) además del p-valor.

5. **Open science:** guardar matrices FC por sujeto, las métricas topológicas y el dataframe maestro en formatos abiertos (CSV + NumPy `.npy`) versionados con DVC o en un repositorio BIDS-compliant.

---

*Basado en Coronel-Oliveros et al., Nature Communications (2025) y Moguilner et al., Nature Medicine (2024). Datos: cohorte BrainLat / ReDLaT.*
