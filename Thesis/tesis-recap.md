# Tesis — Mapa del código y guía de uso

## Qué hace este pipeline

Predice la **edad cerebral** de cada sujeto a partir de su conectividad funcional (fMRI en reposo)
y su volumetría de materia gris (T1w). La diferencia entre la edad predicha y la cronológica es el
**Brain Age Gap (BAG)**: positivo significa que el cerebro "parece más viejo" de lo que corresponde.

El modelo central es una cadena de dos etapas:

```
Matriz FC (116×116)
      ↓
   β-VAE          ← comprime la FC en un espacio latente de baja dimensión (embeddings Z)
      ↓
  Z + T1w features
      ↓
  XGBoost         ← predice edad cronológica → BAG = edad_predicha − edad_real
```

Los hiperparámetros del VAE se optimizan con **Optuna** para minimizar el MAE de predicción de
edad, no solo la reconstrucción. Esto hace que el espacio latente sea útil para el downstream task.

---

## Mapa de archivos

### `src/` — librería modular

| Archivo | Rol |
|---|---|
| `config.py` | Punto central de configuración: rutas a datos y parámetros del experimento (semilla, diagnósticos a usar, tamaño del test, K-Fold, trials de Optuna). Tocar aquí antes de correr cualquier cosa. |
| `utils_ids.py` | Normaliza IDs de sujeto desde distintos formatos de nombre de archivo (`cat_sub-AF025_ses-T0_T1w.xml`, `sub-AF025_timeseries.mat`, `AF025`) a un formato canónico (`AF025`). Necesario porque los mismos sujetos aparecen nombrados distinto en cada fuente de datos. |
| `data_io.py` | Carga los tres tipos de datos: Excel de metadata clínica, CSV de volumetría T1w, y matrices FC desde `.mat`. Convierte matrices simétricas a vectores de triangular superior (6670 valores para AAL-116) y aplica Fisher-z. |
| `cohort.py` | Intersecta las tres fuentes de datos y devuelve un DataFrame con un sujeto por fila. Filtra por diagnóstico (`CN`, `AD`, `FTD`) y descarta sujetos sin edad. Colapsa duplicados T1w promediando. |
| `splits.py` | Genera y guarda en JSON el holdout estratificado (stratifica por sexo × diagnóstico) y los K pliegues de validación cruzada. Al ser JSON se pueden reproducir exactamente en todos los notebooks. |
| `vae_model.py` | Arquitectura β-VAE en TensorFlow/Keras. Encoder devuelve (μ, log\_var, z); decoder reconstruye la FC. Loss = reconstrucción (MSE/MAE/Huber) + β × KL divergence. |
| `vae_callbacks.py` | Dos callbacks Keras: `BetaScheduler` (warmup lineal de β desde 0 para estabilizar el entrenamiento) y `ValidationCorrelations` (monitorea Pearson y cosine similarity de reconstrucción en validación). |
| `vae_train.py` | Funciones de entrenamiento del VAE: K-Fold con early stopping y ReduceLR; entrenamiento final en todo trainval; carga/guarda pesos + hiperparámetros. |
| `vae_optuna_age.py` | Optimización Bayesiana (Optuna, TPE + MedianPruner) del VAE. El objetivo no es la reconstrucción sino el **MAE de brain age** predicho por XGBoost sobre los embeddings. Así se asegura que el espacio latente sea útil para predecir edad. |
| `embeddings.py` | Extrae los embeddings μ del encoder entrenado y los guarda como `.npy` + `.json` (con lista de IDs). |
| `xgb_train.py` | Construye la feature matrix concatenando bloques (Z del VAE + volumetría T1w + sexo opcional + diagnóstico opcional) y entrena `XGBRegressor`. |
| `optuna_xgb.py` | Tuning de XGBoost con Optuna + K-Fold, independiente del VAE (usado en el experimento de FC crudo como baseline). |
| `metrics.py` | Calcula MAE, RMSE, R² y Pearson para evaluar la predicción de edad en el test set. |
| `figures.py` | Genera paneles visuales (heatmaps FC original vs reconstruida) para las animaciones Manim de la tesis. |

---

### `notebooks/` — experimentos

| Notebook | Qué hace | Cuándo correrlo |
|---|---|---|
| `data_analysis.ipynb` | Estadísticas descriptivas de la cohorte: edad, sexo, educación, distribución por sitio y diagnóstico. Figuras para el capítulo 4. | Primero, para entender los datos. |
| `main.ipynb` | **Pipeline principal**: carga cohorte → splits → Optuna VAE → VAE final → XGBoost → evaluación en test. Genera todas las métricas y figuras del capítulo 6. | Después de tener los datos listos. Es el notebook central. |
| `experiments.ipynb` | Experimentos exploratorios: prueba configuraciones alternativas, decisiones de diseño (por qué β-VAE y no PCA, por qué XGBoost y no Ridge). | Como referencia de las decisiones tomadas. |
| `optuna_raw_fc.ipynb` | Baseline: XGBoost directo sobre FC crudo + T1w sin pasar por VAE. Sirve para demostrar que el VAE agrega valor. | Para comparación con el modelo principal. |
| `model_comparison.ipynb` | Compara Ridge vs SVR vs XGBoost usando los mismos features (embeddings Z + T1w). Justifica la elección de XGBoost. | Para la sección de comparación de modelos. |
| `demographics_analysis.ipynb` | Analiza cómo las variables demográficas (sitio, educación, sexo) afectan el BAG. Correlaciones parciales. | Para el análisis de covariables. |
| `demographics_optuna.ipynb` | Re-optimiza XGBoost con Optuna incluyendo educación y sitio como features. | Experimento adicional de ablación. |
| `cn_only_optuna.ipynb` | Modelo entrenado exclusivamente en controles sanos (CN). Más limpio teóricamente: predice edad sin confusión por patología. | Para el experimento CN-only del capítulo 6. |
| `cn_demographics_experiment.ipynb` | CN-only pero agregando educación y sitio como features del XGBoost. | Extensión del experimento anterior. |
| `regression_trees.ipynb` | Genera figuras del marco teórico (árboles de regresión sobre dataset público Advertising). No usa datos de neuroimagen. | Para el capítulo 2 (marco teórico). |

---

### `scripts/`

| Script | Qué hace |
|---|---|
| `pick_best_recon.py` | Carga el VAE final entrenado, reconstruye las FC del test set y elige el sujeto con mejor reconstrucción (percentil 75 de Pearson). Guarda `panel_original.png` y `panel_recon.png` sin bordes para insertar en las animaciones Manim. Requiere haber corrido `main.ipynb` primero. |
| `rerun_exp2_cn_only.py` | Re-corre el experimento CN-only con los parámetros definitivos, de forma reproducible desde línea de comandos sin abrir Jupyter. |

---

### `Animations/`

Animaciones en **Manim** para presentar el pipeline visualmente en la tesis:

| Archivo | Qué anima |
|---|---|
| `scene_01_fc.py` | Construcción de la matriz de conectividad funcional |
| `scene_02_vae.py` | Compresión de la FC por el β-VAE (encoder → espacio latente → decoder) |
| `scene_03_multimodal.py` | Fusión de embeddings Z + volumetría T1w |
| `scene_04_xgboost.py` | Predicción de edad con XGBoost y cálculo del BAG |
| `scene_05_optuna_vae.py` | Optimización Bayesiana del VAE con Optuna |
| `scene_06_optuna_xgb.py` | Tuning de XGBoost con Optuna |
| `pipeline_full.py` | Animación completa del pipeline de principio a fin |
| `common.py` | Estilos y utilidades compartidas entre escenas |
| `render_all.sh` | Renderiza todas las escenas en secuencia |

---

## Datos que necesitás tener en `Thesis/Data/`

```
Thesis/Data/
├── datos-redlat.xlsx          ← metadata clínica: record_id, demo_age, demo_sex,
│                                 clinical_diagnosis (columnas requeridas)
├── fc_mats/                   ← una carpeta con un .mat por sujeto, nombrado como
│                                 sub-XXXX_timeseries.mat (contiene la matriz FC 116×116)
└── Redlat_VGM_AAL_.csv       ← volumetría AAL ya extraída: primera columna = record_id,
                                  columnas restantes = probabilidad media de GM por ROI
```

**Importante:** el pipeline de `Thesis/Code` NO usa imágenes NIfTI ni outputs de fMRIPrep
directamente. Asume que la conectividad funcional **ya fue calculada** y está en `.mat`,
y que la morfometría **ya fue extraída** y está en el CSV.
El pipeline de los notebooks `TPFinal` (01, 02, 03) es la etapa anterior que genera estos archivos.

---

## Pasos para usar el pipeline

### Paso 1 — Verificar que los datos estén listos

Confirmá que `Thesis/Data/` tiene el Excel, la carpeta `fc_mats/` con los `.mat` y el CSV de T1w.
Sin estos tres archivos nada corre.

### Paso 2 — Configurar rutas y parámetros

Abrí `src/config.py` y ajustá `Paths` con las rutas absolutas a tus datos y la carpeta de outputs.
`ExperimentConfig` se puede dejar con los valores por defecto para la primera corrida.

### Paso 3 — Análisis descriptivo (opcional pero recomendado)

```
data_analysis.ipynb
```
Genera tablas y figuras de la cohorte. Verificá que los conteos de sujetos sean los esperados
y que no haya columnas faltantes antes de entrenar.

### Paso 4 — Pipeline principal

```
main.ipynb  (celdas en orden)
```

Lo que hace internamente, en secuencia:
1. Carga y cruza las tres fuentes de datos → cohorte final
2. Genera y guarda los splits (holdout + K-Fold) en `Outputs/splits/`
3. Corre Optuna para el VAE (60 trials por defecto, ~varias horas) → guarda mejor configuración
4. Entrena el VAE final con los mejores hiperparámetros → guarda pesos en `Outputs/vae/`
5. Extrae embeddings μ → los guarda en `Outputs/embeddings/`
6. Entrena XGBoost (con Optuna, 100 trials) → guarda modelo en `Outputs/xgb/`
7. Evalúa en el test set → imprime MAE, RMSE, R², Pearson y genera figuras

Si `reuse_artifacts=True` en `ExperimentConfig` (default), los pasos con outputs ya guardados
se saltean. Útil para retomar si el kernel se interrumpe.

### Paso 5 — Experimentos de comparación

Correr en cualquier orden según lo que necesites para la tesis:

```
optuna_raw_fc.ipynb       → baseline: FC crudo sin VAE
model_comparison.ipynb    → Ridge vs SVR vs XGBoost
cn_only_optuna.ipynb      → modelo solo en controles sanos
demographics_analysis.ipynb → efecto de sitio y educación en el BAG
```

### Paso 6 — Animaciones (opcional)

Desde la carpeta `Animations/`:
```bash
bash render_all.sh
```
O escena por escena:
```bash
manim -pql scene_01_fc.py
```
Requiere Manim instalado y haber corrido `pick_best_recon.py` para los paneles de reconstrucción.

---

## Qué obtenés al final

| Output | Dónde se guarda | Para qué sirve |
|---|---|---|
| Splits reproducibles | `Outputs/splits/splits_seed42_test0.1.json` | Garantizan que todos los experimentos usan la misma partición |
| Hiperparámetros óptimos del VAE | `Outputs/vae_optuna/vae_optuna_age_best.json` | Configuración que minimiza MAE de brain age |
| Pesos del VAE final | `Outputs/vae/vae_final_trainval_optuna/vae.weights.h5` | Modelo listo para inferencia |
| Embeddings Z de todos los sujetos | `Outputs/embeddings/Z_final.npy` + `.json` | Representación comprimida de la FC para análisis downstream |
| Hiperparámetros óptimos de XGBoost | `Outputs/xgb/xgb_best_params.json` | Configuración del predictor de edad |
| Predicciones de edad + BAG | `Outputs/predictions/test_predictions.csv` | `record_id`, `age`, `pred_age`, `BAG` por sujeto |
| Métricas finales | `Outputs/metrics/test_metrics.json` | MAE, RMSE, R², Pearson en el test set |
| Figuras | `Outputs/figures/` | Scatter edad real vs predicha, distribución BAG por diagnóstico, importancia de features |
| Paneles de reconstrucción FC | `Outputs/figures/vae_recons/` | Para las animaciones Manim |
