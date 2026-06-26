# Notebooks

Cada notebook corresponde a una parte del pipeline o de los experimentos reportados en la tesis. Los outputs de las celdas se conservan para cotejar con las figuras y tablas del documento.

| Notebook | Rol | Capítulo |
|----------|-----|----------|
| **main.ipynb** | Pipeline principal: cohorte, β-VAE (Optuna), XGBoost, evaluación final y ablaciones base | 5 (Metodología), 6 (Resultados) |
| **data_analysis.ipynb** | Análisis descriptivo de la cohorte; figuras y tablas de datos | 4 (Datos y cohorte) |
| **experiments.ipynb** | Experimentos exploratorios (baseline, decisiones de diseño) | 6 |
| **optuna_raw_fc.ipynb** | Optimización de XGBoost para FC crudo + T1w; comparación con VAE + T1w | 6 |
| **model_comparison.ipynb** | Comparación Ridge vs SVR vs XGBoost sobre mismas features (VAE μ + T1w) | 6 |
| **demographics_analysis.ipynb** | Impacto de variables demográficas en brain age gap; correlaciones y ablaciones | 6 |
| **demographics_optuna.ipynb** | Re-optimización de XGBoost con Optuna para configuraciones con demográficos | 6 |
| **cn_only_optuna.ipynb** | Modelo entrenado solo en controles sanos (CN) con Optuna dedicada | 6 |
| **cn_demographics_experiment.ipynb** | Experimento CN-only incluyendo educación y sitio como features | 6 |
| **regression_trees.ipynb** | Figuras del marco teórico: árboles de regresión (dataset público Advertising) | 2 (Marco teórico) |

Ejecución: desde `Code/` o desde `Code/notebooks/`; los notebooks resuelven la raíz del proyecto para importar `src` y acceder a `Data/` y `Outputs/` vía la raíz del repositorio.
