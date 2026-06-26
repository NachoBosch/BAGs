# BAG, topología y criticalidad en conectividad funcional

El repositorio implementa un pipeline multimodal para estimar la **brecha de edad cerebral** (*Brain Age Gap*, **BAG**) a partir de conectividad funcional en reposo, morfometría estructural y variables sociodemográficas, e integra análisis de **topología de grafos**, **mundo pequeño** y **criticalidad espectral** en cohortes latinoamericanas y de coma.

---

## ¿Qué es el BAG?

El **BAG** es la diferencia entre la edad biológica estimada a partir de neuroimagen y la edad cronológica:

$$\text{BAG} = \hat{\text{edad}}_{\text{cerebral}} - \text{edad}_{\text{cronológica}}$$

Tras una corrección de sesgo en controles sanos, un **BAG positivo** indica un cerebro que “aparenta” ser más viejo de lo esperado. En este proyecto el BAG se deriva de modelos entrenados principalmente en **controles cognitivamente normales (CN)** y se relaciona con métricas topológicas de la red funcional (eficiencia, clustering, hubs frontoparietales).

---

## Cohortes

### ReDLaT / BrainLat

- Cohorte multicéntrica latinoamericana con fMRI en reposo, T1w (VBM AAL-116) y metadatos clínicos.
- Grupos principales: **CN**, **AD**, **FTD**.
- Subconjunto de análisis principal: **n ≈ 1245** sujetos con FC, T1w y variables completas.
- Datos **no incluidos** en el repositorio (ver [Datos](#datos)).

### Coma / inflamación

- FC pre-parcelada AAL-116 (`.mat` 116×116) en carpetas `controls`, `anoxia`, `traumatic`.
- Sin T1w ni edad cronológica en la cohorte coma.
- Pipeline de transferencia en `09_coma.ipynb`: modelo **Z + TOPO** calibrado en CN de `data-iipsi`, aplicado a coma y comparado con CN / AD / FTD.

---

## Pipeline principal (ReDLaT)

```
FC (6670, Fisher-z) ──► β-VAE ──► Z (64)
        │                              │
        ├──► TOPO (6 métricas grafos) ─┤
        │                              ├──► Ridge / XGB / SVR ──► edad cerebral
T1w (116 ROIs) ──────────────────────┤                              │
sexo, educación, sitio ──────────────┘                              ▼
                                                              BAG (solo CN train)
```

| Bloque | Descripción |
|--------|-------------|
| **Z** | Embeddings del encoder del β-VAE sobre el triángulo superior de la matriz FC (Fisher-z). |
| **TOPO** | 6 métricas: $E_{\text{loc}}$, $E_{\text{glob}}$, clustering, y tres frontoparietales ($E_{\text{loc}}^{\text{FP}}$, betweenness, grado). Umbral Fisher-z = 0.20. |
| **T1w** | Probabilidad media de materia gris por ROI (AAL-116). |
| **Split** | Holdout 90/10 estratificado por **sexo × diagnóstico**. |
| **BAG** | Ridge híbrido entrenado en CN; corrección de sesgo LOSO por sitio. |

**Resultado destacado (holdout):** Ridge con Z + T1w + sexo + TOPO + educación + sitio → **MAE ≈ 5.06 años** (test).

---

## Notebooks

| Notebook | Rol |
|----------|-----|
| `01_exploracion_datos.ipynb` | Exploración demográfica ReDLaT |
| `02_conectividad_funcional_y_grafos.ipynb` | Fisher-z, métricas TOPO, `outputs/graph_metrics_table.csv` |
| `03_BAG_estadistica_ML.ipynb` | BAG + estadística (exploración inicial) |
| `04_mundo_pequeño.ipynb` | Índices σ y ω (prototipo) |
| `05_replicacion_thesis.ipynb` | Réplica pipeline tesis |
| `06_pruebas_finales.ipynb` | Validación final |
| `07_criticalidad_fixed.ipynb` | λ₁, avalanchas, percolación (prototipo) |
| **`08_exploracion_data_iipsi_thesis.ipynb`** | **Pipeline integrado principal** (VAE, modelos, BAG, SW, criticalidad) |
| **`09_coma.ipynb`** | Cohorte coma: TOPO, Z transferido, predicción Z+TOPO vs RedLaT, SW, criticalidad |

Código reutilizable en `Thesis/Code/src/` (`data_io`, `vae_train`, `cohort`, `splits`, `coma_data_io`, etc.).

---

## Análisis complementarios

- **Mundo pequeño:** índices $\sigma$ y $\omega$ con referencias WS y ER (BCT + NetworkX).
- **Criticalidad:** autovalor máximo $\lambda_1$, rango dinámico, avalanchas, percolación ($\chi$), razón de ramificación $\sigma_{\text{br}}$.
- **Asociaciones BAG–topología:** comparación BAG+ vs BAG−; correlaciones con MMSE cuando hay datos clínicos.

Informe completo: [`informe_final.tex`](informe_final.tex).

---

## Instalación

```bash
git clone <url-del-repo>
cd TPFinal-NeuroComp
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r Thesis/requirements.txt
pip install bctpy statsmodels pingouin powerlaw joblib seaborn
```

Requisitos principales: Python 3.10+, TensorFlow 2.x, scikit-learn, XGBoost, bctpy, NetworkX.

---

## Datos

Los datos de neuroimagen **no se versionan** (`.gitignore`: `dataset/`, `brainlat/`, etc.). Para ReDLaT se espera la estructura bajo `data-iipsi/data/`:

```
data-iipsi/data/
├── matrices-redlat/
│   ├── datos-redlat.xlsx
│   └── matrices-redlat/          # .mat FC por sujeto
└── Redlat_VGM_AAL_.csv           # T1w por ROI
```

Para coma, configurar la ruta a los `.mat` en la primera celda de `09_coma.ipynb` (p. ej. `.../databases/fc/inflamacion/{controls,anoxia,traumatic}/`).

---

## Salidas

| Carpeta | Contenido |
|---------|-----------|
| `outputs/graph_metrics_table.csv` | TOPO por sujeto (nb02) |
| `outputs/nb08_thesis/` | VAE, métricas SW, criticalidad, resultados modelos |
| `outputs/nb09_coma/` | TOPO coma, predicciones transferidas, SW, criticalidad |
| `figs/` | Figuras para informe y notebooks |

---

## Estructura del repositorio

```
├── 01–09_*.ipynb          # Notebooks de análisis
├── informe_final.tex      # Informe del TP
├── Thesis/Code/src/       # Módulos Python del pipeline
├── scripts/               # Utilidades (p. ej. generar nb09)
├── figs/                  # Figuras
└── outputs/               # Resultados (generados al ejecutar)
```

---

## Objetivo central del estudio (ReDLaT)

Evaluar si sujetos con **BAG positivo** muestran diferencias en eficiencia local y centralidad de hubs frontoparietales respecto de BAG negativo, mediante grafos sobre FC en reposo (AAL-116), controlando edad, educación y sitio.

---

## Autor

**Ignacio Bosch** — Doctorado en Neurociencias  
Contacto: nachobosch5@gmail.com

---

## Referencias sugeridas

- Prado et al. (2023). The BrainLat project. *Scientific Data*.
- Rubinov & Sporns (2010). Complex network measures of brain connectivity. *NeuroImage*.
- Watts & Strogatz (1998). Small-world networks. *Nature*.

---

## Licencia

Uso académico. Consultar con el autor antes de redistribuir datos o resultados derivados de cohortes ReDLaT/BrainLat.
