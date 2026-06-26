# Especificaciones de archivos de neuroimagen — Pipeline BrainLat / ReDLaT

## 1. Modalidades de imagen disponibles en el dataset

### 1.1 T1w (T1-weighted, imagen estructural anatómica)

- **Qué es:** Imagen estática tridimensional del cerebro. Captura la anatomía: la materia gris aparece en gris medio, la materia blanca en blanco brillante y el LCR en negro.
- **Dimensiones:** 3D, típicamente 1×1×1 mm de resolución, un único volumen (~256×256×176 vóxeles).
- **Extensión en BIDS:** `sub-XXX_T1w.nii.gz`
- **Para qué se usa en este proyecto:** Segmentación de tejidos (GM, WM, LCR), morfometría basada en vóxeles (VBM), referencia anatómica para el registro al espacio MNI.
- **Estado en el dataset:** Disponible para un subconjunto de sujetos en `brainlat/` y `brainlat2/`.

---

### 1.2 BOLD (Blood-Oxygen-Level-Dependent, imagen funcional de resting-state)

- **Qué es:** Imagen dinámica cuatridimensional (3D espacio + tiempo). Mide indirectamente la actividad neuronal a través de los cambios en la oxigenación de la sangre. En resting-state el sujeto está en reposo con los ojos cerrados.
- **Dimensiones:** 4D. Cada volumen 3D tiene baja resolución (~3×3×3 mm) pero se adquieren cientos de volúmenes consecutivos. Un archivo típico: 64×64×36 vóxeles × 200 TR = ~400 MB.
- **Extensión en BIDS:** `sub-XXX_task-rest_bold.nii.gz`
- **Sidecar JSON asociado:** `sub-XXX_task-rest_bold.json` — contiene metadatos de adquisición: TR, tiempo de eco, dirección de codificación de fase, etc.
- **Para qué se usa en este proyecto:** Extracción de series temporales por ROI, cálculo de matrices de conectividad funcional, métricas de teoría de grafos.
- **Estado en el dataset:** **FALTANTE.** Solo existen los JSON sidecars en `brainlat/`. Los archivos `.nii.gz` BOLD no fueron descargados o no están en el equipo local.

---

### 1.3 FLAIR (Fluid-Attenuated Inversion Recovery, imagen estructural)

- **Qué es:** Imagen estructural similar a T2 pero con supresión del líquido cefalorraquídeo. Resalta lesiones de materia blanca, placas de desmielinización y patología vascular.
- **Dimensiones:** 3D, resolución milimétrica, un único volumen.
- **Extensión en BIDS:** `sub-XXX_FLAIR.nii` o `sub-XXX_FLAIR.nii.gz`
- **Para qué se usa en este proyecto:** No está contemplada en el pipeline actual. Podría usarse como covariable de carga de lesiones en análisis secundarios.
- **Estado en el dataset:** Disponible para varios sujetos en `brainlat/` y `brainlat2/`.

---

### 1.4 DWI (Diffusion-Weighted Imaging, imagen de difusión)

- **Qué es:** Mide la difusión de moléculas de agua en el tejido cerebral. Permite reconstruir tractos de sustancia blanca (tractografía) y calcular métricas como FA (anisotropía fraccional) y MD (difusividad media).
- **Archivos asociados:** `.nii.gz` (imagen), `.bval` (valores de gradiente b), `.bvec` (direcciones de gradiente).
- **Extensión en BIDS:** `sub-XXX_dwi.nii.gz`, `sub-XXX_dwi.bval`, `sub-XXX_dwi.bvec`
- **Para qué se usa en este proyecto:** No está contemplada en el pipeline actual. Podría incorporarse para conectividad estructural en extensiones futuras.
- **Estado en el dataset:** Disponible para un subconjunto de sujetos en `brainlat/` y `brainlat2/`.

---

## 2. Outputs de fMRIPrep requeridos por el Notebook 02

fMRIPrep toma como entrada los datos BIDS crudos (T1w + BOLD) y produce archivos preprocesados por sujeto. El Notebook 02 requiere específicamente dos archivos por sujeto:

### 2.1 BOLD preprocesado en espacio MNI

```
<fmriprep_dir>/sub-XXX/func/
    sub-XXX_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
```

- **Qué contiene:** La serie temporal BOLD alineada al espacio estándar MNI152 a resolución 2 mm, con corrección de movimiento y distorsiones aplicadas.
- **Por qué se necesita:** Nilearn aplica el atlas AAL (también en MNI) directamente sobre este volumen para extraer la señal por ROI. Sin alineación al mismo espacio, la superposición atlas–BOLD es inválida.

### 2.2 Archivo de confounders

```
<fmriprep_dir>/sub-XXX/func/
    sub-XXX_task-rest_desc-confounds_timeseries.tsv
```

- **Qué contiene:** Tabla con un regresor por columna y un valor por fila (un fila = un TR). Incluye: 6 parámetros de movimiento (trans_x/y/z, rot_x/y/z), sus derivadas de primer orden, señal media de materia blanca (white_matter), señal media de LCR (csf), framewise displacement (FD), componentes aCompCor, y más.
- **Por qué se necesita:** Se regresa contra el BOLD para eliminar varianza debida a movimiento y ruido fisiológico antes de calcular conectividad. Equivale a incluir regresores de no interés en el GLM de FSL FEAT.

### 2.3 Mapa de probabilidad de materia gris (para VBM)

```
<fmriprep_dir>/sub-XXX/anat/
    sub-XXX_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz
```

- **Qué contiene:** Imagen 3D donde cada vóxel tiene un valor entre 0 y 1 que representa la probabilidad de ser materia gris (output de la segmentación de tejidos realizada por fMRIPrep via FastSurfer/FreeSurfer).
- **Por qué se necesita:** El Notebook 02 (sección 10) extrae el volumen total de GM y la probabilidad media de GM por ROI del atlas AAL como features para el modelo de BAG.

---

## 3. Estructura de carpetas esperada por el Notebook 02

```
<FMRIPREP_DIR>/
├── sub-COA00001/
│   ├── func/
│   │   ├── sub-COA00001_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
│   │   └── sub-COA00001_task-rest_desc-confounds_timeseries.tsv
│   └── anat/
│       └── sub-COA00001_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz
├── sub-COA00002/
│   └── ...
└── ...
```

---

## 4. Resumen del estado actual del dataset local

| Modalidad | Necesaria para | Archivos presentes | Archivos faltantes |
|---|---|---|---|
| T1w raw | Entrada a fMRIPrep | Sí, parcial (`brainlat/`, `brainlat2/`) | Sujetos sin T1w |
| BOLD raw (`.nii.gz`) | Entrada a fMRIPrep | **NO** | Todos los sujetos |
| BOLD JSON sidecar | Metadata de adquisición | Sí (`brainlat/`) | — |
| BOLD preprocesado fMRIPrep | Notebook 02 (FC) | **NO** | Todos los sujetos |
| Confounds TSV fMRIPrep | Notebook 02 (QC + regresión) | **NO** | Todos los sujetos |
| GM probseg fMRIPrep | Notebook 02 (VBM) | **NO** | Todos los sujetos |
| FLAIR | No contemplada | Sí, parcial | — |
| DWI | No contemplada | Sí, parcial | — |

---

## 5. Opciones para generar los outputs requeridos

### Opción A — Correr fMRIPrep (recomendada, gold standard)

fMRIPrep es un pipeline de preprocesamiento de fMRI escrito en Python (Nipype + ANTs + FreeSurfer + FSL). Se ejecuta típicamente en Linux/HPC o vía Docker/Singularity.

**Requisitos de entrada:**
- Dataset en formato BIDS con T1w y BOLD raw por sujeto
- 16–32 GB RAM por sujeto, ~2–4 horas de procesamiento por sujeto

**Comando básico:**
```bash
fmriprep <bids_dir> <output_dir> participant \
  --participant-label sub-COA00001 \
  --output-spaces MNI152NLin2009cAsym:res-2 \
  --fs-no-reconall \
  --nthreads 8 --mem-gb 16
```

### Opción B — Pipeline en Python con nilearn + nipype + ANTsPy

Replicable completamente en Python si se tienen los BOLD raw. Usa las mismas herramientas subyacentes que fMRIPrep (ANTs para registro, FSL para corrección de movimiento). Menos robusto que fMRIPrep pero funciona en cualquier entorno con las dependencias instaladas.

**Pasos equivalentes:**
1. Brain extraction: `ANTsPy` o `nipype.fsl.BET`
2. Corrección de movimiento: `nipype.fsl.MCFLIRT` (conocido del trabajo con FSL)
3. Registro T1w → MNI: `ANTsPy.registration`
4. Registro BOLD → T1w → MNI: `nipype.fsl.FLIRT` (lineal) + `FNIRT` (no-lineal)
5. Estimación de confounders: `nilearn.signal.clean` con parámetros de movimiento extraídos de MCFLIRT
6. Segmentación GM: `ANTsPy` o `nipype.fsl.FAST`

### Opción C — Usar datos ya preprocesados del repositorio BrainLat

Si el dataset BrainLat tiene derivatives disponibles en el repositorio original (OpenNeuro u otro), se pueden descargar directamente los outputs de fMRIPrep sin necesidad de correrlo localmente.
