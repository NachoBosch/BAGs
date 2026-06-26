# Prompt para Claude Code — Notebook de Criticalidad en Redes Cerebrales

## Contexto del proyecto

Estoy trabajando en un proyecto de doctorado en Neurociencias que analiza propiedades
topológicas de redes cerebrales funcionales en una cohorte latinoamericana (BrainLat/ReDLaT).
Tengo un notebook previo 02 que ya generó grafos binarizados a partir de
matrices de conectividad funcional (FC) derivadas de fMRI en reposo, usando el atlas AAL de
116 regiones. Hay aproximadamente 1300 sujetos con etiquetas diagnósticas: CN (controles
sanos), AD (Alzheimer) y FTD (demencia frontotemporal).

**Restricción computacional importante:** No simular por sujeto. Todas las simulaciones
estocásticas deben correr sobre una única matriz de adyacencia promedio por grupo
(CN, AD, FTD). Esto reduce el costo de horas a segundos sin perder capacidad comparativa
entre grupos.

---

## Tarea

Creá un Jupyter Notebook llamado `07_criticalidad.ipynb` que analice dónde operan los
grupos CN, AD y FTD en relación al punto crítico de sus redes cerebrales.
El notebook debe construirse paso a paso, con celdas Markdown explicando cada sección
antes del código, comentarios en el código, y visualizaciones claras al final de cada análisis.

---

## Estructura del notebook

### Sección 0 — Setup e imports
- Importar: numpy, pandas, scipy, matplotlib, seaborn, networkx, powerlaw (pip install powerlaw).
- Cargar los grafos binarizados y matrices de correlación continuas desde notebook 02.
- Calcular la matriz de adyacencia promedio por grupo (CN, AD, FTD) promediando las
  matrices de correlación continuas y luego binarizando con el mismo umbral del notebook 02.
- Verificar dimensiones y N por grupo.

---

### Sección 1 — Autovalor máximo (λ₁) y rango dinámico

**Concepto:** λ₁ de la matriz de adyacencia A indica criticalidad. λ₁ ≈ 1 es crítico,
λ₁ < 1 subcrítico, λ₁ > 1 supercrítico.

**Implementar:**
1. Calcular λ₁ para cada sujeto individual: `numpy.linalg.eigvalsh(A)[-1]`.
2. Calcular λ₁ también para las tres matrices promedio de grupo.
3. Comparar distribuciones de λ₁ entre grupos (CN, AD, FTD).

**Curva de rango dinámico (sobre matrices promedio):**
- Simular respuesta F(h) de cada red promedio para estímulos h en rango logarítmico [1e-5, 1e1].
- Rango dinámico: Δ = 10 * log10(h_0.9 / h_0.1).
- Plotear Δ vs λ₁ para los tres grupos, con línea vertical en λ₁ = 1.

**Gráficos:**
- Violinplot de λ₁ por grupo con línea de referencia en λ₁ = 1.
- Scatterplot Δ vs λ₁ para los tres grupos promedio.

---

### Sección 2 — Curvas de avalancha neuronal (sobre matrices promedio)

**Concepto:** En criticalidad, P(s) ~ s^(-α) con α ≈ 1.5. Subcrítico: caída más rápida.
Supercrítico: caída más lenta.

**Implementar sobre las tres matrices promedio (CN, AD, FTD):**
1. Para cada matriz promedio, simular N = 50000 avalanchas:
   - Activar un nodo semilla aleatorio.
   - Cada nodo activo activa vecinos con probabilidad p = 1/λ₁_grupo.
   - Registrar tamaño s y duración T hasta extinción.
   - Vectorizar con numpy. Semilla: `np.random.seed(42)`.
2. Ajustar ley de potencias con el paquete `powerlaw` (MLE de Clauset).
3. Reportar exponente α y xmin por grupo.

**Curvas:**
- Log-log P(s) para CN, AD, FTD superpuestos.
- Log-log P(T) para CN, AD, FTD superpuestos.
- Barplot de α por grupo.
- Scatterplot <s> vs T en log-log (scaling relation).

**Nota en Markdown:** Aclarar que las simulaciones operan sobre redes promedio grupales,
no sobre sujetos individuales. Metodológicamente válido para comparación entre grupos.

---

### Sección 3 — Curva del segundo cluster máximo (susceptibilidad de percolación)

**Concepto:** El pico de S₂ y de la susceptibilidad χ identifica el umbral crítico de
percolación. Este análisis es determinístico (sin simulación estocástica) y se puede
correr sobre todos los sujetos individuales.

**Implementar sobre todos los sujetos:**
1. Para cada sujeto, reconstruir grafos a umbrales T = [0.01, 0.02, ..., 0.40].
2. Para cada umbral calcular con networkx:
   - S₁ = tamaño del componente gigante / 116.
   - S₂ = tamaño del segundo componente / 116.
   - χ = S₂² / 116.
3. Promediar por grupo diagnóstico ± SEM.

**Curvas:**
- Tres paneles: S₁, S₂ y χ vs umbral, con banda de error ±SEM.
- Línea vertical en el umbral donde χ es máximo para cada grupo.

---

### Sección 4 — Branching Ratio (σ)

**Concepto:** σ = activaciones generación t+1 / generación t. σ = 1 es crítico.

**Implementar sobre las tres matrices promedio:**
1. Usar las avalanchas de Sección 2.
2. Para cada avalancha registrar n por generación: n₀=1, n₁, n₂...
3. σ = promedio de (n_{t+1}/n_t) sobre todas las generaciones y avalanchas.
4. IC por bootstrapping (1000 muestras) sobre las 50000 avalanchas.

**Gráficos:**
- Barplot de σ_CN, σ_AD, σ_FTD con IC, línea de referencia en σ = 1.
- Correlación λ₁ vs σ (tres puntos, uno por grupo).

---

### Sección 5 — Análisis estadístico

**Implementar sobre métricas individuales (λ₁, S₂, χ):**
1. Kruskal-Wallis para comparación de tres grupos.
2. Post-hoc Mann-Whitney con corrección FDR (Benjamini-Hochberg): CN-AD, CN-FTD, AD-FTD.
3. Tamaños de efecto rank-biserial r.
4. Tabla formateada de p-valores y effect sizes.

---

### Sección 6 — Sensibilidad al umbral de binarización

1. Repetir cálculo de λ₁ individual para T = 0.02 a 0.20 en pasos de 0.02.
2. Plotear λ₁ promedio por grupo vs umbral con banda ±1 SD.
3. Verificar que la separación entre grupos se mantiene estable.

---

## Requerimientos técnicos

- `np.random.seed(42)` en todas las simulaciones.
- Paleta consistente: CN=azul (#185fa5), AD=rojo (#e24b4a), FTD=naranja (#ef9f27).
- Todas las figuras: título, ejes etiquetados, leyenda.
- Guardar figuras en `figures/03_criticalidad/`.
- Guardar tabla de métricas individuales en `results/metricas_criticalidad.csv` con
  columnas: `subject_id | diagnosis | lambda1 | S2_peak | chi_peak | critical_threshold`.

---

## Notas metodológicas para incluir en Markdown del notebook

- Las simulaciones de avalanchas operan sobre redes promedio por grupo, no por sujeto,
  por restricción computacional (1300 sujetos × 50000 avalanchas = inviable).
- La percolación (Sección 3) sí corre por sujeto porque es determinística y rápida.
- λ₁ (Sección 1) sí corre por sujeto porque es instantáneo.
- σ y α son métricas grupales, no individuales, y deben reportarse como tales.
