#!/usr/bin/env python3
"""Genera PDF de revisión crítica de resultados nb09."""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figs"
OUT = ROOT / "docs" / "09_coma_revision_resultados.pdf"


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=22, leading=26,
                                alignment=TA_CENTER, spaceAfter=14, textColor=colors.HexColor("#1a1a2e")),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=12, leading=16,
                                   alignment=TA_CENTER, textColor=colors.HexColor("#444")),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=16, leading=20,
                              spaceBefore=10, spaceAfter=8, textColor=colors.HexColor("#2166ac")),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, leading=16,
                              spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#542788")),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=10.5, leading=14,
                               alignment=TA_JUSTIFY, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"], fontSize=10.5, leading=14,
                                   leftIndent=14, spaceAfter=4),
        "crit": ParagraphStyle("crit", parent=base["Normal"], fontSize=10.5, leading=14,
                                 alignment=TA_JUSTIFY, spaceAfter=6,
                                 textColor=colors.HexColor("#8b0000")),
        "caption": ParagraphStyle("caption", parent=base["Normal"], fontSize=9, leading=12,
                                   alignment=TA_CENTER, textColor=colors.HexColor("#555")),
        "small": ParagraphStyle("small", parent=base["Normal"], fontSize=9, leading=12),
    }


def tbl(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#ccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    t.setStyle(TableStyle(style))
    return t


def fig_page(story, path: Path, caption: str, styles, width=17 * cm):
    if not path.exists():
        story.append(Paragraph(f"[Figura no encontrada: {path.name}]", styles["caption"]))
        return
    story.append(Spacer(1, 0.3 * cm))
    story.append(Image(str(path), width=width, height=width * 0.55))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(caption, styles["caption"]))
    story.append(PageBreak())


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    S = build_styles()
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    )
    story = []

    # --- Portada ---
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Notebook 09 — Coma / inflamación", S["title"]))
    story.append(Paragraph("Revisión crítica de resultados", S["title"]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(
        "Transferencia de edad cerebral (Z + TOPO desde ReDLaT) · Mundo pequeño · Criticalidad completa",
        S["subtitle"],
    ))
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph(
        "Cohorte coma: n = 42 (CTRL 19, ANOX 9, TRAU 14) · ReDLaT: n = 1245 · "
        "Parámetros robustos: N_PER_DX = 50, N_NULL = 50, N_AVALANCHAS = 50 000",
        S["subtitle"],
    ))
    story.append(PageBreak())

    # --- 1. Diseño ---
    story.append(Paragraph("1. Diseño experimental y alcance", S["h1"]))
    story.append(Paragraph(
        "El notebook 09 aplica el mismo β-VAE y métricas TOPO (6 variables, umbral Fisher-z = 0.20) "
        "entrenados/calibrados en data-iipsi (CN, AD, FTD) a una cohorte externa de FC AAL-116 en coma "
        "(controles, anoxia, trauma). No hay edad cronológica en coma: la edad cerebral es "
        "<b>predicha</b> con Ridge(Z + TOPO → edad) entrenado solo en CN de ReDLaT (MAE test CN = 6.72 años). "
        "La comparación con patología neurodegenerativa es exploratoria y transversal.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Fortaleza:</b> mismo espacio latente Z y mismo modelo normativo para ambas cohortes, "
        "permitiendo comparar distribuciones en un eje común.",
        S["bullet"],
    ))
    story.append(Paragraph(
        "<b>Debilidad estructural:</b> diferencias de adquisición, preprocesado FC, ausencia de covariables "
        "(edad, sedación, tiempo post-lesión) y n pequeño (especialmente ANOX n = 9) limitan inferencia causal.",
        S["crit"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    # --- 2. Edad cerebral ---
    story.append(Paragraph("2. Edad cerebral transferida (Ridge Z + TOPO)", S["h1"]))
    story.append(tbl([
        ["Grupo", "n", "Edad predicha (media)", "Interpretación"],
        ["CTRL (coma)", "19", "57.7 años", "Referencia local coma"],
        ["TRAU", "14", "57.4 años", "≈ CTRL coma"],
        ["ANOX", "9", "62.1 años", "≈ +4.4 años vs CTRL coma"],
        ["CN (ReDLaT)", "374", "—", "Referencia normativa entrenamiento"],
        ["AD / FTD (ReDLaT)", "—", "—", "Ver fig. comparativa"],
    ], col_widths=[3.2 * cm, 1.2 * cm, 3.5 * cm, 7.5 * cm]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<b>Lectura:</b> ANOX muestra la edad predicha más alta dentro de coma. TRAU y CTRL son casi "
        "indistinguibles en media. En el boxplot conjunto (fig. 09), las distribuciones de coma se "
        "superponen parcialmente con CN de ReDLaT pero ANOX queda desplazada hacia valores más altos, "
        "en dirección compatible con un envejecimiento cerebral acelerado relativo — no demostrable "
        "sin edad real ni ajuste por confusores.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Crítica:</b> el BAG en coma se define como desvío vs media CN ReDLaT, no vs edad cronológica. "
        "Sesgos de dominio (VAE entrenado en envejecimiento sano latinoamericano) pueden comprimir o "
        "expandir predicciones en coma. Un MAE de 6.7 años en CN no garantiza calibración en lesión aguda.",
        S["crit"],
    ))
    story.append(PageBreak())

    # --- 3. Mundo pequeño ---
    story.append(Paragraph("3. Mundo pequeño (σ, ω) — N_NULL = 50", S["h1"]))
    story.append(tbl([
        ["Grupo", "σ_WS", "σ_ER", "ω_WS", "¿SW clásico (σ>1)?"],
        ["CTRL", "1.630", "1.874", "0.110", "Sí (WS y ER)"],
        ["ANOX", "1.381", "1.812", "0.074", "Sí"],
        ["TRAU", "1.584", "2.044", "0.140", "Sí"],
    ], col_widths=[2.5 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 4.5 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(tbl([
        ["Métrica", "Kruskal-Wallis p", "Conclusión"],
        ["σ_WS", "0.126", "No significativo"],
        ["σ_ER", "0.252", "No significativo"],
        ["ω_WS", "0.066", "Tendencia (α≈0.05), no FDR"],
    ], col_widths=[3 * cm, 3 * cm, 9.5 * cm]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<b>Lectura:</b> los tres grupos mantienen σ > 1 frente a redes aleatorias (WS y ER): topología "
        "tipo mundo pequeño conservada. ANOX presenta el σ_WS más bajo (1.38 vs 1.63 CTRL), sugiriendo "
        "menor balance clustering/path-length relativo, pero sin significación estadística con n = 42.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Comparación nb08 (ReDLaT):</b> CN σ_WS ≈ 1.19 vs coma CTRL ≈ 1.63. Coma parece más "
        "'small-world' en términos relativos, posiblemente por diferencias de densidad/umbral o cohorte. "
        "No es comparable directamente sin armonizar densidad.",
        S["crit"],
    ))
    story.append(PageBreak())

    # --- 4. Lambda1 ---
    story.append(Paragraph("4. Criticalidad — λ₁ y rango dinámico Δ", S["h1"]))
    story.append(tbl([
        ["Grupo", "λ₁ individual (media±DE)", "λ₁ matriz promedio", "λ₁/λ₁_ER", "Δ (dB)"],
        ["CTRL", "17.37 ± 1.93", "14.99", "1.21", "13.44"],
        ["ANOX", "21.98 ± 3.15", "19.53", "1.57", "18.66"],
        ["TRAU", "18.38 ± 2.43", "15.93", "1.28", "13.44"],
    ], col_widths=[2.2 * cm, 3.5 * cm, 3 * cm, 2.3 * cm, 2.5 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(tbl([
        ["Contraste", "p (Mann-Whitney)", "p_FDR"],
        ["CTRL vs ANOX", "0.0002", "0.0007"],
        ["ANOX vs TRAU", "0.0089", "0.0119"],
        ["CTRL vs TRAU", "0.283", "0.283"],
    ], col_widths=[4 * cm, 3.5 * cm, 3.5 * cm]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<b>Hallazgo principal:</b> ANOX se separa claramente con λ₁ más alto (Kruskal-Wallis global "
        "p = 0.0007). Esto indica mayor conectividad efectiva al 10% de aristas más fuertes — compatible "
        "con integración aumentada o menor fragmentación local en anoxia respecto a controles de coma.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Rango dinámico Δ:</b> ANOX también muestra Δ = 18.7 dB vs ~13.4 dB en CTRL/TRAU, sugiriendo mayor "
        "capacidad de respuesta no lineal del promedio grupal. TRAU replica el Δ de CTRL.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Comparación nb08:</b> CN ReDLaT λ₁ ≈ 20.3 (PROP 10%) — coma ANOX (22.0) se acerca más a "
        "patrones de conectividad densa que CTRL coma (17.4). No implica 'criticalidad' en sentido "
        "de punto crítico dinámico sin más evidencia (avalanchas, σ_br).",
        S["crit"],
    ))
    story.append(PageBreak())

    # --- 5. Avalanchas ---
    story.append(Paragraph("5. Avalanchas (N = 50 000, p = 1/λ₁_CTRL = 0.0667)", S["h1"]))
    story.append(tbl([
        ["Grupo", "⟨s⟩", "α(s) power-law", "¿Cerca de α = 2?"],
        ["CTRL", "3.70", "1.932", "Moderado"],
        ["ANOX", "5.24", "1.762", "Alejado (más plano)"],
        ["TRAU", "4.35", "1.851", "Moderado"],
    ], col_widths=[2.5 * cm, 2 * cm, 3 * cm, 4 * cm]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<b>Lectura:</b> ANOX produce avalanchas más grandes (⟨s⟩ = 5.24 vs 3.70). Ningún grupo muestra "
        "α(s) ≈ 2 típico de criticalidad auto-organizada estricta (nb08 CN: α ≈ 2.01). Las distribuciones "
        "son más cortas que en ReDLaT — esperable en matrices promedio de n pequeño y FC de lesión.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Crítica metodológica:</b> p_crit se fija con λ₁ de CTRL coma, no de cada grupo. Las "
        "avalanchas en ANOX/TRAU usan la misma p, lo que puede subestimar/sobreestimar propagación "
        "relativa. Ideal: sensibilidad con p = 1/λ₁_grupo.",
        S["crit"],
    ))
    story.append(PageBreak())

    # --- 6. Percolación y sigma_br ---
    story.append(Paragraph("6. Percolación y razón de ramificación σ_br", S["h1"]))
    story.append(tbl([
        ["Grupo", "χ peak (% conexiones)", "Interpretación"],
        ["CTRL", "2.2%", "Transición temprana"],
        ["ANOX", "3.0%", "Transición más tardía"],
        ["TRAU", "1.6%", "Transición más temprana"],
    ], col_widths=[2.5 * cm, 4 * cm, 8 * cm]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(tbl([
        ["Grupo", "σ_br", "IC95%"],
        ["CTRL", "1.068", "[1.066, 1.070]"],
        ["ANOX", "1.045", "[1.044, 1.046]"],
        ["TRAU", "1.054", "[1.053, 1.055]"],
    ], col_widths=[2.5 * cm, 3 * cm, 5 * cm]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<b>Percolación:</b> curvas S₁/S₂/χ (fig. 09) muestran que ANOX mantiene componente gigante "
        "mayor a umbral bajo — coherente con λ₁ elevado. Picos de χ desplazados sugieren distinta "
        "robustez topológica al podar conexiones débiles.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>σ_br:</b> todos los grupos tienen σ_br ≈ 1.05–1.07 (cerca de criticalidad de avalanchas "
        "subcríticas/ligeramente supercríticas). Contraste nb08: σ ≈ 0.63 — discrepancia importante "
        "por definición del estimador (nb09 omite el último término de padres en la fórmula corregida "
        "de nb08). <b>Recomendación:</b> unificar fórmula antes de interpretar σ_br entre notebooks.",
        S["crit"],
    ))
    story.append(PageBreak())

    # --- 7. Sensibilidad ---
    story.append(Paragraph("7. Sensibilidad λ₁ (2%–20%, cohorte completa)", S["h1"]))
    story.append(tbl([
        ["PROP", "CTRL", "ANOX", "TRAU"],
        ["10%", "17.37", "21.98", "18.38"],
        ["6%", "12.65", "16.18", "12.64"],
        ["20%", "28.47", "32.84", "30.67"],
    ], col_widths=[2 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "Las curvas (fig. sensibilidad) son monótonas crecientes y <b>paralelas</b>: el orden "
        "ANOX > TRAU ≈ CTRL se mantiene en todo el rango 2–20%. El hallazgo de λ₁ en ANOX no es "
        "artefacto del umbral PROP = 10% — es robusto a la elección de densidad.",
        S["body"],
    ))
    story.append(PageBreak())

    # --- 8. Síntesis crítica ---
    story.append(Paragraph("8. Síntesis crítica integrada", S["h1"]))
    story.append(Paragraph("<b>Consistente y robusto</b>", S["h2"]))
    for t in [
        "Separación ANOX vs CTRL en λ₁ (p_FDR = 0.0007) replicada en sensibilidad 2–20%.",
        "Mundo pequeño preservado (σ > 1) en los tres grupos; sin evidencia de 'desorganización' global.",
        "ANOX: mayor λ₁, mayor Δ, avalanchas más grandes — patrón coherente hacia mayor integración/redundancia.",
    ]:
        story.append(Paragraph(f"• {t}", S["bullet"]))

    story.append(Paragraph("<b>Débil o no concluyente</b>", S["h2"]))
    for t in [
        "Edad cerebral transferida: ANOX +4 años vs CTRL coma sin test reportado ni edad real.",
        "Mundo pequeño grupal: p > 0.05; ω_WS tendencia (p = 0.066) no confirma diferencias.",
        "Power-law α(s) lejos de 2 en todos los grupos — no soporta criticalidad SOC clásica.",
        "σ_br no comparable con nb08 por diferencia de implementación.",
        "n = 9 en ANOX: intervalos amplios, alto riesgo de outlier-driven effects.",
    ]:
        story.append(Paragraph(f"• {t}", S["bullet"]))

    story.append(Paragraph("<b>Interpretación clínica cautelosa</b>", S["h2"]))
    story.append(Paragraph(
        "Los datos favorecen la hipótesis de que <b>anoxia</b> (no trauma leve en esta muestra) "
        "asocia un perfil de FC con conectividad principal más fuerte y dinámica de avalanchas "
        "amplificada, posiblemente reflejando reorganización post-insulto o estado de red más "
        "integrada/sincronizada. TRAU se comporta como CTRL en casi todas las métricas. "
        "La edad predicha elevada en ANOX es sugerente pero requiere validación con covariables "
        "y cohorte expandida.",
        S["body"],
    ))
    story.append(PageBreak())

    # --- 9. Conclusiones ---
    story.append(Paragraph("9. Conclusiones para el informe final", S["h1"]))
    for i, t in enumerate([
        "El pipeline de transferencia Z + TOPO es ejecutable y produce comparaciones exploratorias "
        "entre coma y ReDLaT; MAE CN = 6.72 años acota error en referencia normativa.",
        "El hallazgo estadísticamente más sólido es λ₁ elevado en ANOX vs CTRL (FDR < 0.001), "
        "robusto al umbral proporcional.",
        "No hay evidencia de pérdida de mundo pequeño en coma; las diferencias entre etiologías "
        "son sutiles en σ/ω.",
        "La criticalidad tipo SOC (α ≈ 2, σ_br ≈ 1 con misma fórmula que nb08) no se confirma; "
        "interpretar 'cercanía al punto crítico' requiere cautela.",
        "Priorizar: ampliar ANOX, unificar σ_br, tests formales edad predicha coma vs ReDLaT, "
        "y sensibilidad de avalanchas con p por grupo.",
    ], 1):
        story.append(Paragraph(f"{i}. {t}", S["body"]))

    story.append(PageBreak())

    # --- Figuras ---
    story.append(Paragraph("Anexo — Figuras del notebook 09", S["h1"]))
    figures = [
        ("09_predicted_age_coma_vs_redlat.png",
         "Fig. A1. Edad predicha y BAG/desvío vs CN ReDLaT. Coma (CTRL, ANOX, TRAU) vs CN, AD, FTD."),
        ("09_sw_by_diagnosis.png",
         "Fig. A2. Mundo pequeño: σ_WS, σ_ER y ω_WS por diagnóstico (n = 42, N_NULL = 50)."),
        ("09_crit_lambda1.png",
         "Fig. A3. λ₁ individual, λ₁ normalizado (ER) y rango dinámico Δ vs λ₁ promedio."),
        ("09_crit_avalanchas.png",
         "Fig. A4. Distribución de tamaños de avalancha, α(s) y relación s–T (N = 50 000)."),
        ("09_crit_percolacion.png",
         "Fig. A5. Curvas de percolación S₁, S₂ y χ (42 sujetos × 32 umbrales)."),
        ("09_crit_branching.png",
         "Fig. A6. σ_br con IC95% y relación λ₁–σ_br."),
        ("09_crit_sensibilidad.png",
         "Fig. A7. Sensibilidad de λ₁ medio al umbral proporcional (2%–20%)."),
    ]
    for fname, cap in figures:
        fig_page(story, FIG / fname, cap, S)

    doc.build(story)
    print(f"PDF generado: {OUT}")


if __name__ == "__main__":
    main()
