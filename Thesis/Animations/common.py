"""
common.py — Paleta, rutas de imágenes y helpers compartidos por todas las escenas.
"""
from pathlib import Path
from manim import *
import numpy as np

# --- Rutas (raíz del repo: padre de Animations/) ---
BASE     = Path(__file__).resolve().parent.parent
IMG_FC   = str(BASE / "Latex/04_marco_teorico/images/fc_example.png")
IMG_RECON= str(BASE / "Outputs/figures/vae_recons/recon_00_idx0.png")

# --- Paleta ---
BG      = "#0D1117"   # fondo oscuro
C_BLUE  = "#4FC3F7"   # FC / inputs
C_VAE   = "#CE93D8"   # VAE / encoder
C_T1W   = "#80CBC4"   # T1w estructural
C_XGB   = "#FFAB40"   # XGBoost
C_LAT   = "#F48FB1"   # embeddings latentes μ
C_TEXT  = "#ECEFF1"   # texto principal
C_DIM   = "#78909C"   # texto secundario / faded
C_GOLD  = "#FFE082"   # highlight
C_OUT   = "#A5D6A7"   # resultado / output

# --- Helpers ---

def feat_cell(v: float, label: str, color=C_BLUE, w=0.52, h=0.52) -> VGroup:
    """Celda de feature: caja con valor numérico y etiqueta inferior."""
    box = Rectangle(
        width=w, height=h,
        fill_color=color,
        fill_opacity=max(0.12, min(0.78, 0.12 + abs(float(v)) * 0.66)),
        stroke_color=color, stroke_width=1.8,
    )
    val = Text(f"{float(v):+.2f}", font_size=12, color=C_TEXT)
    val.move_to(box)
    lbl = Text(label, font_size=9, color=C_DIM)
    lbl.next_to(box, DOWN, buff=0.05)
    return VGroup(box, val, lbl)


def feat_strip(n_cells: int, total: int, color=C_BLUE,
               rng_seed: int = 42) -> VGroup:
    """
    Vector de features resumido: n_cells celdas + ··· + última celda + brace.
    Retorna VGroup(cells, dots, last_cell, brace, brace_label).
    """
    rng = np.random.default_rng(rng_seed)
    vals = rng.uniform(-0.9, 0.9, n_cells + 1).tolist()

    cells = VGroup(*[
        feat_cell(vals[i], f"f{i+1}", color) for i in range(n_cells)
    ])
    cells.arrange(RIGHT, buff=0.10)

    dots = Text("·  ·  ·", font_size=22, color=C_DIM)

    last = feat_cell(vals[-1], f"f{total}", color)

    row = VGroup(cells, dots, last)
    row.arrange(RIGHT, buff=0.16)

    brace = Brace(row, DOWN, color=color, buff=0.08)
    blabel = Text(f"{total:,} features", font_size=18, color=color, weight=BOLD)
    blabel.next_to(brace, DOWN, buff=0.10)

    return VGroup(row, brace, blabel)


def rounded_box(label: str, sublabel: str = "", color=C_VAE,
                width=1.8, height=0.85) -> VGroup:
    """Caja redondeada con etiqueta principal y opcional sublabel."""
    rect = RoundedRectangle(
        corner_radius=0.12, width=width, height=height,
        fill_color=color, fill_opacity=0.22,
        stroke_color=color, stroke_width=2.0,
    )
    lab = Text(label, font_size=15, color=C_TEXT, weight=BOLD)
    lab.move_to(rect)
    grp = VGroup(rect, lab)
    if sublabel:
        sub = Text(sublabel, font_size=12, color=color)
        sub.next_to(lab, DOWN, buff=0.06)
        lab.shift(UP * 0.12)
        sub.next_to(lab, DOWN, buff=0.06)
        grp.add(sub)
    return grp


def trap_connector(top_rect: Rectangle, bot_rect: Rectangle,
                   color=C_VAE, opacity=0.22) -> Polygon:
    """Trapecio VERTICAL: conecta dos barras horizontales (flujo top→bottom)."""
    tl = top_rect.get_corner(DL)
    tr = top_rect.get_corner(DR)
    bl = bot_rect.get_corner(UL)
    br = bot_rect.get_corner(UR)
    return Polygon(tl, tr, br, bl,
                   fill_color=color, fill_opacity=opacity,
                   stroke_width=0)


def h_trap_connector(left_rect, right_rect, color=C_VAE, opacity=0.22) -> Polygon:
    """Trapecio HORIZONTAL: conecta dos barras verticales de distinta altura."""
    p1 = left_rect.get_corner(UR)   # arriba-derecha del bar izquierdo
    p2 = left_rect.get_corner(DR)   # abajo-derecha del bar izquierdo
    p3 = right_rect.get_corner(DL)  # abajo-izquierda del bar derecho
    p4 = right_rect.get_corner(UL)  # arriba-izquierda del bar derecho
    return Polygon(p1, p2, p3, p4,
                   fill_color=color, fill_opacity=opacity,
                   stroke_width=0)


def make_tree(color=C_XGB, s=1.0) -> VGroup:
    """Boceto simplificado de un árbol de decisión (3 niveles)."""
    def circ(pos, r=0.13):
        c = Circle(radius=r * s, fill_color=color, fill_opacity=0.30,
                   stroke_color=color, stroke_width=1.5)
        c.move_to(pos)
        return c

    def leaf_rect(pos):
        r = Rectangle(width=0.22 * s, height=0.16 * s,
                      fill_color=color, fill_opacity=0.42,
                      stroke_color=color, stroke_width=1.2)
        r.move_to(pos)
        return r

    def edge(a, b):
        return Line(a.get_bottom(), b.get_top(),
                    color=color, stroke_width=1.2 * s)

    n0 = circ([0,        0,       0])
    n1 = circ([-0.45*s, -0.55*s, 0])
    n2 = circ([ 0.45*s, -0.55*s, 0])
    l1 = leaf_rect([-0.72*s, -1.10*s, 0])
    l2 = leaf_rect([-0.22*s, -1.10*s, 0])
    l3 = leaf_rect([ 0.22*s, -1.10*s, 0])
    l4 = leaf_rect([ 0.72*s, -1.10*s, 0])

    return VGroup(
        edge(n0, n1), edge(n0, n2),
        edge(n1, l1), edge(n1, l2),
        edge(n2, l3), edge(n2, l4),
        n0, n1, n2, l1, l2, l3, l4,
    )
