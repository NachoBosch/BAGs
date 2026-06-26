"""
Escena 1 — De la Matriz de Conectividad Funcional a un Vector de Features.

Muestra:
  • La imagen real de la matriz FC (116×116) del proyecto.
  • El proceso → strip de features (4 celdas + ··· + f6670).
  • Brace "6,670 features" y nota de Fisher z-transform.
"""
from manim import *
from common import (
    BG, C_BLUE, C_TEXT, C_DIM, C_GOLD,
    IMG_FC, feat_cell,
)
import numpy as np


class FCToFeatures(Scene):
    def construct(self):
        self.camera.background_color = BG

        # --- Header ---
        header = Text(
            "Conectividad Funcional  →  Vector de Features",
            font_size=26, color=C_TEXT,
        )
        header.to_edge(UP, buff=0.30)
        underline = Underline(header, color=C_BLUE, stroke_width=1.5)
        self.play(Write(header), Create(underline), run_time=1.4)
        self.wait(0.6)

        # --- Imagen de la matriz FC ---
        fc = ImageMobject(IMG_FC)
        fc.set_height(3.6)
        fc.move_to(LEFT * 3.1)

        fc_label = Text("Matriz FC  (116 × 116)", font_size=18, color=C_BLUE)
        fc_label.next_to(fc, DOWN, buff=0.20)

        self.play(FadeIn(fc, scale=0.88), run_time=1.3)
        self.play(FadeIn(fc_label, shift=UP * 0.1), run_time=0.8)
        self.wait(1.0)

        # --- Arrow FC → strip ---
        # Posición del strip (calculada para no salir de bounds)
        # strip centrado en x≈2.8, la flecha va desde el borde derecho de fc
        strip_center_x = 2.9

        rng = np.random.default_rng(42)
        vals = rng.uniform(-0.9, 0.9, 5).tolist()

        cells = VGroup(*[
            feat_cell(vals[i], f"f{i+1}") for i in range(4)
        ])
        cells.arrange(RIGHT, buff=0.10)

        dots = Text("·  ·  ·", font_size=22, color=C_DIM)
        last = feat_cell(vals[4], "f6670")

        row = VGroup(cells, dots, last)
        row.arrange(RIGHT, buff=0.16)
        row.move_to([strip_center_x, 0.3, 0])

        # Brace
        brace = Brace(row, DOWN, color=C_BLUE, buff=0.08)
        blabel = Text("6,670 features", font_size=19, color=C_BLUE, weight=BOLD)
        blabel.next_to(brace, DOWN, buff=0.10)

        # Nota Fisher-z
        fisher_note = Text(
            "(Fisher z-transformadas antes de entrar al VAE)",
            font_size=13, color=C_DIM,
        )
        fisher_note.next_to(blabel, DOWN, buff=0.10)

        # Flecha horizontal: misma altura que el strip
        y_arrow = row.get_center()[1]
        arrow = Arrow(
            [fc.get_right()[0] + 0.15, y_arrow, 0],
            [row.get_left()[0] - 0.15, y_arrow, 0],
            color=C_BLUE, stroke_width=2.5, buff=0,
            max_tip_length_to_length_ratio=0.14,
        )
        self.play(GrowArrow(arrow), run_time=1.0)

        # Celdas aparecen en cascada
        self.play(
            LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cells],
                        lag_ratio=0.2),
            run_time=1.7,
        )
        self.play(
            FadeIn(dots, shift=UP * 0.15),
            FadeIn(last, shift=UP * 0.15),
            run_time=1.0,
        )
        self.play(Create(brace), Write(blabel), run_time=1.1)
        self.play(FadeIn(fisher_note), run_time=0.8)

        self.wait(4.0)
