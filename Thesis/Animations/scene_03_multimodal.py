"""
Escena 3 — Integración Multimodal: VAE embeddings + T1w → 180 features.

Muestra:
  • Strip de embeddings μ del VAE (64 dims, color latente/púrpura).
  • Strip de features T1w estructurales (116 dims, teal).
  • Símbolo "+" y flecha de concatenación.
  • Strip combinado (180 features) con brace.
"""
from manim import *
from common import (
    BG, C_VAE, C_T1W, C_LAT, C_TEXT, C_DIM, C_GOLD, C_OUT,
    feat_cell,
)
import numpy as np


class MultimodalFusion(Scene):
    def construct(self):
        self.camera.background_color = BG

        # --- Header ---
        header = Text(
            "Integración Multimodal:  VAE  +  T1w  →  180 features",
            font_size=23, color=C_TEXT,
        )
        header.to_edge(UP, buff=0.28)
        ul = Underline(header, color=C_OUT, stroke_width=1.5)
        self.play(Write(header), Create(ul), run_time=1.4)

        rng = np.random.default_rng(7)

        # --- Helpers ---
        def make_strip(n_cells, total, color, seed_offset=0):
            vs = rng.uniform(-0.9, 0.9, n_cells + 1)
            cells = VGroup(*[
                feat_cell(float(vs[i]), f"μ{i+1}" if color == C_LAT else f"t{i+1}",
                          color)
                for i in range(n_cells)
            ])
            cells.arrange(RIGHT, buff=0.09)
            dots = Text("·  ·  ·", font_size=20, color=C_DIM)
            last = feat_cell(float(vs[-1]),
                             f"μ{total}" if color == C_LAT else f"t{total}",
                             color)
            row = VGroup(cells, dots, last)
            row.arrange(RIGHT, buff=0.14)
            return row

        def titled_strip(row, title_str, color):
            title = Text(title_str, font_size=16, color=color, weight=BOLD)
            title.next_to(row, UP, buff=0.14)
            return VGroup(title, row)

        # --- Strip VAE mu: 64 ---
        z_row = make_strip(3, 64, C_LAT)
        z_block = titled_strip(z_row, "VAE Embeddings  μ  (64 dims)", C_LAT)
        z_block.move_to([-3.0, 0.5, 0])
        note_z = Text("β-VAE encoder output", font_size=12, color=C_DIM)
        note_z.next_to(z_row, DOWN, buff=0.10)
        brace_vae = Brace(z_row, DOWN, color=C_LAT, buff=0.06)
        brace_vae.next_to(note_z, DOWN, buff=0.08)
        bl_vae = Text("64", font_size=14, color=C_LAT)
        bl_vae.next_to(brace_vae, DOWN, buff=0.05)

        # --- Strip T1w: 116 ---
        t_row = make_strip(3, 116, C_T1W)
        t_block = titled_strip(t_row, "T1w Estructural  (116 features)", C_T1W)
        t_block.move_to([-3.0, -1.8, 0])
        note_t1w = Text("Volúmenes materia gris (atlas AAL)", font_size=12, color=C_DIM)
        note_t1w.next_to(t_row, DOWN, buff=0.10)
        brace_t1w = Brace(t_row, DOWN, color=C_T1W, buff=0.06)
        brace_t1w.next_to(note_t1w, DOWN, buff=0.08)
        bl_t1w = Text("116", font_size=14, color=C_T1W)
        bl_t1w.next_to(brace_t1w, DOWN, buff=0.05)

        # --- Símbolo + ---
        plus = Text("+", font_size=52, color=C_GOLD)
        plus.move_to([-0.4, -0.6, 0])

        # --- Flecha de concatenación ---
        concat_arrow = Arrow(
            [0.5, -0.6, 0], [1.6, -0.6, 0],
            color=C_OUT, stroke_width=2.8, buff=0,
            max_tip_length_to_length_ratio=0.15,
        )

        # --- Strip combinado: 180 ---
        rng2 = np.random.default_rng(99)
        # 3 celdas VAE (púrpura) + dots + 3 celdas T1w (teal) + dots + brace
        zvals = rng2.uniform(-0.9, 0.9, 4)
        tvals = rng2.uniform(-0.9, 0.9, 4)

        z_cells_c = VGroup(*[feat_cell(float(zvals[i]), f"μ{i+1}", C_LAT)
                              for i in range(3)])
        z_cells_c.arrange(RIGHT, buff=0.09)
        zdots = Text("·", font_size=22, color=C_DIM)
        t_cells_c = VGroup(*[feat_cell(float(tvals[i]), f"t{i+1}", C_T1W)
                              for i in range(3)])
        t_cells_c.arrange(RIGHT, buff=0.09)
        tdots = Text("·", font_size=22, color=C_DIM)
        last_c = feat_cell(float(tvals[-1]), "t116", C_T1W)

        sep = Text("|", font_size=26, color=C_DIM)

        combined_row = VGroup(z_cells_c, zdots, sep, t_cells_c, tdots, last_c)
        combined_row.arrange(RIGHT, buff=0.10)
        combined_row.move_to([4.0, -0.6, 0])

        brace_c = Brace(combined_row, DOWN, color=C_OUT, buff=0.08)
        lbl_c = Text("180 features", font_size=19, color=C_OUT, weight=BOLD)
        lbl_c.next_to(brace_c, DOWN, buff=0.10)

        combined_title = Text(
            "Feature vector para XGBoost", font_size=15, color=C_OUT,
        )
        combined_title.next_to(combined_row, UP, buff=0.14)

        # Bracket coloreado (nombre distinto para no pisar note_z de arriba)
        combined_note_z = Text("← 64 (VAE)",  font_size=12, color=C_LAT)
        combined_note_t1 = Text("116 (T1w) →", font_size=12, color=C_T1W)
        combined_note_z.next_to(lbl_c, LEFT,  buff=0.3)
        combined_note_t1.next_to(lbl_c, RIGHT, buff=0.3)

        # --- Animaciones ---
        self.play(
            FadeIn(z_block, shift=RIGHT * 0.2),
            FadeIn(note_z),
            Create(brace_vae), FadeIn(bl_vae),
            run_time=1.3,
        )
        self.wait(0.6)
        self.play(
            FadeIn(t_block, shift=RIGHT * 0.2),
            FadeIn(note_t1w),
            Create(brace_t1w), FadeIn(bl_t1w),
            run_time=1.3,
        )
        self.wait(0.6)
        self.play(Write(plus), run_time=0.8)
        self.wait(0.8)

        self.play(GrowArrow(concat_arrow), run_time=1.0)

        self.play(
            LaggedStart(
                FadeIn(z_cells_c, shift=LEFT * 0.15),
                FadeIn(zdots),
                FadeIn(sep),
                FadeIn(t_cells_c, shift=LEFT * 0.15),
                FadeIn(tdots),
                FadeIn(last_c),
                lag_ratio=0.18,
            ),
            run_time=2.0,
        )
        self.play(
            Create(brace_c), Write(lbl_c),
            FadeIn(combined_title),
            run_time=1.1,
        )
        self.play(FadeIn(combined_note_z), FadeIn(combined_note_t1), run_time=0.8)

        self.wait(4.0)
