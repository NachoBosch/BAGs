"""
Pipeline completo — Brain Age Estimation Pipeline.

Escena única que construye el pipeline de izquierda a derecha en 5 fases:
  1. FC matrix → 6,670 features
  2. β-VAE → μ: 64 dims
  3. + T1w: 116 features
  4. → 180 → XGBoost → ŷ
  5. Resultado final con métricas

Todos los elementos permanecen en pantalla al final para el "frame resumen".
"""
from manim import *
from common import (
    BG, C_BLUE, C_VAE, C_T1W, C_XGB, C_LAT, C_TEXT, C_DIM,
    C_GOLD, C_OUT, IMG_FC, make_tree, rounded_box,
)
import numpy as np


# --- Pequeño helper local ---
def compact_arrow(start, end, color=C_DIM):
    return Arrow(
        start, end,
        color=color, stroke_width=2.2, buff=0,
        max_tip_length_to_length_ratio=0.18,
    )


class FullPipeline(Scene):
    def construct(self):
        self.camera.background_color = BG
        rng = np.random.default_rng(42)

        # --- TÍTULO ---
        title = Text(
            "Brain Age Estimation  —  Pipeline Completo",
            font_size=28, color=C_TEXT,
        )
        title.to_edge(UP, buff=0.25)
        subtitle = Text(
            "FC (fMRI)  +  T1w (MRI)  →  β-VAE  →  XGBoost  →  Edad cerebral",
            font_size=16, color=C_DIM,
        )
        subtitle.next_to(title, DOWN, buff=0.10)

        self.play(Write(title), run_time=1.4)
        self.play(FadeIn(subtitle, shift=UP * 0.1), run_time=1.0)
        self.wait(1.0)

        # Línea divisoria
        hline = Line([-6.8, 2.55, 0], [6.8, 2.55, 0],
                     stroke_color=C_DIM, stroke_width=0.7)
        self.play(Create(hline), run_time=0.6)

        # --- FASE 1 — Matriz FC → 6,670 features ---

        # Imagen FC (pequeña)
        fc = ImageMobject(IMG_FC)
        fc.set_height(1.85)
        fc.move_to([-5.6, 0.5, 0])

        fc_lbl = Text("Matriz FC\n116×116", font_size=13, color=C_BLUE)
        fc_lbl.next_to(fc, DOWN, buff=0.12)

        # Caja "6,670"
        box_fc = rounded_box("6,670", "features FC", C_BLUE, width=1.45, height=0.82)
        box_fc.move_to([-3.7, 0.5, 0])

        arr_fc = compact_arrow([-4.65, 0.5, 0], [-4.50, 0.5, 0], C_BLUE)

        self.play(FadeIn(fc, scale=0.85), FadeIn(fc_lbl), run_time=1.1)
        self.play(GrowArrow(arr_fc), run_time=0.7)
        self.play(FadeIn(box_fc, shift=RIGHT * 0.15), run_time=0.9)
        self.wait(0.6)

        # --- FASE 2 — β-VAE encoder → μ: 64 ---

        arr_vae_in = compact_arrow([-2.97, 0.5, 0], [-2.20, 0.5, 0], C_VAE)

        # Mini embudo VAE (3 barras compactas, centradas en x=-1.2)
        cx_vae = -1.2
        bar_w  = [1.8, 1.0, 0.36]   # widths: 6670, 512, 64
        bar_y  = [1.15, 0.5, -0.15]  # y positions
        bar_colors = [C_BLUE, C_VAE, C_LAT]
        bar_fo     = [0.22,   0.25,  0.45]
        bar_labels = ["6,670", "512", "64"]

        bars = []
        bar_rects = []
        for i in range(3):
            r = Rectangle(
                width=bar_w[i], height=0.32,
                fill_color=bar_colors[i], fill_opacity=bar_fo[i],
                stroke_color=bar_colors[i], stroke_width=1.6,
            )
            r.move_to([cx_vae, bar_y[i], 0])
            t = Text(bar_labels[i], font_size=11, color=C_TEXT)
            t.move_to(r)
            bars.append(VGroup(r, t))
            bar_rects.append(r)

        # Trapecios
        def mini_trap(r1, r2, color):
            return Polygon(
                r1.get_corner(DL), r1.get_corner(DR),
                r2.get_corner(UR), r2.get_corner(UL),
                fill_color=color, fill_opacity=0.18, stroke_width=0,
            )

        t12v = mini_trap(bar_rects[0], bar_rects[1], C_VAE)
        t23v = mini_trap(bar_rects[1], bar_rects[2], C_VAE)

        vae_brace = Brace(VGroup(*bar_rects), LEFT, color=C_VAE, buff=0.08)
        vae_lbl = Text("β-VAE", font_size=13, color=C_VAE, weight=BOLD)
        vae_lbl.next_to(vae_brace, LEFT, buff=0.10)

        # Caja μ: 64 (a la derecha del embudo)
        box_z = rounded_box("μ : 64", "latent dims", C_LAT, width=1.35, height=0.82)
        box_z.move_to([0.35, 0.5, 0])

        arr_z_out = compact_arrow([-0.03, -0.15, 0], [0.35 - 0.68, 0.5 - 0.41 + 0.02, 0], C_LAT)
        # Usamos flecha diagonal desde barra latente hasta caja z
        arr_z = Arrow(
            bar_rects[2].get_right() + RIGHT * 0.08,
            box_z.get_left() + LEFT * 0.08,
            color=C_LAT, stroke_width=2.0, buff=0,
            max_tip_length_to_length_ratio=0.16,
        )

        self.play(GrowArrow(arr_vae_in), run_time=0.7)
        self.play(
            LaggedStart(
                FadeIn(bars[0]), Create(t12v), FadeIn(bars[1]),
                Create(t23v), FadeIn(bars[2]),
                lag_ratio=0.3,
            ),
            run_time=1.5,
        )
        self.play(Create(vae_brace), FadeIn(vae_lbl), run_time=0.8)

        # Highlight latente
        glow_z = SurroundingRectangle(bar_rects[2], color=C_GOLD,
                                       stroke_width=2.2, buff=0.07)
        self.play(Create(glow_z), run_time=0.6)
        self.play(GrowArrow(arr_z), FadeIn(box_z, shift=RIGHT * 0.1), run_time=0.9)
        self.wait(0.6)

        # --- FASE 3 — T1w: 116 se une desde abajo ---

        box_t1w = rounded_box("T1w", "116 features", C_T1W, width=1.35, height=0.82)
        box_t1w.move_to([0.35, -1.55, 0])

        t1w_note = Text(
            "Volúmenes materia gris\n(atlas AAL, 116 regiones)",
            font_size=12, color=C_DIM,
        )
        t1w_note.next_to(box_t1w, LEFT, buff=0.22)

        arr_t1w_up = Arrow(
            [0.35, -1.08, 0], [0.35, 0.08, 0],
            color=C_T1W, stroke_width=2.0, buff=0,
            max_tip_length_to_length_ratio=0.16,
        )

        plus_sym = Text("+", font_size=40, color=C_GOLD)
        plus_sym.move_to([0.35, -0.48, 0])

        self.play(FadeIn(box_t1w, shift=UP * 0.2), FadeIn(t1w_note), run_time=1.0)
        self.play(Write(plus_sym), GrowArrow(arr_t1w_up), run_time=0.9)
        self.wait(0.6)

        # --- FASE 4 — 180 features → XGBoost → ŷ ---

        # Caja 180
        box_180 = rounded_box("180", "64 + 116", C_OUT, width=1.35, height=0.82)
        box_180.move_to([2.05, 0.5, 0])

        arr_to_180 = compact_arrow([1.02, 0.5, 0], [1.36, 0.5, 0], C_OUT)

        # XGBoost como caja compacta (evita overflow de árboles)
        box_xgb = rounded_box("XGBoost", "100 estim.", C_XGB, width=1.70, height=0.82)
        box_xgb.move_to([4.05, 0.5, 0])

        arr_to_xgb = compact_arrow([2.74, 0.5, 0], [3.19, 0.5, 0], C_XGB)
        lbl_xgb = Text("predict", font_size=12, color=C_DIM)
        lbl_xgb.next_to(arr_to_xgb, UP, buff=0.06)

        # Predicción
        pred_box = RoundedRectangle(
            corner_radius=0.14, width=1.55, height=0.95,
            fill_color=C_OUT, fill_opacity=0.22,
            stroke_color=C_OUT, stroke_width=2.4,
        )
        pred_box.move_to([5.85, 0.5, 0])
        yhat = Text("ŷ = 63.2", font_size=20, color=C_OUT, weight=BOLD)
        yhat.move_to(pred_box.get_center() + UP * 0.15)
        yanos = Text("años", font_size=14, color=C_OUT)
        yanos.next_to(yhat, DOWN, buff=0.04)
        pred_title = Text("Edad\ncerebral", font_size=12, color=C_DIM)
        pred_title.next_to(pred_box, UP, buff=0.10)

        arr_to_pred = compact_arrow(
            box_xgb.get_right() + RIGHT * 0.08,
            pred_box.get_left() + LEFT * 0.08,
            C_GOLD,
        )

        self.play(
            GrowArrow(arr_to_180),
            FadeIn(box_180, shift=RIGHT * 0.1),
            run_time=0.9,
        )
        self.play(GrowArrow(arr_to_xgb), FadeIn(lbl_xgb), run_time=0.7)
        self.play(FadeIn(box_xgb, shift=RIGHT * 0.1), run_time=0.9)
        self.play(GrowArrow(arr_to_pred), run_time=0.7)
        self.play(
            FadeIn(pred_box, scale=0.85),
            FadeIn(pred_title),
            Write(yhat), FadeIn(yanos),
            run_time=1.1,
        )
        self.wait(1.0)

        # --- FASE 5 — Métricas finales ---

        divider = Line([-6.8, -2.0, 0], [6.8, -2.0, 0],
                       stroke_color=C_DIM, stroke_width=0.8)
        self.play(Create(divider), run_time=0.5)

        metrics = VGroup(
            Text("MAE = 5.62 años", font_size=18, color=C_GOLD),
            Text("R²  = 0.411",     font_size=18, color=C_GOLD),
            Text("N = 1,245  |  cohorte RedLaT  (CN, AD, FTD)",
                 font_size=14, color=C_DIM),
        )
        metrics.arrange(RIGHT, buff=0.7)
        metrics.move_to([0, -2.55, 0])

        self.play(
            LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in metrics],
                        lag_ratio=0.3),
            run_time=1.3,
        )

        # Resaltar todo el pipeline con un recuadro suave
        pipeline_rect = SurroundingRectangle(
            Group(fc, box_z, box_t1w, pred_box),
            color=C_DIM, stroke_width=0.9, buff=0.25, corner_radius=0.18,
        )
        self.play(Create(pipeline_rect), run_time=1.0)

        self.wait(4.0)
