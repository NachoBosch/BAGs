"""
Escena 4 — XGBoost: Predicción de Edad Cerebral.

Mejoras respecto a v1:
  • Flechas horizontales que NO pasan por los árboles (posiciones corregidas).
  • Input representado como dos barras coloreadas (z + T1w) en lugar de celdas.
  • Scatter plot de predicción vs. edad real (imagen del proyecto).
  • Métricas finales al pie.
"""
from manim import *
from common import (
    BG, C_VAE, C_T1W, C_XGB, C_LAT, C_TEXT, C_DIM,
    C_GOLD, C_OUT, make_tree, BASE,
)



class XGBoostPrediction(Scene):
    def construct(self):
        self.camera.background_color = BG

        # --- Header ---
        header = Text(
            "XGBoost:  Predicción de Edad Cerebral",
            font_size=25, color=C_TEXT,
        )
        header.to_edge(UP, buff=0.28)
        ul = Underline(header, color=C_XGB, stroke_width=1.5)
        self.play(Write(header), Create(ul), run_time=1.4)

        # --- Input: barra z (64) + barra T1w (116) ---
        bar_z = Rectangle(
            width=0.85, height=1.85,
            fill_color=C_LAT, fill_opacity=0.40,
            stroke_color=C_LAT, stroke_width=2.0,
        )
        bar_z.move_to([-4.6, 0.35, 0])
        lbl_z = Text("μ\n64", font_size=12, color=C_TEXT)
        lbl_z.move_to(bar_z)

        bar_t1 = Rectangle(
            width=1.05, height=1.85,
            fill_color=C_T1W, fill_opacity=0.35,
            stroke_color=C_T1W, stroke_width=2.0,
        )
        bar_t1.next_to(bar_z, RIGHT, buff=0.12)
        lbl_t1 = Text("T1w\n116", font_size=12, color=C_TEXT)
        lbl_t1.move_to(bar_t1)

        in_bars = VGroup(bar_z, bar_t1)

        brace_in = Brace(in_bars, DOWN, color=C_OUT, buff=0.08)
        lbl_180 = Text("180 features", font_size=17, color=C_OUT, weight=BOLD)
        lbl_180.next_to(brace_in, DOWN, buff=0.10)

        in_title = Text("Feature vector\n(VAE + T1w)", font_size=13, color=C_DIM)
        in_title.next_to(in_bars, UP, buff=0.14)

        # Posición real del borde derecho del grupo de barras de entrada
        # bar_z: center=(-4.6, 0.35), w=0.85  → right = -4.6+0.425 = -4.175
        # bar_t1: next to bar_z RIGHT buff=0.12 → center_x = -4.175+0.12+0.525 = -3.535
        #         right = -3.535 + 0.525 = -3.010
        IN_RIGHT_X = -3.010

        # --- 3 árboles XGBoost ---
        trees = VGroup(*[make_tree(C_XGB, s=0.78) for _ in range(3)])
        trees.arrange(RIGHT, buff=0.40)
        # Centrar árboles para que el flujo sea horizontal respecto a y=0.35
        trees.move_to([0.5, 0.0, 0])

        trees_lbl = Text("XGBoost  (ensemble de árboles)", font_size=15,
                          color=C_XGB, weight=BOLD)
        trees_lbl.next_to(trees, UP, buff=0.22)

        # Borde izquierdo/derecho del grupo de árboles
        # s=0.78: width per tree ≈ 1.62*0.78 = 1.264
        # 3 trees + 2×buff(0.40) = 3×1.264 + 0.80 = 4.592 wide
        # center x=0.5 → left = 0.5 - 2.296 = -1.796, right = 0.5 + 2.296 = 2.796
        TREES_LEFT_X  = -1.796
        TREES_RIGHT_X =  2.796

        # --- Flecha input → árboles (HORIZONTAL en y=0.35) ---
        arr1 = Arrow(
            [IN_RIGHT_X + 0.14, 0.35, 0],
            [TREES_LEFT_X - 0.14, 0.35, 0],
            color=C_XGB, stroke_width=2.5, buff=0,
            max_tip_length_to_length_ratio=0.13,
        )

        # --- Caja de predicción ---
        pred_box = RoundedRectangle(
            corner_radius=0.14, width=2.15, height=1.10,
            fill_color=C_OUT, fill_opacity=0.20,
            stroke_color=C_OUT, stroke_width=2.5,
        )
        pred_box.move_to([4.85, 0.35, 0])

        y_hat = Text("ŷ = 63.2", font_size=26, color=C_OUT, weight=BOLD)
        y_hat.move_to(pred_box.get_center() + UP * 0.17)
        y_anos = Text("años", font_size=16, color=C_OUT)
        y_anos.next_to(y_hat, DOWN, buff=0.04)

        pred_title = Text("Edad cerebral\npredicción", font_size=13, color=C_DIM)
        pred_title.next_to(pred_box, UP, buff=0.12)

        # pred_box left = 4.85 - 1.075 = 3.775
        PRED_LEFT_X = 3.775

        # --- Flecha árboles → predicción (HORIZONTAL en y=0.35) ---
        arr2 = Arrow(
            [TREES_RIGHT_X + 0.14, 0.35, 0],
            [PRED_LEFT_X - 0.14, 0.35, 0],
            color=C_GOLD, stroke_width=2.5, buff=0,
            max_tip_length_to_length_ratio=0.13,
        )

        # --- Línea divisoria ---
        divider = Line([-6.5, -1.42, 0], [6.5, -1.42, 0],
                       stroke_color=C_DIM, stroke_width=0.7)

        met1 = Text("MAE  =  5.62 años", font_size=20, color=C_GOLD)
        met2 = Text("R²   =  0.411",     font_size=20, color=C_GOLD)
        met3 = Text("N = 125  (hold-out test)  |  cohorte RedLaT", font_size=15, color=C_DIM)

        metrics = VGroup(met1, met2, met3)
        metrics.arrange(RIGHT, buff=0.70)
        metrics.move_to([0.0, -1.88, 0])

        met_title = Text("Rendimiento en test set:", font_size=15, color=C_DIM)
        met_title.next_to(metrics, UP, buff=0.16)

        # --- Animaciones ---

        # Input
        self.play(
            FadeIn(in_title),
            FadeIn(VGroup(bar_z, lbl_z), shift=UP * 0.15),
            FadeIn(VGroup(bar_t1, lbl_t1), shift=UP * 0.15),
            run_time=1.3,
        )
        self.play(Create(brace_in), FadeIn(lbl_180), run_time=0.9)

        # Flecha → árboles
        self.play(GrowArrow(arr1), run_time=0.7)

        # Árboles aparecen en cascada
        self.play(FadeIn(trees_lbl), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(t, shift=UP * 0.15) for t in trees],
                        lag_ratio=0.30),
            run_time=1.4,
        )
        self.wait(0.6)

        # Flecha → predicción
        self.play(GrowArrow(arr2), run_time=0.9)
        self.play(
            FadeIn(pred_box, scale=0.85),
            FadeIn(pred_title),
            run_time=0.8,
        )
        self.play(Write(y_hat), FadeIn(y_anos), run_time=1.0)
        self.wait(1.0)

        # Divisor + métricas
        self.play(Create(divider), run_time=0.6)

        self.play(FadeIn(met_title), run_time=0.5)
        self.play(
            LaggedStart(*[Write(m) for m in metrics], lag_ratio=0.4),
            run_time=1.3,
        )

        self.wait(4.0)
