"""
Escena 6 -- Optimizacion de hiperparametros de XGBoost con Optuna.

Muestra: VAE/embeddings fijos, Optuna sugiere params XGBoost,
entrena y evalua en K folds, MAE, Optuna recibe score.
"""
from manim import *
from common import (
    BG, C_BLUE, C_VAE, C_T1W, C_XGB, C_LAT, C_TEXT, C_DIM,
    C_GOLD, C_OUT, rounded_box,
)


class OptunaXGBScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text(
            "Etapa 2: Optimizacion de XGBoost con Optuna",
            font_size=28, color=C_TEXT,
        )
        title.to_edge(UP, buff=0.25)
        subtitle = Text(
            "Embeddings del mejor VAE fijados; solo se optimizan parametros de XGBoost",
            font_size=15, color=C_DIM,
        )
        subtitle.next_to(title, DOWN, buff=0.10)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.1), run_time=0.8)
        self.wait(0.8)

        hline = Line(LEFT * 6.8 + UP * 2.5, RIGHT * 6.8 + UP * 2.5,
                     stroke_color=C_DIM, stroke_width=0.7)
        self.play(Create(hline), run_time=0.4)

        # Top row: frozen VAE -> mu -> Optuna -> Params XGB
        vae_frozen = RoundedRectangle(
            corner_radius=0.12, width=1.8, height=0.75,
            fill_color=C_VAE, fill_opacity=0.08,
            stroke_color=C_VAE, stroke_width=1.8,
        )
        vae_frozen.set_stroke(opacity=0.5)
        vae_frozen.move_to(LEFT * 5.8 + UP * 1.5)
        vae_lbl = Text("beta-VAE", font_size=13, color=C_VAE, weight=BOLD)
        vae_lbl.move_to(vae_frozen.get_center() + UP * 0.08)
        vae_sub = Text("FIJO", font_size=9, color=C_DIM)
        vae_sub.next_to(vae_lbl, DOWN, buff=0.04)

        emb_box = rounded_box("μ: 64 dims", "fijos", C_LAT, width=1.6, height=0.75)
        emb_box.move_to(LEFT * 3.6 + UP * 1.5)

        arr_emb = Arrow(vae_frozen.get_right(), emb_box.get_left(),
                        color=C_VAE, stroke_width=1.8, buff=0.08,
                        max_tip_length_to_length_ratio=0.12)

        optuna_box = rounded_box("Optuna", "TPE Sampler", C_GOLD, width=1.8, height=0.85)
        optuna_box.move_to(LEFT * 1.2 + UP * 1.5)

        trial_label = Text("100 trials", font_size=13, color=C_GOLD)
        trial_label.next_to(optuna_box, DOWN, buff=0.12)

        params_box = rounded_box("Params XGB", "n_est, depth, lr, ...", C_XGB,
                                 width=2.3, height=0.85)
        params_box.move_to(RIGHT * 1.6 + UP * 1.5)

        arr_opt = Arrow(optuna_box.get_right(), params_box.get_left(),
                        color=C_GOLD, stroke_width=2.0, buff=0.08,
                        max_tip_length_to_length_ratio=0.12)

        self.play(
            FadeIn(vae_frozen), Write(vae_lbl), FadeIn(vae_sub),
            run_time=0.8,
        )
        self.play(GrowArrow(arr_emb), FadeIn(emb_box, shift=RIGHT * 0.1), run_time=0.7)
        self.wait(0.3)
        self.play(FadeIn(optuna_box), run_time=0.6)
        self.play(Write(trial_label), run_time=0.5)
        self.play(GrowArrow(arr_opt), FadeIn(params_box, shift=RIGHT * 0.1), run_time=0.7)
        self.wait(0.6)

        # Fold frame
        fold_frame = RoundedRectangle(
            corner_radius=0.12, width=10.0, height=2.2,
            stroke_color=C_DIM, stroke_width=1.2,
        )
        fold_frame.move_to(DOWN * 0.7)

        fold_title = Text("CV 5-fold: train/val rotativo", font_size=13,
                          color=C_DIM, slant=ITALIC)
        fold_title.next_to(fold_frame, UP, buff=0.08)
        fold_title.align_to(fold_frame, LEFT).shift(RIGHT * 0.2)

        self.play(Create(fold_frame), FadeIn(fold_title), run_time=0.8)

        fold_cells = VGroup()
        for i in range(5):
            cell = Rectangle(
                width=0.55, height=0.20,
                fill_color=C_BLUE, fill_opacity=0.22,
                stroke_color=C_DIM, stroke_width=0.8,
            )
            fold_cells.add(cell)
        fold_cells.arrange(RIGHT, buff=0.08)
        fold_cells.move_to(fold_frame.get_top() + DOWN * 0.40 + RIGHT * 2.5)
        fold_lbl = Text("fold val rota", font_size=9, color=C_DIM)
        fold_lbl.next_to(fold_cells, LEFT, buff=0.15)
        val_marker = SurroundingRectangle(
            fold_cells[0], color=C_GOLD, stroke_width=1.8, buff=0.03,
        )
        self.play(FadeIn(fold_lbl), FadeIn(fold_cells), Create(val_marker), run_time=0.6)

        arr_emb_down = Arrow(
            emb_box.get_bottom(),
            fold_frame.get_top() + LEFT * 3.0 + DOWN * 0.05,
            color=C_LAT, stroke_width=1.5, buff=0.08,
            max_tip_length_to_length_ratio=0.12,
        )
        arr_params_down = Arrow(
            params_box.get_bottom(),
            fold_frame.get_top() + RIGHT * 0.5 + DOWN * 0.05,
            color=C_XGB, stroke_width=1.5, buff=0.08,
            max_tip_length_to_length_ratio=0.12,
        )
        self.play(GrowArrow(arr_emb_down), GrowArrow(arr_params_down), run_time=0.7)

        # Steps inside fold
        step_y = -0.65

        feat_box = rounded_box("μ + T1w", "180 feats", C_OUT, width=1.6, height=0.6)
        feat_box.set_y(step_y).set_x(-3.0)

        xgb_step = RoundedRectangle(
            corner_radius=0.1, width=2.0, height=0.6,
            fill_color=C_XGB, fill_opacity=0.18,
            stroke_color=C_XGB, stroke_width=2.0,
        )
        xgb_step.set_y(step_y).set_x(0.0)
        xgb_lbl2 = Text("Entrenar XGB", font_size=13, color=C_XGB, weight=BOLD)
        xgb_lbl2.move_to(xgb_step.get_center() + UP * 0.08)
        xgb_sub2 = Text("fold train", font_size=9, color=C_XGB)
        xgb_sub2.next_to(xgb_lbl2, DOWN, buff=0.03)

        mae_box = rounded_box("MAE", "fold val", C_OUT, width=1.3, height=0.6)
        mae_box.set_y(step_y).set_x(2.8)

        a1 = Arrow(feat_box.get_right(), xgb_step.get_left(),
                   color=C_OUT, stroke_width=1.8, buff=0.08,
                   max_tip_length_to_length_ratio=0.12)
        a2 = Arrow(xgb_step.get_right(), mae_box.get_left(),
                   color=C_XGB, stroke_width=1.8, buff=0.08,
                   max_tip_length_to_length_ratio=0.12)

        self.play(FadeIn(feat_box, shift=RIGHT * 0.1), run_time=0.7)
        self.play(GrowArrow(a1), run_time=0.5)
        self.play(
            FadeIn(xgb_step), Write(xgb_lbl2), FadeIn(xgb_sub2),
            run_time=0.8,
        )
        self.play(GrowArrow(a2), run_time=0.5)
        self.play(FadeIn(mae_box, shift=RIGHT * 0.1), run_time=0.7)
        self.wait(0.8)

        # Fold cycling
        fold_maes = [6.5, 6.2, 6.4, 6.3, 6.4]
        fold_counter = Text("k = 1", font_size=15, color=C_GOLD, weight=BOLD)
        fold_counter.move_to(fold_frame.get_corner(UR) + LEFT * 0.5 + DOWN * 0.25)
        fold_error = Text(f"MAE = {fold_maes[0]:.1f}", font_size=12, color=C_OUT)
        fold_error.next_to(fold_counter, DOWN, buff=0.10)
        self.play(FadeIn(fold_counter), FadeIn(fold_error), run_time=0.5)
        self.wait(0.4)

        for k in range(2, 6):
            new_counter = Text(f"k = {k}", font_size=15, color=C_GOLD, weight=BOLD)
            new_counter.move_to(fold_counter)
            new_error = Text(f"MAE = {fold_maes[k - 1]:.1f}", font_size=12, color=C_OUT)
            new_error.move_to(fold_error)
            new_marker = SurroundingRectangle(
                fold_cells[k - 1], color=C_GOLD, stroke_width=1.8, buff=0.03,
            )
            self.play(
                FadeOut(fold_counter, shift=UP * 0.15),
                FadeIn(new_counter, shift=UP * 0.15),
                FadeOut(fold_error, shift=UP * 0.10),
                FadeIn(new_error, shift=UP * 0.10),
                Transform(val_marker, new_marker),
                Flash(mae_box, color=C_OUT, flash_radius=0.35, line_length=0.12),
                run_time=0.5,
            )
            fold_counter = new_counter
            fold_error = new_error
            self.wait(0.2)

        self.wait(0.5)

        # Report back to Optuna: path goes RIGHT, UP, LEFT (above all
        # top-row boxes), then DOWN into Optuna from the top.
        right_x = 5.8
        above_y = 2.2
        optuna_top = optuna_box.get_top()

        fb_right = Line(
            mae_box.get_right(),
            [right_x, mae_box.get_center()[1], 0],
            color=C_GOLD, stroke_width=1.5,
        )
        fb_up = Line(
            [right_x, mae_box.get_center()[1], 0],
            [right_x, above_y, 0],
            color=C_GOLD, stroke_width=1.5,
        )
        fb_left = Line(
            [right_x, above_y, 0],
            [optuna_top[0], above_y, 0],
            color=C_GOLD, stroke_width=1.5,
        )
        fb_down = Arrow(
            [optuna_top[0], above_y, 0],
            optuna_top + UP * 0.02,
            color=C_GOLD, stroke_width=1.5, buff=0,
            max_tip_length_to_length_ratio=0.5,
        )
        report_lbl = Text("mean(MAE) = 6.36", font_size=11, color=C_GOLD)
        report_lbl.move_to([2.5, above_y + 0.13, 0])

        self.play(Create(fb_right), run_time=0.4)
        self.play(Create(fb_up), run_time=0.5)
        self.play(Create(fb_left), FadeIn(report_lbl), run_time=0.7)
        self.play(GrowArrow(fb_down), run_time=0.5)
        self.wait(0.8)

        self.wait(2.5)
