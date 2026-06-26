"""
Escena 2 — β-VAE: Layout horizontal.

  FC Original  →  Encoder (6670→512)  →  Distribución (μ, σ → z)  →  Decoder  →  FC Reconstruida

El espacio latente se representa como distribución q(z|x): el encoder produce μ y σ,
z se muestrea como z = μ + σ·ε (como en la figura del paper).
"""
from manim import *
import numpy as np
from common import (
    BG, C_BLUE, C_VAE, C_LAT, C_TEXT, C_DIM, C_GOLD,
    h_trap_connector, feat_cell, BASE,
)

# --- Rutas de imágenes ---
# Usar paneles del MISMO sujeto (original + reconstruida). Para una reconstrucción
# con buena r, ejecutar: python Code/scripts/pick_best_recon.py
RECON_DIR = BASE / "Outputs/figures/vae_recons"
IMG_FC_ORIG  = str(RECON_DIR / "panel_original.png")
IMG_FC_RECON = str(RECON_DIR / "panel_recon.png")
BEST_R_FILE  = RECON_DIR / "best_r.txt"


class BetaVAEScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # --- Header ---
        header = Text(
            "β-VAE: Compresión de Conectividad Funcional",
            font_size=24, color=C_TEXT,
        )
        header.to_edge(UP, buff=0.28)
        ul = Underline(header, color=C_VAE, stroke_width=1.5)
        self.play(Write(header), Create(ul), run_time=1.4)

        # --- IMÁGENES DE FC ---
        # FC original (izquierda) — IMG_FC tiene ratio 1.215
        fc_orig = ImageMobject(IMG_FC_ORIG)
        fc_orig.set_height(2.6)
        fc_orig.move_to([-4.8, 0.2, 0])

        fc_orig_lbl = Text("FC original", font_size=13, color=C_BLUE)
        fc_orig_lbl.next_to(fc_orig, DOWN, buff=0.15)

        # FC reconstruida (derecha) — panel cropeado 800×600 (ratio 1.333)
        fc_recon = ImageMobject(IMG_FC_RECON)
        fc_recon.set_height(2.6)
        fc_recon.move_to([4.8, 0.2, 0])

        fc_recon_box = SurroundingRectangle(fc_recon, color=C_DIM, stroke_width=1.0, buff=0.08)
        r_str = BEST_R_FILE.read_text().strip() if BEST_R_FILE.exists() else None
        fc_recon_lbl = Text(
            f"FC reconstruida\n(Pearson r = {r_str})" if r_str else "FC reconstruida",
            font_size=13, color=C_DIM,
        )
        fc_recon_lbl.next_to(fc_recon, DOWN, buff=0.15)

        # --- BARRAS DEL VAE (verticales, centradas en y=0.2) ---
        # Encoder: decreciente de izquierda a derecha
        # Decoder: creciente de izquierda a derecha (atenuado)
        BAR_W = 0.55
        Y0    = 0.2   # y centro de todas las barras

        def vbar(x, height, color, fo=0.28, sw=2.0, label="", fs=13, dim=False):
            """Barra vertical centrada en (x, Y0)."""
            r = Rectangle(
                width=BAR_W, height=height,
                fill_color=color, fill_opacity=fo,
                stroke_color=color, stroke_width=sw,
            )
            r.move_to([x, Y0, 0])
            txt_color = C_DIM if dim else C_TEXT
            t = Text(label, font_size=fs, color=txt_color)
            t.move_to(r)
            return VGroup(r, t), r   # (grupo completo, solo el rect para trapecios)

        # Encoder
        g_in,  r_in  = vbar(-2.2, 2.8, C_BLUE,  fo=0.30, sw=2.0, label="6,670", fs=12)
        g_enc, r_enc = vbar(-1.1, 1.7, C_VAE,   fo=0.30, sw=2.0, label="512",   fs=13)

        # Bottleneck: solo μ, σ, z (círculos bien separados). Fórmula debajo; "64 dims" al fondo con flecha.
        r_z = RoundedRectangle(
            width=1.5, height=0.88,
            fill_color=C_LAT, fill_opacity=0.35,
            stroke_color=C_LAT, stroke_width=2.4, corner_radius=0.14,
        )
        r_z.move_to([0.0, Y0, 0])
        # Tres nodos: μ arriba, σ abajo (sin encimarse), z a la derecha
        circ_r = 0.15
        mu_circ = Circle(radius=circ_r, fill_color=C_GOLD, fill_opacity=0.5,
                         stroke_color=C_GOLD, stroke_width=2.0)
        sigma_circ = Circle(radius=circ_r, fill_color="#FFB74D", fill_opacity=0.5,
                            stroke_color="#FFB74D", stroke_width=2.0)
        z_circ = Circle(radius=circ_r, fill_color="#A5D6A7", fill_opacity=0.5,
                        stroke_color="#A5D6A7", stroke_width=2.0)
        mu_circ.move_to(r_z.get_center() + UP * 0.30 + LEFT * 0.36)
        sigma_circ.move_to(r_z.get_center() + DOWN * 0.26 + LEFT * 0.36)
        z_circ.move_to(r_z.get_center() + RIGHT * 0.36)
        lbl_mu = Text("μ", font_size=13, color=C_TEXT, weight=BOLD)
        lbl_sigma = Text("σ", font_size=13, color=C_TEXT, weight=BOLD)
        lbl_z = Text("z", font_size=13, color=C_TEXT, weight=BOLD)
        lbl_mu.move_to(mu_circ)
        lbl_sigma.move_to(sigma_circ)
        lbl_z.move_to(z_circ)
        arr_mu_z = Arrow(mu_circ.get_right(), z_circ.get_left(), buff=0.10,
                         color=C_DIM, stroke_width=1.6, max_tip_length_to_length_ratio=0.35)
        arr_sigma_z = Arrow(sigma_circ.get_right(), z_circ.get_left(), buff=0.10,
                            color=C_DIM, stroke_width=1.6, max_tip_length_to_length_ratio=0.35)
        g_z = VGroup(r_z, mu_circ, sigma_circ, z_circ, lbl_mu, lbl_sigma, lbl_z,
                     arr_mu_z, arr_sigma_z)
        # Fórmula debajo del bottleneck
        formula = Text("z = μ + σ·ε", font_size=13, color=C_TEXT, slant=ITALIC)
        formula.next_to(r_z, DOWN, buff=0.24)
        # 64 dims al fondo centro, con flecha desde el bottleneck (μ, σ, z son vectores de dim 64)
        dims_lbl = Text("μ, σ, z  →  64 dims", font_size=14, color=C_LAT, weight=BOLD)
        dims_lbl.move_to([0.0, -2.0, 0])
        arr_to_64 = Arrow(
            formula.get_bottom() + DOWN * 0.05,
            dims_lbl.get_top() + UP * 0.08,
            color=C_LAT, stroke_width=2.0, buff=0,
            max_tip_length_to_length_ratio=0.12,
        )

        # Decoder (atenuado)
        g_dec, r_dec = vbar( 1.1, 1.7, C_VAE,   fo=0.10, sw=1.2, label="512",   fs=12, dim=True)
        g_out, r_out = vbar( 2.2, 2.8, C_BLUE,  fo=0.08, sw=1.2, label="6,670", fs=12, dim=True)

        # --- Trapecios horizontales ---
        t1 = h_trap_connector(r_in,  r_enc, C_VAE,  0.22)
        t2 = h_trap_connector(r_enc, r_z,   C_VAE,  0.22)
        t3 = h_trap_connector(r_z,   r_dec, C_VAE,  0.07)
        t4 = h_trap_connector(r_dec, r_out, C_VAE,  0.05)

        # Brace β-VAE (solo figura). Tras fade-out del decoder usamos brace solo encoder+bottleneck.
        all_bars = VGroup(r_in, r_enc, r_z, r_dec, r_out)
        vae_brace = Brace(all_bars, UP, color=C_VAE, buff=0.30)
        vae_brace_lbl = Text("β-VAE", font_size=16, color=C_VAE, weight=BOLD)
        vae_brace_lbl.next_to(vae_brace, UP, buff=0.08)
        encoder_bars = VGroup(r_in, r_enc, r_z)
        vae_brace_short = Brace(encoder_bars, UP, color=C_VAE, buff=0.30)
        vae_brace_short_lbl = Text("β-VAE", font_size=16, color=C_VAE, weight=BOLD)
        vae_brace_short_lbl.next_to(vae_brace_short, UP, buff=0.08)

        glow = SurroundingRectangle(r_z, color=C_GOLD, stroke_width=3.0, buff=0.14,
                                    corner_radius=0.10)

        # --- Flechas FC → Encoder y Decoder → FC ---
        arr_in = Arrow(
            fc_orig.get_right() + RIGHT * 0.12,
            r_in.get_left() + LEFT * 0.12,
            color=C_BLUE, stroke_width=2.2, buff=0,
            max_tip_length_to_length_ratio=0.15,
        )
        arr_out = Arrow(
            r_out.get_right() + RIGHT * 0.12,
            fc_recon.get_left() + LEFT * 0.12,
            color=C_DIM, stroke_width=1.8, buff=0,
            max_tip_length_to_length_ratio=0.15,
        )
        # β note al pie (solo números y parámetros)
        beta_note = Text(
            "β = 0.0567   |   warmup = 73 épocas   |   Optuna (70 trials)   |   μ: 64 dims",
            font_size=12, color=C_DIM,
        )
        beta_note.to_edge(DOWN, buff=0.20)

        # --- Animación ---

        # 1. FC original aparece
        self.play(FadeIn(fc_orig, scale=0.88), FadeIn(fc_orig_lbl), run_time=1.2)
        self.play(GrowArrow(arr_in), run_time=0.9)

        # 2. Encoder (izq → derecha)
        self.play(FadeIn(g_in, shift=RIGHT * 0.15), run_time=0.8)
        self.play(Create(t1), run_time=0.6)
        self.play(FadeIn(g_enc, shift=RIGHT * 0.15), run_time=0.8)
        self.play(Create(t2), run_time=0.6)
        self.play(FadeIn(g_z, shift=RIGHT * 0.15), run_time=1.0)

        # 3. Highlight bottleneck + fórmula + flecha a "64 dims" al fondo
        self.play(Create(glow), FadeIn(formula), run_time=0.8)
        self.play(GrowArrow(arr_to_64), FadeIn(dims_lbl), run_time=0.8)
        self.wait(0.8)

        # 4. Brace β-VAE
        self.play(Create(vae_brace), FadeIn(vae_brace_lbl), run_time=0.9)

        # 5. Decoder (atenuado, left → right)
        self.play(Create(t3), FadeIn(g_dec), run_time=0.8)
        self.play(Create(t4), FadeIn(g_out), run_time=0.8)

        # 6. FC reconstruida aparece
        self.play(GrowArrow(arr_out), run_time=0.7)
        self.play(
            FadeIn(fc_recon, scale=0.88),
            Create(fc_recon_box),
            FadeIn(fc_recon_lbl),
            run_time=1.3,
        )
        self.wait(1.4)

        # 7. Transición: desaparece el decoder; extraemos solo μ (vector como en la escena siguiente)
        decoder_side = Group(t3, t4, g_dec, g_out, arr_out, fc_recon, fc_recon_box, fc_recon_lbl)
        self.play(
            FadeOut(decoder_side, shift=RIGHT * 0.5),
            run_time=1.5,
        )
        self.play(
            ReplacementTransform(vae_brace, vae_brace_short),
            ReplacementTransform(vae_brace_lbl, vae_brace_short_lbl),
            run_time=0.7,
        )
        self.wait(0.6)
        # Vector μ (64) más separado; flecha larga (sin texto explicativo)
        rng_mu = np.random.default_rng(88)
        vs = rng_mu.uniform(-0.8, 0.8, 4)
        mu_cells = VGroup(*[feat_cell(float(vs[i]), f"μ{i+1}", C_LAT) for i in range(3)])
        mu_cells.arrange(RIGHT, buff=0.08)
        mu_dots = Text("·  ·  ·", font_size=18, color=C_DIM)
        mu_last = feat_cell(float(vs[-1]), "μ64", C_LAT)
        mu_row = VGroup(mu_cells, mu_dots, mu_last)
        mu_row.arrange(RIGHT, buff=0.12)
        mu_row.move_to([3.4, Y0, 0])
        mu_brace = Brace(mu_row, DOWN, color=C_LAT, buff=0.06)
        mu_brace_lbl = Text("64", font_size=14, color=C_LAT)
        mu_brace_lbl.next_to(mu_brace, DOWN, buff=0.05)
        arr_to_mu = Arrow(
            r_z.get_right() + RIGHT * 0.15,
            mu_row.get_left() + LEFT * 0.15,
            color=C_GOLD, stroke_width=2.5, buff=0,
            max_tip_length_to_length_ratio=0.12,
        )
        self.play(GrowArrow(arr_to_mu), run_time=0.7)
        self.play(
            LaggedStart(*[FadeIn(c, shift=LEFT * 0.1) for c in [mu_cells, mu_dots, mu_last]],
                        lag_ratio=0.15),
            run_time=0.9,
        )
        self.play(Create(mu_brace), FadeIn(mu_brace_lbl), run_time=0.7)
        self.wait(1.0)

        # 8. β params al pie
        self.play(FadeIn(beta_note), run_time=0.8)

        self.wait(4.0)
