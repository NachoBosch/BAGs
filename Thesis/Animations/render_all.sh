#!/usr/bin/env bash
# render_all.sh — Renderiza todas las escenas y combina en un video final.
#
# Uso:
#   cd Thesis-comp-sci/Animations
#   bash render_all.sh
#
# Calidad:
#   -ql   480p  (rapido, para probar)
#   -qm   720p  (intermedio)
#   -qh   1080p (final)
#
# Cambia QUALITY según lo que necesites.

set -e
QUALITY="-qh"    # Cambia a -qh para la versión final
OUT_DIR="media/videos"

echo "Rendering Brain Age Pipeline — Manim"
echo "Calidad: $QUALITY"

# --- Escenas individuales ---
manim $QUALITY scene_01_fc.py         FCToFeatures       && echo "OK Scene 1: FC to Features"
manim $QUALITY scene_02_vae.py        BetaVAEScene       && echo "OK Scene 2: beta-VAE"
manim $QUALITY scene_03_multimodal.py MultimodalFusion   && echo "OK Scene 3: Multimodal Fusion"
manim $QUALITY scene_04_xgboost.py    XGBoostPrediction  && echo "OK Scene 4: XGBoost"
manim $QUALITY scene_05_optuna_vae.py OptunaVAEScene     && echo "OK Scene 5: Optuna VAE"
manim $QUALITY scene_06_optuna_xgb.py OptunaXGBScene     && echo "OK Scene 6: Optuna XGBoost"

# --- Pipeline completo ---
manim $QUALITY pipeline_full.py       FullPipeline       && echo "OK Full Pipeline"

# --- Combinar con ffmpeg ---
# Detectar la carpeta de salida según calidad
case "$QUALITY" in
  "-ql") RES="480p15" ;;
  "-qm") RES="720p30" ;;
  "-qh") RES="1080p60" ;;
  *) RES="720p30" ;;
esac

echo ""
echo "Combinando Optuna + escenas del pipeline con ffmpeg..."

# Crear lista de archivos (rutas relativas a Animations/)
LIST_FILE="media/concat_list.txt"
mkdir -p media
ANIM_DIR="$(cd "$(dirname "$0")" && pwd)"

cat > "$LIST_FILE" <<EOF
file '${ANIM_DIR}/media/videos/scene_05_optuna_vae/$RES/OptunaVAEScene.mp4'
file '${ANIM_DIR}/media/videos/scene_06_optuna_xgb/$RES/OptunaXGBScene.mp4'
file '${ANIM_DIR}/media/videos/scene_01_fc/$RES/FCToFeatures.mp4'
file '${ANIM_DIR}/media/videos/scene_02_vae/$RES/BetaVAEScene.mp4'
file '${ANIM_DIR}/media/videos/scene_03_multimodal/$RES/MultimodalFusion.mp4'
file '${ANIM_DIR}/media/videos/scene_04_xgboost/$RES/XGBoostPrediction.mp4'
EOF

COMBINED="media/brain_age_pipeline_combined.mp4"
ffmpeg -y -f concat -safe 0 -i "$LIST_FILE" -c copy "$COMBINED" \
  && echo "OK Video combinado: $COMBINED" \
  || echo "ERROR: No se pudo combinar (verifica que los archivos existan)"

echo ""
echo "Pipeline completo: media/videos/pipeline_full/$RES/FullPipeline.mp4"
echo "Video combinado:   $COMBINED"
