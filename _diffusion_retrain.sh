#!/bin/bash
# Refine a splat by retraining splatfacto-big on the edited images set.
# Usage: bash _diffusion_retrain.sh <scene> <model>
#   scene = data4 | room
#   model = kontext | qwen
# Writes: /c/Users/mgallai/Downloads/diffusion/<scene>/<model>/refined_scene.ply
set -eu

SCENE="$1"
MODEL="$2"
PROJECT="/c/Users/mgallai/Projects/3d_automated/reconstruction_project"
DIFF="/c/Users/mgallai/Downloads/diffusion"
DOCKER="/c/Program Files/Docker/Docker/resources/bin/docker"

# Prompts / paths
case "$SCENE" in
  data4)
    SRC_TRANSFORMS="$PROJECT/output/diffusion_prep/data4/transforms.json"
    ;;
  room)
    SRC_TRANSFORMS="$PROJECT/colmap/dense/transforms.json"
    ;;
  data3)
    SRC_TRANSFORMS="$PROJECT/output/data3_workspace/nerfstudio_data/transforms.json"
    ;;
  *) echo "unknown scene $SCENE"; exit 2 ;;
esac

case "$MODEL" in
  kontext|qwen) : ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac

# Prepare a fresh nerfstudio_data directory that points at edited images + shared transforms
STAGING="$PROJECT/output/diffusion_prep/${SCENE}/retrain_${MODEL}_ns"
mkdir -p "$STAGING/images"

echo "[retrain:$SCENE:$MODEL] copying transforms.json ..."
cp "$SRC_TRANSFORMS" "$STAGING/transforms.json"

echo "[retrain:$SCENE:$MODEL] hard-linking edited images (fast) ..."
SRC_IMG="$PROJECT/output/diffusion_prep/${SCENE}/edited_${MODEL}/images"
n_src=$(ls "$SRC_IMG" | wc -l)
echo "  source: $SRC_IMG ($n_src files)"
# Hard-link to avoid duplicating data
rm -f "$STAGING/images"/*
for f in "$SRC_IMG"/*; do
  b=$(basename "$f")
  ln "$f" "$STAGING/images/$b" 2>/dev/null || cp "$f" "$STAGING/images/$b"
done
n_dst=$(ls "$STAGING/images" | wc -l)
echo "  staged: $STAGING/images ($n_dst files)"

# If transforms.json has file_path prefixes like "images/DSCF4667.jpg", we need
# to keep them. Splatfacto will look under $STAGING/images/DSCF4667.jpg. OK.

# Docker path (Windows -> WSL2)
WORKSPACE=$(cygpath -m "$PROJECT")
DATA_REL="output/diffusion_prep/${SCENE}/retrain_${MODEL}_ns"

TS=$(date +%s)
OUT_DIR_REL="nerfstudio/${SCENE}_${MODEL}_${TS}"
mkdir -p "$PROJECT/$OUT_DIR_REL"

echo "[retrain:$SCENE:$MODEL] launching splatfacto-big (30k iters, native res, cache-images cpu, stop-split 12000) ..."
"$DOCKER" run --rm --gpus all --shm-size=4g \
  -v "$WORKSPACE":/workspace \
  nerfstudio-blackwell:latest \
  bash -c "cd /workspace && \
    python3 -c 'import torch; print(\"cuda:\", torch.cuda.is_available())' && \
    yes | ns-train splatfacto-big \
      --data /workspace/${DATA_REL} \
      --output-dir /workspace/${OUT_DIR_REL} \
      --vis tensorboard \
      --max-num-iterations 30000 \
      --steps-per-save 500 \
      --pipeline.datamanager.cache-images cpu \
      --pipeline.model.stop-split-at 12000 \
      --pipeline.model.cull-alpha-thresh 0.01 \
      nerfstudio-data \
      --downscale-factor 1" 2>&1

# Find the trained checkpoint + export splat (nerfstudio may write to splatfacto/ or splatfacto-big/)
CFG=$(ls "$PROJECT/${OUT_DIR_REL}"/*/splatfacto*/*/config.yml 2>/dev/null | tail -1)
if [ -z "$CFG" ]; then
  echo "[retrain:$SCENE:$MODEL] FATAL: no config.yml found under $PROJECT/${OUT_DIR_REL}"
  exit 1
fi
# $CFG is git-bash style (/c/Users/...). Strip $PROJECT prefix, keep the tail as the container-relative path.
CFG_REL=$(echo "$CFG" | sed "s|^$PROJECT/||")
echo "[retrain:$SCENE:$MODEL] exporting splat from $CFG_REL"
"$DOCKER" run --rm --gpus all -v "$WORKSPACE":/workspace \
  nerfstudio-blackwell:latest \
  bash -c "cd /workspace && python3 fix_weights.py && \
    ns-export gaussian-splat \
      --load-config /workspace/${CFG_REL} \
      --output-dir /workspace/output/diffusion_prep/${SCENE}/retrain_${MODEL}_export" 2>&1

# Copy the exported PLY to the Downloads deliverable
# Folder names in Downloads: kontext -> "flux", qwen -> "Qwen"
case "$MODEL" in
  kontext) DST_SUBDIR="flux" ;;
  qwen)    DST_SUBDIR="Qwen" ;;
  *)       DST_SUBDIR="$MODEL" ;;
esac
SRC_PLY="$PROJECT/output/diffusion_prep/${SCENE}/retrain_${MODEL}_export/splat.ply"
DST_PLY="$DIFF/${SCENE}/${DST_SUBDIR}/refined_scene.ply"
mkdir -p "$(dirname "$DST_PLY")"
cp "$SRC_PLY" "$DST_PLY"
echo "[retrain:$SCENE:$MODEL] DONE. splat -> $DST_PLY"
echo "  size: $(du -h "$DST_PLY" | cut -f1)"
