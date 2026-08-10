#!/bin/bash
# Data3 fresh reconstruction:
#  1. COLMAP SfM (colmap docker) -> sparse/0/
#  2. nerfstudio conversion COLMAP -> transforms.json
#  3. splatfacto-big training
#  4. Export splat.ply
set -eu
PROJECT="/c/Users/mgallai/Projects/3d_automated/reconstruction_project"
DIFF="/c/Users/mgallai/Downloads/diffusion"
DOCKER="/c/Program Files/Docker/Docker/resources/bin/docker"
WORKSPACE=$(cygpath -m "$PROJECT")

cd "$PROJECT"

# Container paths (Linux inside container)
CONT_WS="/workspace"
NS_DATA="$CONT_WS/output/data3_workspace/nerfstudio_data"
IMG_DIR="$NS_DATA/images"
COLMAP_DIR="$NS_DATA/colmap"

# =============================
# STAGE 1: COLMAP SfM
# =============================
echo "===== STAGE 1: COLMAP SfM (feature extractor + matcher + mapper) ====="
mkdir -p output/data3_workspace/nerfstudio_data/colmap/sparse
mkdir -p output/data3_workspace/nerfstudio_data/colmap/database

DB_PATH="$COLMAP_DIR/database.db"
rm -f "$PROJECT/output/data3_workspace/nerfstudio_data/colmap/database.db"

"$DOCKER" run --rm --gpus all -v "$WORKSPACE":$CONT_WS colmap/colmap:latest bash -c "
  set -e
  # 1a. Feature extraction (SIFT, GPU on by default)
  colmap feature_extractor \
    --database_path $DB_PATH \
    --image_path $IMG_DIR \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE
  echo '=== features extracted ==='

  # 1b. Exhaustive matching (fine for 140 images)
  colmap exhaustive_matcher \
    --database_path $DB_PATH
  echo '=== matching done ==='

  # 1c. SfM (mapper)
  colmap mapper \
    --database_path $DB_PATH \
    --image_path $IMG_DIR \
    --output_path $COLMAP_DIR/sparse
  echo '=== mapper done ==='

  ls -la $COLMAP_DIR/sparse
" 2>&1

echo "===== STAGE 1 DONE ====="
ls output/data3_workspace/nerfstudio_data/colmap/sparse/

# =============================
# STAGE 2: COLMAP -> nerfstudio transforms.json
# =============================
echo "===== STAGE 2: convert COLMAP -> nerfstudio transforms.json ====="
"$DOCKER" run --rm -v "$WORKSPACE":$CONT_WS nerfstudio-blackwell:latest bash -c "
  cd $CONT_WS
  python3 -c \"
from pathlib import Path
from nerfstudio.process_data.colmap_utils import colmap_to_json
n = colmap_to_json(
    recon_dir=Path('$COLMAP_DIR/sparse/0'),
    output_dir=Path('$NS_DATA'),
    image_id_to_depth_path=None,
    camera_mask_path=None,
    image_rename_map=None,
    keep_original_world_coordinate=False,
)
print(f'wrote transforms.json with {n} cameras')
\"
" 2>&1
echo "===== STAGE 2 DONE ====="

# =============================
# STAGE 3: splatfacto-big training
# =============================
echo "===== STAGE 3: splatfacto-big training ====="
"$DOCKER" run --rm --gpus all --shm-size=4g -v "$WORKSPACE":$CONT_WS nerfstudio-blackwell:latest \
  bash -c "cd $CONT_WS && python3 fix_weights.py && \
    yes | ns-train splatfacto-big \
      --data $NS_DATA \
      --output-dir $CONT_WS/output/data3_workspace/nerfstudio \
      --vis tensorboard \
      --max-num-iterations 30000 \
      --steps-per-save 500 \
      --pipeline.datamanager.cache-images cpu \
      --pipeline.model.stop-split-at 12000 \
      --pipeline.model.cull-alpha-thresh 0.01 \
      nerfstudio-data \
      --downscale-factor 1" 2>&1
echo "===== STAGE 3 DONE ====="

# =============================
# STAGE 4: export splat.ply
# =============================
echo "===== STAGE 4: export splat.ply ====="
CFG=$(ls "$PROJECT/output/data3_workspace/nerfstudio"/*/splatfacto*/*/config.yml 2>/dev/null | tail -1)
if [ -z "$CFG" ]; then
  echo "FATAL: no config.yml"; exit 1
fi
CFG_REL=$(echo "$CFG" | sed "s|^$PROJECT/||")
mkdir -p output/data3_workspace/splat_export
"$DOCKER" run --rm --gpus all -v "$WORKSPACE":$CONT_WS nerfstudio-blackwell:latest \
  bash -c "cd $CONT_WS && python3 fix_weights.py && \
    ns-export gaussian-splat \
      --load-config $CONT_WS/${CFG_REL} \
      --output-dir $CONT_WS/output/data3_workspace/splat_export" 2>&1

cp output/data3_workspace/splat_export/splat.ply "$DIFF/data3/full_scene.ply"
cp output/data3_workspace/splat_export/splat.ply "$DIFF/data3/cleaned_scene.ply"
echo "===== STAGE 4 DONE ====="
ls -la "$DIFF/data3/"
