#!/bin/bash
# Resume the data4 post-chain from STEP 5 (steps 1-3 already succeeded,
# step 4 = Clean-GS failed and is skipped; we use v5_shcolor per-object
# splats as the input to step 7 instead of v6_cleaned).
set -e
export PYTHONIOENCODING=utf-8
cd "/c/Users/mgallai/Projects/3d_automated/reconstruction_project"
PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"
PROMPTS_SLUGS="tape_measure,potted_artificial_plant,cardboard_box"

DATAPARSER="$(ls -td nerfstudio/dense/splatfacto/*/ | head -1)dataparser_transforms.json"
echo "Using dataparser: ${DATAPARSER}"

SEGMENTED_DIR="output/segmented_data4_v9a_fp_v2"
SPLAT_V5_DIR="output/segmented_data4_v9a_fp_v2_splat_v5_shcolor"
SAM3_MASK_CACHE="${SEGMENTED_DIR}/masks"
UNSEEN_DIR="output/unseen_core_data4"
PATCH_DIR="output/desk_patch_data4"
SCENE_MESH="output/mesh_v35/mesh_v35_openmvs.ply"
SCENE_MESH_TEX="output/mesh_v35/scene_textured0.png"
SCENE_SPLAT="output/splat_v35_noinit_pruned.ply"

echo "========================================================================"
echo "STEP 5/7: _unseen_core_map.py (--auto-desk plane fit + unseen mask)"
echo "========================================================================"
"$PY" _unseen_core_map.py --auto-desk \
  --scene-mesh "${SCENE_MESH}" \
  --segmented-dir "${SEGMENTED_DIR}" \
  --project-root . \
  --prompts "$PROMPTS_SLUGS" \
  --out-dir "${UNSEEN_DIR}"

echo "========================================================================"
echo "STEP 6/7: _seed_desk_patch.py (K-NN clone desk-patch Gaussians)"
echo "========================================================================"
"$PY" _seed_desk_patch.py \
  --project-root . \
  --scene-mesh "${SCENE_MESH}" \
  --dataparser "${DATAPARSER}" \
  --unseen-dir "${UNSEEN_DIR}" \
  --prompts "$PROMPTS_SLUGS" \
  --out-dir "${PATCH_DIR}" \
  --sam3-cache "${SAM3_MASK_CACHE}" \
  --scene-splat "${SCENE_SPLAT}" \
  --object-mesh-dir "${SEGMENTED_DIR}"

echo "========================================================================"
echo "STEP 7/7: _pipeline_full.py (assemble final deliverable — uses v5_shcolor)"
echo "========================================================================"
"$PY" _pipeline_full.py --skip-reconstruction \
  --scene-name data4 \
  --prompts "$PROMPTS_SLUGS" \
  --dataparser "${DATAPARSER}" \
  --scene-splat "${SCENE_SPLAT}" \
  --scene-mesh "${SCENE_MESH}" \
  --scene-mesh-texture "${SCENE_MESH_TEX}" \
  --object-mesh-dir "${SEGMENTED_DIR}" \
  --object-splat-dir "${SPLAT_V5_DIR}" \
  --patch-splat-dir "${PATCH_DIR}" \
  --unseen-dir "${UNSEEN_DIR}" \
  --out-root output/pipeline_data4_full_run

echo "========================================================================"
echo "ALL DONE."
echo "========================================================================"
ls -la output/pipeline_data4_full_run/
