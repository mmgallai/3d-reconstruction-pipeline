#!/bin/bash
# Room v1 extraction chain: extract 3 objects + scene_without_objects.
# Assumes prerequisites are done:
#   - nerfstudio_data/images/ = 311 room JPGs (1556x1038)
#   - nerfstudio_data/femto_intrinsics.json = room PINHOLE intrinsics
#   - colmap/dense/transforms.json = 311 frames w/ OPENCV intrinsics
#   - nerfstudio/dense/splatfacto/*/dataparser_transforms.json = ROOM dataparser
#   - output/mesh_v37/mesh_v37_openmvs.ply + scene_textured0.png = room mesh
#   - output/splat_v37_da3_pruned.ply = room splat
#   - output/segmented_room_v9a_fp_v2/masks/ = 311*3 = 933 SAM3 PNGs
#
# Differences from data4 chain:
#   - No --exclude-color (data4-specific desk-brown)
#   - No Clean-GS step (broken on data4; use v5 splats directly for scene_without)
#   - Slugs are underscored versions of prompts
set -e
export PYTHONIOENCODING=utf-8

cd "/c/Users/mgallai/Projects/3d_automated/reconstruction_project"
PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"

PROMPTS_SPACES="blue armchair,wooden coffee table,black subwoofer"
PROMPTS_SLUGS="blue_armchair,wooden_coffee_table,black_subwoofer"

# Use extraction-mode dataparser (scale=1.0) because splat positions in this
# splatfacto config are in COLMAP-world, not dp-normalized. Confirmed via
# empirical test: (splat_xyz - t) @ R matches mesh_v37 bbox to ~10%.
DATAPARSER="nerfstudio/dataparser_transforms_extraction.json"
echo "Using dataparser: ${DATAPARSER}"
if [ ! -f "${DATAPARSER}" ]; then
  echo "FATAL: dataparser not found"
  exit 1
fi

# --- Paths --------------------------------------------------------------------
SEGMENTED_DIR="output/segmented_room_v9a_fp_v2"
SPLAT_V5_DIR="output/segmented_room_v9a_fp_v2_splat_v5_shcolor"
SAM3_MASK_CACHE="${SEGMENTED_DIR}/masks"
UNSEEN_DIR="output/unseen_core_room"
PATCH_DIR="output/desk_patch_room"
SCENE_MESH="output/mesh_v37/mesh_v37_openmvs.ply"
SCENE_MESH_TEX="output/mesh_v37/scene_textured0.png"
SCENE_SPLAT="output/splat_v37_da3_pruned.ply"
OUT_ROOT="output/july/room_v1"

mkdir -p "${OUT_ROOT}"

# --- STEP 1: scene_segmenter.pipeline (SAM3 already cached → skip SAM3, run votes)
echo "========================================================================"
echo "STEP 1/6: scene_segmenter.pipeline (uses cached SAM3 masks)"
echo "========================================================================"
"$PY" -m scene_segmenter.pipeline \
  --mesh "${SCENE_MESH}" \
  --atlas "${SCENE_MESH_TEX}" \
  --project-root . \
  --prompts "$PROMPTS_SPACES" \
  --output-dir "${SEGMENTED_DIR}"

# --- STEP 2 SKIPPED: scene_segmenter.pipeline already produced per-object
#    <slug>_extracted.{obj,ply,glb,collider.json}. _spatial_crop.py's
#    PROMPTS constant is hardcoded and it doesn't accept --prompts anyway.

# --- STEP 3: _spatial_crop_splat.py (v5_shcolor per-object splat) -------------
echo "========================================================================"
echo "STEP 3/6: _spatial_crop_splat.py (v5_shcolor per-object splat)"
echo "========================================================================"
"$PY" _spatial_crop_splat.py \
  --splat "${SCENE_SPLAT}" \
  --dataparser "${DATAPARSER}" \
  --segmented-dir "${SEGMENTED_DIR}" \
  --prompts "$PROMPTS_SPACES" \
  --output-dir "${SPLAT_V5_DIR}" \
  --crop-style footprint --xz-dilate-cm 0.5 --y-margin-cm 0.5

# --- STEP 4: _unseen_core_map.py (plane fit + unseen mask) --------------------
echo "========================================================================"
echo "STEP 4/6: _unseen_core_map.py (--auto-desk)"
echo "========================================================================"
"$PY" _unseen_core_map.py --auto-desk \
  --scene-mesh "${SCENE_MESH}" \
  --segmented-dir "${SEGMENTED_DIR}" \
  --project-root . \
  --prompts "$PROMPTS_SLUGS" \
  --out-dir "${UNSEEN_DIR}"

# --- STEP 5: _seed_desk_patch.py (K-NN clone patch Gaussians) -----------------
echo "========================================================================"
echo "STEP 5/6: _seed_desk_patch.py (clone mode)"
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

# --- STEP 6: _pipeline_full.py (assemble final deliverable) -------------------
echo "========================================================================"
echo "STEP 6/6: _pipeline_full.py --skip-reconstruction"
echo "========================================================================"
"$PY" _pipeline_full.py --skip-reconstruction \
  --scene-name room_v1 \
  --prompts "$PROMPTS_SLUGS" \
  --dataparser "${DATAPARSER}" \
  --scene-splat "${SCENE_SPLAT}" \
  --scene-mesh "${SCENE_MESH}" \
  --scene-mesh-texture "${SCENE_MESH_TEX}" \
  --object-mesh-dir "${SEGMENTED_DIR}" \
  --object-splat-dir "${SPLAT_V5_DIR}" \
  --patch-splat-dir "${PATCH_DIR}" \
  --unseen-dir "${UNSEEN_DIR}" \
  --out-root "${OUT_ROOT}"

echo "========================================================================"
echo "ALL DONE."
echo "========================================================================"
ls -la "${OUT_ROOT}/"
echo "---"
ls -la "${OUT_ROOT}/splat/" 2>/dev/null || true
echo "---"
ls -la "${OUT_ROOT}/mesh/" 2>/dev/null || true
