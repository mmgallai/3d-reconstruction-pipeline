#!/bin/bash
# Runs the entire post-reconstruction chain on V32 to produce the final
# pipeline_v32_full_scratch_run output. Called as one background job so
# we don't have to chain task notifications across many short jobs.
set -e
export PYTHONIOENCODING=utf-8

cd "/c/Users/mgallai/Projects/3d_automated/reconstruction_project"
PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"
PROMPTS_SPACES="white water bottle,blue box,red lobster figurine"
PROMPTS_SLUGS="white_water_bottle,blue_box,red_lobster_figurine"

echo "========================================================================"
echo "STEP 1/7: scene_segmenter.pipeline (SAM3 voting + per-face scores)"
echo "========================================================================"
"$PY" -m scene_segmenter.pipeline \
  --mesh output/mesh_v32_data3/mesh_v32_data3_openmvs.ply \
  --atlas output/mesh_v32_data3/scene_textured0.png \
  --project-root . \
  --prompts "$PROMPTS_SPACES" \
  --output-dir output/segmented_v32_data3_v9a_fp_v2

echo "========================================================================"
echo "STEP 2/7: _spatial_crop.py (v9a_fp_v2 per-object mesh extraction)"
echo "========================================================================"
"$PY" _spatial_crop.py --mode a \
  --output-dir output/segmented_v32_data3_v9a_fp_v2 \
  --crop-style footprint \
  --xz-dilate-cm 0.5 \
  --y-margin-cm 0.5 \
  --seed-threshold 0.7 \
  --no-remainder

echo "========================================================================"
echo "STEP 3/7: _spatial_crop_splat.py (v5_shcolor per-object splat)"
echo "========================================================================"
"$PY" _spatial_crop_splat.py \
  --splat output/splat_v32_data3_noinit_pruned.ply \
  --dataparser nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json \
  --segmented-dir output/segmented_v32_data3_v9a_fp_v2 \
  --prompts "$PROMPTS_SPACES" \
  --output-dir output/segmented_v32_data3_v9a_fp_v2_splat_v5_shcolor \
  --crop-style footprint --xz-dilate-cm 0.5 --y-margin-cm 0.5 \
  --exclude-color "124,113,71" --color-tolerance 18 --sh-view-dir "0,0,1"

echo "========================================================================"
echo "STEP 4/7: _clean_splat.py (v6_cleaned via Clean-GS prune)"
echo "========================================================================"
"$PY" _clean_splat.py

echo "========================================================================"
echo "STEP 5/7: _unseen_core_map.py (--auto-desk plane fit + unseen mask)"
echo "========================================================================"
"$PY" _unseen_core_map.py --auto-desk

echo "========================================================================"
echo "STEP 6/7: _seed_desk_patch.py (NN-copy desk-patch Gaussians)"
echo "========================================================================"
"$PY" _seed_desk_patch.py

echo "========================================================================"
echo "STEP 7/7: _pipeline_full.py (assemble final deliverable)"
echo "========================================================================"
"$PY" _pipeline_full.py --skip-reconstruction \
  --out-root output/pipeline_v32_full_scratch_run

echo "========================================================================"
echo "ALL DONE."
echo "========================================================================"
ls -la output/pipeline_v32_full_scratch_run/
echo "---"
ls -la output/pipeline_v32_full_scratch_run/mesh/
echo "---"
ls -la output/pipeline_v32_full_scratch_run/splat/
