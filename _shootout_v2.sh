#!/bin/bash
set -u
OUT_ROOT="output/model_shootout"
PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"
IMG="colmap/dense/images.orig_with_chair/DSCF4907.jpg"
MASK="output/segmented_room_v9a_fp_v2/masks/DSCF4907__blue_armchair.png"
REMOVE="Remove the blue armchair and teddy bear from the image while preserving the background and remaining elements, maintaining realism and original details."
REPLACE="Replace the blue armchair with a large potted plant with tall broad leaves in a black ceramic pot, indoor plant, preserve surrounding scene."

# --- FLUX Kontext WITHOUT mask (instruct-only), guidance 4.0 ---
mkdir -p "$OUT_ROOT/flux_kontext"
echo "===== FLUX Kontext (no-mask, instruct) ====="
echo "-- Test A: REMOVE --"
"$PY" _shootout_run_one.py --model-family flux_kontext \
  --model-id black-forest-labs/FLUX.1-Kontext-dev \
  --image "$IMG" --prompt "$REMOVE" \
  --out "$OUT_ROOT/flux_kontext/remove_v2.jpg" \
  --steps 28 --guidance 4.0 --long-edge 1024 --seed 42 --use-nf4 || echo "kontext-remove FAILED"
echo "-- Test B: REPLACE --"
"$PY" _shootout_run_one.py --model-family flux_kontext \
  --model-id black-forest-labs/FLUX.1-Kontext-dev \
  --image "$IMG" --prompt "$REPLACE" \
  --out "$OUT_ROOT/flux_kontext/replace_v2.jpg" \
  --steps 28 --guidance 4.0 --long-edge 1024 --seed 42 --use-nf4 || echo "kontext-replace FAILED"

# --- Qwen-Image-Edit-2511 (fixed: nf4 transformer + text encoder + cpu_offload) ---
mkdir -p "$OUT_ROOT/qwen_edit"
echo "===== Qwen-Image-Edit-2511 ====="
echo "-- Test A: REMOVE --"
"$PY" _shootout_run_one.py --model-family qwen_edit \
  --model-id Qwen/Qwen-Image-Edit-2511 \
  --image "$IMG" --prompt "$REMOVE" \
  --out "$OUT_ROOT/qwen_edit/remove.jpg" \
  --steps 28 --guidance 1.0 --long-edge 1024 --seed 42 --use-nf4 || echo "qwen-remove FAILED"
echo "-- Test B: REPLACE --"
"$PY" _shootout_run_one.py --model-family qwen_edit \
  --model-id Qwen/Qwen-Image-Edit-2511 \
  --image "$IMG" --prompt "$REPLACE" \
  --out "$OUT_ROOT/qwen_edit/replace.jpg" \
  --steps 28 --guidance 1.0 --long-edge 1024 --seed 42 --use-nf4 || echo "qwen-replace FAILED"

# --- FireRed-Image-Edit-1.1 (same architecture, same fix) ---
mkdir -p "$OUT_ROOT/fired_edit"
echo "===== FireRed-Image-Edit-1.1 ====="
echo "-- Test A: REMOVE --"
"$PY" _shootout_run_one.py --model-family fired_edit \
  --model-id FireRedTeam/FireRed-Image-Edit-1.1 \
  --image "$IMG" --prompt "$REMOVE" \
  --out "$OUT_ROOT/fired_edit/remove.jpg" \
  --steps 28 --guidance 1.0 --long-edge 1024 --seed 42 --use-nf4 || echo "fired-remove FAILED"
echo "-- Test B: REPLACE --"
"$PY" _shootout_run_one.py --model-family fired_edit \
  --model-id FireRedTeam/FireRed-Image-Edit-1.1 \
  --image "$IMG" --prompt "$REPLACE" \
  --out "$OUT_ROOT/fired_edit/replace.jpg" \
  --steps 28 --guidance 1.0 --long-edge 1024 --seed 42 --use-nf4 || echo "fired-replace FAILED"

echo "===== ALL TESTS DONE ====="
