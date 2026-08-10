#!/bin/bash
# 5 models × 8 images = 40 edits for the to_edit dataset.
# Each output goes to output/to_edit/outputs/<model>/<image_stem>.jpg
set -u

OUT_ROOT="output/to_edit/outputs"
IMG_DIR="output/to_edit/images"
MASK_DIR="output/to_edit/masks"
PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"

# Per-image removal-instruction prompts (used by Kontext no-mask, Qwen, FireRed).
# These are natural-language edit instructions.
declare -A INSTRUCT_PROMPT
INSTRUCT_PROMPT["data2_a"]="Remove the white water bottle, the blue Arducam box, and the red plastic lobster toy from the desk. Show only the empty desk surface where they were."
INSTRUCT_PROMPT["data2_b"]="Remove the white water bottle, the blue Arducam box, and the red plastic lobster toy from the desk. Show only the empty desk surface where they were."
INSTRUCT_PROMPT["data4_a"]="Remove the yellow tape measure, the small green artificial plant, and the tan cardboard box from the desk. Show only the empty desk surface where they were."
INSTRUCT_PROMPT["data4_b"]="Remove the yellow tape measure and the green artificial plant in the black pot from the desk. Show only the empty desk surface where they were."
INSTRUCT_PROMPT["data5_a"]="Remove the blue armchair and the cow-print teddy bear pillow from the room. Show the empty floor and curtains behind them."
INSTRUCT_PROMPT["data5_b"]="Remove the blue armchair and the cow-print teddy bear pillow from the room. Show the empty floor and curtains behind them."
INSTRUCT_PROMPT["data6_a"]="Remove the pair of tan slippers from the rug. Show the empty rug where they were."
INSTRUCT_PROMPT["data6_b"]="Remove the pair of tan slippers from the rug. Show the empty rug where they were."

# Per-image positive-fill prompts (used by FLUX Fill dev — describes what SHOULD be there).
declare -A FILL_PROMPT
FILL_PROMPT["data2_a"]="clean empty cream-colored office desk surface, uniform tan laminate, cubicle wall background, no objects, no shadows"
FILL_PROMPT["data2_b"]="clean empty cream-colored office desk surface, uniform tan laminate, cubicle wall background, no objects, no shadows"
FILL_PROMPT["data4_a"]="clean empty cream-colored office desk surface, uniform tan laminate, cubicle wall background, no objects, no shadows"
FILL_PROMPT["data4_b"]="clean empty cream-colored office desk surface, uniform tan laminate, cubicle wall background, no objects, no shadows"
FILL_PROMPT["data5_a"]="empty polished pine hardwood floor, grey linen curtains, warm indoor tungsten lighting, photorealistic living room, no chair, no pillow, no shadows"
FILL_PROMPT["data5_b"]="empty polished pine hardwood floor, grey linen curtains, warm indoor tungsten lighting, photorealistic living room, no chair, no pillow, no shadows"
FILL_PROMPT["data6_a"]="clean empty grey rug with dark grey square accent, warm indoor lighting, photorealistic living room floor, no slippers, no shoes"
FILL_PROMPT["data6_b"]="clean empty grey rug with dark grey square accent, warm indoor lighting, photorealistic living room floor, no slippers, no shoes"

STEMS=("data2_a" "data2_b" "data4_a" "data4_b" "data5_a" "data5_b" "data6_a" "data6_b")

mkdir -p "$OUT_ROOT"/{lama,flux_fill,flux_kontext,qwen_edit,fired_edit}

# --- LaMa: mask + no prompt (fastest, native res) ---
echo "========================================"
echo "MODEL 1/5: LaMa (mask only, native res)"
echo "========================================"
for stem in "${STEMS[@]}"; do
  echo "-- $stem --"
  "$PY" _shootout_run_one.py --model-family lama \
    --image "$IMG_DIR/$stem.jpg" \
    --mask  "$MASK_DIR/${stem}__union.png" \
    --prompt "n/a" \
    --out "$OUT_ROOT/lama/$stem.jpg" \
    --steps 20 || echo "lama-$stem FAILED"
done

# --- FLUX Fill: mask + positive fill prompt ---
echo "========================================"
echo "MODEL 2/5: FLUX Fill dev (mask + fill prompt, nf4)"
echo "========================================"
for stem in "${STEMS[@]}"; do
  echo "-- $stem --"
  "$PY" _shootout_run_one.py --model-family flux_fill \
    --model-id black-forest-labs/FLUX.1-Fill-dev \
    --image "$IMG_DIR/$stem.jpg" \
    --mask  "$MASK_DIR/${stem}__union.png" \
    --prompt "${FILL_PROMPT[$stem]}" \
    --out "$OUT_ROOT/flux_fill/$stem.jpg" \
    --steps 28 --guidance 30 --long-edge 1024 --seed 42 --use-nf4 || echo "flux_fill-$stem FAILED"
done

# --- FLUX Kontext (no-mask, instruct) ---
echo "========================================"
echo "MODEL 3/5: FLUX Kontext dev (instruct, no mask, nf4)"
echo "========================================"
for stem in "${STEMS[@]}"; do
  echo "-- $stem --"
  "$PY" _shootout_run_one.py --model-family flux_kontext \
    --model-id black-forest-labs/FLUX.1-Kontext-dev \
    --image "$IMG_DIR/$stem.jpg" \
    --prompt "${INSTRUCT_PROMPT[$stem]}" \
    --out "$OUT_ROOT/flux_kontext/$stem.jpg" \
    --steps 28 --guidance 4.0 --long-edge 1024 --seed 42 --use-nf4 || echo "flux_kontext-$stem FAILED"
done

# --- Qwen-Image-Edit-2511 (instruct, no mask) ---
echo "========================================"
echo "MODEL 4/5: Qwen-Image-Edit-2511 (instruct, nf4 both, cpu_offload)"
echo "========================================"
for stem in "${STEMS[@]}"; do
  echo "-- $stem --"
  "$PY" _shootout_run_one.py --model-family qwen_edit \
    --model-id Qwen/Qwen-Image-Edit-2511 \
    --image "$IMG_DIR/$stem.jpg" \
    --prompt "${INSTRUCT_PROMPT[$stem]}" \
    --out "$OUT_ROOT/qwen_edit/$stem.jpg" \
    --steps 28 --guidance 1.0 --long-edge 1024 --seed 42 --use-nf4 || echo "qwen-$stem FAILED"
done

# --- FireRed-Image-Edit-1.1 ---
echo "========================================"
echo "MODEL 5/5: FireRed-Image-Edit-1.1 (instruct, nf4 both, cpu_offload)"
echo "========================================"
for stem in "${STEMS[@]}"; do
  echo "-- $stem --"
  "$PY" _shootout_run_one.py --model-family fired_edit \
    --model-id FireRedTeam/FireRed-Image-Edit-1.1 \
    --image "$IMG_DIR/$stem.jpg" \
    --prompt "${INSTRUCT_PROMPT[$stem]}" \
    --out "$OUT_ROOT/fired_edit/$stem.jpg" \
    --steps 28 --guidance 1.0 --long-edge 1024 --seed 42 --use-nf4 || echo "fired-$stem FAILED"
done

echo "========================================"
echo "ALL 40 EDITS DONE"
echo "========================================"
ls -R "$OUT_ROOT"
