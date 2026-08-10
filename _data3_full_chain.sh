#!/bin/bash
# Chain for data3: Kontext edit -> Kontext retrain -> Qwen edit -> Qwen retrain -> copy artifacts
set -eu
PROJECT="/c/Users/mgallai/Projects/3d_automated/reconstruction_project"
DIFF="/c/Users/mgallai/Downloads/diffusion"
PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN env var before running}"

cd "$PROJECT"

# STAGE 1: Kontext editing (140 views ~ 4hr)
echo "[data3-chain] STAGE 1: Kontext editing ..."
"$PY" _diffusion_edit_persistent.py --scene data3 --model kontext > _diffusion_edit_data3_kontext.log 2>&1

# STAGE 2: copy edited images + prompts to Downloads
echo "[data3-chain] STAGE 2: copy Kontext deliverables ..."
mkdir -p "$DIFF/data3/flux/edited_images" "$DIFF/data3/flux/masks_or_prompts"
for f in output/diffusion_prep/data3/edited_kontext/images/*; do
  ln "$f" "$DIFF/data3/flux/edited_images/$(basename $f)" 2>/dev/null || cp "$f" "$DIFF/data3/flux/edited_images/$(basename $f)"
done
cat > "$DIFF/data3/flux/masks_or_prompts/prompt.txt" <<'PROMPT'
FLUX.1 Kontext dev (no-mask, instruction-only, nf4 quantized)
guidance_scale=4.0, num_inference_steps=28, seed=42, long_edge=1024

Instruction: "Remove the white water bottle, the blue Arducam box, and the red plastic lobster toy from the desk. Show only the empty desk surface where they were."

Note: All 140/140 views were edited (every view contained target objects per SAM3 filter with >=0.5% coverage).
PROMPT

# STAGE 3: Kontext splatfacto retrain (~30 min)
echo "[data3-chain] STAGE 3: Kontext splatfacto retrain ..."
bash _diffusion_retrain.sh data3 kontext > _diffusion_retrain_data3_kontext.log 2>&1

# STAGE 4: Qwen editing (140 views ~ 7hr)
echo "[data3-chain] STAGE 4: Qwen editing ..."
"$PY" _diffusion_edit_persistent.py --scene data3 --model qwen > _diffusion_edit_data3_qwen.log 2>&1

# STAGE 5: copy Qwen edited images + prompts
echo "[data3-chain] STAGE 5: copy Qwen deliverables ..."
mkdir -p "$DIFF/data3/Qwen/edited_images" "$DIFF/data3/Qwen/masks_or_prompts"
for f in output/diffusion_prep/data3/edited_qwen/images/*; do
  ln "$f" "$DIFF/data3/Qwen/edited_images/$(basename $f)" 2>/dev/null || cp "$f" "$DIFF/data3/Qwen/edited_images/$(basename $f)"
done
cat > "$DIFF/data3/Qwen/masks_or_prompts/prompt.txt" <<'PROMPT'
Qwen-Image-Edit-2511 (20B DiT + Qwen2.5-VL-7B text encoder, nf4 quantized both, cpu_offload enabled)
true_cfg_scale=4.0, guidance_scale=1.0, num_inference_steps=28, seed=42, long_edge=1024

Instruction: "Remove the white water bottle, the blue Arducam box, and the red plastic lobster toy from the desk. Show only the empty desk surface where they were."

Note: All 140/140 views were edited.
PROMPT

# STAGE 6: Qwen splatfacto retrain (~30 min)
echo "[data3-chain] STAGE 6: Qwen splatfacto retrain ..."
bash _diffusion_retrain.sh data3 qwen > _diffusion_retrain_data3_qwen.log 2>&1

echo "[data3-chain] ALL DATA3 STAGES DONE."
