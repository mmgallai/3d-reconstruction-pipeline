#!/bin/bash
# Chain: data4 qwen edit -> room qwen edit -> data4 qwen retrain -> room qwen retrain
set -eu
PROJECT="/c/Users/mgallai/Projects/3d_automated/reconstruction_project"
PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"
DIFF="/c/Users/mgallai/Downloads/diffusion"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN env var before running}"

cd "$PROJECT"

echo "[qwen-chain] STAGE 1: data4 qwen editing (198 views ~ 10hr) ..."
"$PY" _diffusion_edit_persistent.py --scene data4 --model qwen > _diffusion_edit_data4_qwen.log 2>&1

echo "[qwen-chain] STAGE 2: room qwen editing (229 views ~ 11.5hr) ..."
"$PY" _diffusion_edit_persistent.py --scene room --model qwen > _diffusion_edit_room_qwen.log 2>&1

echo "[qwen-chain] STAGE 3: copy edited images + prompts to Downloads deliverables ..."
for scene in data4 room; do
  mkdir -p "$DIFF/$scene/Qwen/edited_images" "$DIFF/$scene/Qwen/masks_or_prompts"
  for f in output/diffusion_prep/$scene/edited_qwen/images/*; do
    ln "$f" "$DIFF/$scene/Qwen/edited_images/$(basename $f)" 2>/dev/null || cp "$f" "$DIFF/$scene/Qwen/edited_images/$(basename $f)"
  done
done
cat > "$DIFF/data4/Qwen/masks_or_prompts/prompt.txt" <<'PROMPT'
Qwen-Image-Edit-2511 (20B DiT + Qwen2.5-VL-7B text encoder, nf4 quantized both, cpu_offload enabled)
true_cfg_scale=4.0, guidance_scale=1.0, num_inference_steps=28, seed=42, long_edge=1024

Instruction: "Remove the yellow tape measure, the small green artificial plant, and the tan cardboard box from the desk. Show only the empty desk surface where they were."

Note: 198/229 views were edited (those where the target objects were visible per SAM3 filter with >=0.5% coverage). The remaining 31 views were copied through unchanged.
PROMPT
cat > "$DIFF/room/Qwen/masks_or_prompts/prompt.txt" <<'PROMPT'
Qwen-Image-Edit-2511 (20B DiT + Qwen2.5-VL-7B text encoder, nf4 quantized both, cpu_offload enabled)
true_cfg_scale=4.0, guidance_scale=1.0, num_inference_steps=28, seed=42, long_edge=1024

Instruction: "Remove the blue armchair and the cow-print teddy bear pillow from the room. Show the empty floor and curtains behind them."

Note: 229/311 views were edited (those where the target objects were visible per SAM3 filter with >=0.5% coverage). The remaining 82 views were copied through unchanged.
PROMPT
echo "[qwen-chain] edited-images + prompts copied"

echo "[qwen-chain] STAGE 4: data4 qwen splatfacto retrain (~30 min) ..."
bash _diffusion_retrain.sh data4 qwen > _diffusion_retrain_data4_qwen.log 2>&1

echo "[qwen-chain] STAGE 5: room qwen splatfacto retrain (~30 min) ..."
bash _diffusion_retrain.sh room qwen > _diffusion_retrain_room_qwen.log 2>&1

echo "[qwen-chain] ALL QWEN STAGES DONE."
