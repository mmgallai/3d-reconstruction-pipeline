#!/bin/bash
# Chain: wait for data4 kontext to fully finish (all 229), then launch room kontext.
set -eu
PROJECT="/c/Users/mgallai/Projects/3d_automated/reconstruction_project"
PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN env var before running}"

echo "[chain] waiting for data4 kontext to finish ..."
while [ "$(ls $PROJECT/output/diffusion_prep/data4/edited_kontext/images/ 2>/dev/null | wc -l)" -lt 229 ]; do
  sleep 60
done
echo "[chain] data4 kontext DONE (229/229). launching room kontext ..."

cd "$PROJECT"
"$PY" _diffusion_edit_persistent.py --scene room --model kontext > _diffusion_edit_room_kontext.log 2>&1
echo "[chain] room kontext DONE"
