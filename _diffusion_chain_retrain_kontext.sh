#!/bin/bash
# Chain: wait for room kontext editing (all 311), then retrain data4 splatfacto,
# then room splatfacto. Both retrains produce refined splats saved to
# Downloads/diffusion/<scene>/flux/refined_scene.ply
set -eu
PROJECT="/c/Users/mgallai/Projects/3d_automated/reconstruction_project"

echo "[retrain-chain] waiting for room kontext editing to finish (need 311 files) ..."
while [ "$(ls $PROJECT/output/diffusion_prep/room/edited_kontext/images/ 2>/dev/null | wc -l)" -lt 311 ]; do
  sleep 60
done
echo "[retrain-chain] room kontext editing DONE (311/311). launching data4 retrain ..."

cd "$PROJECT"
bash _diffusion_retrain.sh data4 kontext > _diffusion_retrain_data4_kontext.log 2>&1
echo "[retrain-chain] data4 kontext retrain DONE. launching room retrain ..."

bash _diffusion_retrain.sh room kontext > _diffusion_retrain_room_kontext.log 2>&1
echo "[retrain-chain] room kontext retrain DONE. all kontext done."
