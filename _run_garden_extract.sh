#!/bin/bash
# Garden extraction v1 — uses same direct-projection approach as room v6.
# Prerequisites (from _run_garden_reconstruct.sh):
#   - nerfstudio_data/images/ has 185 garden JPGs (1296x840)
#   - colmap/dense/transforms.json for garden
#   - nerfstudio/dense/splatfacto/<latest>/dataparser_transforms.json for garden
#   - output/mesh_v??/mesh_v??_openmvs.ply (auto-picks latest)
#   - output/splat_v??_da3_pruned.ply
#   - output/segmented_garden_v9a_fp_v2/masks/ (from _run_sam3_garden.py)
set -e
export PYTHONIOENCODING=utf-8
cd "/c/Users/mgallai/Projects/3d_automated/reconstruction_project"

PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"

# Auto-detect latest mesh + splat produced by garden reconstruction
MESH_V=$(ls -1d output/mesh_v* 2>/dev/null | sed -E 's|.*mesh_v([0-9]+)$|\1|' | grep -E '^[0-9]+$' | sort -n | tail -1)
if [ -z "$MESH_V" ]; then echo "FATAL: no mesh_v* dir"; exit 1; fi
SCENE_MESH="output/mesh_v${MESH_V}/mesh_v${MESH_V}_openmvs.ply"
SCENE_SPLAT=$(ls output/splat_v${MESH_V}_*_pruned.ply 2>/dev/null | head -1)
if [ -z "$SCENE_SPLAT" ]; then echo "FATAL: no matching splat_v${MESH_V}_*_pruned.ply"; exit 1; fi

DATAPARSER="$(ls -td nerfstudio/dense/splatfacto/*/ | head -1)dataparser_transforms.json"
if [ ! -f "$DATAPARSER" ]; then echo "FATAL: no dataparser"; exit 1; fi

# Create extraction-mode dataparser with scale=1.0 (splat is in COLMAP-world for splatfacto-big)
EXTRACT_DP="nerfstudio/dataparser_transforms_extraction_garden.json"
"$PY" -c "
import json
d = json.load(open('$DATAPARSER'))
d['scale'] = 1.0
json.dump(d, open('$EXTRACT_DP', 'w'), indent=2)
print('  wrote', '$EXTRACT_DP')
"

echo "== garden extraction v1 =="
echo "   mesh:  $SCENE_MESH"
echo "   splat: $SCENE_SPLAT"
echo "   dp:    $EXTRACT_DP"

"$PY" _extract_by_projection.py \
  --splat "$SCENE_SPLAT" \
  --transforms colmap/dense/transforms.json \
  --dataparser "$DATAPARSER" \
  --masks-dir output/segmented_garden_v9a_fp_v2/masks \
  --prompts "ceramic_pot:ceramic_pot+dried_leaves,green_ball,wooden_table" \
  --out-dir output/july/garden_v1 \
  --min-views 5 --keep-thresh 0.25 \
  --drop-anisotropy 80 \
  --fill-holes --fill-k 4 --fill-mode nearest --fill-shell-inner-m 0.02 --fill-shell-outer-m 0.35

mkdir -p output/july/garden_v1/qa
for ply_tag in \
  "scene_without_objects_filled.ply scenewout_filled" \
  "objects/ceramic_pot.ply obj_ceramic_pot" \
  "objects/green_ball.ply obj_green_ball" \
  "objects/wooden_table.ply obj_wooden_table"; do
  ply=$(echo $ply_tag | awk '{print $1}')
  tag=$(echo $ply_tag | awk '{print $2}')
  bash _render_qa_docker.sh "output/july/garden_v1/splat/$ply" "0,45,90,135,175" "output/july/garden_v1/qa/${tag}_grid.png" 2>&1 | grep saved || true
done

echo "== garden v1 DONE =="
ls -la output/july/garden_v1/splat/objects/ 2>/dev/null
