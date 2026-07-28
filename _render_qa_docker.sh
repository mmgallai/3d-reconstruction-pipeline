#!/bin/bash
# Wrapper: run _render_splat_views.py inside nerfstudio-blackwell docker image.
# Usage: ./_render_qa_docker.sh <ply> <indices_csv> <out_png>
#   e.g. ./_render_qa_docker.sh output/july/room_v1/scene_without_objects_splat.ply 0,60,120,180,240 output/july/room_v1/qa/scene_without_grid.png
set -e
export MSYS_NO_PATHCONV=1  # prevent git-bash from mangling /workspace paths passed to docker
PLY_REL="$1"
INDICES="${2:-0,60,120,180,240}"
OUT_REL="$3"
# Optional override: 4th arg = transforms.json path, 5th arg = dataparser.json path
TRANSFORMS_REL="${4:-colmap/dense/transforms.json}"
if [ -n "$5" ]; then
    DP_REL="$5"
else
    # Newest dataparser under nerfstudio/dense/splatfacto/ (fallback)
    DP_REL="nerfstudio/$(ls -td nerfstudio/dense/splatfacto/*/ 2>/dev/null | head -1 | sed 's|^nerfstudio/||')dataparser_transforms.json"
fi

echo "  ply         = $PLY_REL"
echo "  indices     = $INDICES"
echo "  transforms  = $TRANSFORMS_REL"
echo "  dp          = $DP_REL"
echo "  out         = $OUT_REL"

docker run --rm --gpus all \
    -v "C:/Users/mgallai/Projects/3d_automated/reconstruction_project:/workspace" \
    -e TORCH_HOME=/workspace/torch_cache \
    nerfstudio-blackwell \
    python3 /workspace/_render_splat_views.py \
    --ply "/workspace/${PLY_REL}" \
    --transforms "/workspace/${TRANSFORMS_REL}" \
    --dataparser "/workspace/${DP_REL}" \
    --indices "${INDICES}" \
    --out "/workspace/${OUT_REL}"
