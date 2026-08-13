#!/bin/bash
# Retrain the ROOM splat with the blue armchair + teddy bear removed.
#
# Uses the 311 LaMa-inpainted images produced by _inpaint_chair_lama.py
# (canonical batch, dilate=40) as the *only* difference vs the full-quality
# room reconstruction that produced splat_v39_da3_fullq.ply.
#
# What this script does (mirrors the pattern in _run_garden_reconstruct.sh):
#   1. Timestamped backup of colmap/, nerfstudio/, nerfstudio_data/ so the
#      current room_fullq run (splat_v39_da3_fullq.ply lineage) is preserved.
#   2. Populate nerfstudio_data/images/ with the inpainted JPGs from
#      output/lama_chair_removal/canonical/images/. Filenames are already
#      DSCF####.JPG (matches the source room dataset exactly).
#   3. Drop in reference intrinsics from the Mip-NeRF 360 room source dir.
#      (COLMAP re-solves them from EXIF anyway; kept for parity + audit trail.)
#   4. Even-crop images (defensive; the Fuji frames are already even but the
#      pipeline expects HxW divisible by 2).
#   5. Build the stub HEIC-scan directory that reconstruct_realityscan.py
#      requires as a positional arg (its HEIC->JPG stage no-ops because the
#      JPGs are already in place; the stubs just satisfy the precondition
#      that the scan dir contains one file per view).
#   6. Launch: python reconstruct_realityscan.py <stub> \
#              --downscale-factor 1 --no-openmvs-mesh --no-mesh
#      --downscale-factor 1  : same native-res regime as splat_v39_da3_fullq
#      --no-openmvs-mesh     : skip the ~10 h textured-mesh path that OOM'd
#                              on the last full-quality run (see
#                              FULL_QUALITY_RUN_NOTES.md "OpenMVS Stage 8b:
#                              CRASHED THE PC"). Splat-only run is the goal.
#      --no-mesh             : also skip nerfstudio's Poisson mesh export.
#   7. Export the splat via ns-export gaussian-splat once training finishes,
#      and rename it to output/splat_v40_da3_fullq_no_chair.ply so it does
#      not clobber v39.
#
# Expected wall clock (RTX 5070 Ti, 16 GB VRAM, 32 GB RAM):
#   Stage 2 COLMAP SfM  : ~30-45 min  (fresh run; no cached features because
#                                      the input pixels are different from v39)
#   Stage 3 COLMAP MVS  : ~1 h 45 min (matches v39; MVS operates on the same
#                                      views, only the pixel content changed)
#   Stage 4 DA3 depth   : ~5 min
#   Stage 7 splatfacto  : ~1 h        (matches v39's 58 min at native res)
#   Splat export        : ~2 min
#   ---
#   TOTAL               : ~3.5-4 h    (well under the 20+ h that would
#                                      include OpenMVS mesh)
#
# ============================================================================
# PREREQUISITES — read BEFORE running
# ============================================================================
#   * The BGR/RGB debate on the LaMa outputs was RESOLVED in favor of the
#     current writer (the "blue cast" observed in QA was in the rendered
#     preview, not on disk). If a re-QA changes that verdict, re-run the
#     inpaint before this script.
#   * Nvidia driver: update to a post-Aug-2 build first. The RefineMesh crash
#     on 2026-07-28 was RAM-driven, but the Aug 2 BSOD pattern targeted the
#     GPU driver. Even splat-only training holds ~14 GB VRAM for an hour.
#   * WSL2 memory cap: set to ~24 GB (of the 32 GB physical) in
#     %USERPROFILE%\.wslconfig  ->  [wsl2]  memory=24GB  swap=8GB
#     Prevents WSL/docker from starving Windows.
#   * ~2 GB free VRAM headroom recommended before launch (close browsers,
#     Slack, etc.).
#   * ~40 GB free disk under reconstruction_project/  (COLMAP dense + PLYs).
#
# ============================================================================

set -e
export PYTHONIOENCODING=utf-8
cd "/c/Users/mgallai/Projects/3d_automated/reconstruction_project"

PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"
ROOM_SRC="/c/Users/mgallai/Downloads/gs_test_scenes/room_mipnerf360_source"
INPAINTED_IMGS="/c/Users/mgallai/Projects/3d_automated/reconstruction_project/output/lama_chair_removal/canonical/images"
STAMP="$(date +%s)"

# Sanity checks up front so we fail before touching state.
[ -d "$INPAINTED_IMGS" ] || { echo "ERROR: inpainted images dir missing: $INPAINTED_IMGS"; exit 2; }
N_IN=$(ls "$INPAINTED_IMGS" | wc -l)
[ "$N_IN" -eq 311 ] || { echo "ERROR: expected 311 inpainted images, found $N_IN"; exit 2; }
[ -f "$ROOM_SRC/intrinsics.json" ] || { echo "ERROR: room intrinsics.json missing: $ROOM_SRC/intrinsics.json"; exit 2; }
[ -x "$PY" ] || { echo "ERROR: sam3 python missing: $PY"; exit 2; }

echo "== 1. Backup current workspace (room_fullq / v39 lineage) =="
if [ -d "colmap" ]; then
  mv "colmap" "colmap.room_fullq_no_chair_saved_${STAMP}"
  echo "   colmap/ -> colmap.room_fullq_no_chair_saved_${STAMP}"
fi
if [ -d "nerfstudio" ]; then
  mv "nerfstudio" "nerfstudio.room_fullq_no_chair_saved_${STAMP}"
  echo "   nerfstudio/ -> nerfstudio.room_fullq_no_chair_saved_${STAMP}"
fi
if [ -d "nerfstudio_data" ]; then
  mv "nerfstudio_data" "nerfstudio_data.room_fullq_no_chair_saved_${STAMP}"
  echo "   nerfstudio_data/ -> nerfstudio_data.room_fullq_no_chair_saved_${STAMP}"
fi

echo "== 2. Populate nerfstudio_data with LaMa-inpainted images =="
mkdir -p nerfstudio_data/images
cp -r "${INPAINTED_IMGS}/." nerfstudio_data/images/
N=$(ls nerfstudio_data/images/ | wc -l)
echo "   copied $N inpainted images (canonical dilate=40 batch)"
[ "$N" -eq 311 ] || { echo "ERROR: post-copy count $N != 311"; exit 3; }

echo "== 3. Copy reference intrinsics (audit trail; COLMAP re-solves from EXIF) =="
cp "${ROOM_SRC}/intrinsics.json" nerfstudio_data/femto_intrinsics.json
echo "   $(head -c 80 nerfstudio_data/femto_intrinsics.json)..."

echo "== 4. Even-crop images (pipeline compatibility) =="
"$PY" -c "
from PIL import Image
from pathlib import Path
d = Path('nerfstudio_data/images')
n_c = n_ok = 0
last = None
for f in sorted(d.iterdir()):
    if f.suffix.lower() not in {'.jpg', '.jpeg'}: continue
    im = Image.open(f)
    w, h = im.size
    tw = w - (w % 2); th = h - (h % 2)
    if (tw, th) == (w, h):
        n_ok += 1
    else:
        im.crop((0, 0, tw, th)).save(f, quality=95)
        n_c += 1
    last = (tw, th)
print(f'   even-cropped {n_c}, already even {n_ok}, size {last}')
"

echo "== 5. Build stub HEIC scan dir (satisfies pipeline precondition) =="
STUB="_stub_heic_scan_room_no_chair_v${STAMP}"
rm -rf "$STUB"; mkdir -p "$STUB"
for jpg in nerfstudio_data/images/*.JPG nerfstudio_data/images/*.jpg; do
  [ -f "$jpg" ] || continue
  stem="$(basename "$jpg")"; stem="${stem%.*}"
  : > "${STUB}/${stem}.heic"
done
echo "   $STUB with $(ls -1 $STUB | wc -l) stubs"

echo "== 6. Launch reconstruct_realityscan.py (splat-only, native res) =="
# --downscale-factor 1 : matches splat_v39_da3_fullq training regime.
# --no-openmvs-mesh    : skip OpenMVS textured mesh (the ~10 h stage that OOM'd on v39).
# --no-mesh            : also skip nerfstudio Poisson mesh export.
"$PY" reconstruct_realityscan.py "$STUB" \
    --downscale-factor 1 \
    --no-openmvs-mesh \
    --no-mesh

echo "== 7. Export splat from the trained checkpoint =="
# Find the latest splatfacto run just produced under nerfstudio/dense/splatfacto/.
LATEST_CFG=$(ls -1t nerfstudio/dense/splatfacto/*/config.yml 2>/dev/null | head -1)
if [ -z "$LATEST_CFG" ]; then
  echo "WARN: could not find a config.yml under nerfstudio/dense/splatfacto/*/"
  echo "      Skipping export step. Splat may already have been exported by the pipeline."
else
  echo "   config: $LATEST_CFG"
  # The pipeline typically already exports splat.ply as part of stage 9;
  # rename it (or a fresh ns-export result) to a v40 tag.
  if [ -f "output/splat.ply" ]; then
    mv "output/splat.ply" "output/splat_v40_da3_fullq_no_chair.ply"
    echo "   -> output/splat_v40_da3_fullq_no_chair.ply"
  else
    echo "   No output/splat.ply found; run ns-export manually via docker:"
    echo "     docker run --rm --gpus all -v \$(pwd):/workspace nerfstudio-blackwell \\"
    echo "       bash -c 'python3 /workspace/fix_weights.py && \\"
    echo "         ns-export gaussian-splat \\"
    echo "           --load-config /workspace/${LATEST_CFG} \\"
    echo "           --output-dir /workspace/output'"
    echo "     mv output/splat.ply output/splat_v40_da3_fullq_no_chair.ply"
  fi
fi

echo "== ROOM (no-chair) RECONSTRUCTION DONE =="
LATEST_SPLAT=$(ls -1t output/splat_v*_no_chair.ply 2>/dev/null | head -1)
[ -n "$LATEST_SPLAT" ] && echo "   Latest no-chair splat: $LATEST_SPLAT"

rm -rf "$STUB"
