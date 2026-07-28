#!/bin/bash
# Reconstruct garden scene from scratch.
# Backs up current room state (colmap/, nerfstudio/, nerfstudio_data/) first.
# Runs full reconstruct_realityscan.py on 185 garden images.
# Wall clock: ~2.5 hr (COLMAP MVS ~50 min + splatfacto ~20 min + OpenMVS ~70 min)
set -e
export PYTHONIOENCODING=utf-8
cd "/c/Users/mgallai/Projects/3d_automated/reconstruction_project"

PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"
GARDEN_SRC="/c/Users/mgallai/Downloads/gs_test_scenes/garden_mipnerf360_source"
STAMP="$(date +%s)"

echo "== 1. Backup current ROOM state =="
if [ -d "colmap" ]; then
  mv "colmap" "colmap.room_v6_saved_${STAMP}"
  echo "   colmap/ -> colmap.room_v6_saved_${STAMP}"
fi
if [ -d "nerfstudio" ]; then
  mv "nerfstudio" "nerfstudio.room_v6_saved_${STAMP}"
  echo "   nerfstudio/ -> nerfstudio.room_v6_saved_${STAMP}"
fi
if [ -d "nerfstudio_data" ]; then
  mv "nerfstudio_data" "nerfstudio_data.room_saved_${STAMP}"
  echo "   nerfstudio_data/ -> nerfstudio_data.room_saved_${STAMP}"
fi

echo "== 2. Populate nerfstudio_data with garden images =="
mkdir -p nerfstudio_data/images
cp -r "${GARDEN_SRC}/images_4/." nerfstudio_data/images/
N=$(ls nerfstudio_data/images/ | wc -l)
echo "   copied $N images"
cp "${GARDEN_SRC}/intrinsics.json" nerfstudio_data/femto_intrinsics.json

echo "== 3. Even-crop images (pipeline compatibility) =="
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

echo "== 4. Adjust intrinsics for images_4 tier (divide by 4) =="
"$PY" -c "
import json
p = 'nerfstudio_data/femto_intrinsics.json'
d = json.load(open(p))
c = d['color_intrinsics']
for k in ['fx','fy','cx','cy']:
    c[k] /= 4.0
c['width'] = 1296  # even-cropped from 1297
c['height'] = 840
json.dump(d, open(p, 'w'), indent=2)
print(f'   scaled intrinsics: fx={c[\"fx\"]:.2f} width={c[\"width\"]} height={c[\"height\"]}')
"

echo "== 5. Build stub HEIC scan dir for pipeline precondition =="
STUB="_stub_heic_scan_garden_v${STAMP}"
rm -rf "$STUB"; mkdir -p "$STUB"
for jpg in nerfstudio_data/images/*.jpg; do
  stem="$(basename "${jpg%.jpg}")"
  : > "${STUB}/${stem}.heic"
done
echo "   $STUB with $(ls -1 $STUB | wc -l) stubs"

echo "== 6. Launch reconstruct_realityscan.py =="
"$PY" reconstruct_realityscan.py "$STUB"
echo "== GARDEN RECONSTRUCTION DONE =="

# Report which mesh version was produced
LATEST=$(ls output/mesh_v* -d 2>/dev/null | sed -E 's|.*mesh_v([0-9]+).*|\1|' | sort -n | tail -1)
echo "== Latest mesh version: v${LATEST} =="

rm -rf "$STUB"
