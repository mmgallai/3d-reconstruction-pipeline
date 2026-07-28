#!/usr/bin/env bash
# =============================================================================
# _run_test_scenes_garden_and_room.sh
#
# Runs the V32 reconstruction pipeline (reconstruct_realityscan.py) on the
# Mip-NeRF 360 "garden" (images_4/, 185 imgs) and "room" (images_2/, 311 imgs)
# test scenes back-to-back.
#
# The pipeline is invoked WITHOUT --use-femto-depth (Mip-NeRF 360 has no ToF
# depth), so it falls back to the DA3 monocular-depth init path.
#
# --- Stage-1 HEIC bypass ------------------------------------------------------
# The default pipeline stage 1 calls heic_converter.convert_directory(
# scan_dir, jpeg_dir) which hard-errors when scan_dir has no *.heic/*.heif.
# Trick used here:
#   1. Pre-seed nerfstudio_data/images/ with our JPGs (renamed lowercase .jpg).
#   2. Hand the pipeline a stub scan_dir containing zero-byte *.heic files
#      whose stems match the pre-seeded JPGs.  heic_converter then finds
#      dst.exists() for every entry (Windows FS is case-insensitive), appends
#      the existing JPG to the output list, and never calls Image.open.
# Net effect: stage 1 no-ops, downstream stages consume our real JPGs.
# =============================================================================

set -euo pipefail
export PYTHONIOENCODING=utf-8

PROJECT="/c/Users/mgallai/Projects/3d_automated/reconstruction_project"
PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"

GARDEN_SRC="/c/Users/mgallai/Downloads/gs_test_scenes/garden_mipnerf360_source"
ROOM_SRC="/c/Users/mgallai/Downloads/gs_test_scenes/room_mipnerf360_source"

cd "$PROJECT"

# -----------------------------------------------------------------------------
# 1. Snapshot the current data4 state so we can restore after both scenes run.
# -----------------------------------------------------------------------------
BAK_STAMP="$(date +%s)"

# nerfstudio_data: back up only if an obvious pre-existing backup isn't there.
if compgen -G "nerfstudio_data.bak_before_test_scenes.*" > /dev/null; then
    echo "== data4 nerfstudio_data already backed up under nerfstudio_data.bak_before_test_scenes.*; skipping snapshot"
else
    if [ -d "nerfstudio_data" ]; then
        mv "nerfstudio_data" "nerfstudio_data.bak_before_test_scenes.${BAK_STAMP}"
        echo "== Snapshotted nerfstudio_data → nerfstudio_data.bak_before_test_scenes.${BAK_STAMP}"
    fi
fi

# colmap/ and nerfstudio/ (both git-ignored) — snapshot if present
for name in colmap nerfstudio; do
    if [ -d "$name" ]; then
        mv "$name" "${name}.bak_before_test_scenes.${BAK_STAMP}"
        echo "== Snapshotted $name → ${name}.bak_before_test_scenes.${BAK_STAMP}"
    fi
done

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Find the highest existing mesh_v<N> integer in output/ (0 if none).
latest_mesh_version() {
    local n=0
    if [ -d "output" ]; then
        # shellcheck disable=SC2012
        n="$(ls -1 output 2>/dev/null \
             | grep -E '^mesh_v[0-9]+$' \
             | sed -E 's/^mesh_v([0-9]+)$/\1/' \
             | sort -n | tail -1 || true)"
    fi
    echo "${n:-0}"
}

# -----------------------------------------------------------------------------
# run_scene <name> <src_dir> <images_tier>
# -----------------------------------------------------------------------------
run_scene() {
    local name="$1"
    local src="$2"
    local tier="$3"

    echo ""
    echo "############################################################"
    echo "# RUN_SCENE: ${name}"
    echo "#   src   = ${src}"
    echo "#   tier  = ${tier}"
    echo "############################################################"

    local src_images="${src}/${tier}"
    local src_intrinsics="${src}/intrinsics.json"

    if [ ! -d "$src_images" ]; then
        echo "!! Source images dir not found: $src_images" >&2
        return 1
    fi
    if [ ! -f "$src_intrinsics" ]; then
        echo "!! Source intrinsics not found: $src_intrinsics" >&2
        return 1
    fi

    # --- Fresh nerfstudio_data/ ---------------------------------------------
    rm -rf "nerfstudio_data"
    mkdir -p "nerfstudio_data/images"

    echo "== Copying JPGs → nerfstudio_data/images/ (lowercase .jpg suffix)"
    local jpg_count=0
    # Copy every image; force lowercase .jpg suffix so downstream (case-
    # sensitive on POSIX tools inside Docker) globs succeed.
    for f in "$src_images"/*; do
        [ -f "$f" ] || continue
        local base
        base="$(basename "$f")"
        local stem="${base%.*}"
        cp -f "$f" "nerfstudio_data/images/${stem}.jpg"
        jpg_count=$((jpg_count + 1))
    done
    echo "   Copied ${jpg_count} images."

    cp -f "$src_intrinsics" "nerfstudio_data/femto_intrinsics.json"
    echo "== Copied intrinsics.json → nerfstudio_data/femto_intrinsics.json"

    # --- Crop images to even x even dimensions ------------------------------
    # Pipeline's internal 2x downscale rounds odd dimensions inconsistently
    # in image vs camera-params, producing a 1-px mismatch that crashes
    # splatfacto training. Trimming to even removes the ambiguity.
    echo "== Cropping images to even x even dimensions (pipeline compatibility)"
    "$PY" - <<'PYEND'
from PIL import Image
from pathlib import Path
import json
d = Path("nerfstudio_data/images")
n_cropped = n_ok = 0
new_w = new_h = None
for f in sorted(d.iterdir()):
    if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
    im = Image.open(f)
    w, h = im.size
    tw = w - (w % 2); th = h - (h % 2)
    if (tw, th) == (w, h):
        n_ok += 1
    else:
        im.crop((0, 0, tw, th)).save(f, quality=95)
        n_cropped += 1
    new_w, new_h = tw, th
print(f"   Even-cropped {n_cropped}, already even {n_ok}, final size {new_w}x{new_h}")
# Adjust intrinsics.json width/height if they changed (rare but safe).
p = Path("nerfstudio_data/femto_intrinsics.json")
if p.is_file():
    j = json.loads(p.read_text())
    if "color_intrinsics" in j:
        old_w = j["color_intrinsics"].get("width")
        old_h = j["color_intrinsics"].get("height")
        if (old_w, old_h) != (new_w, new_h):
            j["color_intrinsics"]["width"] = new_w
            j["color_intrinsics"]["height"] = new_h
            p.write_text(json.dumps(j, indent=2))
            print(f"   Adjusted intrinsics width/height {old_w}x{old_h} -> {new_w}x{new_h}")
PYEND

    # --- Build a stub scan_dir full of empty *.heic files ------------------
    # heic_converter iterates *.heic, but skips per-file convert when the
    # destination JPG already exists (idempotency guard).  Stubs let us
    # satisfy the "at least one HEIC" precondition without hitting PIL.
    local stub_dir="_stub_heic_scan_${name}"
    rm -rf "$stub_dir"
    mkdir -p "$stub_dir"
    echo "== Building stub HEIC scan_dir → ${stub_dir}"
    for jpg in nerfstudio_data/images/*.jpg; do
        local stem
        stem="$(basename "${jpg%.jpg}")"
        : > "${stub_dir}/${stem}.heic"
    done
    local stub_count
    stub_count="$(ls -1 "$stub_dir" | wc -l | tr -d '[:space:]')"
    echo "   Created ${stub_count} stub .heic files."

    # --- Clean colmap/ and nerfstudio/ so this scene starts pristine -------
    rm -rf "colmap" "nerfstudio"

    # --- Record baseline version so we can tag output afterwards -----------
    local v_before
    v_before="$(latest_mesh_version)"
    echo "== Baseline mesh version before run: v${v_before}"

    # --- Invoke pipeline ---------------------------------------------------
    local log="test_scene_${name}.log"
    echo "== Launching reconstruct_realityscan.py (log → ${log})"
    echo "   $PY reconstruct_realityscan.py \"$stub_dir\""
    "$PY" reconstruct_realityscan.py "$stub_dir" > "$log" 2>&1

    # --- Report ------------------------------------------------------------
    local v_after
    v_after="$(latest_mesh_version)"
    echo "== ${name} DONE =="
    echo "   Baseline version: v${v_before}"
    echo "   Latest version  : v${v_after}"
    echo "   Mesh dirs produced (new since baseline):"
    if [ -d "output" ]; then
        # shellcheck disable=SC2012
        ls -1 output 2>/dev/null \
            | grep -E '^mesh_v[0-9]+$' \
            | awk -v b="$v_before" -F'v' '($2+0) > (b+0)' \
            || true
    fi

    # Publish scene→version mapping for the final summary.
    if [ "$name" = "garden" ]; then
        GARDEN_VERSION="$v_after"
    elif [ "$name" = "room" ]; then
        ROOM_VERSION="$v_after"
    fi

    # Stub scan_dir isn't needed anymore.
    rm -rf "$stub_dir"
}

# -----------------------------------------------------------------------------
# 2. Run the two scenes sequentially.
# -----------------------------------------------------------------------------
GARDEN_VERSION=""
ROOM_VERSION=""

run_scene garden "$GARDEN_SRC" images_4
run_scene room   "$ROOM_SRC"   images_2

# -----------------------------------------------------------------------------
# 3. Restore data4 nerfstudio_data from the latest backup.
# -----------------------------------------------------------------------------
echo ""
echo "== Restoring data4 nerfstudio_data from latest backup"
rm -rf "nerfstudio_data"

# Pick the newest nerfstudio_data.bak_before_test_scenes.* directory.
LATEST_BAK=""
for d in nerfstudio_data.bak_before_test_scenes.*; do
    [ -d "$d" ] || continue
    if [ -z "$LATEST_BAK" ] || [ "$d" \> "$LATEST_BAK" ]; then
        LATEST_BAK="$d"
    fi
done

if [ -n "$LATEST_BAK" ]; then
    mv "$LATEST_BAK" "nerfstudio_data"
    echo "   Restored ${LATEST_BAK} → nerfstudio_data"
else
    echo "   (no backup to restore — nerfstudio_data left empty)"
fi

# -----------------------------------------------------------------------------
# 4. Final report — parent orchestrator parses this.
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  TEST SCENE RUN COMPLETE"
echo "  garden -> mesh_v${GARDEN_VERSION}  (output/mesh_v${GARDEN_VERSION}/)"
echo "  room   -> mesh_v${ROOM_VERSION}   (output/mesh_v${ROOM_VERSION}/)"
echo "============================================================"
