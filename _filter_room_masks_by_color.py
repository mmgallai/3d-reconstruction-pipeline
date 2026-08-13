"""Color-sanity filter for room masks (RGB-based, simple + robust).

Problem: SAM3's "blue armchair" prompt matches shape > color, catching WOODEN
STOOLS, SIDE TABLES, and the GREY 3-SEAT SOFA in views where the actual blue
chair isn't visible.

Fix per view:
  1. Split the union mask into connected components.
  2. For each component, sample its pixels' RGB from the ORIGINAL image.
  3. Compute % of pixels that are "blue-dominant": B > R + 8 AND B > G - 5.
     - Blue chair fabric: B channel is clearly higher than R → passes
     - Grey couch: B ≈ R ≈ G → fails
     - Wood: R > B → fails
  4. Keep component only if blue-dominant fraction >= 0.10  (10%)
     — Blue chair typically 40-70% blue-dominant pixels; grey/wood <5%
     — The 10% threshold accepts partial-view chairs but rejects
        pure-non-chair objects.
  5. If NO components pass, mark the view as invisible (dropped from
     visible list, mask file deleted).

Additionally updates visibility.json to move rejected views to invisible.
Backs up the original visibility.json first.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label, binary_erosion

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
IMG_DIR  = PROJECT / "colmap/dense/images.orig_with_chair"
MASK_DIR = PROJECT / "output/diffusion_prep/room/masks"
VIS_FILE = PROJECT / "output/diffusion_prep/room/visibility.json"

MIN_COMPONENT_PIXELS = 2000       # ignore tiny mask specks
MIN_BLUE_FRAC        = 0.06       # >=6% pixels must be B > R + 3
BLUE_R_MARGIN        = 3          # B > R + margin
INTERIOR_ERODE_PX    = 20         # sample only interior (not dilation halo)


def blue_frac(rgb: np.ndarray) -> float:
    """Fraction of pixels where B > R + BLUE_R_MARGIN (blue-tinted vs warm)."""
    if len(rgb) == 0:
        return 0.0
    R = rgb[:, 0].astype(np.int32)
    B = rgb[:, 2].astype(np.int32)
    return float((B > R + BLUE_R_MARGIN).mean())


def process_view(name: str) -> tuple[bool, str, float]:
    img = np.array(Image.open(IMG_DIR / name).convert("RGB"))
    mask_p = MASK_DIR / f"{Path(name).stem}__union.png"
    if not mask_p.exists():
        return False, "no_mask_file", 0.0
    mask = np.array(Image.open(mask_p).convert("L")) > 127
    if mask.sum() == 0:
        return False, "empty_mask", 0.0

    lbl, n_cc = cc_label(mask)
    if n_cc == 0:
        return False, "no_components", 0.0

    kept = np.zeros_like(mask)
    max_blue_seen = 0.0
    for cc in range(1, n_cc + 1):
        cc_mask = (lbl == cc)
        n = int(cc_mask.sum())
        if n < MIN_COMPONENT_PIXELS:
            continue
        # Erode to interior to skip the dilation halo (which is BACKGROUND
        # pixels — floor, curtain — that would dilute the color signal)
        core = binary_erosion(cc_mask, iterations=INTERIOR_ERODE_PX)
        if core.sum() < 1000:
            core = cc_mask   # fall back on the raw component for tiny views
        pixels = img[core]
        if len(pixels) > 30000:
            idx = np.random.default_rng(42).choice(len(pixels), 30000, replace=False)
            pixels = pixels[idx]
        bf = blue_frac(pixels)
        max_blue_seen = max(max_blue_seen, bf)
        if bf >= MIN_BLUE_FRAC:
            kept |= cc_mask

    if kept.sum() == 0:
        return False, f"all_components_rejected (max_blue_frac={max_blue_seen:.2f})", max_blue_seen

    Image.fromarray((kept.astype(np.uint8) * 255)).save(mask_p)
    return True, f"kept (max_blue_frac={max_blue_seen:.2f})", max_blue_seen


def main():
    # Backup + read visibility.json
    if VIS_FILE.exists():
        shutil.copy2(VIS_FILE, VIS_FILE.with_suffix(".json.pre_color_filter"))
    vis = json.loads(VIS_FILE.read_text())
    visible = list(vis["visible"])
    invisible = list(vis["invisible"])

    print(f"[filter] processing {len(visible)} visible-view masks in room ...")
    print(f"[filter] thresholds: MIN_BLUE_FRAC={MIN_BLUE_FRAC} "
          f"BLUE_R_MARGIN={BLUE_R_MARGIN}")

    newly_invisible = []
    kept_count = 0
    for i, name in enumerate(visible, 1):
        kept, reason, bf = process_view(name)
        if not kept:
            newly_invisible.append(name)
            (MASK_DIR / f"{Path(name).stem}__union.png").unlink(missing_ok=True)
            if i <= 20 or len(newly_invisible) <= 30 or i % 50 == 0:
                print(f"  [{i:3d}/{len(visible)}] {name} REJECTED  {reason}")
        else:
            kept_count += 1
            if i % 50 == 0:
                print(f"  [{i:3d}/{len(visible)}] {name} kept  {reason}")

    print(f"\n[filter] kept {kept_count}/{len(visible)} views | "
          f"rejected {len(newly_invisible)}")

    new_visible = [n for n in visible if n not in newly_invisible]
    new_invisible = sorted(set(invisible + newly_invisible))
    vis["visible"] = new_visible
    vis["invisible"] = new_invisible
    vis["n_visible"] = len(new_visible)
    vis["n_invisible"] = len(new_invisible)
    vis["color_filter_rejected"] = sorted(newly_invisible)
    vis["color_filter_thresholds"] = {
        "min_blue_frac": MIN_BLUE_FRAC,
        "blue_r_margin": BLUE_R_MARGIN,
        "min_component_pixels": MIN_COMPONENT_PIXELS,
    }
    VIS_FILE.write_text(json.dumps(vis, indent=2))
    print(f"[filter] visibility.json updated: visible={len(new_visible)} "
          f"invisible={len(new_invisible)}")


if __name__ == "__main__":
    main()
