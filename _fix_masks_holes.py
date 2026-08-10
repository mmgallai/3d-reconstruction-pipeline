"""Fix SAM3 union masks to become solid inpaint-safe blobs.

SAM3 exclusion issues we need to defeat:
  - Interior labels/QR/patterns inside a "cardboard box" ("that's not cardboard, it's paper")
  - Fluffy edges that leave the object outline with 1-pixel-thin gaps
  - Legs of a footstool leaving a large concave gap the model would inpaint as-is

The straightforward `binary_fill_holes` only fills FULLY-enclosed regions —
so a hole with a 1-pixel gap to the outside stays black, and any big concave
opening (footstool legs, chair back) never fills.

Fix per mask (in order):
  1. binary_closing (10 px)        — bridge tiny inter-fragment gaps
  2. convex_hull_object            — replace each connected component with
                                      its convex hull → guarantees a solid
                                      convex blob, no interior holes possible
  3. binary_dilation (+6 px)       — small extra slack for edge blending

Runs IN PLACE on all *__union.png under:
  output/diffusion_prep/room/masks/
  output/diffusion_prep/data4/masks/
  output/diffusion_prep/data3/masks/
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_closing
from skimage.morphology import convex_hull_object

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
CLOSE_PX = 10
EXTRA_DILATE_PX = 6

targets = [PROJECT / "output/diffusion_prep" / s / "masks" for s in ("room", "data4", "data3")]

grand_files = 0
grand_expanded = 0
for mask_dir in targets:
    if not mask_dir.exists():
        print(f"[skip] {mask_dir} does not exist")
        continue
    files = sorted(mask_dir.glob("*__union.png"))
    print(f"[{mask_dir.parent.name}] fixing {len(files)} masks in {mask_dir}")
    n_expanded = 0
    for p in files:
        m = np.array(Image.open(p).convert("L")) > 127
        cov_before = m.sum()
        if cov_before == 0:
            continue

        # 1. closing to bridge small gaps
        m = binary_closing(m, iterations=CLOSE_PX)

        # 2. convex hull PER connected component (fills any hole, incl. leaked ones)
        m = convex_hull_object(m)

        # 3. extra dilation
        if EXTRA_DILATE_PX > 0:
            m = binary_dilation(m, iterations=EXTRA_DILATE_PX)

        cov_after = m.sum()
        if cov_after > cov_before * 1.05:  # >5% expansion means we filled a real hole
            n_expanded += 1

        Image.fromarray((m.astype(np.uint8) * 255)).save(p)

    print(f"[{mask_dir.parent.name}] done — {n_expanded}/{len(files)} masks materially expanded (holes filled)")
    grand_files += len(files)
    grand_expanded += n_expanded

print(f"\nTOTAL: {grand_expanded}/{grand_files} masks materially expanded across all scenes.")
