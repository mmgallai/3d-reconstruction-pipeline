"""Fix interior holes in SAM3 union masks used for diffusion inpainting.

SAM3 sometimes excludes visually distinct sub-regions (a white product label
inside a tan cardboard box; the cream/brown pattern of a teddy). Inpaint
pipelines then leave those interior regions untouched, producing floating
label / pattern artifacts.

Fix per mask:
  1. binary_closing (5 px)       — bridge tiny inter-fragment gaps
  2. binary_fill_holes            — close any interior black region
  3. binary_dilation (+4 px)      — small extra slack for edge blending

Runs IN PLACE on all *__union.png masks under:
  output/diffusion_prep/room/masks/
  output/diffusion_prep/data4/masks/
  output/diffusion_prep/data3/masks/     (if present)
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_closing

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
EXTRA_DILATE_PX = 4
CLOSE_PX = 5

targets = [PROJECT / "output/diffusion_prep" / s / "masks" for s in ("room", "data4", "data3")]

grand_holes = 0
grand_files = 0
for mask_dir in targets:
    if not mask_dir.exists():
        print(f"[skip] {mask_dir} does not exist")
        continue
    files = sorted(mask_dir.glob("*__union.png"))
    print(f"[{mask_dir.parent.name}] fixing {len(files)} masks in {mask_dir}")
    n_holes = 0
    for p in files:
        m = np.array(Image.open(p).convert("L")) > 127

        m = binary_closing(m, iterations=CLOSE_PX)
        m_filled = binary_fill_holes(m)
        n_new = m_filled.sum() - m.sum()
        m = m_filled
        if EXTRA_DILATE_PX > 0:
            m = binary_dilation(m, iterations=EXTRA_DILATE_PX)

        Image.fromarray((m.astype(np.uint8) * 255)).save(p)
        if n_new > 100:
            n_holes += 1

    print(f"[{mask_dir.parent.name}] done — {n_holes}/{len(files)} masks had interior holes filled")
    grand_holes += n_holes
    grand_files += len(files)

print(f"\nTOTAL: {grand_holes}/{grand_files} masks had holes filled across all scenes.")
