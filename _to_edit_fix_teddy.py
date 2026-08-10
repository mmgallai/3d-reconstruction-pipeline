"""Fix data5_a/b union masks: extend them upward to cover the teddy bear
sticking above the chair back. Pure numpy — no GPU (avoids conflict with
in-flight FLUX Fill)."""
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_dilation

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
MASK_DIR = PROJECT / "output/to_edit/masks"
IMG_DIR  = PROJECT / "output/to_edit/images"
PREV     = PROJECT / "output/to_edit/previews"

# For each data5 image, take the existing union, find its top-most row,
# extend the mask upward by extra pixels to capture the teddy that sits
# on top of the chair.  Also horizontally dilate around the chair top.
for stem in ("data5_a", "data5_b"):
    union_path = MASK_DIR / f"{stem}__union.png"
    union = np.array(Image.open(union_path).convert("L")) > 127
    h, w = union.shape

    # Find each column's top-most set-pixel row; propagate the mask upward.
    # Only for columns where the mask exists in the chair area (avoids
    # expanding random false-positives).
    col_any = union.any(axis=0)
    top_rows = np.where(col_any, union.argmax(axis=0), h)  # h means no mask
    UP_EXTEND_PX = 250  # tops of teddy stick up ~200px above chair back
    for c in range(w):
        t = top_rows[c]
        if t < h:
            new_top = max(0, t - UP_EXTEND_PX)
            union[new_top:t, c] = True

    # Also horizontally dilate the top slice a bit (teddy widens above chair)
    union = binary_dilation(union, iterations=6)

    # Save updated union
    Image.fromarray((union.astype(np.uint8) * 255)).save(union_path)
    cov = union.mean() * 100
    print(f"[fix-teddy] {stem}: union extended upward {UP_EXTEND_PX}px, cov={cov:.1f}%")

    # Preview
    img = np.array(Image.open(IMG_DIR / f"{stem}.jpg").convert("RGB"))
    over = img.copy()
    u = union.astype(np.uint8)
    over[u > 0] = (over[u > 0].astype(int) * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    Image.fromarray(over).save(PREV / f"preview_{stem}.jpg", quality=85)

print("[fix-teddy] done — will re-run LaMa on data5_a/b at end of run.")
