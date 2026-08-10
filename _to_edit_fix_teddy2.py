"""Take 2: add explicit bounding box for the cow-print teddy pillow that
SAM3 excluded from the chair mask (they are distinct objects)."""
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_dilation

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
MASK_DIR = PROJECT / "output/to_edit/masks"
IMG_DIR  = PROJECT / "output/to_edit/images"
PREV     = PROJECT / "output/to_edit/previews"

# Teddy bounding boxes tuned from mask_check previews. Both images are 3114x2074.
TEDDY_BOX = {
    "data5_a": (1580, 400, 2150, 850),   # (x0, y0, x1, y1)
    "data5_b": (2100, 400, 2700, 950),
}

for stem, (x0, y0, x1, y1) in TEDDY_BOX.items():
    union_path = MASK_DIR / f"{stem}__union.png"
    union = np.array(Image.open(union_path).convert("L")) > 127
    h, w = union.shape

    # Add teddy bounding box
    teddy = np.zeros_like(union)
    teddy[y0:y1, x0:x1] = True
    union = union | teddy

    # Small dilation to smooth the transition
    union = binary_dilation(union, iterations=6)

    Image.fromarray((union.astype(np.uint8) * 255)).save(union_path)
    print(f"[fix2] {stem}: added teddy box ({x0},{y0})-({x1},{y1}), union cov={union.mean()*100:.1f}%")

    # Preview
    img = np.array(Image.open(IMG_DIR / f"{stem}.jpg").convert("RGB"))
    over = img.copy()
    u = union.astype(np.uint8)
    over[u > 0] = (over[u > 0].astype(int) * 0.35 + np.array([255, 0, 0]) * 0.65).astype(np.uint8)
    Image.fromarray(over).save(PREV / f"mask_check_{stem}.jpg", quality=85)

print("[fix2] done")
