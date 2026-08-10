"""Regenerate diffusion union masks from raw SAM3, then apply a NON-DESTRUCTIVE
hole-fill (bridge outline gaps + fill enclosed interiors; NO convex hull).

Why: prior _fix_masks_holes.py using convex_hull_object was too aggressive —
irregular objects like a tape measure (body + curled strap) collapsed into a
huge triangle grabbing empty background.

Correct approach:
  1. Re-run SAM3 per-object for each visible view
  2. Union the per-object masks + dilate 12 px (matches visibility.py original)
  3. Fix per union mask:
       a. binary_closing (iterations=15)  — bridges 1-30 px outline gaps
       b. binary_fill_holes                — closes any interior region now enclosed
       c. binary_dilation (+4 px)          — small slack
     No convex hull. Outer contour stays true to the object.

Outputs replace masks at:
  output/diffusion_prep/<scene>/masks/<stem>__union.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_closing, binary_fill_holes

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(PROJECT))
from scene_segmenter.sam3_segment import segment_view, _load_sam3, SAM3_REPO_DEFAULT

# Same prompts + dilation as the original visibility filter
SCENE_CFG = {
    "room": {
        "images_dir": PROJECT / "colmap/dense/images.orig_with_chair",
        "prompts": [
            "blue upholstered single-seat armchair recliner",
            "cow-print teddy bear pillow",
        ],
        "sam3_dilate_px": 16,
    },
    "data4": {
        "images_dir": PROJECT / "assets/data4/nerfstudio_data/images",
        "prompts": [
            "yellow and grey retractable tape measure",
            "small round green artificial topiary plant",
            "tan cardboard shipping box",
        ],
        "sam3_dilate_px": 12,
    },
    "data3": {
        "images_dir": PROJECT / "output/data3_workspace/nerfstudio_data/images",
        "prompts": [
            "white water bottle with silver cap",
            "small blue product box",
            "red plastic toy lobster figurine",
        ],
        "sam3_dilate_px": 12,
    },
}

# Non-destructive fix parameters
CLOSE_ITERS  = 15   # bridge outline gaps up to ~15 px (much more than before)
DILATE_ITERS = 4    # small slack


def fix_union(m: np.ndarray) -> np.ndarray:
    """Bridge outline gaps → fill enclosed interior → small dilation.
    Preserves object outer contour (no convex hull)."""
    m = binary_closing(m, iterations=CLOSE_ITERS)
    m = binary_fill_holes(m)
    if DILATE_ITERS > 0:
        m = binary_dilation(m, iterations=DILATE_ITERS)
    return m


def rebuild_scene(scene: str, processor) -> None:
    cfg = SCENE_CFG[scene]
    prep_dir = PROJECT / "output/diffusion_prep" / scene
    vis = json.loads((prep_dir / "visibility.json").read_text())
    src_dir = cfg["images_dir"]
    mask_dir = prep_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    prompts = cfg["prompts"]
    sam_dilate = cfg["sam3_dilate_px"]
    visible = vis["visible"]

    print(f"[{scene}] rebuilding {len(visible)} visible-view union masks "
          f"| {len(prompts)} prompts | sam_dilate={sam_dilate} | "
          f"post: close={CLOSE_ITERS} dilate={DILATE_ITERS}")

    for i, name in enumerate(visible, 1):
        img_path = src_dir / name
        pil = Image.open(img_path)
        h, w = pil.size[1], pil.size[0]

        union = np.zeros((h, w), dtype=bool)
        for prompt in prompts:
            m = segment_view(processor, pil, prompt, mode="best")
            if m.shape != (h, w):
                m_pil = Image.fromarray((m.astype(np.uint8) * 255)).resize(
                    (w, h), Image.NEAREST)
                m = np.array(m_pil) > 127
            union |= m

        if sam_dilate > 0 and union.any():
            union = binary_dilation(union, iterations=sam_dilate)

        # Non-destructive fix
        union_fixed = fix_union(union)

        out_p = mask_dir / f"{Path(name).stem}__union.png"
        Image.fromarray((union_fixed.astype(np.uint8) * 255)).save(out_p)

        if i % 20 == 0 or i == len(visible):
            cov = union_fixed.mean() * 100
            print(f"  [{i:4d}/{len(visible)}] {name}  cov={cov:.1f}%")

    print(f"[{scene}] done.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["room", "data4", "data3"])
    args = ap.parse_args()

    print("[sam3] loading model ...")
    from sam3.model.sam3_image_processor import Sam3Processor
    ckpt = SAM3_REPO_DEFAULT / "checkpoints" / "sam3.pt"
    processor = Sam3Processor(_load_sam3(ckpt, "cuda"))

    for s in args.scenes:
        rebuild_scene(s, processor)


if __name__ == "__main__":
    main()
