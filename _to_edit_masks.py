"""SAM3 mask generation for the to_edit shootout dataset.

Per-image object prompts (from user's instrutions.txt):
  data2_a, data2_b: white water bottle, small blue box, red toy lobster
  data4_a:          yellow tape measure, artificial plant, cardboard box
  data4_b:          yellow tape measure, artificial plant (no box in this view)
  data5_a, data5_b: blue armchair, cow-print teddy bear pillow
  data6_a, data6_b: pair of tan slippers

Writes:
  output/to_edit/masks/<stem>__<prompt-slug>.png  (per-object mask)
  output/to_edit/masks/<stem>__union.png          (all target objects unioned + dilated)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(PROJECT))
from scene_segmenter.sam3_segment import segment_view, _load_sam3, _slugify, SAM3_REPO_DEFAULT

IMG_DIR   = PROJECT / "output" / "to_edit" / "images"
MASK_DIR  = PROJECT / "output" / "to_edit" / "masks"
DILATE_PX = 12  # dilate union mask slightly so inpainters get a soft border

# Per-image target-object prompt lists (best-scoring SAM3 mask per prompt).
PER_IMAGE_PROMPTS = {
    "data2_a": ["white water bottle with silver cap", "small blue product box", "red plastic toy lobster figurine"],
    "data2_b": ["white water bottle with silver cap", "small blue product box", "red plastic toy lobster figurine"],
    "data4_a": ["yellow and grey retractable tape measure", "small round green artificial topiary plant", "tan cardboard shipping box"],
    "data4_b": ["yellow and grey retractable tape measure", "small green artificial plant in black plastic pot"],
    "data5_a": ["blue upholstered single-seat armchair recliner", "cow-print teddy bear pillow"],
    "data5_b": ["blue upholstered single-seat armchair recliner", "cow-print teddy bear pillow"],
    "data6_a": ["pair of tan suede open-back house slippers"],
    "data6_b": ["pair of tan suede open-back house slippers"],
}


def main() -> None:
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = SAM3_REPO_DEFAULT / "checkpoints" / "sam3.pt"
    print(f"[sam3] loading model from {ckpt} on cuda ...")
    from sam3.model.sam3_image_processor import Sam3Processor
    processor = Sam3Processor(_load_sam3(ckpt, "cuda"))

    total = sum(len(v) for v in PER_IMAGE_PROMPTS.values())
    done = 0
    for stem, prompts in PER_IMAGE_PROMPTS.items():
        img_path = IMG_DIR / f"{stem}.jpg"
        if not img_path.exists():
            print(f"[sam3] SKIP {stem}: file missing")
            continue
        pil = Image.open(img_path)
        h, w = pil.size[1], pil.size[0]
        union = np.zeros((h, w), dtype=bool)

        for prompt in prompts:
            mask = segment_view(processor, pil, prompt, mode="best")  # best = single highest-score
            slug = _slugify(prompt)
            out_p = MASK_DIR / f"{stem}__{slug}.png"
            Image.fromarray((mask.astype(np.uint8) * 255)).save(out_p)
            union |= mask
            cov = mask.mean() * 100
            done += 1
            print(f"[sam3] {done}/{total}  {stem}  '{prompt[:50]}'  cov={cov:.2f}%")

        if DILATE_PX > 0 and union.any():
            union = binary_dilation(union, iterations=DILATE_PX)
        union_p = MASK_DIR / f"{stem}__union.png"
        Image.fromarray((union.astype(np.uint8) * 255)).save(union_p)
        print(f"[sam3]   union saved -> {union_p.name}  cov={union.mean()*100:.1f}%")

    print(f"[sam3] DONE. Masks in {MASK_DIR}")


if __name__ == "__main__":
    main()
