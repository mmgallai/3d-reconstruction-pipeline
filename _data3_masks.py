"""SAM3 masks for data3 objects: white water bottle, blue Arducam box, red lobster.
Runs SAM3 once per (image, prompt) and saves masks in the flat format expected by
_extract_by_projection.py: <masks_dir>/<stem>__<slug>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(PROJECT))
from scene_segmenter.sam3_segment import segment_view, _load_sam3, _slugify, SAM3_REPO_DEFAULT

IMG_DIR   = PROJECT / "output/data3_workspace/nerfstudio_data/images"
MASK_DIR  = PROJECT / "output/data3_workspace/masks"

PROMPTS = [
    "white water bottle with silver cap",
    "small blue product box",
    "red plastic toy lobster figurine",
]


def main() -> None:
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    print("[sam3] loading model ...")
    from sam3.model.sam3_image_processor import Sam3Processor
    ckpt = SAM3_REPO_DEFAULT / "checkpoints" / "sam3.pt"
    processor = Sam3Processor(_load_sam3(ckpt, "cuda"))

    imgs = sorted(IMG_DIR.glob("*.jpg"))
    total = len(imgs) * len(PROMPTS)
    done = 0
    for img_path in imgs:
        pil = Image.open(img_path)
        for prompt in PROMPTS:
            slug = _slugify(prompt)
            m = segment_view(processor, pil, prompt, mode="best")
            out_p = MASK_DIR / f"{img_path.stem}__{slug}.png"
            Image.fromarray((m.astype(np.uint8) * 255)).save(out_p)
            done += 1
            if done % 30 == 0:
                cov = m.mean() * 100
                print(f"[sam3] {done}/{total}  last: {img_path.stem} '{prompt[:40]}' cov={cov:.1f}%")

    print(f"[sam3] done — {done} masks written to {MASK_DIR}")


if __name__ == "__main__":
    main()
