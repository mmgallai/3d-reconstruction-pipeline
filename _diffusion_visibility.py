"""SAM3 visibility filter: for each dataset, decide per-view whether it
contains the target objects. A view is "visible" if the union of all target
masks covers >= MIN_COV_PCT of the image.

Writes:
  output/diffusion_prep/<scene>/visibility.json
    { "visible": ["IMG_1.jpg", ...], "invisible": [...], "coverage_pct": {...} }
  output/diffusion_prep/<scene>/masks/<img_stem>__union.png   (dilated union mask, only for VISIBLE views)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(PROJECT))
from scene_segmenter.sam3_segment import segment_view, _load_sam3, SAM3_REPO_DEFAULT

OUT_ROOT = PROJECT / "output/diffusion_prep"

# Minimum union-mask coverage (%) for a view to be "visible".
# Below this threshold, the object is too small to bother editing.
MIN_COV_PCT = 0.5

DATASETS = {
    "data4": {
        "images_dir": PROJECT / "assets/data4/nerfstudio_data/images",
        "prompts":    ["yellow and grey retractable tape measure",
                       "small round green artificial topiary plant",
                       "tan cardboard shipping box"],
        "dilate_px":  12,
    },
    "room": {
        "images_dir": PROJECT / "colmap/dense/images.orig_with_chair",
        "prompts":    ["blue upholstered single-seat armchair recliner",
                       "cow-print teddy bear pillow"],
        "dilate_px":  16,
    },
}


def process_dataset(name: str, cfg: dict, processor) -> None:
    imgs_dir = cfg["images_dir"]
    prompts  = cfg["prompts"]
    dilate   = cfg["dilate_px"]

    out_dir = OUT_ROOT / name
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(p for p in imgs_dir.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    n = len(all_imgs)
    print(f"[vis:{name}] {n} images | prompts={prompts}")

    visible:   List[str] = []
    invisible: List[str] = []
    coverage:  Dict[str, float] = {}

    for i, p in enumerate(all_imgs, 1):
        pil = Image.open(p)
        h, w = pil.size[1], pil.size[0]
        union = np.zeros((h, w), dtype=bool)

        for prompt in prompts:
            m = segment_view(processor, pil, prompt, mode="best")
            if m.shape != (h, w):
                # rare SAM3 quirk — resize
                m = np.array(Image.fromarray((m.astype(np.uint8) * 255)).resize(
                    (w, h), Image.NEAREST)) > 127
            union |= m

        raw_cov = union.mean() * 100
        coverage[p.name] = float(raw_cov)

        if raw_cov >= MIN_COV_PCT:
            if dilate > 0:
                union = binary_dilation(union, iterations=dilate)
            Image.fromarray((union.astype(np.uint8) * 255)).save(
                out_dir / "masks" / f"{p.stem}__union.png")
            visible.append(p.name)
        else:
            invisible.append(p.name)

        if i % 20 == 0 or i == n:
            print(f"[vis:{name}] {i:4d}/{n}  visible={len(visible)}  invisible={len(invisible)}  "
                  f"last_cov={raw_cov:.2f}%")

    result = {
        "visible": visible,
        "invisible": invisible,
        "coverage_pct": coverage,
        "threshold_pct": MIN_COV_PCT,
        "prompts": prompts,
        "n_total": n,
        "n_visible": len(visible),
        "n_invisible": len(invisible),
        "images_dir": str(imgs_dir),
    }
    (out_dir / "visibility.json").write_text(json.dumps(result, indent=2))
    print(f"[vis:{name}] DONE. visible={len(visible)}/{n}  ({100*len(visible)/n:.1f}%)")


def main():
    print("[vis] loading SAM3 ...")
    from sam3.model.sam3_image_processor import Sam3Processor
    proc = Sam3Processor(_load_sam3(SAM3_REPO_DEFAULT / "checkpoints/sam3.pt", "cuda"))

    for name, cfg in DATASETS.items():
        process_dataset(name, cfg, proc)


if __name__ == "__main__":
    main()
