"""SAM3 mask generation for garden scene extraction.

Prompts (union of variants):
  - ceramic_pot           = "ceramic pot" | "clay vase"
  - dried_leaves          = "dried palm leaves in pot"
  - green_soccer_ball     = "green soccer ball"
  - wooden_garden_table   = "round wooden garden table"

Writes masks to output/segmented_garden_v9a_fp_v2/masks/<view>__<slug>.png
"""
from pathlib import Path
import sys

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(PROJECT))
from scene_segmenter.sam3_segment import segment_multi_prompt

image_dir = PROJECT / "nerfstudio_data" / "images"
image_paths = sorted(image_dir.glob("*.jpg"))
print(f"[sam3-garden] found {len(image_paths)} images")

masks_dir = PROJECT / "output" / "segmented_garden_v9a_fp_v2" / "masks"

# Simple prompts (empirically better than compound prompts for SAM3)
prompts = [
    "ceramic pot",
    "dried leaves",
    "green ball",
    "wooden table",
]

segment_multi_prompt(
    image_paths=image_paths,
    prompts=prompts,
    masks_dir=masks_dir,
    mode="union", threshold=0.5, erode_px=1, device="cuda",
)
print(f"[sam3-garden] DONE at {masks_dir}")
