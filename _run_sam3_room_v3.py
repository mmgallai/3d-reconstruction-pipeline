"""Regenerate SAM3 masks with tuned prompts for room v3.

Changes vs v2:
  - "blue armchair with teddy bear" (was "blue armchair") — include teddy in chair
  - "large black subwoofer tower speaker" (was "black subwoofer") — sharper prompt
  - "wooden coffee table" — unchanged

Writes to output/segmented_room_v3/masks/ (NEW dir so v2 masks stay).
"""
from pathlib import Path
import sys

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(PROJECT))

from scene_segmenter.sam3_segment import segment_multi_prompt

image_dir = PROJECT / "nerfstudio_data" / "images"
image_paths = sorted(image_dir.glob("*.jpg"))
print(f"[sam3-v3] found {len(image_paths)} images")

masks_dir = PROJECT / "output" / "segmented_room_v3" / "masks"
prompts = [
    "blue armchair with teddy bear",
    "large black subwoofer tower speaker",
    "wooden coffee table",
]

segment_multi_prompt(
    image_paths=image_paths,
    prompts=prompts,
    masks_dir=masks_dir,
    mode="union",
    threshold=0.5,
    erode_px=1,  # less erosion to keep tight extraction; was 2
    device="cuda",
)
print(f"[sam3-v3] DONE at {masks_dir}")
