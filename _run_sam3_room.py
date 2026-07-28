"""One-shot SAM3 mask generation for the room scene.

Prompts: blue armchair, wooden coffee table, black subwoofer.
Writes to output/segmented_room_v9a_fp_v2/masks/<frame_stem>__<slug>.png
"""
from pathlib import Path
import sys

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(PROJECT))

from scene_segmenter.sam3_segment import segment_multi_prompt

image_dir = PROJECT / "nerfstudio_data" / "images"
image_paths = sorted(image_dir.glob("*.jpg"))
print(f"[sam3-room] found {len(image_paths)} images in {image_dir}")

masks_dir = PROJECT / "output" / "segmented_room_v9a_fp_v2" / "masks"
prompts = ["blue armchair", "wooden coffee table", "black subwoofer"]

segment_multi_prompt(
    image_paths=image_paths,
    prompts=prompts,
    masks_dir=masks_dir,
    mode="union",
    threshold=0.5,
    erode_px=2,
    device="cuda",
)
print(f"[sam3-room] DONE — masks at {masks_dir}")
