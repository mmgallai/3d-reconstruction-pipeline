"""SAM3 for alternate garden 3rd objects."""
from pathlib import Path
import sys
PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(PROJECT))
from scene_segmenter.sam3_segment import segment_multi_prompt

image_dir = PROJECT / "nerfstudio_data" / "images"
image_paths = sorted(image_dir.glob("*.jpg"))
masks_dir = PROJECT / "output" / "segmented_garden_v9a_fp_v2" / "masks"

prompts = [
    "silver garden sphere",   # reflective silver balls in background
    "black door",              # black wooden door on left of garden
    "potted plant",            # dark pot with green plant on right
]

segment_multi_prompt(
    image_paths=image_paths,
    prompts=prompts,
    masks_dir=masks_dir,
    mode="union", threshold=0.5, erode_px=1, device="cuda",
)
print("DONE")
