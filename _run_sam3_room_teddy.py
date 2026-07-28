"""Generate SAM3 masks for just 'teddy bear' — will be UNIONed with 'blue_armchair' at extraction time."""
from pathlib import Path
import sys
PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(PROJECT))
from scene_segmenter.sam3_segment import segment_multi_prompt

image_dir = PROJECT / "nerfstudio_data" / "images"
image_paths = sorted(image_dir.glob("*.jpg"))
print(f"[teddy] found {len(image_paths)} images")
masks_dir = PROJECT / "output" / "segmented_room_v9a_fp_v2" / "masks"   # dump into SAME dir as v2 masks
segment_multi_prompt(
    image_paths=image_paths,
    prompts=["teddy bear"],
    masks_dir=masks_dir,
    mode="union", threshold=0.5, erode_px=1, device="cuda",
)
print(f"[teddy] DONE")
