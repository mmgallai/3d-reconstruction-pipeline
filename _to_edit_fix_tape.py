"""Fix data4_b tape mask (SAM3 missed on first try). Retry with simpler prompt +
merge into union mask."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(PROJECT))
from scene_segmenter.sam3_segment import segment_view, _load_sam3, SAM3_REPO_DEFAULT

IMG   = PROJECT / "output/to_edit/images/data4_b.jpg"
MASKS = PROJECT / "output/to_edit/masks"

print("[fix] loading SAM3 ...")
from sam3.model.sam3_image_processor import Sam3Processor
proc = Sam3Processor(_load_sam3(SAM3_REPO_DEFAULT / "checkpoints/sam3.pt", "cuda"))

pil = Image.open(IMG)
# Try simpler / more visual prompts. SAM3 sometimes fails on multi-adjective.
candidates = ["tape measure", "yellow tape measure", "measuring tape tool",
              "black and yellow object", "hand tool with strap"]
best_mask = None
best_cov  = 0.0
best_prompt = ""
for p in candidates:
    m = segment_view(proc, pil, p, mode="best")
    cov = m.mean() * 100
    print(f"[fix]  '{p}'  cov={cov:.2f}%")
    if cov > best_cov and cov < 20:  # sanity: tape can't be >20% of image
        best_cov = cov
        best_mask = m
        best_prompt = p

if best_mask is None or best_cov < 0.2:
    print("[fix] SAM3 could not find tape — falling back to a manual box.")
    # From visual inspection: tape is around x=[450, 900], y=[600, 850] in 1920×1080
    h, w = pil.size[1], pil.size[0]
    best_mask = np.zeros((h, w), dtype=bool)
    best_mask[600:850, 450:900] = True
    best_prompt = "manual bounding box (SAM3 failed)"

# Save fresh tape mask
Image.fromarray((best_mask.astype(np.uint8) * 255)).save(
    MASKS / "data4_b__yellow_and_grey_retractable_tape_measure.png")
print(f"[fix] saved tape mask cov={best_mask.mean()*100:.2f}% prompt='{best_prompt}'")

# Rebuild union: tape ∪ plant (data4_b has only these 2)
plant = np.array(Image.open(MASKS / "data4_b__small_green_artificial_plant_in_black_pl.png").convert("L")) > 127
union = best_mask | plant
union = binary_dilation(union, iterations=12)
Image.fromarray((union.astype(np.uint8) * 255)).save(MASKS / "data4_b__union.png")
print(f"[fix] union rebuilt cov={union.mean()*100:.1f}%")

# Preview
img = np.array(pil.convert("RGB"))
over = img.copy()
u = (union.astype(np.uint8))
over[u > 0] = (over[u > 0].astype(int) * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
Image.fromarray(over).save(PROJECT / "output/to_edit/previews/preview_data4_b.jpg", quality=85)
print("[fix] preview updated")
