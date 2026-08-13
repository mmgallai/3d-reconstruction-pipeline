"""Final cross-check: what should desk_y be such that the patch sits where the
object splat bottoms are?
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
import numpy as np
import trimesh
from plyfile import PlyData

ROOT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(ROOT))

MESH_PATH = ROOT / "output/mesh_v32_data3/mesh_v32_data3_openmvs.ply"
SPLAT_PATH = ROOT / "output/splat_v32_data3_noinit_pruned.ply"
DP_PATH = ROOT / "nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json"
BOUNDS_PATH = ROOT / "colmap/dense/tof_bounds.json"
OBJ_SPLAT_DIR = ROOT / "output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned"
PATCH_PATH = ROOT / "output/desk_patch_v32_data3/desk_patch_white_water_bottle.ply"

d = json.loads(DP_PATH.read_text())
T = np.asarray(d["transform"], dtype=np.float64)
R = T[:3, :3]; t = T[:3, 3]; dp_scale = float(d["scale"])
b = json.loads(BOUNDS_PATH.read_text())
colmap_to_metric = 1.0 / float(b["scale_factor_da3_to_colmap"])

# ============================================================
# Where SHOULD the patch be in SPLAT space such that it visually
# sits where the bottle bottom is?
# ============================================================
print("=" * 70)
print("Each object splat: bottom in SPLAT space (where the patch SHOULD sit)")
print("=" * 70)
for fname in sorted(os.listdir(OBJ_SPLAT_DIR)):
    if not fname.endswith("_splat.ply"):
        continue
    p = PlyData.read(str(OBJ_SPLAT_DIR / fname))
    v = p["vertex"]
    xyz = np.stack([np.asarray(v["x"], float),
                    np.asarray(v["y"], float),
                    np.asarray(v["z"], float)], axis=1)
    print(f"  {fname}:")
    print(f"    splat-space Y min  = {xyz[:,1].min():+.4f}")
    print(f"    splat-space Y p02  = {np.percentile(xyz[:,1], 2):+.4f}")
    print(f"    splat-space Y p50  = {np.percentile(xyz[:,1], 50):+.4f}")
    print(f"    centroid splat XYZ = {xyz.mean(axis=0).round(3)}")

# ============================================================
# The patch is at splat Y in [0.66, 0.99] but with centroid +0.83.
# Bottle bottom is at splat Y = 0.69. Bottle centroid splat Y = ~0.81.
# So actually the patch IS at roughly the right splat Y!
# What if the issue is somewhere ELSE? Let me check the actual patch positions
# vs object positions in 3D.
# ============================================================
print("\n" + "=" * 70)
print("Compare PATCH vs OBJECT SPLAT bounding boxes IN SPLAT SPACE")
print("=" * 70)
patch = PlyData.read(str(PATCH_PATH))
pv = patch["vertex"]
pxyz = np.stack([np.asarray(pv["x"], float),
                 np.asarray(pv["y"], float),
                 np.asarray(pv["z"], float)], axis=1)
print(f"  WHITE_WATER_BOTTLE patch (splat space):")
print(f"    X: [{pxyz[:,0].min():+.4f}, {pxyz[:,0].max():+.4f}]  centroid={pxyz[:,0].mean():+.4f}")
print(f"    Y: [{pxyz[:,1].min():+.4f}, {pxyz[:,1].max():+.4f}]  centroid={pxyz[:,1].mean():+.4f}")
print(f"    Z: [{pxyz[:,2].min():+.4f}, {pxyz[:,2].max():+.4f}]  centroid={pxyz[:,2].mean():+.4f}")

obj = PlyData.read(str(OBJ_SPLAT_DIR / "white_water_bottle_splat.ply"))
ov = obj["vertex"]
oxyz = np.stack([np.asarray(ov["x"], float),
                 np.asarray(ov["y"], float),
                 np.asarray(ov["z"], float)], axis=1)
print(f"  WHITE_WATER_BOTTLE object splat (splat space):")
print(f"    X: [{oxyz[:,0].min():+.4f}, {oxyz[:,0].max():+.4f}]  centroid={oxyz[:,0].mean():+.4f}")
print(f"    Y: [{oxyz[:,1].min():+.4f}, {oxyz[:,1].max():+.4f}]  centroid={oxyz[:,1].mean():+.4f}")
print(f"    Z: [{oxyz[:,2].min():+.4f}, {oxyz[:,2].max():+.4f}]  centroid={oxyz[:,2].mean():+.4f}")

print("\n  ==> Differences (patch - object centroid):")
print(f"    dX = {pxyz[:,0].mean() - oxyz[:,0].mean():+.4f}")
print(f"    dY = {pxyz[:,1].mean() - oxyz[:,1].mean():+.4f}")
print(f"    dZ = {pxyz[:,2].mean() - oxyz[:,2].mean():+.4f}")

# ============================================================
# Now let's understand: in splat space, which axis is UP visually?
# We can check via camera positions transformed to splat space.
# ============================================================
print("\n" + "=" * 70)
print("Camera positions in SPLAT space (visualised up axis)")
print("=" * 70)
sys.path.insert(0, str(ROOT))
from scene_segmenter.views import V32ViewSource
vs = V32ViewSource(ROOT)
cam_m = np.array([v.c2w[:3, 3] for v in vs.all_views()])  # metric
# metric -> splat
cam_c = cam_m / colmap_to_metric
cam_s = dp_scale * (cam_c @ R.T + t[None, :])
print(f"  Camera SPLAT-space ranges:")
print(f"    X: [{cam_s[:,0].min():+.3f}, {cam_s[:,0].max():+.3f}]")
print(f"    Y: [{cam_s[:,1].min():+.3f}, {cam_s[:,1].max():+.3f}]")
print(f"    Z: [{cam_s[:,2].min():+.3f}, {cam_s[:,2].max():+.3f}]")
print(f"  Camera SPLAT-space median XYZ: {np.median(cam_s, axis=0).round(3)}")
print(f"  Camera SPLAT-space p05 Y = {np.percentile(cam_s[:,1], 5):+.3f}")
print(f"  Camera SPLAT-space p95 Y = {np.percentile(cam_s[:,1], 95):+.3f}")

# Cameras orbit looking DOWN at desk. In splat space, where do they sit relative
# to objects?
print(f"\n  Bottle object splat-Y range: [{oxyz[:,1].min():.3f}, {oxyz[:,1].max():.3f}]")
print(f"  Cameras splat-Y median: {np.median(cam_s[:,1]):+.3f}")
print(f"  ==> Cameras are {'BELOW' if np.median(cam_s[:,1]) < oxyz[:,1].mean() else 'ABOVE'} bottle in splat Y")

# Also: which splat axis do cameras DESCEND on as they orbit higher in metric?
# Check correlation of metric Y and splat axes
print("\n  Correlation of metric Y with each splat axis (per camera):")
for k, ax in enumerate("XYZ"):
    corr = np.corrcoef(cam_m[:, 1], cam_s[:, k])[0, 1]
    print(f"    corr(metric Y, splat {ax}) = {corr:+.4f}")

# ============================================================
# So what IS the desk plane in SPLAT space directly?
# A horizontal plane in metric (normal +Y) becomes a tilted plane in splat,
# but objects sit at definite splat positions. The "desk surface in splat space"
# is just where the object bottoms are.
# ============================================================
print("\n" + "=" * 70)
print("INFERRED desk surface IN SPLAT SPACE (from object bottoms)")
print("=" * 70)
all_obj_bottoms = []
for fname in sorted(os.listdir(OBJ_SPLAT_DIR)):
    if not fname.endswith("_splat.ply"):
        continue
    p = PlyData.read(str(OBJ_SPLAT_DIR / fname))
    v = p["vertex"]
    xyz = np.stack([np.asarray(v["x"], float),
                    np.asarray(v["y"], float),
                    np.asarray(v["z"], float)], axis=1)
    # Bottom 5% of points in splat-Y
    bot = xyz[xyz[:, 1] < np.percentile(xyz[:, 1], 5)]
    print(f"  {fname}: bottom-5% centroid splat XYZ = {bot.mean(axis=0).round(4)}")
    all_obj_bottoms.append(bot.mean(axis=0))
all_obj_bottoms = np.array(all_obj_bottoms)
mean_bottom_y = all_obj_bottoms[:, 1].mean()
print(f"  Mean of object-bottom splat-Y across 3 objects: {mean_bottom_y:+.4f}")

# ============================================================
# Now invert THAT splat Y back to metric: what metric Y should we use to seed?
# ============================================================
print("\n  If we want the patch to sit at splat Y = {:.4f},".format(mean_bottom_y))
print("  what METRIC Y must we feed to the forward map?")
# Forward: splat = dp_scale * (R @ metric/colmap_to_metric + t)
# Inverse: metric = ((splat/dp_scale - t) @ R) * colmap_to_metric
target_splat_y = mean_bottom_y
# Use median X/Z of patch as reference
mean_patch_xz = pxyz[:, [0, 2]].mean(axis=0)
target_s = np.array([[mean_patch_xz[0], target_splat_y, mean_patch_xz[1]]])
target_m = ((target_s / dp_scale - t[None, :]) @ R) * colmap_to_metric
print(f"  Target splat XYZ = {target_s[0].round(4)}")
print(f"  Inverted target METRIC XYZ = {target_m[0].round(4)} m")
print(f"  ==> Should have used --desk-y ~ {target_m[0,1]:.4f} m  "
      f"(instead of 0.0020)")

# Difference vs current
patch_inv = ((pxyz / dp_scale - t[None, :]) @ R) * colmap_to_metric
print(f"\n  Patch currently at metric Y = {np.median(patch_inv[:,1]):+.4f} m")
print(f"  Patch should be at metric Y  = {target_m[0,1]:+.4f} m")
print(f"  delta metric Y = {target_m[0,1] - np.median(patch_inv[:,1]):+.4f} m")
print(f"  delta splat Y  = {target_splat_y - pxyz[:,1].mean():+.4f}")

# ============================================================
# Sanity: the bottle "extracted-mesh's 2nd percentile vertex Y" gave 0.002 m.
# Where is the bottle's extracted MESH? Find it.
# ============================================================
print("\n" + "=" * 70)
print("Find bottle extracted mesh and check its 2nd percentile vertex Y")
print("=" * 70)
for root, dirs, files in os.walk(ROOT / "output"):
    for f in files:
        if "white_water_bottle" in f.lower() and (f.endswith(".ply") or f.endswith(".obj") or f.endswith(".glb")):
            fp = Path(root) / f
            print(f"  {fp}")
