"""Check where the desk surface sits in splat space vs. where the patch sits."""
import numpy as np
from plyfile import PlyData

SCENE_PLY = r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project/output/splat_v32_data3_noinit_pruned.ply"
PATCH_PLY = r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project/output/desk_patch_v32_data3/desk_patch_white_water_bottle.ply"

# Desk normal in splat space (from assumption 3 in the prompt)
N = np.array([0.051, 0.064, -0.997], dtype=np.float64)
N = N / np.linalg.norm(N)


def load_xyz(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    return xyz


print("=" * 70)
print("Loading scene splat:", SCENE_PLY)
scene_xyz = load_xyz(SCENE_PLY)
print(f"  scene Gaussians: {scene_xyz.shape[0]}")
print(f"  scene bbox: min={scene_xyz.min(axis=0)}, max={scene_xyz.max(axis=0)}")

print("\nLoading patch splat:", PATCH_PLY)
patch_xyz = load_xyz(PATCH_PLY)
print(f"  patch Gaussians: {patch_xyz.shape[0]}")
print(f"  patch bbox: min={patch_xyz.min(axis=0)}, max={patch_xyz.max(axis=0)}")

# Projections onto desk normal
proj_scene = scene_xyz @ N
proj_patch = patch_xyz @ N

print(f"\nDesk normal axis (splat space): {N}")
print(f"\nScene projection stats:")
print(f"  min={proj_scene.min():.4f}, max={proj_scene.max():.4f}")
print(f"  mean={proj_scene.mean():.4f}, median={np.median(proj_scene):.4f}")
print(f"  percentiles: 1%={np.percentile(proj_scene,1):.4f}, "
      f"5%={np.percentile(proj_scene,5):.4f}, "
      f"50%={np.percentile(proj_scene,50):.4f}, "
      f"95%={np.percentile(proj_scene,95):.4f}, "
      f"99%={np.percentile(proj_scene,99):.4f}")

print(f"\nPatch projection stats:")
print(f"  min={proj_patch.min():.4f}, max={proj_patch.max():.4f}")
print(f"  mean={proj_patch.mean():.4f}, median={np.median(proj_patch):.4f}")

# Build a histogram of scene projections to find the modal value
# Use 1mm bins in splat units (~0.0015 splat units = 1mm if scale_factor ~1.5)
# But we don't know the splat scale, so use 200 bins across the proj range
print("\n" + "=" * 70)
print("Histogram of scene projections (200 bins) — find modal value")
hist, edges = np.histogram(proj_scene, bins=200)
mode_bin = np.argmax(hist)
mode_lo, mode_hi = edges[mode_bin], edges[mode_bin + 1]
mode_center = 0.5 * (mode_lo + mode_hi)
print(f"  modal bin: [{mode_lo:.4f}, {mode_hi:.4f}], center={mode_center:.4f}, "
      f"count={hist[mode_bin]}")

# Top 10 densest bins
top10 = np.argsort(hist)[-10:][::-1]
print(f"\n  Top 10 densest bins (sorted by count):")
for b in top10:
    c = 0.5 * (edges[b] + edges[b + 1])
    print(f"    proj~{c:+.4f}  count={hist[b]}")

# Use a finer histogram limited to the densest region for refinement
# The desk should be near the HIGH end of proj (positive) since normal points
# from desk into the air above. Actually: above the desk = AWAY from desk normal,
# so going up subtracts from proj. So desk surface = MAXIMUM proj-cluster.
# But scene also contains the floor, walls etc, so find the densest cluster.
print("\n" + "=" * 70)
print("Refined: histogram with 1000 bins over full range")
hist2, edges2 = np.histogram(proj_scene, bins=1000)
mode_bin2 = np.argmax(hist2)
mode_center2 = 0.5 * (edges2[mode_bin2] + edges2[mode_bin2 + 1])
print(f"  modal proj (1000-bin): {mode_center2:.4f}  count={hist2[mode_bin2]}")

# Compare with patch median
patch_median = np.median(proj_patch)
offset_splat = patch_median - mode_center2
print("\n" + "=" * 70)
print("COMPARISON:")
print(f"  P_desk_splat (scene modal proj): {mode_center2:+.4f}")
print(f"  patch median proj:               {patch_median:+.4f}")
print(f"  offset (patch - desk):           {offset_splat:+.4f} splat units")

# Convert splat units to metric using dataparser scale and colmap scale
# We need: splat_unit -> metric meters
# splat_pos = dp_scale * (R @ colmap_pos + t)
# colmap_pos = metric / colmap_to_metric  ;  colmap_to_metric = 1/8.798 m
# So splat distance ~= dp_scale * colmap distance = dp_scale * 8.798 * metric distance
# Need dp_scale from dataparser_transforms.json
import json
DP = r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project/nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json"
try:
    dp = json.load(open(DP))
    print(f"\n  dataparser keys: {list(dp.keys())}")
    if "scale" in dp:
        dp_scale = dp["scale"]
        print(f"  dp_scale = {dp_scale}")
        # splat_dist = dp_scale * colmap_dist
        # colmap_dist = 8.798 * metric_dist  (since colmap_to_metric = 1/8.798)
        splat_per_metric = dp_scale * 8.798
        print(f"  1 metric meter = {splat_per_metric:.4f} splat units")
        print(f"  offset in metric meters = {offset_splat / splat_per_metric * 1000:+.2f} mm")
except Exception as e:
    print(f"  could not load dp transforms: {e}")
