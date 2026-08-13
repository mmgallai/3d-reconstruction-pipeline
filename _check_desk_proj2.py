"""Refined: find the desk plane by looking near where objects sit, and
sanity-check the offset using the white_water_bottle per-object splat
(whose base = desk surface)."""
import numpy as np
from plyfile import PlyData

SCENE_PLY   = r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project/output/splat_v32_data3_noinit_pruned.ply"
PATCH_PLY   = r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project/output/desk_patch_v32_data3/desk_patch_white_water_bottle.ply"
BOTTLE_PLY  = r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project/output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned/white_water_bottle_splat.ply"
BLUE_PLY    = r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project/output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned/blue_box_splat.ply"
LOBSTER_PLY = r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project/output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned/red_lobster_figurine_splat.ply"

# Desk normal in splat space
N = np.array([0.051, 0.064, -0.997], dtype=np.float64)
N = N / np.linalg.norm(N)

def load_xyz(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


scene = load_xyz(SCENE_PLY)
patch = load_xyz(PATCH_PLY)
bottle = load_xyz(BOTTLE_PLY)
blue   = load_xyz(BLUE_PLY)
lob    = load_xyz(LOBSTER_PLY)

proj_scene = scene @ N
proj_patch = patch @ N
proj_bot   = bottle @ N
proj_blue  = blue @ N
proj_lob   = lob @ N

print(f"Patch (white_water_bottle) bbox in splat space:")
print(f"  min={patch.min(axis=0)}")
print(f"  max={patch.max(axis=0)}")
print(f"Patch centroid: {patch.mean(axis=0)}")
print(f"Patch proj on N: min={proj_patch.min():.4f}, max={proj_patch.max():.4f}, "
      f"median={np.median(proj_patch):.4f}")

print()
print(f"White-water-bottle object splat ({len(bottle)} G) bbox:")
print(f"  min={bottle.min(axis=0)}")
print(f"  max={bottle.max(axis=0)}")
print(f"  proj on N: min={proj_bot.min():.4f}, max={proj_bot.max():.4f}, "
      f"median={np.median(proj_bot):.4f}, 99%={np.percentile(proj_bot,99):.4f}, "
      f"p98={np.percentile(proj_bot,98):.4f}")
print(f"  The BASE of the bottle (resting on desk) = MAX proj direction.")
print(f"  Top 1% percentile proj (likely on desk): {np.percentile(proj_bot,99):.4f}")

print()
print(f"Blue box object splat ({len(blue)} G) bbox:")
print(f"  min={blue.min(axis=0)}, max={blue.max(axis=0)}")
print(f"  proj on N: min={proj_blue.min():.4f}, max={proj_blue.max():.4f}, "
      f"median={np.median(proj_blue):.4f}, p99={np.percentile(proj_blue,99):.4f}")

print()
print(f"Red lobster object splat ({len(lob)} G) bbox:")
print(f"  min={lob.min(axis=0)}, max={lob.max(axis=0)}")
print(f"  proj on N: min={proj_lob.min():.4f}, max={proj_lob.max():.4f}, "
      f"median={np.median(proj_lob):.4f}, p99={np.percentile(proj_lob,99):.4f}")

# Look at scene Gaussians IN the patch XY region (near bottle's XZ footprint),
# and find the densest proj-value there — that's the local desk surface.
print("\n" + "=" * 70)
print("LOCAL desk search: scene Gaussians near patch XY footprint")
pxmin, pymin, pzmin = patch.min(axis=0)
pxmax, pymax, pzmax = patch.max(axis=0)
# Use a generous box around the patch XY (since N is mostly -Z, the patch
# is a horizontal-ish disc and the desk plane extends in X,Y here)
print(f"Patch footprint X=[{pxmin:.3f},{pxmax:.3f}] Y=[{pymin:.3f},{pymax:.3f}] Z=[{pzmin:.3f},{pzmax:.3f}]")
# Pull scene points with X,Y near patch and Z anywhere
mask = ((scene[:, 0] >= pxmin - 0.05) & (scene[:, 0] <= pxmax + 0.05) &
        (scene[:, 1] >= pymin - 0.05) & (scene[:, 1] <= pymax + 0.05))
nearby = scene[mask]
print(f"Scene Gaussians near patch footprint (XY box): {nearby.shape[0]}")
if nearby.shape[0] > 100:
    proj_nearby = nearby @ N
    print(f"  proj stats: min={proj_nearby.min():.4f}, max={proj_nearby.max():.4f}, "
          f"median={np.median(proj_nearby):.4f}")
    # Modal value (densest cluster)
    hist, edges = np.histogram(proj_nearby, bins=200)
    mode_bin = np.argmax(hist)
    mode_c = 0.5 * (edges[mode_bin] + edges[mode_bin + 1])
    print(f"  Local modal proj (densest cluster): {mode_c:.4f}, count={hist[mode_bin]}")
    # Show top 5 bins
    top5 = np.argsort(hist)[-5:][::-1]
    print("  Top 5 densest local bins:")
    for b in top5:
        c = 0.5 * (edges[b] + edges[b + 1])
        print(f"    proj~{c:+.4f}  count={hist[b]}")

# Convert splat -> metric
SPLAT_PER_M = 0.17305755937705114 * 8.798
print("\n" + "=" * 70)
print("METRIC OFFSETS (1 metric meter = %.4f splat units):" % SPLAT_PER_M)

p_patch = np.median(proj_patch)
p_bot_base = np.percentile(proj_bot, 99)  # base of bottle = max proj
p_bot_top  = np.percentile(proj_bot, 1)   # top of bottle = min proj
print(f"  patch median proj           : {p_patch:+.4f}")
print(f"  bottle base (99%) proj      : {p_bot_base:+.4f}")
print(f"  bottle top  (1%)  proj      : {p_bot_top:+.4f}")
print(f"  bottle height (splat units) : {p_bot_base - p_bot_top:.4f}")
print(f"  bottle height (metric mm)   : {(p_bot_base - p_bot_top) / SPLAT_PER_M * 1000:.1f} mm")

offset_to_bot_base = p_patch - p_bot_base
print(f"\n  offset patch_med - bottle_base : {offset_to_bot_base:+.4f} splat")
print(f"  offset in metric mm           : {offset_to_bot_base / SPLAT_PER_M * 1000:+.1f} mm")
print("\n  Sign interpretation: N points UP relative to desk-floor direction;")
print("    bottle base on desk = high proj; bottle top in air = low proj.")
print(f"    If patch_med << bottle_base, patch is BELOW desk (sunk).")
print(f"    If patch_med >> bottle_base, patch is ABOVE desk (floating).")
