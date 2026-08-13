"""Cross-check: all three object bases should cluster at the same desk proj.
Then check the patch proj for each of the three patch PLYs."""
import numpy as np
from plyfile import PlyData

def load_xyz(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

N = np.array([0.051, 0.064, -0.997], dtype=np.float64); N /= np.linalg.norm(N)
SPLAT_PER_M = 0.17305755937705114 * 8.798  # 1.5226

objs = {
    "white_water_bottle":     (r"output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned/white_water_bottle_splat.ply",
                                r"output/desk_patch_v32_data3/desk_patch_white_water_bottle.ply"),
    "blue_box":               (r"output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned/blue_box_splat.ply",
                                r"output/desk_patch_v32_data3/desk_patch_blue_box.ply"),
    "red_lobster_figurine":   (r"output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned/red_lobster_figurine_splat.ply",
                                r"output/desk_patch_v32_data3/desk_patch_red_lobster_figurine.ply"),
}

print(f"{'slug':<25} {'base_p99':>10} {'top_p1':>10} {'h_mm':>8}   {'patch_med':>10}  {'offset_mm':>10}")
for slug, (obj_p, patch_p) in objs.items():
    obj = load_xyz("C:/Users/mgallai/Projects/3d_automated/reconstruction_project/" + obj_p)
    patch = load_xyz("C:/Users/mgallai/Projects/3d_automated/reconstruction_project/" + patch_p)
    p_obj = obj @ N
    p_patch = patch @ N
    base = np.percentile(p_obj, 99)
    top  = np.percentile(p_obj, 1)
    pmed = np.median(p_patch)
    off_mm = (pmed - base) / SPLAT_PER_M * 1000
    h_mm = (base - top) / SPLAT_PER_M * 1000
    print(f"{slug:<25} {base:>10.4f} {top:>10.4f} {h_mm:>8.1f}   {pmed:>10.4f}  {off_mm:>10.1f}")

# Also: where is the floor? Most-negative proj cluster:
SCENE = load_xyz(r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project/output/splat_v32_data3_noinit_pruned.ply")
proj_scene = SCENE @ N

# Locate desk plane more robustly: take a 1000-bin histogram restricted to the
# desk-likely band (around object bases ~ 0.35)
band = proj_scene[(proj_scene > 0.30) & (proj_scene < 0.45)]
print(f"\nScene Gaussians in desk band [0.30, 0.45]: {band.size}")
hist, edges = np.histogram(band, bins=100)
mode_bin = np.argmax(hist)
mode_c = 0.5*(edges[mode_bin]+edges[mode_bin+1])
print(f"  modal proj in desk band: {mode_c:.4f}  count={hist[mode_bin]}")
top5 = np.argsort(hist)[-5:][::-1]
for b in top5:
    c = 0.5*(edges[b]+edges[b+1])
    print(f"    proj~{c:+.4f}  count={hist[b]}")
