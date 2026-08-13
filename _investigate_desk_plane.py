"""Empirical investigation: where IS the desk plane in metric world coords?"""
from __future__ import annotations
import json
import sys
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
PATCH_PATH = ROOT / "output/desk_patch_v32_data3/desk_patch_white_water_bottle.ply"
NPZ_PATH = ROOT / "output/unseen_core_v32_data3/unseen_core_white_water_bottle.npz"


def load_dataparser():
    d = json.loads(DP_PATH.read_text())
    T = np.asarray(d["transform"], dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    s = float(d["scale"])
    return R, t, s


def load_colmap_to_metric():
    b = json.loads(BOUNDS_PATH.read_text())
    return 1.0 / float(b["scale_factor_da3_to_colmap"])


# ============================================================
# 1) MESH - vertex Y histogram
# ============================================================
print("=" * 70)
print("1) MESH vertex Y histogram (mesh_v32_data3_openmvs.ply)")
print("=" * 70)
mesh = trimesh.load(str(MESH_PATH), force="mesh", process=False)
V = np.asarray(mesh.vertices, dtype=np.float64)
F = np.asarray(mesh.faces, dtype=np.int64)
print(f"  n_vertices = {len(V):,}, n_faces = {len(F):,}")
print(f"  X range: [{V[:,0].min():.4f}, {V[:,0].max():.4f}] m")
print(f"  Y range: [{V[:,1].min():.4f}, {V[:,1].max():.4f}] m")
print(f"  Z range: [{V[:,2].min():.4f}, {V[:,2].max():.4f}] m")
print(f"  Y percentiles (raw vertices):")
for p in [1, 2, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    p{p:02d} = {np.percentile(V[:,1], p):+.4f} m")

# Histogram of vertex Y values in 1 cm bins
y_min, y_max = V[:, 1].min(), V[:, 1].max()
bins = np.arange(np.floor(y_min * 100) / 100, np.ceil(y_max * 100) / 100 + 0.01, 0.01)
counts, edges = np.histogram(V[:, 1], bins=bins)
top_idx = np.argsort(counts)[-10:][::-1]
print(f"\n  Top 10 vertex-Y bins (1cm bins):")
for i in top_idx:
    print(f"    Y in [{edges[i]:+.3f}, {edges[i+1]:+.3f}] m  ->  {counts[i]:>7d} verts")

# Identify nearly-horizontal faces (normal dot +Y > 0.95)
print("\n  Finding nearly-horizontal faces (face normal . +Y > 0.95):")
fn = mesh.face_normals
horiz_mask = fn[:, 1] > 0.95
print(f"    {horiz_mask.sum():,} horizontal faces out of {len(F):,}")
# Centroid Y of horizontal faces
face_centroids = V[F].mean(axis=1)
horiz_cy = face_centroids[horiz_mask, 1]
print(f"    Horizontal face centroid Y percentiles:")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"      p{p:02d} = {np.percentile(horiz_cy, p):+.4f} m")
counts_h, edges_h = np.histogram(horiz_cy, bins=bins)
top_idx_h = np.argsort(counts_h)[-10:][::-1]
print(f"    Top 10 horizontal-face-centroid Y bins:")
for i in top_idx_h:
    print(f"      Y in [{edges_h[i]:+.3f}, {edges_h[i+1]:+.3f}] m  ->  {counts_h[i]:>7d} faces")

# Weighted by area
face_areas = mesh.area_faces
print("\n  Area-weighted horizontal-face Y (weighted histogram, 1cm):")
counts_a, _ = np.histogram(horiz_cy, bins=bins, weights=face_areas[horiz_mask])
top_idx_a = np.argsort(counts_a)[-10:][::-1]
for i in top_idx_a:
    print(f"      Y in [{edges_h[i]:+.3f}, {edges_h[i+1]:+.3f}] m  ->  {counts_a[i]:>8.4f} m^2")

mesh_desk_y_mode = edges_h[np.argmax(counts_a)] + 0.005
print(f"\n  >>> MESH-DERIVED DESK Y (mode of area-weighted horizontal faces): "
      f"{mesh_desk_y_mode:+.4f} m  (center of 1cm bin)")

# ============================================================
# 2) V32 splat - invert to metric, then histogram Y
# ============================================================
print("\n" + "=" * 70)
print("2) V32 SPLAT inverted to metric (splat_v32_data3_noinit_pruned.ply)")
print("=" * 70)
splat = PlyData.read(str(SPLAT_PATH))
v = splat["vertex"]
sx = np.asarray(v["x"], dtype=np.float64)
sy = np.asarray(v["y"], dtype=np.float64)
sz = np.asarray(v["z"], dtype=np.float64)
splat_xyz = np.stack([sx, sy, sz], axis=1)
print(f"  n_gaussians = {len(splat_xyz):,}")
print(f"  splat-space X range: [{sx.min():.3f}, {sx.max():.3f}]")
print(f"  splat-space Y range: [{sy.min():.3f}, {sy.max():.3f}]")
print(f"  splat-space Z range: [{sz.min():.3f}, {sz.max():.3f}]")

# Inverse transform: metric = (splat/dp_scale - t) @ R * colmap_to_metric
dp_R, dp_t, dp_scale = load_dataparser()
colmap_to_metric = load_colmap_to_metric()
print(f"\n  dp_scale = {dp_scale:.6f}")
print(f"  colmap_to_metric = {colmap_to_metric:.6f} m/colmap_unit  "
      f"(scale_factor_da3_to_colmap = {1/colmap_to_metric:.4f})")
print(f"  dp_t = {dp_t}")
print(f"  dp_R =\n{dp_R}")

# Forward: splat = dp_scale * ((metric/colmap_to_metric) @ R^T + t)
#          splat = dp_scale * (R @ colmap + t)  where colmap = metric/colmap_to_metric
# Inverse: colmap = R^T @ (splat/dp_scale - t)
#          metric = colmap * colmap_to_metric
colmap_xyz = (splat_xyz / dp_scale - dp_t[None, :]) @ dp_R  # = R^T @ ...
metric_xyz = colmap_xyz * colmap_to_metric

print(f"\n  Inverted metric-space ranges:")
print(f"    X: [{metric_xyz[:,0].min():.4f}, {metric_xyz[:,0].max():.4f}] m")
print(f"    Y: [{metric_xyz[:,1].min():.4f}, {metric_xyz[:,1].max():.4f}] m")
print(f"    Z: [{metric_xyz[:,2].min():.4f}, {metric_xyz[:,2].max():.4f}] m")

my = metric_xyz[:, 1]
print(f"\n  Inverted-metric Y percentiles:")
for p in [1, 2, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    p{p:02d} = {np.percentile(my, p):+.4f} m")

bins2 = np.arange(np.floor(my.min() * 100) / 100,
                  np.ceil(my.max() * 100) / 100 + 0.01, 0.01)
counts2, edges2 = np.histogram(my, bins=bins2)
top_idx2 = np.argsort(counts2)[-10:][::-1]
print(f"\n  Top 10 metric-Y bins (1cm bins, inverted V32 splat):")
for i in top_idx2:
    print(f"    Y in [{edges2[i]:+.3f}, {edges2[i+1]:+.3f}] m  ->  {counts2[i]:>7d} gauss")

splat_desk_y_mode = edges2[np.argmax(counts2)] + 0.005
print(f"\n  >>> SPLAT-DERIVED desk Y (densest 1cm bin in inverted splat): "
      f"{splat_desk_y_mode:+.4f} m")

# ============================================================
# 3) V32ViewSource - camera positions metric Y
# ============================================================
print("\n" + "=" * 70)
print("3) Camera positions (V32ViewSource): median metric Y")
print("=" * 70)
from scene_segmenter.views import V32ViewSource
vs = V32ViewSource(ROOT)
cam_pos = np.array([v.c2w[:3, 3] for v in vs.all_views()])
print(f"  n_views = {len(cam_pos)}")
print(f"  Camera X range: [{cam_pos[:,0].min():.3f}, {cam_pos[:,0].max():.3f}] m")
print(f"  Camera Y range: [{cam_pos[:,1].min():.3f}, {cam_pos[:,1].max():.3f}] m")
print(f"  Camera Z range: [{cam_pos[:,2].min():.3f}, {cam_pos[:,2].max():.3f}] m")
print(f"  Camera Y percentiles:")
for p in [5, 25, 50, 75, 95]:
    print(f"    p{p:02d} = {np.percentile(cam_pos[:,1], p):+.4f} m")
print(f"  Camera Y median = {np.median(cam_pos[:,1]):+.4f} m")
print(f"  Camera Y mean   = {np.mean(cam_pos[:,1]):+.4f} m")

# ============================================================
# 4) Where is the PATCH currently sitting?
# ============================================================
print("\n" + "=" * 70)
print("4) The actual patch PLY in splat space")
print("=" * 70)
patch = PlyData.read(str(PATCH_PATH))
pv = patch["vertex"]
pxyz_splat = np.stack([
    np.asarray(pv["x"], dtype=np.float64),
    np.asarray(pv["y"], dtype=np.float64),
    np.asarray(pv["z"], dtype=np.float64),
], axis=1)
print(f"  patch n_gauss = {len(pxyz_splat):,}")
print(f"  patch splat-space X: [{pxyz_splat[:,0].min():.3f}, {pxyz_splat[:,0].max():.3f}]")
print(f"  patch splat-space Y: [{pxyz_splat[:,1].min():.3f}, {pxyz_splat[:,1].max():.3f}]")
print(f"  patch splat-space Z: [{pxyz_splat[:,2].min():.3f}, {pxyz_splat[:,2].max():.3f}]")
print(f"  patch CENTROID (splat space): "
      f"{pxyz_splat.mean(axis=0).round(4)}")

# Invert patch back to metric to see what metric Y it was actually placed at
patch_colmap = (pxyz_splat / dp_scale - dp_t[None, :]) @ dp_R
patch_metric = patch_colmap * colmap_to_metric
print(f"  patch inverted-metric Y range: [{patch_metric[:,1].min():.4f}, "
      f"{patch_metric[:,1].max():.4f}] m")
print(f"  patch inverted-metric Y median: {np.median(patch_metric[:,1]):+.4f} m")

# ============================================================
# 5) What does the unseen-core NPZ say about grid Y?
# ============================================================
print("\n" + "=" * 70)
print("5) Unseen-core grid (the source points for the patch)")
print("=" * 70)
data = np.load(NPZ_PATH)
print(f"  keys: {list(data.keys())}")
gpts = data["grid_points"].astype(np.float64)
print(f"  grid shape = {gpts.shape}")
print(f"  grid X: [{gpts[:,0].min():.4f}, {gpts[:,0].max():.4f}] m")
print(f"  grid Y: [{gpts[:,1].min():.4f}, {gpts[:,1].max():.4f}] m")
print(f"  grid Z: [{gpts[:,2].min():.4f}, {gpts[:,2].max():.4f}] m")
print(f"  grid Y unique values (first 5): {np.unique(gpts[:,1])[:5]}")
print(f"  grid Y median = {np.median(gpts[:,1]):+.6f} m")

# ============================================================
# 6) SUMMARY: compare hardcoded vs measured
# ============================================================
print("\n" + "=" * 70)
print("6) SUMMARY")
print("=" * 70)
print(f"  Hardcoded --desk-y (used by orchestrator):     +0.0020 m")
print(f"  Mesh area-weighted horizontal-face mode:       {mesh_desk_y_mode:+.4f} m")
print(f"  V32 splat (inverted to metric) densest bin:    {splat_desk_y_mode:+.4f} m")
print(f"  Patch median metric Y (inverted from output):  {np.median(patch_metric[:,1]):+.4f} m")
print(f"  Camera median metric Y:                        {np.median(cam_pos[:,1]):+.4f} m")
print(f"\n  Vertical offset (patch - true desk via mesh):  "
      f"{np.median(patch_metric[:,1]) - mesh_desk_y_mode:+.4f} m")
print(f"  Vertical offset (patch - true desk via splat): "
      f"{np.median(patch_metric[:,1]) - splat_desk_y_mode:+.4f} m")

# Convert offset to splat-space units for direct visibility scale
metric_to_splat_scalar = dp_scale / colmap_to_metric
print(f"\n  metric_to_splat scalar = {metric_to_splat_scalar:.5f} "
      f"(1 m metric -> {metric_to_splat_scalar:.3f} splat units)")
offset_mesh = (np.median(patch_metric[:,1]) - mesh_desk_y_mode) * metric_to_splat_scalar
offset_splat = (np.median(patch_metric[:,1]) - splat_desk_y_mode) * metric_to_splat_scalar
print(f"  Splat-space offset (vs mesh desk):  {offset_mesh:+.4f} splat units")
print(f"  Splat-space offset (vs splat mode): {offset_splat:+.4f} splat units")
