"""Deeper investigation: mesh and splat disagree about desk Y sign.
Look at face normals both ways, and locate the actual desk geometry."""
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
OBJ_SPLAT_DIR = ROOT / "output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned"


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


dp_R, dp_t, dp_scale = load_dataparser()
colmap_to_metric = load_colmap_to_metric()

# ============================================================
# Load mesh
# ============================================================
print("=" * 70)
print("MESH normal/orientation diagnosis")
print("=" * 70)
mesh = trimesh.load(str(MESH_PATH), force="mesh", process=False)
V = np.asarray(mesh.vertices, dtype=np.float64)
F = np.asarray(mesh.faces, dtype=np.int64)
fn = mesh.face_normals
fa = mesh.area_faces
fc = V[F].mean(axis=1)

# What fraction of faces have normal aligned with each of +-X, +-Y, +-Z?
for ax, axis in zip(["X", "Y", "Z"], np.eye(3)):
    pos = np.dot(fn, axis) > 0.95
    neg = np.dot(fn, axis) < -0.95
    print(f"  +{ax}: {pos.sum():>7d} faces, area = {fa[pos].sum():.4f} m^2")
    print(f"  -{ax}: {neg.sum():>7d} faces, area = {fa[neg].sum():.4f} m^2")

# Look at BOTH +Y and -Y horizontal faces and their Y-centroid hist
print("\n  +Y-facing (up) face centroid Y bins:")
mask_up = fn[:, 1] > 0.95
bins = np.arange(-0.55, 0.55, 0.01)
counts_up_a, edges = np.histogram(fc[mask_up, 1], bins=bins, weights=fa[mask_up])
top_idx = np.argsort(counts_up_a)[-8:][::-1]
for i in top_idx:
    print(f"    Y in [{edges[i]:+.3f}, {edges[i+1]:+.3f}] m  ->  {counts_up_a[i]:.5f} m^2")

print("\n  -Y-facing (down) face centroid Y bins:")
mask_dn = fn[:, 1] < -0.95
counts_dn_a, _ = np.histogram(fc[mask_dn, 1], bins=bins, weights=fa[mask_dn])
top_idx = np.argsort(counts_dn_a)[-8:][::-1]
for i in top_idx:
    print(f"    Y in [{edges[i]:+.3f}, {edges[i+1]:+.3f}] m  ->  {counts_dn_a[i]:.5f} m^2")

# Maybe mesh "up" is some other axis. Look at total face area per dominant axis.
print("\n  Total area by dominant face-normal axis (cos > 0.95):")
for name, vec in [("+X", [1,0,0]), ("-X", [-1,0,0]),
                   ("+Y", [0,1,0]), ("-Y", [0,-1,0]),
                   ("+Z", [0,0,1]), ("-Z", [0,0,-1])]:
    m = np.dot(fn, vec) > 0.95
    print(f"    {name}: {fa[m].sum():.4f} m^2  ({m.sum()} faces)")

# ============================================================
# Check: invert one of the OBJECT splat PLYs to metric and see where they sit
# ============================================================
print("\n" + "=" * 70)
print("OBJECT splat PLYs (per-object): invert to metric, find Y range")
print("=" * 70)
import os
for fname in os.listdir(OBJ_SPLAT_DIR):
    if not fname.endswith("_splat.ply"):
        continue
    fp = OBJ_SPLAT_DIR / fname
    p = PlyData.read(str(fp))
    v = p["vertex"]
    xyz_s = np.stack([
        np.asarray(v["x"], dtype=np.float64),
        np.asarray(v["y"], dtype=np.float64),
        np.asarray(v["z"], dtype=np.float64),
    ], axis=1)
    xyz_colmap = (xyz_s / dp_scale - dp_t[None, :]) @ dp_R
    xyz_m = xyz_colmap * colmap_to_metric
    print(f"  {fname}: n={len(xyz_s):,}")
    print(f"    splat Y: [{xyz_s[:,1].min():.3f}, {xyz_s[:,1].max():.3f}]")
    print(f"    metric Y: [{xyz_m[:,1].min():.4f}, {xyz_m[:,1].max():.4f}] m")
    print(f"    metric centroid: {xyz_m.mean(axis=0).round(4)}")
    print(f"    metric Y p02, p50, p98: "
          f"{np.percentile(xyz_m[:,1],2):.4f}, "
          f"{np.percentile(xyz_m[:,1],50):.4f}, "
          f"{np.percentile(xyz_m[:,1],98):.4f}")

# ============================================================
# Re-examine: where does the SPLAT's densest horizontal slab sit?
# A desk should appear as a thick slab in Y. Find the LOWEST high-density slab.
# ============================================================
print("\n" + "=" * 70)
print("SPLAT inverted-metric Y - looking for the desk slab")
print("=" * 70)
splat = PlyData.read(str(SPLAT_PATH))
v = splat["vertex"]
sxyz = np.stack([
    np.asarray(v["x"], dtype=np.float64),
    np.asarray(v["y"], dtype=np.float64),
    np.asarray(v["z"], dtype=np.float64),
], axis=1)
mxyz = ((sxyz / dp_scale - dp_t[None, :]) @ dp_R) * colmap_to_metric

# The desk top should have lots of Gaussians at one Y level over a wide XZ area.
# Pick gaussians with Y in [-0.5, 0.5] and look at their XZ extent per Y bin.
bins3 = np.arange(-0.55, 0.55, 0.01)
print("  Per-bin: count, XZ extent (range in X)*(range in Z) - desk should be wide")
for lo in np.arange(-0.5, 0.45, 0.05):
    m = (mxyz[:, 1] >= lo) & (mxyz[:, 1] < lo + 0.05)
    if m.sum() < 50:
        continue
    xrng = mxyz[m, 0].ptp()
    zrng = mxyz[m, 2].ptp()
    print(f"    Y in [{lo:+.2f}, {lo+0.05:+.2f}]  n={m.sum():>5d}  "
          f"X-range={xrng:.3f}  Z-range={zrng:.3f}  area~{xrng*zrng:.3f}")

# ============================================================
# What if "desk" should be defined from the OBJECT bottoms, not from a plane fit?
# The bottle, blue_box, and lobster all sit ON the desk; their BOTTOMS = desk plane
# ============================================================
print("\n" + "=" * 70)
print("Object splat BOTTOM Y - the desk is where objects sit")
print("=" * 70)
for fname in os.listdir(OBJ_SPLAT_DIR):
    if not fname.endswith("_splat.ply"):
        continue
    fp = OBJ_SPLAT_DIR / fname
    p = PlyData.read(str(fp))
    v = p["vertex"]
    xyz_s = np.stack([
        np.asarray(v["x"], dtype=np.float64),
        np.asarray(v["y"], dtype=np.float64),
        np.asarray(v["z"], dtype=np.float64),
    ], axis=1)
    xyz_m = ((xyz_s / dp_scale - dp_t[None, :]) @ dp_R) * colmap_to_metric
    # Find what Y the desk would be at for this object
    print(f"  {fname}:")
    print(f"    splat Y min (object bottom in splat space): {xyz_s[:,1].min():.4f}")
    print(f"    splat Y p02: {np.percentile(xyz_s[:,1], 2):.4f}")
    print(f"    metric Y min (object bottom in metric): {xyz_m[:,1].min():.4f}")
    print(f"    metric Y p02: {np.percentile(xyz_m[:,1], 2):.4f}")
    print(f"    metric Y p98 (object top): {np.percentile(xyz_m[:,1], 98):.4f}")
    print(f"    => object metric height = "
          f"{np.percentile(xyz_m[:,1],98) - np.percentile(xyz_m[:,1],2):.4f} m")

# ============================================================
# What's the relationship between splat Y and metric Y?
# Pick a few sample splat Y values and see the metric Y they invert to.
# ============================================================
print("\n" + "=" * 70)
print("Splat-Y <-> Metric-Y mapping (at XZ=0)")
print("=" * 70)
for sy in np.arange(-0.5, 2.0, 0.25):
    test_s = np.array([[0.0, sy, 0.0]])
    test_m = ((test_s / dp_scale - dp_t[None, :]) @ dp_R) * colmap_to_metric
    print(f"  splat XYZ=(0, {sy:+.2f}, 0)  ->  metric XYZ="
          f"({test_m[0,0]:+.4f}, {test_m[0,1]:+.4f}, {test_m[0,2]:+.4f}) m")

# And forward: a few metric Y test values
print("\n  Forward (metric -> splat) at metric XZ=0:")
for my in np.arange(-0.5, 0.5, 0.1):
    test_m = np.array([[0.0, my, 0.0]])
    test_colmap = test_m / colmap_to_metric
    test_s = dp_scale * (test_colmap @ dp_R.T + dp_t[None, :])
    print(f"  metric XYZ=(0, {my:+.2f}, 0)  ->  splat XYZ="
          f"({test_s[0,0]:+.4f}, {test_s[0,1]:+.4f}, {test_s[0,2]:+.4f})")
