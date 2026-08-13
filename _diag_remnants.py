"""Spatial diagnosis of scene_without_objects.ply remnants for data4.

For each object:
  1. Load scene_without_objects.ply mesh and extracted_object.ply mesh.
  2. Compute object AABB.
  3. Find scene verts inside expanded XZ AABB, above Y_min + 2 cm.
  4. Compare centroid of remnants vs centroid of extracted mesh (XZ plane).
  5. Coverage ratio: fraction of AABB XZ area with remnant density > 0.
  6. Also compute radial distribution to distinguish periphery vs center.
"""
from pathlib import Path
import numpy as np
import trimesh
import json
import sys

ROOT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
SCENE_WITHOUT = ROOT / "output/pipeline_data4_full_run/mesh/scene_without_objects.ply"
SEG_ROOT = ROOT / "output/segmented_data4_v9a_fp_v2"

OBJECTS = ["tape_measure", "potted_artificial_plant", "cardboard_box"]

print(f"Loading scene_without_objects: {SCENE_WITHOUT}")
scene = trimesh.load(str(SCENE_WITHOUT), process=False)
scene_verts = np.asarray(scene.vertices)
scene_colors = None
try:
    scene_colors = np.asarray(scene.visual.vertex_colors)[:, :3]
except Exception:
    pass
print(f"Scene verts: {len(scene_verts)}")
print(f"Scene AABB: min={scene_verts.min(0)}, max={scene_verts.max(0)}")

results = {}
for obj in OBJECTS:
    ext_path = SEG_ROOT / obj / f"{obj}_extracted.ply"
    if not ext_path.exists():
        print(f"MISSING: {ext_path}")
        continue
    ext = trimesh.load(str(ext_path), process=False)
    ext_verts = np.asarray(ext.vertices)
    aabb_min = ext_verts.min(0)
    aabb_max = ext_verts.max(0)
    ext_center = ext_verts.mean(0)

    # AABB in XZ, plus 2cm cushion for boundary
    xmin, ymin, zmin = aabb_min
    xmax, ymax, zmax = aabb_max
    size = aabb_max - aabb_min

    # Filter scene verts: inside XZ AABB, and Y > ymin + 2cm
    mask = (
        (scene_verts[:, 0] >= xmin) & (scene_verts[:, 0] <= xmax) &
        (scene_verts[:, 2] >= zmin) & (scene_verts[:, 2] <= zmax) &
        (scene_verts[:, 1] >= ymin + 0.02) & (scene_verts[:, 1] <= ymax + 0.02)
    )
    remnant = scene_verts[mask]
    remnant_colors = scene_colors[mask] if scene_colors is not None else None
    n_rem = int(mask.sum())

    print(f"\n=== {obj} ===")
    print(f"  AABB min: [{xmin:.3f}, {ymin:.3f}, {zmin:.3f}]")
    print(f"  AABB max: [{xmax:.3f}, {ymax:.3f}, {zmax:.3f}]")
    print(f"  Size XYZ (cm): [{size[0]*100:.1f}, {size[1]*100:.1f}, {size[2]*100:.1f}]")
    print(f"  Ext center: [{ext_center[0]:.3f}, {ext_center[1]:.3f}, {ext_center[2]:.3f}]")
    print(f"  Remnants (in XZ AABB, y>ymin+2cm): {n_rem}")

    if n_rem == 0:
        results[obj] = {"n_remnants": 0}
        continue

    rem_center = remnant.mean(0)
    print(f"  Rem center: [{rem_center[0]:.3f}, {rem_center[1]:.3f}, {rem_center[2]:.3f}]")

    # Centroid offset from object center (XZ only)
    offset_xz = np.linalg.norm(rem_center[[0, 2]] - ext_center[[0, 2]])
    half_diag_xz = np.linalg.norm([size[0], size[2]]) / 2.0
    offset_ratio = offset_xz / half_diag_xz if half_diag_xz > 0 else 0
    print(f"  Centroid offset (XZ) from ext center: {offset_xz*100:.2f} cm")
    print(f"  Half XZ diag: {half_diag_xz*100:.2f} cm, offset ratio: {offset_ratio:.3f}")

    # Radial distribution: distance from object XZ center normalized by half XZ diag
    dxz = np.linalg.norm(remnant[:, [0, 2]] - ext_center[[0, 2]], axis=1)
    # Normalize by max in-AABB XZ distance
    max_xz_r = np.linalg.norm([size[0]/2, size[2]/2])
    d_norm = dxz / max_xz_r if max_xz_r > 0 else dxz
    frac_center = float((d_norm < 0.33).mean())  # inner 1/3
    frac_mid = float(((d_norm >= 0.33) & (d_norm < 0.67)).mean())
    frac_edge = float((d_norm >= 0.67).mean())
    print(f"  Radial distribution (fraction):")
    print(f"    inner (0-0.33 R): {frac_center:.3f}")
    print(f"    mid   (0.33-0.67 R): {frac_mid:.3f}")
    print(f"    edge  (0.67+ R): {frac_edge:.3f}")

    # Coverage ratio: XZ grid at 2cm resolution
    cell = 0.02  # 2 cm
    nx = max(1, int(np.ceil(size[0] / cell)))
    nz = max(1, int(np.ceil(size[2] / cell)))
    ix = np.clip(((remnant[:, 0] - xmin) / cell).astype(int), 0, nx - 1)
    iz = np.clip(((remnant[:, 2] - zmin) / cell).astype(int), 0, nz - 1)
    grid = np.zeros((nx, nz), dtype=bool)
    grid[ix, iz] = True
    coverage = float(grid.sum() / (nx * nz))
    print(f"  XZ grid: {nx}x{nz} cells @ 2cm, occupied {int(grid.sum())} / {nx*nz}")
    print(f"  Coverage ratio: {coverage:.3f}")

    # Y distribution: how far above the object bottom
    y_rel = remnant[:, 1] - ymin
    y_stats = {
        "mean_cm": float(y_rel.mean() * 100),
        "median_cm": float(np.median(y_rel) * 100),
        "p90_cm": float(np.percentile(y_rel, 90) * 100),
        "obj_height_cm": float(size[1] * 100),
    }
    print(f"  Y relative to object bottom (cm): mean={y_stats['mean_cm']:.1f}, median={y_stats['median_cm']:.1f}, p90={y_stats['p90_cm']:.1f}, obj_h={y_stats['obj_height_cm']:.1f}")

    # Distance to nearest extracted-mesh face (3D)
    try:
        # Use ProximityQuery for exact distance
        pq = trimesh.proximity.ProximityQuery(ext)
        # Distance may be slow for many verts, use random sample
        if len(remnant) > 5000:
            idx = np.random.default_rng(42).choice(len(remnant), 5000, replace=False)
            sample = remnant[idx]
        else:
            sample = remnant
        d = pq.signed_distance(sample)
        # Use unsigned
        d = np.abs(d)
        dist_stats = {
            "mean_cm": float(d.mean() * 100),
            "median_cm": float(np.median(d) * 100),
            "p10_cm": float(np.percentile(d, 10) * 100),
            "p90_cm": float(np.percentile(d, 90) * 100),
        }
        frac_within_5cm = float((d <= 0.05).mean())
        frac_within_10cm = float((d <= 0.10).mean())
        frac_within_15cm = float((d <= 0.15).mean())
        print(f"  3D dist to extracted mesh (cm): mean={dist_stats['mean_cm']:.1f}, median={dist_stats['median_cm']:.1f}, p10={dist_stats['p10_cm']:.1f}, p90={dist_stats['p90_cm']:.1f}")
        print(f"  Fraction of remnants within: 5cm={frac_within_5cm:.3f}, 10cm={frac_within_10cm:.3f}, 15cm={frac_within_15cm:.3f}")
    except Exception as e:
        print(f"  ProximityQuery failed: {e}")
        dist_stats = None
        frac_within_5cm = frac_within_10cm = frac_within_15cm = None

    # Color analysis
    if remnant_colors is not None:
        rgb_mean = remnant_colors.mean(0)
        # Tan-like: R,G ~150-170, B ~90-110. Extracted object colors for reference
        print(f"  Remnant mean RGB: ({rgb_mean[0]:.0f}, {rgb_mean[1]:.0f}, {rgb_mean[2]:.0f})")
        # Compare against extracted mesh color
        try:
            ext_c = np.asarray(ext.visual.vertex_colors)[:, :3]
            ext_rgb = ext_c.mean(0)
            print(f"  Extracted mesh mean RGB: ({ext_rgb[0]:.0f}, {ext_rgb[1]:.0f}, {ext_rgb[2]:.0f})")
        except Exception:
            pass

    results[obj] = {
        "n_remnants": n_rem,
        "aabb_size_cm": [float(size[0]*100), float(size[1]*100), float(size[2]*100)],
        "ext_center": ext_center.tolist(),
        "rem_center": rem_center.tolist(),
        "offset_xz_cm": float(offset_xz*100),
        "offset_ratio": float(offset_ratio),
        "radial_frac": {"inner": frac_center, "mid": frac_mid, "edge": frac_edge},
        "coverage_ratio": coverage,
        "y_stats_cm": y_stats,
        "dist_to_ext_cm": dist_stats,
        "frac_within_5cm": frac_within_5cm,
        "frac_within_10cm": frac_within_10cm,
        "frac_within_15cm": frac_within_15cm,
    }

print("\n\n=== SUMMARY JSON ===")
print(json.dumps(results, indent=2, default=str))

out_json = ROOT / "_diag_remnants.json"
out_json.write_text(json.dumps(results, indent=2, default=str))
print(f"\nWrote {out_json}")
