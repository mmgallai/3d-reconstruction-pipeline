"""Reviewer D — refinement: threshold sensitivity + bottle stub geometry
   deep-dive."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

_ROOT = Path(r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(_ROOT))
from _spatial_crop_splat import _load_dataparser_transform, _splat_to_metric  # noqa

V4 = _ROOT / "output" / "pipeline_v32_v11_patches_v4"
V2 = _ROOT / "output" / "pipeline_v32_v11_patches_v2"
SEG = _ROOT / "output" / "segmented_v32_data3_v9a_fp_v2"
UNSEEN = _ROOT / "output" / "unseen_core_v32_data3"
DP_JSON = _ROOT / "nerfstudio" / "dense" / "splatfacto" / "2026-06-07_203807" / "dataparser_transforms.json"
BOUNDS_JSON = _ROOT / "colmap" / "dense" / "tof_bounds.json"

_SH_C0 = 0.28209479177387814


def load_splat(path):
    R, t, dp_scale = _load_dataparser_transform(DP_JSON)
    bounds = json.loads(BOUNDS_JSON.read_text())
    colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])
    pd = PlyData.read(str(path))
    v = pd["vertex"].data
    pos = _splat_to_metric(
        np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64),
        R, t, dp_scale, colmap_to_metric)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], -1).astype(np.float32)
    rgb = np.clip(dc * _SH_C0 + 0.5, 0, 1)
    raw_s = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], -1).astype(np.float32)
    scales_colmap = np.exp(raw_s)
    scales_metric = scales_colmap * (colmap_to_metric * dp_scale)
    return dict(pos=pos, rgb=rgb, scales=scales_metric, scales_colmap=scales_colmap,
                dp_scale=dp_scale, colmap_to_metric=colmap_to_metric)


def dist_fn_from_mesh(mesh_ply):
    m = trimesh.load(str(mesh_ply), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.uint32)
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(verts)
    tm.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)
    y_min = float(verts[:, 1].min())
    y_max = float(verts[:, 1].max())
    def f(p): return scene.compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy()
    return f, y_min, y_max, verts


def analyze_stripe_thresholds(splat, mesh_ply, tag):
    dist_fn, y_min, y_max, _ = dist_fn_from_mesh(mesh_ply)
    dist = dist_fn(splat["pos"])
    within = dist <= 0.10  # 10 cm
    scales_m = splat["scales"]
    max_ax = scales_m.max(1)
    min_ax = np.maximum(scales_m.min(1), 1e-9)
    aniso = max_ax / min_ax
    above = splat["pos"][:, 1] > (y_max + 0.005)
    out = {"tag": tag,
           "n_within_10cm": int(within.sum()),
           "n_above_box_top": int((within & above).sum()),
           "max_axis_stats_cm_within": {
               "p50": float(np.median(max_ax[within]) * 100),
               "p90": float(np.percentile(max_ax[within], 90) * 100),
               "p95": float(np.percentile(max_ax[within], 95) * 100),
               "p99": float(np.percentile(max_ax[within], 99) * 100),
               "max": float(max_ax[within].max() * 100),
           },
           "aniso_stats_within": {
               "p50": float(np.median(aniso[within])),
               "p90": float(np.percentile(aniso[within], 90)),
               "p95": float(np.percentile(aniso[within], 95)),
               "p99": float(np.percentile(aniso[within], 99)),
           }}
    # Combined stripe criteria at descending thresholds:
    for max_cm, aniso_thr in [(3.0, 5.0), (1.0, 5.0), (0.5, 5.0), (0.3, 5.0), (0.3, 3.0), (0.2, 3.0), (0.1, 3.0)]:
        crit = within & (max_ax > max_cm / 100.0) & (aniso > aniso_thr)
        crit_above = crit & above
        out[f"stripe_cand_maxcm={max_cm}_aniso={aniso_thr}"] = {
            "in_10cm": int(crit.sum()),
            "above_box_top": int(crit_above.sum()),
        }
    return out


def bottle_stub_deep(mesh_v4_verts, mesh_v2_verts):
    """Deep dive on bottle stub — v2 vs v4."""
    mesh = SEG / "white_water_bottle" / "white_water_bottle_extracted.ply"
    npz = UNSEEN / "unseen_core_white_water_bottle.npz"
    data = np.load(npz)
    pts = data["grid_points"]
    patch_y_max = float(pts[:, 1].max())
    dist_fn, y_min_b, y_max_b, _ = dist_fn_from_mesh(mesh)
    out = {"bottle_y_min": y_min_b, "bottle_y_max": y_max_b, "patch_y_max": patch_y_max}
    for tag, verts in [("v2", mesh_v2_verts), ("v4", mesh_v4_verts)]:
        dist = dist_fn(verts.astype(np.float32))
        within = dist <= 0.05
        y = verts[:, 1]
        # y distribution of within-verts
        y_within = y[within]
        # "Stub" = verts within 5cm of bottle AND ABOVE the top of the bottle
        # AND their y ~ within the patch band or slightly above.
        above_bottle_top = within & (y > y_max_b + 0.005)
        # Anything at all sticking out above the bottle mesh:
        # "cone-shaped stub" = high aspect verts above the bottle top
        out[tag] = {
            "n_within_5cm": int(within.sum()),
            "n_above_bottle_top": int(above_bottle_top.sum()),
            "n_above_patch_y_max": int((within & (y > patch_y_max + 0.005)).sum()),
            "n_above_bottle_top_by_1cm": int((within & (y > y_max_b + 0.01)).sum()),
            "n_above_bottle_top_by_3cm": int((within & (y > y_max_b + 0.03)).sum()),
            "n_above_bottle_top_by_5cm": int((within & (y > y_max_b + 0.05)).sum()),
            "y_within_p95": float(np.percentile(y_within, 95)) if y_within.size else None,
            "y_within_p99": float(np.percentile(y_within, 99)) if y_within.size else None,
            "y_within_max": float(y_within.max()) if y_within.size else None,
        }
    return out


def load_mesh(path):
    m = trimesh.load(str(path), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float64)
    try:
        vc = np.asarray(m.visual.vertex_colors)
        rgb = vc[:, :3].astype(np.float32) / 255.0
    except Exception:
        rgb = None
    return verts, rgb


def main():
    print("Loading v2 + v4 ...")
    splat_v4 = load_splat(V4 / "splat" / "scene_without_objects.ply")
    splat_v2 = load_splat(V2 / "splat" / "scene_without_objects.ply")
    v4_verts, _ = load_mesh(V4 / "mesh" / "scene_without_objects.ply")
    v2_verts, _ = load_mesh(V2 / "mesh" / "scene_without_objects.ply")

    print(f"dp_scale={splat_v4['dp_scale']:.6f}, colmap_to_metric={splat_v4['colmap_to_metric']:.6f}")

    print("\n=== STRIPE THRESHOLD SWEEP (BLUE BOX) ===")
    bbox = SEG / "blue_box" / "blue_box_extracted.ply"
    r_v2 = analyze_stripe_thresholds(splat_v2, bbox, "v2")
    r_v4 = analyze_stripe_thresholds(splat_v4, bbox, "v4")
    print("V2:", json.dumps(r_v2, indent=2))
    print("V4:", json.dumps(r_v4, indent=2))

    print("\n=== BOTTLE STUB DEEP DIVE ===")
    bs = bottle_stub_deep(v4_verts, v2_verts)
    print(json.dumps(bs, indent=2))

    print("\n=== SPLAT STRIPE CHECK USING COLMAP-SCALE SCALES (unclear scale unit) ===")
    # sanity: verify metric scale interpretation
    print(f"scales_metric.max = {splat_v4['scales'].max():.4f} m")
    print(f"scales_colmap.max = {splat_v4['scales_colmap'].max():.4f}")


if __name__ == "__main__":
    main()
