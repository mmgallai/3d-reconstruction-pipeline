"""Reviewer D — v4 metrics vs user's v2 complaints.

Complaints (from screenshots after v2):
  1. Lobster patch tinted RED with visible red "figurine remnants"
  2. Bottle patch OK but dark cone-shaped mesh stub above it
  3. Blue box patch mis-placed AND blue tint
  4. Blue vertical stripe extending upward on the right in the SPLAT view

For each, compute the v4 answer. Everything is done in metric space
(the extracted meshes are metric; the pipeline splat is in COLMAP frame
and must be transformed via the dataparser_transforms + tof_bounds scale.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

# Local imports
_ROOT = Path(r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(_ROOT))
from _spatial_crop_splat import _load_dataparser_transform, _splat_to_metric  # noqa

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
V4 = _ROOT / "output" / "pipeline_v32_v11_patches_v4"
V2 = _ROOT / "output" / "pipeline_v32_v11_patches_v2"
SEG = _ROOT / "output" / "segmented_v32_data3_v9a_fp_v2"
UNSEEN = _ROOT / "output" / "unseen_core_v32_data3"
DP_JSON = _ROOT / "nerfstudio" / "dense" / "splatfacto" / "2026-06-07_203807" / "dataparser_transforms.json"
BOUNDS_JSON = _ROOT / "colmap" / "dense" / "tof_bounds.json"

PROMPTS = ["white_water_bottle", "blue_box", "red_lobster_figurine"]

# --------------------------------------------------------------------------
# Splat loading
# --------------------------------------------------------------------------

_SH_C0 = 0.28209479177387814


def load_splat(path: Path):
    """Load a Gaussian splat PLY. Returns dict with positions_metric, rgb_dc, scales, rotations, opacity."""
    R, t, dp_scale = _load_dataparser_transform(DP_JSON)
    bounds = json.loads(BOUNDS_JSON.read_text())
    colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])
    pd = PlyData.read(str(path))
    v = pd["vertex"].data
    pos_splat = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)
    pos_m = _splat_to_metric(pos_splat, R, t, dp_scale, colmap_to_metric)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
    rgb = np.clip(dc * _SH_C0 + 0.5, 0.0, 1.0)
    scales = None
    if all(k in v.dtype.names for k in ["scale_0", "scale_1", "scale_2"]):
        # gsplat stores log-scales
        raw_s = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
        scales = np.exp(raw_s)  # in COLMAP units
        # convert to metric
        scales_m = scales * colmap_to_metric * dp_scale
    else:
        scales_m = None
    return dict(
        pos_metric=pos_m,
        rgb=rgb,
        scales_metric=scales_m,
    )


def make_distance_fn(mesh_ply: Path):
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

    def dist(points_metric):
        p32 = points_metric.astype(np.float32)
        return scene.compute_distance(o3d.core.Tensor(p32)).numpy()

    return dist, y_min, y_max, verts


def load_mesh_verts_colors(path: Path):
    m = trimesh.load(str(path), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float64)
    try:
        vc = np.asarray(m.visual.vertex_colors)  # (n, 4)
        rgb = vc[:, :3].astype(np.float32) / 255.0
    except Exception:
        rgb = None
    return verts, rgb, m


# --------------------------------------------------------------------------
# Per-complaint measurements
# --------------------------------------------------------------------------

def complaint_1_lobster(splat_v4, mesh_v4_verts, mesh_v4_rgb):
    """1. Lobster: pure-red (R>0.6 AND G<0.4 AND B<0.4) Gaussians within
    5 cm of extracted lobster mesh. Should be zero. Same for mesh verts."""
    mesh = SEG / "red_lobster_figurine" / "red_lobster_figurine_extracted.ply"
    dist_fn, y_min, y_max, verts_lob = make_distance_fn(mesh)
    # Splat side
    dist = dist_fn(splat_v4["pos_metric"])
    within = dist <= 0.05
    rgb = splat_v4["rgb"]
    red = (rgb[:, 0] > 0.6) & (rgb[:, 1] < 0.4) & (rgb[:, 2] < 0.4)
    n_red_within = int((within & red).sum())
    n_within = int(within.sum())
    # Also count vaguely red (r > g and r > b, r > 0.4)
    red_loose = (rgb[:, 0] > 0.4) & (rgb[:, 0] > rgb[:, 1] + 0.1) & (rgb[:, 0] > rgb[:, 2] + 0.1)
    n_red_loose_within = int((within & red_loose).sum())
    # Mean color inside the region
    mean_rgb_within = rgb[within].mean(axis=0).tolist() if n_within else [0, 0, 0]
    # Mesh side
    if mesh_v4_rgb is not None:
        dist_m = dist_fn(mesh_v4_verts.astype(np.float32))
        within_m = dist_m <= 0.05
        red_m = (mesh_v4_rgb[:, 0] > 0.6) & (mesh_v4_rgb[:, 1] < 0.4) & (mesh_v4_rgb[:, 2] < 0.4)
        n_mesh_red = int((within_m & red_m).sum())
        n_mesh_within = int(within_m.sum())
        mean_mesh_rgb = mesh_v4_rgb[within_m].mean(axis=0).tolist() if n_mesh_within else [0, 0, 0]
    else:
        n_mesh_red = -1
        n_mesh_within = -1
        mean_mesh_rgb = [0, 0, 0]
    return {
        "splat_within_5cm": n_within,
        "splat_pure_red_within_5cm": n_red_within,
        "splat_loose_red_within_5cm": n_red_loose_within,
        "splat_mean_rgb_within_5cm": [round(x, 3) for x in mean_rgb_within],
        "mesh_within_5cm": n_mesh_within,
        "mesh_pure_red_within_5cm": n_mesh_red,
        "mesh_mean_rgb_within_5cm": [round(x, 3) for x in mean_mesh_rgb],
        "lobster_y_min_m": y_min,
        "lobster_y_max_m": y_max,
    }


def complaint_2_bottle_stub(mesh_v4_verts):
    """2. Bottle: mesh vertices within 5 cm of extracted bottle mesh AND
    ABOVE the patch's y_max ceiling. Should be zero (that's the stub)."""
    mesh = SEG / "white_water_bottle" / "white_water_bottle_extracted.ply"
    npz_path = UNSEEN / "unseen_core_white_water_bottle.npz"
    data = np.load(npz_path)
    pts = data["grid_points"]  # metric grid
    patch_y_min = float(pts[:, 1].min())
    patch_y_max = float(pts[:, 1].max())
    # bottle mesh distance
    dist_fn, y_min_b, y_max_b, verts_b = make_distance_fn(mesh)
    dist = dist_fn(mesh_v4_verts.astype(np.float32))
    within = dist <= 0.05
    # ABOVE the patch (y increases upward in this project)
    above = mesh_v4_verts[:, 1] > (patch_y_max + 0.001)
    n_stub = int((within & above).sum())
    # Also: highest vertex within 5 cm of bottle, to characterize any remaining stub height
    within_idx = np.where(within)[0]
    if within_idx.size:
        y_within = mesh_v4_verts[within_idx, 1]
        y_top_of_region = float(y_within.max())
        y_top_over_patch = y_top_of_region - patch_y_max
    else:
        y_top_of_region = None
        y_top_over_patch = None
    # bottle extracted top for reference
    return {
        "bottle_extracted_y_min_m": y_min_b,
        "bottle_extracted_y_max_m": y_max_b,
        "patch_y_min_m": patch_y_min,
        "patch_y_max_m": patch_y_max,
        "mesh_verts_within_5cm_and_above_patch": n_stub,
        "mesh_verts_within_5cm_total": int(within.sum()),
        "top_of_within_region_y_m": y_top_of_region,
        "top_of_within_region_above_patch_m": y_top_over_patch,
    }


def complaint_3_bluebox(splat_v4, mesh_v4_verts, mesh_v4_rgb):
    """3. Blue box: (a) KDTree match of patch geometry to NPZ grid, (b)
    tinted-blue vertices near the box in v4."""
    mesh = SEG / "blue_box" / "blue_box_extracted.ply"
    npz_path = UNSEEN / "unseen_core_blue_box.npz"
    data = np.load(npz_path)
    grid = data["grid_points"].astype(np.float64)  # metric
    # In v4 mesh, find the patch chunk: patch vertices should equal exactly (or
    # extremely close to) the NPZ grid_points. KD-tree query each grid point
    # against the entire mesh vertex set.
    tree = cKDTree(mesh_v4_verts)
    d_kdt, _ = tree.query(grid, k=1)
    frac_matched_1mm = float((d_kdt < 1e-3).mean())
    frac_matched_5mm = float((d_kdt < 5e-3).mean())
    max_kd = float(d_kdt.max())
    mean_kd = float(d_kdt.mean())
    # blue box distance
    dist_fn, y_min, y_max, verts_bb = make_distance_fn(mesh)
    # (b) tinted blue verts near box (5 cm)
    if mesh_v4_rgb is not None:
        dist = dist_fn(mesh_v4_verts.astype(np.float32))
        within = dist <= 0.05
        # "tinted blue": B > 0.5 AND B > R + 0.1 AND B > G + 0.1  (the leaked blue)
        tinted_blue = (mesh_v4_rgb[:, 2] > 0.5) & (mesh_v4_rgb[:, 2] > mesh_v4_rgb[:, 0] + 0.1) & (mesh_v4_rgb[:, 2] > mesh_v4_rgb[:, 1] + 0.1)
        n_tinted = int((within & tinted_blue).sum())
        n_within = int(within.sum())
        mean_rgb = mesh_v4_rgb[within].mean(axis=0).tolist() if n_within else [0, 0, 0]
    else:
        n_tinted = -1
        n_within = -1
        mean_rgb = [0, 0, 0]
    return {
        "npz_grid_n": int(len(grid)),
        "kd_match_frac_1mm": round(frac_matched_1mm, 4),
        "kd_match_frac_5mm": round(frac_matched_5mm, 4),
        "kd_mean_m": mean_kd,
        "kd_max_m": max_kd,
        "mesh_within_5cm_of_box": n_within,
        "mesh_tinted_blue_within_5cm": n_tinted,
        "mesh_mean_rgb_within_5cm": [round(x, 3) for x in mean_rgb],
        "bluebox_y_min_m": y_min,
        "bluebox_y_max_m": y_max,
    }


def complaint_4_stripe(splat_v4):
    """4. Blue vertical stripe: anisotropic Gaussians (max_axis > 3 cm
    metric AND anisotropy > 5x) within 10 cm 3D distance of blue_box
    extracted mesh. Also check the splat's blueness."""
    mesh = SEG / "blue_box" / "blue_box_extracted.ply"
    dist_fn, y_min, y_max, verts_bb = make_distance_fn(mesh)
    dist = dist_fn(splat_v4["pos_metric"])
    within = dist <= 0.10
    scales = splat_v4["scales_metric"]
    if scales is None:
        return {"error": "no scales column"}
    max_axis = scales.max(axis=1)
    min_axis = np.maximum(scales.min(axis=1), 1e-6)
    anisotropy = max_axis / min_axis
    stripe_like = within & (max_axis > 0.03) & (anisotropy > 5.0)
    # Break out spatial distribution ABOVE the box
    above_box = splat_v4["pos_metric"][:, 1] > (y_max + 0.01)
    stripe_and_above = stripe_like & above_box
    # Blueness of stripe-like Gaussians
    rgb = splat_v4["rgb"]
    stripe_rgb_mean = rgb[stripe_like].mean(axis=0).tolist() if stripe_like.any() else [0, 0, 0]
    # How many are blue-tinted
    blue_tinted = (rgb[:, 2] > 0.5) & (rgb[:, 2] > rgb[:, 0] + 0.1) & (rgb[:, 2] > rgb[:, 1] + 0.1)
    n_stripe_and_blue = int((stripe_like & blue_tinted).sum())
    return {
        "n_within_10cm": int(within.sum()),
        "n_stripe_candidates_within_10cm": int(stripe_like.sum()),
        "n_stripe_candidates_above_box_top": int(stripe_and_above.sum()),
        "n_stripe_candidates_blue_tinted": n_stripe_and_blue,
        "stripe_candidates_mean_rgb": [round(x, 3) for x in stripe_rgb_mean],
        "bluebox_y_max_m": y_max,
        "max_axis_p50_m": float(np.median(max_axis[within])) if within.any() else 0.0,
        "max_axis_p95_m": float(np.percentile(max_axis[within], 95)) if within.any() else 0.0,
        "anisotropy_p95_within": float(np.percentile(anisotropy[within], 95)) if within.any() else 0.0,
    }


def compare_v2(splat_v2, mesh_v2_verts, mesh_v2_rgb):
    """Also compute the same numbers on v2 for baseline sanity."""
    print("\n===== V2 BASELINE =====")
    print("Lobster:", json.dumps(complaint_1_lobster(splat_v2, mesh_v2_verts, mesh_v2_rgb), indent=2))
    print("Bottle stub:", json.dumps(complaint_2_bottle_stub(mesh_v2_verts), indent=2))
    print("Blue box:", json.dumps(complaint_3_bluebox(splat_v2, mesh_v2_verts, mesh_v2_rgb), indent=2))
    print("Stripe:", json.dumps(complaint_4_stripe(splat_v2), indent=2))


def main():
    # -------- V4 --------
    print("Loading v4 splat + mesh ...")
    splat_v4 = load_splat(V4 / "splat" / "scene_without_objects.ply")
    mesh_v4_verts, mesh_v4_rgb, _ = load_mesh_verts_colors(V4 / "mesh" / "scene_without_objects.ply")
    print(f"  splat: {len(splat_v4['pos_metric']):,} Gaussians")
    print(f"  mesh : {len(mesh_v4_verts):,} verts (colored={mesh_v4_rgb is not None})")

    print("\n===== V4 METRICS =====")
    r1 = complaint_1_lobster(splat_v4, mesh_v4_verts, mesh_v4_rgb)
    print("[1] Lobster red-remnant:", json.dumps(r1, indent=2))
    r2 = complaint_2_bottle_stub(mesh_v4_verts)
    print("[2] Bottle stub:", json.dumps(r2, indent=2))
    r3 = complaint_3_bluebox(splat_v4, mesh_v4_verts, mesh_v4_rgb)
    print("[3] Blue box:", json.dumps(r3, indent=2))
    r4 = complaint_4_stripe(splat_v4)
    print("[4] Blue vertical stripe:", json.dumps(r4, indent=2))

    # -------- V2 (baseline for comparison) --------
    try:
        splat_v2 = load_splat(V2 / "splat" / "scene_without_objects.ply")
        mesh_v2_verts, mesh_v2_rgb, _ = load_mesh_verts_colors(V2 / "mesh" / "scene_without_objects.ply")
        print(f"\nV2: splat {len(splat_v2['pos_metric']):,} G, mesh {len(mesh_v2_verts):,} verts")
        r1v2 = complaint_1_lobster(splat_v2, mesh_v2_verts, mesh_v2_rgb)
        r2v2 = complaint_2_bottle_stub(mesh_v2_verts)
        r3v2 = complaint_3_bluebox(splat_v2, mesh_v2_verts, mesh_v2_rgb)
        r4v2 = complaint_4_stripe(splat_v2)
        print("[1v2] Lobster:", json.dumps(r1v2, indent=2))
        print("[2v2] Bottle stub:", json.dumps(r2v2, indent=2))
        print("[3v2] Blue box:", json.dumps(r3v2, indent=2))
        print("[4v2] Stripe:", json.dumps(r4v2, indent=2))
    except Exception as e:
        print(f"V2 comparison skipped: {e}")

    out = {
        "complaint_1_lobster_red": r1,
        "complaint_2_bottle_stub": r2,
        "complaint_3_bluebox": r3,
        "complaint_4_stripe": r4,
    }
    (V4 / "reviewer_d_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {V4 / 'reviewer_d_metrics.json'}")


if __name__ == "__main__":
    main()
