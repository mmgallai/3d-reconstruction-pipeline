"""Reviewer A -- numerical verification for pipeline v5 clone-fill output.

Tests the 5 criteria described in the task:
 1) Color match: mesh patch mean RGB vs desk ring mean RGB (5-15 cm shell).
 2) No residual object color inside patch (red-lobster / blue-box tinted).
 3) Splat patch mean-RGB matches surrounding desk splat.
 4) Patch elevation within +/- 3 cm of ring at similar XZ.
 5) No remaining small holes (< 30 cm perimeter) in object XZ areas.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree

_ROOT = Path(r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(_ROOT))
from _spatial_crop_splat import _load_dataparser_transform, _splat_to_metric  # noqa

V5 = _ROOT / "output" / "pipeline_v32_v11_patches_v5_clone"
SEG = _ROOT / "output" / "segmented_v32_data3_v9a_fp_v2"
DP_JSON = _ROOT / "nerfstudio" / "dense" / "splatfacto" / "2026-06-07_203807" / "dataparser_transforms.json"
BOUNDS_JSON = _ROOT / "colmap" / "dense" / "tof_bounds.json"

SLUGS = ["white_water_bottle", "blue_box", "red_lobster_figurine"]

_SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------- helpers

def load_splat(path: Path):
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
    return dict(pos=pos, rgb=rgb, n=int(len(pos)))


def load_mesh_with_colors(path: Path):
    m = trimesh.load(str(path), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    if m.visual.kind == "vertex" and getattr(m.visual, "vertex_colors", None) is not None:
        vc = np.asarray(m.visual.vertex_colors, dtype=np.uint8)
        if vc.shape[1] == 4:
            vc = vc[:, :3]
        rgb01 = vc.astype(np.float32) / 255.0
    else:
        try:
            vc = np.asarray(m.visual.to_color().vertex_colors, dtype=np.uint8)
            if vc.shape[1] == 4:
                vc = vc[:, :3]
            rgb01 = vc.astype(np.float32) / 255.0
        except Exception:
            rgb01 = np.full((len(verts), 3), 0.5, dtype=np.float32)
            vc = (rgb01 * 255).astype(np.uint8)
    return dict(verts=verts, faces=faces, rgb_u8=vc, rgb01=rgb01)


def object_dist_field(obj_ply: Path):
    m = trimesh.load(str(obj_ply), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.uint32)
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(verts)
    tm.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)
    def dist(p):
        p32 = np.ascontiguousarray(p, dtype=np.float32)
        return scene.compute_distance(o3d.core.Tensor(p32)).numpy()
    y_min = float(verts[:, 1].min())
    y_max = float(verts[:, 1].max())
    x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
    z_min, z_max = float(verts[:, 2].min()), float(verts[:, 2].max())
    return dist, dict(y_min=y_min, y_max=y_max,
                      x_min=x_min, x_max=x_max,
                      z_min=z_min, z_max=z_max)


def desk_plane_y_from_ring(scene_verts: np.ndarray, dist_arr: np.ndarray) -> float:
    """Estimate desk plane Y using the ring of scene verts 5-15 cm from the
    object mesh. Using ring MEDIAN Y is robust because the tabletop
    dominates that shell.
    """
    ring = scene_verts[(dist_arr > 0.05) & (dist_arr <= 0.15)]
    if len(ring) == 0:
        return float("nan")
    return float(np.median(ring[:, 1]))


# ---------------------------------------------------------------- main check

def check_slug(slug: str,
               mesh_scene: dict,
               splat_scene: dict) -> dict:
    obj_ply = SEG / slug / f"{slug}_extracted.ply"
    dist_fn, bbox = object_dist_field(obj_ply)

    # === MESH SIDE ===
    verts = mesh_scene["verts"]
    rgb_u8 = mesh_scene["rgb_u8"]

    # Compute distance from every scene vert to the object mesh.
    # Sample in chunks to avoid huge memory.
    N = verts.shape[0]
    d_obj = np.empty(N, dtype=np.float32)
    step = 200_000
    for i in range(0, N, step):
        d_obj[i:i + step] = dist_fn(verts[i:i + step])

    desk_y = desk_plane_y_from_ring(verts, d_obj)

    # Patch verts: inside the 3D crop shell (dist <= 5 cm) AND near desk plane
    # (within +/- 8 cm vertically), so we exclude any residual object body
    # geometry.
    patch_mask = (d_obj <= 0.05) & (np.abs(verts[:, 1] - desk_y) <= 0.08)
    # Ring: donor shell 5-15 cm from the object mesh, near desk plane.
    ring_mask = (d_obj > 0.05) & (d_obj <= 0.15) & (np.abs(verts[:, 1] - desk_y) <= 0.08)

    patch_rgb = rgb_u8[patch_mask].astype(np.float64)
    ring_rgb = rgb_u8[ring_mask].astype(np.float64)

    def mean_rgb(a):
        return a.mean(axis=0) if len(a) else np.array([np.nan, np.nan, np.nan])

    mesh_patch_mean = mean_rgb(patch_rgb)
    mesh_ring_mean = mean_rgb(ring_rgb)
    mesh_delta = np.abs(mesh_patch_mean - mesh_ring_mean)
    mesh_color_pass = bool(np.all(mesh_delta < 10.0))

    # No residual object color inside patch
    if patch_rgb.size:
        R_, G_, B_ = patch_rgb[:, 0], patch_rgb[:, 1], patch_rgb[:, 2]
        if slug == "red_lobster_figurine":
            residual = int(np.sum(R_ > (G_ + 30)))
        elif slug == "blue_box":
            residual = int(np.sum(B_ > (R_ + 20)))
        elif slug == "white_water_bottle":
            # near-white residual: all channels > 220
            residual = int(np.sum((R_ > 220) & (G_ > 220) & (B_ > 220)))
        else:
            residual = -1
    else:
        residual = -1
    residual_pass = (residual == 0) if residual >= 0 else False

    # Elevation check: for each patch vert, find nearest ring vert in XZ,
    # compare Y.
    if patch_mask.sum() and ring_mask.sum():
        ring_xz = verts[ring_mask][:, [0, 2]]
        patch_xz = verts[patch_mask][:, [0, 2]]
        tree = cKDTree(ring_xz)
        _, idx = tree.query(patch_xz, k=1)
        ring_y = verts[ring_mask][:, 1][idx]
        patch_y = verts[patch_mask][:, 1]
        dy = patch_y - ring_y
        dy_p95 = float(np.percentile(np.abs(dy), 95))
        dy_median = float(np.median(np.abs(dy)))
        dy_max = float(np.max(np.abs(dy)))
        elev_pass = bool(dy_p95 <= 0.03)  # 95 % within 3 cm
    else:
        dy_p95 = dy_median = dy_max = float("nan")
        elev_pass = False

    # === SPLAT SIDE ===
    s_pos = splat_scene["pos"]
    s_rgb = splat_scene["rgb"]

    # Distance from splat centers to the object mesh
    Ns = s_pos.shape[0]
    d_sp = np.empty(Ns, dtype=np.float32)
    for i in range(0, Ns, step):
        d_sp[i:i + step] = dist_fn(s_pos[i:i + step])

    s_patch = (d_sp <= 0.05) & (np.abs(s_pos[:, 1] - desk_y) <= 0.08)
    s_ring = (d_sp > 0.05) & (d_sp <= 0.15) & (np.abs(s_pos[:, 1] - desk_y) <= 0.08)

    sp_patch_rgb_u8 = (s_rgb[s_patch] * 255.0)
    sp_ring_rgb_u8 = (s_rgb[s_ring] * 255.0)

    sp_patch_mean = mean_rgb(sp_patch_rgb_u8)
    sp_ring_mean = mean_rgb(sp_ring_rgb_u8)
    sp_delta = np.abs(sp_patch_mean - sp_ring_mean)
    splat_color_pass = bool(np.all(sp_delta < 10.0))

    return dict(
        slug=slug,
        desk_y=desk_y,
        # mesh
        mesh_patch_n=int(patch_mask.sum()),
        mesh_ring_n=int(ring_mask.sum()),
        mesh_patch_mean_rgb=[float(x) for x in mesh_patch_mean],
        mesh_ring_mean_rgb=[float(x) for x in mesh_ring_mean],
        mesh_delta=[float(x) for x in mesh_delta],
        mesh_color_pass=mesh_color_pass,
        # residual
        residual_verts=residual,
        residual_pass=residual_pass,
        # elevation
        elev_dy_median=dy_median,
        elev_dy_p95=dy_p95,
        elev_dy_max=dy_max,
        elev_pass=elev_pass,
        # splat
        splat_patch_n=int(s_patch.sum()),
        splat_ring_n=int(s_ring.sum()),
        splat_patch_mean_rgb=[float(x) for x in sp_patch_mean],
        splat_ring_mean_rgb=[float(x) for x in sp_ring_mean],
        splat_delta=[float(x) for x in sp_delta],
        splat_color_pass=splat_color_pass,
        bbox=bbox,
    )


def check_holes(mesh_path: Path, per_slug_bbox: dict) -> dict:
    """Iterate boundary loops; count small (<30 cm perimeter) loops whose
    centroid falls inside any object's XZ AABB.

    The criterion in the spec is: "there should be zero small (<30 cm
    perimeter) boundary loops in the object XZ areas." We report every
    bin so failures are legible.
    """
    m = trimesh.load(str(mesh_path), force="mesh", process=False)
    outline = m.outline()
    total_loops = 0
    small_loops_total = 0
    small_in_obj = 0
    small_in_obj_by_slug = {slug: 0 for slug in per_slug_bbox}
    peris_in_obj = []
    for entity in outline.entities:
        try:
            pts = outline.vertices[entity.points]
        except Exception:
            continue
        if len(pts) < 3:
            continue
        segs = np.diff(pts, axis=0)
        seg_len = float(np.linalg.norm(segs, axis=1).sum())
        if not np.allclose(pts[0], pts[-1]):
            seg_len += float(np.linalg.norm(pts[-1] - pts[0]))
        total_loops += 1
        if seg_len < 0.30:
            small_loops_total += 1
            cx, _cy, cz = pts.mean(axis=0)
            for slug, bbox in per_slug_bbox.items():
                if (bbox["x_min"] <= cx <= bbox["x_max"] and
                        bbox["z_min"] <= cz <= bbox["z_max"]):
                    small_in_obj += 1
                    small_in_obj_by_slug[slug] += 1
                    peris_in_obj.append(seg_len)
                    break
    return dict(
        total_loops=total_loops,
        small_loops_total=small_loops_total,
        small_loops_in_obj_area=small_in_obj,
        small_in_obj_by_slug=small_in_obj_by_slug,
        peri_in_obj_percentiles=(
            np.percentile(peris_in_obj, [50, 90, 95, 99]).tolist()
            if peris_in_obj else []),
    )


def main():
    mesh_path = V5 / "mesh" / "scene_without_objects.ply"
    splat_path = V5 / "splat" / "scene_without_objects.ply"
    print(f"[load] mesh: {mesh_path}")
    mesh_scene = load_mesh_with_colors(mesh_path)
    print(f"       {mesh_scene['verts'].shape[0]:,} verts, "
          f"{mesh_scene['faces'].shape[0]:,} faces")
    print(f"[load] splat: {splat_path}")
    splat_scene = load_splat(splat_path)
    print(f"       {splat_scene['n']:,} gaussians")

    results = {}
    bboxes = {}
    for slug in SLUGS:
        print(f"\n=== {slug} =========================================")
        r = check_slug(slug, mesh_scene, splat_scene)
        results[slug] = r
        bboxes[slug] = r["bbox"]
        print(f"  mesh patch n={r['mesh_patch_n']:,}  ring n={r['mesh_ring_n']:,}")
        print(f"  mesh patch mean RGB = {np.array(r['mesh_patch_mean_rgb']).round(1).tolist()}")
        print(f"  mesh ring  mean RGB = {np.array(r['mesh_ring_mean_rgb']).round(1).tolist()}")
        print(f"  mesh |delta|        = {np.array(r['mesh_delta']).round(2).tolist()}  "
              f"pass={r['mesh_color_pass']}")
        print(f"  residual verts      = {r['residual_verts']}  pass={r['residual_pass']}")
        print(f"  elev |dy| median/p95/max = "
              f"{r['elev_dy_median']:.4f} / {r['elev_dy_p95']:.4f} / {r['elev_dy_max']:.4f} m  "
              f"pass={r['elev_pass']}")
        print(f"  splat patch n={r['splat_patch_n']:,}  ring n={r['splat_ring_n']:,}")
        print(f"  splat patch mean RGB = {np.array(r['splat_patch_mean_rgb']).round(1).tolist()}")
        print(f"  splat ring  mean RGB = {np.array(r['splat_ring_mean_rgb']).round(1).tolist()}")
        print(f"  splat |delta|        = {np.array(r['splat_delta']).round(2).tolist()}  "
              f"pass={r['splat_color_pass']}")

    print("\n=== hole scan =============================================")
    holes = check_holes(mesh_path, bboxes)
    print(f"  boundary loops total={holes['total_loops']}, "
          f"small (<30 cm)={holes['small_loops_total']}, "
          f"in object XZ areas={holes['small_loops_in_obj_area']}")
    holes_pass = holes["small_loops_in_obj_area"] == 0

    print("\n=== SUMMARY ===============================================")
    all_pass = True
    for slug, r in results.items():
        row = [r['mesh_color_pass'], r['residual_pass'], r['splat_color_pass'], r['elev_pass']]
        print(f"  {slug}: mesh_color={r['mesh_color_pass']}, "
              f"residual={r['residual_pass']}, splat_color={r['splat_color_pass']}, "
              f"elev={r['elev_pass']}")
        all_pass = all_pass and all(row)
    print(f"  holes: {holes_pass}")
    all_pass = all_pass and holes_pass
    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    # dump JSON
    out = V5 / "reviewer_a_v5_check.json"
    out.write_text(json.dumps(dict(results=results, holes=holes,
                                   all_pass=all_pass), indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
