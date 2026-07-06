"""Track 1 of the COMPREHENSIVE_PLAN: measure the truly-unseen desk core
per object.

For each removable object's footprint on the desk, count how many of the
~140 captured views ACTUALLY observed each 5 mm square of desk surface
under it. A square is "truly unseen" if the answer is ZERO across all
views (i.e. the object geometry — or some other occluder — blocked the
camera's line of sight in every frame). Outputs:
    1. per-object % unseen of the footprint area
    2. per-object PNG heatmap (view-count + binary unseen mask)
    3. per-object NPZ with grid + visibility for downstream fill code

This decides whether Track 2a (mesh-seeded amodal fill) needs a 2D
inpainter at all:
    - all 3 objects < 30 % unseen  -> NN-copy from already-seen footprint
                                      pixels is enough; skip LaMa
    - any object > 70 % unseen     -> need the full diffusion path
    - mixed / 30-70 %               -> NN-copy + LaMa hybrid

Why this measurement is correct: the V32 OpenMVS scene mesh INCLUDES
each object as part of the scene. A ray cast from camera through a
desk-plane grid point will hit the object's mesh first in views where
the object is in front of the desk, so "unobstructed" naturally means
the camera had line-of-sight to that 3D point. No need to identify the
object's face indices separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from scene_segmenter.views import V32ViewSource  # noqa: E402

DEFAULT_GRID_RES_M = 0.005      # 5 mm desk-plane grid
DEFAULT_DEPTH_TOL_M = 0.015     # 1.5 cm visibility tolerance
DEFAULT_FOOTPRINT_DILATE_M = 0.02   # 2 cm safety halo around object XZ AABB


def _build_scene(mesh_path: Path):
    import open3d as o3d
    import trimesh
    print(f"[scene] loading {mesh_path}")
    m = trimesh.load(str(mesh_path), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.uint32)
    print(f"[scene]   {len(verts):,} verts, {len(faces):,} faces")
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(verts)
    tm.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)
    return scene


def _object_footprint(extracted_ply: Path,
                      dilate_m: float,
                      desk_y_override: float | None
                      ) -> tuple[float, float, float, float, float]:
    """Return (x_min, x_max, z_min, z_max, y_desk) for the object's
    desk-plane footprint, with a small dilation.

    Important: the per-object extracted-mesh's lowest vertex is NOT a
    reliable desk plane. For objects whose actual contact base was
    occluded during capture (box, lobster), OpenMVS cannot reconstruct
    the bottom of the object, so the extracted mesh's `y_min` is some
    centimetres ABOVE the real desk. Pass `desk_y_override` (from a
    RANSAC plane fit on the scene mesh, or hardcoded from the bottle's
    well-reconstructed base) for trustworthy results.
    """
    import trimesh
    m = trimesh.load(str(extracted_ply), force="mesh", process=False)
    v = np.asarray(m.vertices, dtype=np.float64)
    if desk_y_override is not None:
        y_desk = float(desk_y_override)
    else:
        # 2nd percentile to avoid stray single-vertex outliers.
        y_desk = float(np.percentile(v[:, 1], 2))
    x_min = float(v[:, 0].min()) - dilate_m
    x_max = float(v[:, 0].max()) + dilate_m
    z_min = float(v[:, 2].min()) - dilate_m
    z_max = float(v[:, 2].max()) + dilate_m
    return x_min, x_max, z_min, z_max, y_desk


def _make_grid(x_min, x_max, z_min, z_max, y_desk,
               grid_res: float) -> tuple[np.ndarray, tuple[int, int]]:
    xs = np.arange(x_min, x_max + grid_res * 0.5, grid_res)
    zs = np.arange(z_min, z_max + grid_res * 0.5, grid_res)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    n = gx.size
    pts = np.stack([gx.flatten(),
                    np.full(n, y_desk, dtype=np.float64),
                    gz.flatten()], axis=-1).astype(np.float32)
    return pts, (len(xs), len(zs))


def fit_local_desk_plane(scene_mesh_verts: np.ndarray,
                          x_min: float, x_max: float,
                          z_min: float, z_max: float,
                          halo_outer_m: float = 0.15,
                          halo_inner_m: float = 0.01,
                          ransac_distance_threshold: float = 0.005,
                          ransac_iterations: int = 3000,
                          min_inliers_required: int = 100,
                          max_normal_tilt_deg: float = 45.0,
                          ) -> tuple[np.ndarray | None, float | None, dict]:
    """RANSAC-fit a 3D plane to the desk neighbourhood around an object.

    This is the desk analog of object segmentation: instead of voting on
    which faces belong to an object, we vote on which vertices belong to
    the desk surface around the object. RANSAC handles outliers (random
    nearby geometry, other objects, walls) automatically.

    Returns (normal_unit, offset_d, info_dict):
      ax + by + cz + d = 0  where (a, b, c) = normal_unit.
      If fit fails or is implausible, returns (None, None, {...}).

    Algorithm:
      1. Take vertices in an XZ halo OUTSIDE the object's footprint AABB
         (so we sample REAL desk, not interpolated geometry under the object).
      2. RANSAC plane fit (Open3D segment_plane).
      3. Flip normal so b > 0 (always points "up" in metric world).
      4. Reject if plane is nearly vertical (object was next to a wall and the
         fit locked onto the wall) -- caller falls back.

    The result is INDEPENDENT of world-axis alignment: the plane normal
    captures whatever tilt the captured scene has.
    """
    import open3d as o3d
    info: dict = {"n_outer": 0, "n_inner": 0, "n_halo": 0,
                  "n_inliers": 0, "tilt_deg": 0.0, "rejected": None}

    xz = scene_mesh_verts[:, [0, 2]]
    in_outer = ((xz[:, 0] >= x_min - halo_outer_m)
                & (xz[:, 0] <= x_max + halo_outer_m)
                & (xz[:, 1] >= z_min - halo_outer_m)
                & (xz[:, 1] <= z_max + halo_outer_m))
    in_inner = ((xz[:, 0] >= x_min - halo_inner_m)
                & (xz[:, 0] <= x_max + halo_inner_m)
                & (xz[:, 1] >= z_min - halo_inner_m)
                & (xz[:, 1] <= z_max + halo_inner_m))
    halo_mask = in_outer & ~in_inner
    info["n_outer"] = int(in_outer.sum())
    info["n_inner"] = int(in_inner.sum())
    info["n_halo"] = int(halo_mask.sum())

    halo_verts = scene_mesh_verts[halo_mask]
    if len(halo_verts) < min_inliers_required:
        # fall back: use any verts inside outer halo
        halo_verts = scene_mesh_verts[in_outer]
        info["n_halo"] = int(in_outer.sum())
        if len(halo_verts) < min_inliers_required:
            info["rejected"] = f"too few halo verts ({len(halo_verts)} < {min_inliers_required})"
            return None, None, info

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(halo_verts.astype(np.float64))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=ransac_distance_threshold,
        ransac_n=3,
        num_iterations=ransac_iterations,
    )
    info["n_inliers"] = int(len(inliers))
    a, b, c, d = plane_model
    # Flip so normal points "up" in metric world (positive Y component).
    if b < 0:
        a, b, c, d = -a, -b, -c, -d
    normal = np.array([a, b, c], dtype=np.float64)
    normal_unit = normal / np.linalg.norm(normal)
    tilt_deg = float(np.degrees(np.arccos(np.clip(normal_unit[1], -1, 1))))
    info["tilt_deg"] = tilt_deg
    if tilt_deg > max_normal_tilt_deg:
        info["rejected"] = (f"plane normal too tilted ({tilt_deg:.1f}° > "
                            f"{max_normal_tilt_deg:.1f}°) — RANSAC likely "
                            f"locked onto a wall or other surface")
        return None, None, info
    return normal_unit, float(d), info


def _make_grid_on_plane(x_min: float, x_max: float, z_min: float, z_max: float,
                         normal: np.ndarray, offset_d: float,
                         grid_res: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a 2D grid in XZ but place each point on the fitted plane.

    Plane: ax + by + cz + d = 0  =>  y = -(ax + cz + d) / b
    """
    a, b, c = normal
    xs = np.arange(x_min, x_max + grid_res * 0.5, grid_res)
    zs = np.arange(z_min, z_max + grid_res * 0.5, grid_res)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    gy = -(a * gx + c * gz + offset_d) / b
    pts = np.stack([gx.flatten(), gy.flatten(), gz.flatten()],
                   axis=-1).astype(np.float32)
    return pts, (len(xs), len(zs))


def _count_views_per_point(scene, points: np.ndarray, view_source,
                           depth_tol: float) -> np.ndarray:
    """For each 3D grid point return the number of views that had an
    UNOBSTRUCTED line of sight to it (i.e. the scene-mesh ray hit at
    distance >= ||cam - P|| - depth_tol, AND P projects into the image)."""
    import open3d as o3d
    seen = np.zeros(len(points), dtype=np.int32)
    views = view_source.all_views()
    n_views = len(views)
    print(f"[visibility] casting {len(points):,} points × {n_views} views = "
          f"{len(points) * n_views:,} rays total")
    for vi, view in enumerate(views):
        cam_pos = view.c2w[:3, 3].astype(np.float32)
        dirs = points - cam_pos[None, :]
        dists = np.linalg.norm(dirs, axis=-1)
        valid_dist = dists > 0.05
        # In-frame test (uses w2c, K)
        w2c = view.w2c
        K = view.K
        pts_cam = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
        zc = pts_cam[:, 2]
        in_front = zc > 0.05
        u = K[0, 0] * pts_cam[:, 0] / np.where(in_front, zc, 1.0) + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / np.where(in_front, zc, 1.0) + K[1, 2]
        in_frame = (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height)
        # Cast ray from camera to grid point and check unobstructed
        dirs_norm = dirs / np.maximum(dists[:, None], 1e-9)
        origins = np.broadcast_to(cam_pos, dirs_norm.shape)
        rays_np = np.concatenate([origins, dirs_norm], axis=-1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays_np))
        hit_t = ans["t_hit"].numpy()
        unobstructed = hit_t >= (dists - depth_tol)
        visible = valid_dist & in_front & in_frame & unobstructed
        seen[visible] += 1
        if (vi + 1) % 20 == 0 or vi == n_views - 1:
            print(f"[visibility]   view {vi+1}/{n_views}: "
                  f"{int(visible.sum())} pts visible from this view "
                  f"(cumulative seen >=1: {int((seen > 0).sum())}/{len(points)})")
    return seen


def _save_heatmap(seen_count: np.ndarray, shape_xz: tuple[int, int],
                  x_min, x_max, z_min, z_max,
                  out_png: Path, title: str, pct_unseen: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nx, nz = shape_xz
    sc = seen_count.reshape(nx, nz)
    unseen = (sc == 0).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im0 = axes[0].imshow(sc.T, cmap="viridis", origin="lower",
                         extent=[x_min, x_max, z_min, z_max], aspect="equal")
    axes[0].set_title(f"{title}\nview count per 5 mm desk-plane pixel")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Z (m)")
    plt.colorbar(im0, ax=axes[0], shrink=0.85, label="views observing this point")
    axes[1].imshow(unseen.T, cmap="Reds", origin="lower",
                   extent=[x_min, x_max, z_min, z_max], aspect="equal",
                   vmin=0, vmax=1)
    axes[1].set_title(f"{title}\ntruly-unseen mask = {pct_unseen:.1f}% of footprint")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=140)
    plt.close(fig)


def analyze_object(view_source, scene, scene_mesh_verts: np.ndarray | None,
                   extracted_ply: Path, out_dir: Path,
                   obj_name: str, grid_res: float, depth_tol: float,
                   dilate_m: float,
                   desk_y_override: float | None,
                   auto_desk: bool = False,
                   halo_outer_m: float = 0.15,
                   halo_inner_m: float = 0.01,
                   ransac_distance_threshold: float = 0.005,
                   max_normal_tilt_deg: float = 45.0,
                   ) -> dict:
    print(f"\n=== {obj_name} ===")
    if not extracted_ply.exists():
        print(f"  missing extracted mesh: {extracted_ply}")
        return {"object": obj_name, "error": "missing extracted mesh"}
    x_min, x_max, z_min, z_max, y_desk_fallback = _object_footprint(
        extracted_ply, dilate_m, desk_y_override)
    plane_normal: np.ndarray | None = None
    plane_offset: float | None = None
    plane_info: dict = {}
    if auto_desk and scene_mesh_verts is not None:
        plane_normal, plane_offset, plane_info = fit_local_desk_plane(
            scene_mesh_verts, x_min, x_max, z_min, z_max,
            halo_outer_m=halo_outer_m, halo_inner_m=halo_inner_m,
            ransac_distance_threshold=ransac_distance_threshold,
            max_normal_tilt_deg=max_normal_tilt_deg,
        )
        if plane_normal is None:
            print(f"  AUTO-DESK FALLBACK: {plane_info.get('rejected', 'unknown')}; "
                  f"using y={y_desk_fallback:.3f} m")
        else:
            print(f"  desk plane (auto): "
                  f"normal=({plane_normal[0]:+.3f}, {plane_normal[1]:+.3f}, "
                  f"{plane_normal[2]:+.3f})  d={plane_offset:+.4f}  "
                  f"tilt={plane_info['tilt_deg']:.1f}°  "
                  f"inliers={plane_info['n_inliers']}/{plane_info['n_halo']}")
    if plane_normal is not None:
        pts, shape_xz = _make_grid_on_plane(
            x_min, x_max, z_min, z_max, plane_normal, plane_offset, grid_res)
        y_min_pt = float(pts[:, 1].min())
        y_max_pt = float(pts[:, 1].max())
        print(f"  footprint: X[{x_min:.3f}, {x_max:.3f}] m "
              f"× Z[{z_min:.3f}, {z_max:.3f}] m "
              f"on fitted tilted plane (Y spans {y_min_pt:.3f}-{y_max_pt:.3f})")
    else:
        print(f"  footprint: X[{x_min:.3f}, {x_max:.3f}] m "
              f"× Z[{z_min:.3f}, {z_max:.3f}] m "
              f"on FLAT desk plane y={y_desk_fallback:.3f} m")
        pts, shape_xz = _make_grid(x_min, x_max, z_min, z_max, y_desk_fallback, grid_res)
    print(f"  grid: {shape_xz[0]} × {shape_xz[1]} = {pts.shape[0]} points "
          f"at {grid_res*100:.1f} cm spacing")
    seen = _count_views_per_point(scene, pts, view_source, depth_tol)
    n_total = len(pts)
    n_unseen = int((seen == 0).sum())
    pct_unseen = 100.0 * n_unseen / n_total
    n_seen_once_only = int((seen == 1).sum())
    n_seen_5plus = int((seen >= 5).sum())
    n_low = int((seen < 5).sum())            # "barely seen"
    n_decent = int((seen >= 20).sum())        # decent NN-copy candidates
    pct_low = 100.0 * n_low / n_total
    pct_decent = 100.0 * n_decent / n_total
    median_views = float(np.median(seen))
    summary = {
        "object": obj_name,
        "extracted_ply": str(extracted_ply),
        "footprint_x_cm": (x_max - x_min) * 100,
        "footprint_z_cm": (z_max - z_min) * 100,
        "desk_plane_y_m_fallback": y_desk_fallback,
        "auto_desk_used": plane_normal is not None,
        "desk_normal_metric": (None if plane_normal is None
                                else [float(plane_normal[0]),
                                      float(plane_normal[1]),
                                      float(plane_normal[2])]),
        "desk_plane_d": (None if plane_offset is None else float(plane_offset)),
        "desk_plane_tilt_deg": plane_info.get("tilt_deg"),
        "desk_plane_inliers": plane_info.get("n_inliers"),
        "desk_plane_halo_count": plane_info.get("n_halo"),
        "grid_res_m": grid_res,
        "grid_shape": list(shape_xz),
        "n_total": n_total,
        "n_unseen": n_unseen,
        "pct_unseen": pct_unseen,
        "n_seen_once_only": n_seen_once_only,
        "n_low_under_5": n_low,
        "pct_low_under_5": pct_low,
        "n_decent_20plus": n_decent,
        "pct_decent_20plus": pct_decent,
        "n_seen_5plus": n_seen_5plus,
        "median_views_per_point": median_views,
        "max_views_per_point": int(seen.max()),
    }
    # Save NPZ for downstream fill code
    npz_path = out_dir / f"unseen_core_{obj_name}.npz"
    np_kwargs = dict(
        grid_points=pts,
        seen_count=seen,
        shape_xz=np.array(shape_xz, dtype=np.int32),
        bounds_xyz=np.array([x_min, x_max, y_desk_fallback, z_min, z_max],
                            dtype=np.float64),
    )
    if plane_normal is not None:
        np_kwargs["desk_normal_metric"] = plane_normal.astype(np.float64)
        np_kwargs["desk_plane_d"] = np.array([plane_offset], dtype=np.float64)
    np.savez(npz_path, **np_kwargs)
    # Save heatmap
    png_path = out_dir / f"unseen_core_{obj_name}.png"
    _save_heatmap(seen, shape_xz, x_min, x_max, z_min, z_max,
                  png_path, obj_name, pct_unseen)
    summary["npz_path"] = str(npz_path)
    summary["png_path"] = str(png_path)
    print(f"  -> seen >=1: {n_total - n_unseen}/{n_total}  "
          f"({100 - pct_unseen:.1f}%)")
    print(f"  -> TRULY UNSEEN: {n_unseen}/{n_total}  ({pct_unseen:.1f}%)")
    print(f"  -> median views/pt: {median_views:.0f}  "
          f"max: {summary['max_views_per_point']}")
    print(f"  -> wrote {png_path.name} + {npz_path.name}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-mesh", type=Path,
                   default=Path("output/mesh_v32_data3/mesh_v32_data3_openmvs.ply"))
    p.add_argument("--segmented-dir", type=Path,
                   default=Path("output/segmented_v32_data3_v9a_fp_v2"))
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--prompts", default="white_water_bottle,blue_box,red_lobster_figurine",
                   help="comma-separated object SLUGS (snake_case)")
    p.add_argument("--out-dir", type=Path,
                   default=Path("output/unseen_core_v32_data3"))
    p.add_argument("--grid-res-m", type=float, default=DEFAULT_GRID_RES_M)
    p.add_argument("--depth-tol-m", type=float, default=DEFAULT_DEPTH_TOL_M)
    p.add_argument("--footprint-dilate-m", type=float,
                   default=DEFAULT_FOOTPRINT_DILATE_M)
    p.add_argument("--desk-y", type=float, default=None,
                   help="Override desk-plane Y (m) for ALL objects. "
                        "Use this when per-object extracted-mesh y_min is "
                        "unreliable (objects whose contact base was occluded "
                        "during capture). Hardcoded 0.002 m matches V32 desk.")
    p.add_argument("--auto-desk", action="store_true",
                   help="Auto-detect desk plane per object via RANSAC fit on "
                        "scene-mesh vertices in a halo around the footprint. "
                        "Handles tilted scenes (the plane normal absorbs the "
                        "tilt). Recommended for any scene where the world "
                        "axes are not exactly aligned with the desk surface.")
    p.add_argument("--auto-desk-halo-outer-m", type=float, default=0.15)
    p.add_argument("--auto-desk-halo-inner-m", type=float, default=0.01)
    p.add_argument("--auto-desk-ransac-thresh-m", type=float, default=0.005)
    p.add_argument("--auto-desk-max-tilt-deg", type=float, default=45.0,
                   help="Reject RANSAC plane fits whose normal tilts more "
                        "than this from world +Y (guard against locking "
                        "onto a wall). Default 45° accommodates the ~35° "
                        "desk tilt in V32 with headroom.")
    p.add_argument("--auto-desk-seed", type=int, default=0,
                   help="RNG seed for Open3D RANSAC plane fits (via "
                        "o3d.utility.random.seed). Makes --auto-desk "
                        "deterministic across runs; change if a specific "
                        "seed causes an outlier fit.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Seed Open3D's RNG so RANSAC plane fits are deterministic across runs.
    # Without this, segment_plane's non-seedable internal RNG can flip the
    # tilt-guard decision between runs on identical data (e.g. blue_box
    # fit at 34.9° one run vs 46° the next).
    import open3d as _o3d
    _o3d.utility.random.seed(int(args.auto_desk_seed))

    print("[init] loading V32 view source")
    vs = V32ViewSource(args.project_root)
    print(f"[init]   {len(vs)} views loaded")
    scene = _build_scene(args.scene_mesh)

    # Also keep the raw vertices for RANSAC plane fitting (--auto-desk).
    scene_mesh_verts: np.ndarray | None = None
    if args.auto_desk:
        import trimesh
        _mesh = trimesh.load(str(args.scene_mesh), force="mesh", process=False)
        scene_mesh_verts = np.asarray(_mesh.vertices, dtype=np.float64)
        print(f"[init] scene mesh verts cached for RANSAC: {len(scene_mesh_verts):,}")

    results = []
    for slug in args.prompts.split(","):
        slug = slug.strip()
        extracted_ply = args.segmented_dir / slug / f"{slug}_extracted.ply"
        r = analyze_object(vs, scene, scene_mesh_verts, extracted_ply,
                           args.out_dir, slug,
                           args.grid_res_m, args.depth_tol_m,
                           args.footprint_dilate_m, args.desk_y,
                           auto_desk=args.auto_desk,
                           halo_outer_m=args.auto_desk_halo_outer_m,
                           halo_inner_m=args.auto_desk_halo_inner_m,
                           ransac_distance_threshold=args.auto_desk_ransac_thresh_m,
                           max_normal_tilt_deg=args.auto_desk_max_tilt_deg)
        results.append(r)

    # Summary
    summary_json = args.out_dir / "summary.json"
    summary_json.write_text(json.dumps(results, indent=2))

    print("\n=== SUMMARY ===")
    print(f"{'Object':<30} {'Footprint':<14} {'%unseen':<10} "
          f"{'%<5views':<11} {'%>=20views':<12} {'median':<10}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['object']:<30} ERROR: {r['error']}")
            continue
        fp_str = f"{r['footprint_x_cm']:.0f}x{r['footprint_z_cm']:.0f} cm"
        print(f"{r['object']:<30} {fp_str:<14} "
              f"{r['pct_unseen']:>6.1f} %   "
              f"{r['pct_low_under_5']:>6.1f} %   "
              f"{r['pct_decent_20plus']:>6.1f} %    "
              f"{r['median_views_per_point']:.0f}")

    # Decision hint
    pcts = [r["pct_unseen"] for r in results if "pct_unseen" in r]
    if not pcts:
        return
    max_pct = max(pcts)
    if max_pct < 30:
        verdict = ("DECISION: every object < 30 % unseen → "
                   "NN-copy from seen footprint is enough; "
                   "skip LaMa (Track 2a M2)")
    elif max_pct > 70:
        verdict = ("DECISION: some object > 70 % unseen → "
                   "need the full mesh-seeded + diffusion path "
                   "(Track 2a M1-M4)")
    else:
        verdict = ("DECISION: mixed (30-70 %) → "
                   "NN-copy where possible + LaMa for the central core")
    print()
    print(verdict)
    print(f"\nwrote {summary_json}")


if __name__ == "__main__":
    main()
