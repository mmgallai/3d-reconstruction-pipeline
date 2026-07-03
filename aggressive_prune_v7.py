"""
Aggressive splat pruner v7 — SURFACE-NORMAL PROJECTED COVARIANCE shape mask.

v5 / v6 used a SCALAR mesh-distance threshold (drop if the Gaussian's
*centre* is farther than N cm from the nearest mesh face). That fights
the wrong problem: the reflection-blob "needles" sticking out of the
monitor often have their CENTRE close to the screen surface, but they
extend a huge distance along the surface NORMAL — and that long axis is
exactly what produces the visible blob when the camera moves.

v7 implements Gemini's recommendation: do not threshold scalar distance
to the mesh, threshold the PROJECTED VARIANCE of the Gaussian along the
local surface normal.

Filter pipeline (v7):

  1. **Scale percentile** — same as v1-v6 (default p95).
  2. **Opacity threshold** — same as v1-v6 (default 0.30).
  3. **Surface-normal projected covariance** (NEW)
       For each surviving Gaussian:
         a) Find the closest point on the OpenMVS mesh and the face
            normal N at that hit (Open3D RaycastingScene
            compute_closest_points -> primitive_ids -> trimesh
            face_normals).
         b) Build the metric 3D covariance:
              R   = rotation matrix from quaternion (w, x, y, z) at
                    rot_0..rot_3 (nerfstudio splat convention)
              s_i = exp(scale_i)         (splat-space std-dev)
              Σ_splat  = R · diag(s_i^2) · R^T
              Σ_metric = Σ_splat / SPLAT_TO_METRIC^2
         c) Project Σ onto N:
              V_perp     = N^T · Σ_metric · N         (variance ⟂ surface)
              sigma_perp = sqrt(max(V_perp, 0))       (std-dev, metres)
         d) Keep if sigma_perp <= --max-perp-m (default 0.02 m = 2 cm).

The discriminator:
  * Surface Gaussian — short along N, long in-plane -> tiny V_perp -> KEEP.
  * Needle outward  — long along N -> huge V_perp -> DROP.
  * Independent of overall scale: rotation orientation is the deciding factor.

Usage:
    python aggressive_prune_v7.py output/splat_v32_data3_noinit_pruned.ply \
        [--out output/splat_cleanup_phases/phase2_surface_normal/splat_v7.ply] \
        [--scale-pct 95] [--opacity-min 0.30] \
        [--scene-mesh output/mesh_v32_data3/mesh_v32_data3_openmvs.ply] \
        [--dataparser nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json] \
        [--bounds-json colmap/dense/tof_bounds.json] \
        [--max-perp-m 0.02] [--no-shape-filter]

The input file is never modified. A new .ply is written.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reuse v1 binary PLY IO and v32 dataparser inversion helpers.
from aggressive_prune import _read_ply, _write_ply
from _spatial_crop_splat import _load_dataparser_transform, _splat_to_metric


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_ply", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Output .ply. Default: <input>_cleaned_v7.ply")
    p.add_argument("--scale-pct", type=float, default=95.0,
                   help="Drop Gaussians whose max-scale axis is above this "
                        "percentile of the population (default 95)")
    p.add_argument("--opacity-min", type=float, default=0.30,
                   help="Drop Gaussians with sigmoid(opacity) < this "
                        "(default 0.30)")
    p.add_argument("--scene-mesh", type=Path,
                   default=Path("output/mesh_v32_data3/mesh_v32_data3_openmvs.ply"),
                   help="Triangulated scene mesh used to define local "
                        "surface normals (metric world units)")
    p.add_argument("--dataparser", type=Path,
                   default=Path("nerfstudio/dense/splatfacto/2026-06-07_203807/"
                                "dataparser_transforms.json"),
                   help="nerfstudio dataparser_transforms.json next to the "
                        "splatfacto run's config.yml")
    p.add_argument("--bounds-json", type=Path,
                   default=Path("colmap/dense/tof_bounds.json"),
                   help="tof_bounds.json containing scale_factor_da3_to_colmap")
    p.add_argument("--max-perp-m", type=float, default=0.02,
                   help="KEEP Gaussians whose perpendicular std-dev "
                        "(sigma_perp = sqrt(N^T Σ_metric N)) is <= this in "
                        "metres. THE KEY KNOB. Default 0.02 = 2 cm.")
    p.add_argument("--no-shape-filter", action="store_true",
                   help="Skip the surface-normal projected-covariance filter "
                        "(only run scale + opacity, same as v1).")
    return p.parse_args()


def _build_raycasting_scene_and_normals(mesh_path: Path):
    """Load mesh via trimesh, wrap in Open3D RaycastingScene, also return
    a (n_faces, 3) array of UNIT face normals indexed by primitive_id.

    Returns (scene, face_normals_unit, n_verts, n_faces, t_load, t_bvh).
    """
    import open3d as o3d
    import trimesh

    t0 = time.perf_counter()
    m = trimesh.load(str(mesh_path), force="mesh", process=False)
    if m.is_empty or len(m.faces) == 0:
        raise RuntimeError(f"Scene mesh has no faces: {mesh_path}")
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.uint32)
    # trimesh exposes computed face normals (already unit-length when the
    # mesh is well-formed). Normalize defensively.
    face_normals = np.asarray(m.face_normals, dtype=np.float64)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    face_normals_unit = face_normals / norms
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = o3d.core.Tensor(verts)
    tmesh.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    t_bvh = time.perf_counter() - t0

    return scene, face_normals_unit, len(verts), len(faces), t_load, t_bvh


def _quat_to_rot_matrix(quats_wxyz: np.ndarray) -> np.ndarray:
    """Convert (N, 4) quaternions in (w, x, y, z) order to (N, 3, 3)
    rotation matrices. Quats are normalized before conversion.
    """
    q = quats_wxyz.astype(np.float64)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    q = q / n
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - z * w)
    R[:, 0, 2] = 2.0 * (x * z + y * w)
    R[:, 1, 0] = 2.0 * (x * y + z * w)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - x * w)
    R[:, 2, 0] = 2.0 * (x * z - y * w)
    R[:, 2, 1] = 2.0 * (y * z + x * w)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def _projected_perp_sigma(R: np.ndarray, scales_splat: np.ndarray,
                          normals: np.ndarray,
                          splat_to_metric: float) -> tuple:
    """For each Gaussian compute sigma_perp = sqrt(N^T Σ_metric N).

    Args:
        R: (N, 3, 3) rotation matrices.
        scales_splat: (N, 3) per-axis std-devs in splat units (= exp(stored)).
        normals: (N, 3) unit face normals at the closest mesh hit.
        splat_to_metric: scalar; metric = splat * splat_to_metric.

    Returns:
        sigma_perp: (N,) sqrt(V_perp), in metric metres.
        sigma_max:  (N,) sqrt(largest eigenvalue of Σ_metric), in metric
                    metres. Useful for sanity printouts.
    """
    s2_splat = scales_splat.astype(np.float64) ** 2                  # (N, 3)
    # Σ_splat = R · diag(s^2) · R^T, computed via:
    #   tmp_{n,i,j} = R_{n,i,j} * s2_{n,j}
    #   Σ_{n,i,k}  = sum_j tmp_{n,i,j} * R_{n,k,j}
    tmp = R * s2_splat[:, np.newaxis, :]                             # (N, 3, 3)
    sigma_splat = np.einsum("nij,nkj->nik", tmp, R)                  # (N, 3, 3)
    sigma_metric = sigma_splat * (splat_to_metric ** 2)              # (N, 3, 3)

    # V_perp = N^T · Σ_metric · N -> (N,)
    sig_n = np.einsum("nij,nj->ni", sigma_metric, normals)
    v_perp = np.einsum("ni,ni->n", normals, sig_n)
    sigma_perp = np.sqrt(np.clip(v_perp, 0.0, None))

    # Largest eigenvalue == max axis variance (after rotation, eigenvalues
    # of Σ are the diagonal entries of diag(s^2) scaled to metric).
    sigma_max = scales_splat.max(axis=1) * splat_to_metric
    return sigma_perp, sigma_max


def main():
    args = _parse_args()
    in_path = args.input_ply.resolve()
    out_path = (args.out or in_path.with_name(in_path.stem + "_cleaned_v7.ply")).resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    print(f"=== aggressive_prune_v7 (surface-normal projected covariance) ===")
    print(f"  input       : {in_path}")
    print(f"  output      : {out_path}")
    print(f"  scene-mesh  : {args.scene_mesh}")
    print(f"  dataparser  : {args.dataparser}")
    print(f"  bounds-json : {args.bounds_json}")
    print(f"  max-perp    : {args.max_perp_m:.4f} m "
          f"({args.max_perp_m*100:.2f} cm)")
    if args.no_shape_filter:
        print(f"  (--no-shape-filter set: surface-normal filter DISABLED)")

    # ── Read input splat PLY ─────────────────────────────────────────
    n, props, raw, rec_size, header = _read_ply(in_path)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(n, len(props)).copy()
    name_to_idx = {p: i for i, p in enumerate(props)}
    print(f"\n  loaded {n:,} Gaussians, {len(props)} properties each")

    x = arr[:, name_to_idx["x"]]
    y = arr[:, name_to_idx["y"]]
    z = arr[:, name_to_idx["z"]]
    op_logit = arr[:, name_to_idx["opacity"]]
    op = 1.0 / (1.0 + np.exp(-op_logit))
    scale_keys = sorted([p for p in props if p.startswith("scale_")])
    scales_log = np.stack([arr[:, name_to_idx[k]] for k in scale_keys], axis=1)
    scales_splat_all = np.exp(scales_log)                            # (N, 3) std-dev in splat units
    max_scale = scales_splat_all.max(axis=1)

    print(f"  current stats:")
    print(f"    opacity median {np.median(op):.3f}  min {op.min():.3f}  "
          f"max {op.max():.3f}")
    print(f"    max-scale median {np.median(max_scale):.4f}  "
          f"p95 {np.percentile(max_scale, 95):.4f}  "
          f"max {max_scale.max():.4f}")
    print(f"    splat BB: x[{x.min():.2f},{x.max():.2f}]  "
          f"y[{y.min():.2f},{y.max():.2f}]  "
          f"z[{z.min():.2f},{z.max():.2f}]")

    keep = np.ones(n, dtype=bool)

    # ── Filter 1: scale percentile ────────────────────────────────────
    scale_thresh = float(np.percentile(max_scale, args.scale_pct))
    scale_mask = max_scale <= scale_thresh
    dropped_1 = int((~scale_mask).sum())
    print(f"\n  filter 1 — scale > p{args.scale_pct} ({scale_thresh:.4f}): "
          f"drop {dropped_1:,} Gaussians ({100*dropped_1/n:.1f}%)")
    keep &= scale_mask

    # ── Filter 2: opacity threshold ───────────────────────────────────
    op_mask = op >= args.opacity_min
    dropped_2 = int((~op_mask & keep).sum())
    print(f"  filter 2 — opacity < {args.opacity_min}: drop additional "
          f"{dropped_2:,} Gaussians")
    keep &= op_mask

    n_pre_shape = int(keep.sum())
    print(f"  after scale + opacity: {n_pre_shape:,} / {n:,} "
          f"({100*n_pre_shape/n:.1f}%)")

    # ── Filter 3: surface-normal projected covariance ─────────────────
    if not args.no_shape_filter:
        # Quaternion property names in nerfstudio splat PLYs.
        rot_keys = ["rot_0", "rot_1", "rot_2", "rot_3"]   # (w, x, y, z)
        for k in rot_keys:
            if k not in name_to_idx:
                print(f"ERROR: PLY missing rotation property '{k}'", file=sys.stderr)
                return 2

        # Load + invert dataparser transform.
        dp_path = args.dataparser.resolve()
        if not dp_path.exists():
            print(f"ERROR: dataparser not found: {dp_path}", file=sys.stderr)
            return 2
        R_dp, t_dp, dp_scale = _load_dataparser_transform(dp_path)
        print(f"\n  dataparser: scale={dp_scale:.4f}, "
              f"R det={np.linalg.det(R_dp):.4f}")

        # Load bounds -> colmap_to_metric and splat_to_metric.
        bp = args.bounds_json.resolve()
        if not bp.exists():
            print(f"ERROR: bounds-json not found: {bp}", file=sys.stderr)
            return 2
        bj = json.loads(bp.read_text())
        scale_factor = float(bj["scale_factor_da3_to_colmap"])
        colmap_to_metric = 1.0 / scale_factor
        splat_to_metric = colmap_to_metric / dp_scale
        print(f"  scale_factor_da3_to_colmap = {scale_factor:.6f}")
        print(f"  colmap_to_metric           = {colmap_to_metric:.6f}")
        print(f"  splat_to_metric            = {splat_to_metric:.6f}  "
              f"(1 splat unit = {splat_to_metric*100:.4f} cm)")

        # Build raycasting scene + cache face normals.
        mp = args.scene_mesh.resolve()
        if not mp.exists():
            print(f"ERROR: scene mesh not found: {mp}", file=sys.stderr)
            return 2
        print(f"  loading scene mesh + building BVH ...")
        (scene, face_normals_unit, n_v, n_f,
         t_load, t_bvh) = _build_raycasting_scene_and_normals(mp)
        print(f"    mesh load: {t_load:.2f} s  ({n_v:,} verts, "
              f"{n_f:,} faces)")
        print(f"    BVH build: {t_bvh:.2f} s")

        # Survivors only — to save closest-point queries on already-dropped points.
        survivor_idx = np.where(keep)[0]
        pos_splat = arr[survivor_idx][:,
                       [name_to_idx["x"], name_to_idx["y"], name_to_idx["z"]]
                   ].astype(np.float64)
        scales_splat = scales_splat_all[survivor_idx]                # (M, 3)
        quats = np.stack([arr[survivor_idx, name_to_idx[k]] for k in rot_keys],
                         axis=1)                                     # (M, 4) wxyz

        pos_metric = _splat_to_metric(pos_splat, R_dp, t_dp, dp_scale,
                                       colmap_to_metric)
        print(f"  inverted {len(pos_metric):,} survivor positions to metric")
        print(f"    metric BB: "
              f"x[{pos_metric[:,0].min():.2f},{pos_metric[:,0].max():.2f}]  "
              f"y[{pos_metric[:,1].min():.2f},{pos_metric[:,1].max():.2f}]  "
              f"z[{pos_metric[:,2].min():.2f},{pos_metric[:,2].max():.2f}]")

        # Closest-point query -> primitive_ids -> face normals.
        import open3d as o3d
        t0 = time.perf_counter()
        query = o3d.core.Tensor(pos_metric.astype(np.float32))
        cp_res = scene.compute_closest_points(query)
        prim_ids = cp_res["primitive_ids"].numpy().astype(np.int64)
        t_query = time.perf_counter() - t0
        print(f"  closest-point query: {t_query:.2f} s "
              f"({len(prim_ids):,} points)")

        # Sanity check on primitive_ids.
        if prim_ids.min() < 0 or prim_ids.max() >= n_f:
            print(f"ERROR: primitive_ids out of range "
                  f"[{prim_ids.min()}, {prim_ids.max()}] vs n_faces={n_f}",
                  file=sys.stderr)
            return 2

        normals = face_normals_unit[prim_ids]                        # (M, 3)

        # Build rotation matrices + project covariance.
        t0 = time.perf_counter()
        R_g = _quat_to_rot_matrix(quats)
        sigma_perp, sigma_max = _projected_perp_sigma(
            R_g, scales_splat, normals, splat_to_metric)
        t_cov = time.perf_counter() - t0
        print(f"  covariance projection: {t_cov:.2f} s")

        # Early sanity printout — first 5 survivors.
        print(f"\n  sanity (first 5 survivors):")
        print(f"    {'idx':>10s}  {'sigma_perp(cm)':>14s}  "
              f"{'max_axis(cm)':>14s}  {'normal':>22s}")
        for i in range(min(5, len(sigma_perp))):
            nv = normals[i]
            print(f"    {survivor_idx[i]:>10d}  "
                  f"{sigma_perp[i]*100:>14.4f}  "
                  f"{sigma_max[i]*100:>14.4f}  "
                  f"({nv[0]:+.3f},{nv[1]:+.3f},{nv[2]:+.3f})")

        # Distribution of sigma_perp.
        sp_cm = sigma_perp * 100.0
        print(f"\n  sigma_perp distribution (metric cm): "
              f"median {np.median(sp_cm):.3f}  "
              f"p90 {np.percentile(sp_cm, 90):.3f}  "
              f"p99 {np.percentile(sp_cm, 99):.3f}  "
              f"max {sp_cm.max():.3f}")

        # Apply the threshold.
        shape_mask_survivors = sigma_perp <= args.max_perp_m
        n_kept_shape = int(shape_mask_survivors.sum())
        n_dropped_shape = int((~shape_mask_survivors).sum())
        pct_drop = 100.0 * n_dropped_shape / max(n_pre_shape, 1)
        print(f"\n  filter 3 — sigma_perp > {args.max_perp_m*100:.2f} cm:")
        print(f"    n_pre_shape    = {n_pre_shape:,}")
        print(f"    n_kept_shape   = {n_kept_shape:,}")
        print(f"    n_dropped_shape= {n_dropped_shape:,} ({pct_drop:.1f}%)")

        # Project survivor-mask back onto full keep array.
        full_shape_mask = np.zeros(n, dtype=bool)
        full_shape_mask[survivor_idx] = shape_mask_survivors
        keep &= full_shape_mask

    # ── Write output ──────────────────────────────────────────────────
    kept = arr[keep]
    print(f"\n  kept {int(keep.sum()):,} / {n:,} Gaussians "
          f"({100*int(keep.sum())/n:.1f}%)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(out_path, header, kept)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
