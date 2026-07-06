"""Full V32 pipeline orchestrator -- single entry point that assembles the
final deliverable directory for a scene.

Pipeline stages (in order):

  1. Reconstruction (optional; skip if --skip-reconstruction or outputs exist).
     Calls `reconstruct_realityscan.py --use-femto-depth`. Produces:
       output/splat_<scene>_noinit_pruned.ply
       output/mesh_<scene>/mesh_<scene>_openmvs.ply

  2. SPLAT CLEANUP -- v11: mesh-distance smoothstep, mode='both' (scale +
     opacity fade). For each Gaussian, computes unsigned distance to the
     nearest OpenMVS mesh face (Open3D RaycastingScene) and applies a
     smoothstep multiplier that reaches 1.0 at `--cleanup-inner-m` and
     0.0 at `--cleanup-outer-m`; the multiplier scales down the
     Gaussian's three scale axes AND its opacity. Gaussians whose scale
     or opacity fall below floor thresholds are then dropped. Defaults:
     mode='both', inner=2 cm, outer=20 cm (PI-suggested soft-fade
     approach, selected after v10/v11 sweeps -- see PROGRESS_LOG.md
     "PI soft-fade" section). Writes <out>/splat/scene_cleaned.ply

  3. Per-object reuse (already done by scene_segmenter v9a_fp_v2).
     Copies the pre-built per-object meshes (PLY) and Clean-GS-pruned splats
     into <out>/{mesh,splat}/objects/<slug>.ply.

  4. Scene-without-objects (mesh side + splat side, combined for all prompts).
     - Object volume = 3D distance to the extracted object mesh <= 5 cm,
       capped at Y_min + 1 m. Follows the object's real shape (bottle
       neck, overhangs) instead of a coarse XZ silhouette + Y range.
     - MESH: load V32 mesh, bake per-vertex atlas colours, drop any face
             with ANY vertex inside ANY object volume (any-vertex test).
             For each prompt, add a per-object planar mesh patch from
             the NPZ grid points and colour every patch vertex via K-NN
             clone (K=200, per-channel median) of scene-mesh donor verts
             in a 5-30 cm shell around all objects. Small, targeted;
             mesh grows by ~a few thousand faces per object.
     - SPLAT: start from stage-2 output, drop Gaussians whose CENTER is
              inside any object volume, concatenate the 3 desk patches
              from output/desk_patch_<scene>/desk_patch_<slug>.ply.

  5. Assembly + scene_full copies.

  6. Summary.

CLI:
    python _pipeline_full.py [--scene-name v32_data3]
                             [--skip-reconstruction]
                             [--prompts white_water_bottle,blue_box,red_lobster_figurine]
                             [--cleanup-mode both]
                             [--cleanup-inner-m 0.02]
                             [--cleanup-outer-m 0.20]
                             [--out-root output/pipeline_v32_test_run]

Default behaviour: V32, skip reconstruction, the 3 grabbable prompts,
v11 both-mode smoothstep 2->20 cm, writes to
output/pipeline_v32_test_run/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

# Ensure local imports work no matter where the script is run from.
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from _spatial_crop_splat import (                  # noqa: E402
    _load_dataparser_transform,
    _splat_to_metric,
)

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _log(msg: str):
    print(msg, flush=True)


def _section(title: str):
    bar = "=" * max(8, 72 - len(title))
    _log(f"\n=== {title} {bar}")


def _require(path: Path, what: str):
    if not path.exists():
        raise SystemExit(f"[fatal] missing {what}: {path}")
    return path


def _copy(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise SystemExit(f"[fatal] cannot copy, source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return dst


def _mb(p: Path) -> float:
    return p.stat().st_size / 1_048_576 if p.exists() else 0.0


def _read_ply(path: Path):
    """Return the raw structured ndarray + dtype for a splat PLY."""
    pd = PlyData.read(str(path))
    v = pd["vertex"]
    return v.data, v.data.dtype


def _write_ply(rows: np.ndarray, dst: Path):
    """Write a structured ndarray as a binary PLY 'vertex' element."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    el = PlyElement.describe(rows, "vertex")
    PlyData([el], text=False).write(str(dst))


# ---------------------------------------------------------------------------
# Stage 1 -- reconstruction
# ---------------------------------------------------------------------------

def stage_reconstruction(*, project_root: Path, scene_name: str,
                          skip: bool, splat_path: Path, mesh_path: Path):
    if skip:
        _log("[stage 1] reconstruction skipped (--skip-reconstruction)")
        if not splat_path.exists() or not mesh_path.exists():
            raise SystemExit(
                "[fatal] skip-reconstruction requested but expected outputs "
                "are missing.\n"
                f"  splat: {splat_path} (exists={splat_path.exists()})\n"
                f"  mesh : {mesh_path} (exists={mesh_path.exists()})"
            )
        _log(f"  reusing splat: {splat_path}  ({_mb(splat_path):.1f} MB)")
        _log(f"  reusing mesh : {mesh_path}  ({_mb(mesh_path):.1f} MB)")
        return

    if splat_path.exists() and mesh_path.exists():
        _log("[stage 1] reconstruction outputs already exist -- skipping run")
        _log(f"  splat: {splat_path}  ({_mb(splat_path):.1f} MB)")
        _log(f"  mesh : {mesh_path}  ({_mb(mesh_path):.1f} MB)")
        return

    _log("[stage 1] running reconstruct_realityscan.py --use-femto-depth ...")
    cmd = [sys.executable, str(project_root / "reconstruct_realityscan.py"),
           "--use-femto-depth"]
    _log(f"  cmd: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(project_root))
    if res.returncode != 0:
        raise SystemExit(
            f"[fatal] reconstruction failed (exit {res.returncode}) after "
            f"{time.time()-t0:.1f}s"
        )
    _log(f"  reconstruction OK in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Stage 2 -- splat cleanup (v11: mesh-distance smoothstep, scale + opacity)
# ---------------------------------------------------------------------------

def stage_splat_cleanup(*, splat_in: Path, mesh_in: Path,
                         dp_json: Path, bounds_json: Path,
                         inner_m: float, outer_m: float, mode: str,
                         out_ply: Path, project_root: Path) -> dict:
    _log(f"[stage 2] splat cleanup (v11 smoothstep, mode={mode}, "
         f"inner={inner_m*100:.1f} cm, outer={outer_m*100:.1f} cm)")
    _log(f"  splat in : {splat_in}")
    _log(f"  mesh in  : {mesh_in}")
    _log(f"  dp json  : {dp_json}")
    _log(f"  bounds   : {bounds_json}")
    _log(f"  out ply  : {out_ply}")

    script = project_root / "aggressive_prune_v11.py"
    if not script.exists():
        raise SystemExit(f"[fatal] aggressive_prune_v11.py not found: {script}")

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        str(splat_in),
        "--scene-mesh", str(mesh_in),
        "--dataparser", str(dp_json),
        "--bounds-json", str(bounds_json),
        "--mode", mode,
        "--inner-m", str(inner_m),
        "--outer-m", str(outer_m),
        "--out", str(out_ply),
    ]
    _log(f"  cmd: {' '.join(cmd)}")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(cmd, env=env, check=False,
                            capture_output=True, text=True)
    for line in (result.stdout or "").splitlines():
        _log(f"    | {line}")
    if result.returncode != 0:
        for line in (result.stderr or "").splitlines():
            _log(f"    ! {line}")
        raise SystemExit(
            f"[fatal] aggressive_prune_v11.py exited with code {result.returncode}")

    data_in, _ = _read_ply(splat_in)
    n_in = len(data_in)
    data_out, _ = _read_ply(out_ply)
    n_kept = len(data_out)
    _log(f"  kept {n_kept:,}/{n_in:,} = {100.0*n_kept/max(n_in,1):.2f}%  "
         f"(v11 {mode}-mode smoothstep, {inner_m*100:.1f}->{outer_m*100:.1f} cm)")
    _log(f"  wrote {out_ply}  ({_mb(out_ply):.1f} MB)")

    return {
        "method": "v11_smoothstep",
        "mode": mode,
        "inner_m": float(inner_m),
        "outer_m": float(outer_m),
        "n_in": int(n_in),
        "n_kept": int(n_kept),
        "pct_kept": round(100.0 * n_kept / max(n_in, 1), 2),
        "out_path": str(out_ply),
    }


# ---------------------------------------------------------------------------
# Stage 3 -- per-object outputs (copy from existing segmenter runs)
# ---------------------------------------------------------------------------

def stage_per_object(*, prompts: list, object_mesh_dir: Path,
                      object_splat_dir: Path, out_mesh_dir: Path,
                      out_splat_dir: Path) -> list:
    _log("[stage 3] per-object outputs (reuse v9a_fp_v2 + v6_cleaned)")
    out_mesh_dir.mkdir(parents=True, exist_ok=True)
    out_splat_dir.mkdir(parents=True, exist_ok=True)
    import trimesh
    records = []
    for slug in prompts:
        rec = {"slug": slug}
        # MESH side: copy the segmenter's PLY into the deliverable directory,
        # then bake the V32 atlas into per-vertex RGB so the output PLY is
        # self-contained (no `comment TextureFile` sidecar needed).
        src_mesh_ply = object_mesh_dir / slug / f"{slug}_extracted.ply"
        dst_mesh = out_mesh_dir / f"{slug}.ply"
        _copy(_require(src_mesh_ply, f"object mesh for {slug}"), dst_mesh)

        # Load via the segmenter's sibling OBJ (whose UVs are oriented to
        # match the atlas's PIL/top-left origin), bake atlas -> per-vertex
        # RGBA via trimesh.visual.to_color(), and overwrite the destination
        # PLY in place. Loading from the VCG-format PLY directly samples the
        # atlas with a V flip and lands on the orange background padding;
        # the OBJ avoids that.
        obj_src = object_mesh_dir / slug / f"{slug}_extracted.obj"
        if obj_src.exists():
            m = trimesh.load(str(obj_src), force="mesh", process=False)
            if hasattr(m.visual, "to_color"):
                color_visual = m.visual.to_color()
                vc = np.asarray(color_visual.vertex_colors)
                if vc.shape[0] == len(m.vertices):
                    m.visual = trimesh.visual.ColorVisuals(
                        mesh=m, vertex_colors=vc)
                    m.export(str(dst_mesh))
                    mean_rgb = tuple(int(x) for x in
                                     vc[:, :3].mean(axis=0).round(0))
                    _log(f"  [{slug}] baked per-vertex RGB into {dst_mesh.name} "
                         f"(mean = {mean_rgb})")
                else:
                    _log(f"  [{slug}] WARN: to_color() returned {vc.shape}, "
                         f"expected ({len(m.vertices)}, 4); kept original PLY")
            else:
                _log(f"  [{slug}] WARN: visual is "
                     f"{type(m.visual).__name__}, cannot bake; kept original PLY")
        else:
            _log(f"  [{slug}] WARN: no OBJ source at {obj_src}; "
                 f"per-object PLY left untouched (texture sidecar required)")
        # NOTE: scene_textured0.png is intentionally NOT copied here. The
        # per-object PLYs are self-contained vertex-coloured meshes; there
        # is no `comment TextureFile` reference to satisfy.

        rec["mesh_ply"] = str(dst_mesh)
        rec["mesh_mb"] = _mb(dst_mesh)
        _log(f"  [{slug}] mesh  : {dst_mesh.name}  ({rec['mesh_mb']:.2f} MB)")

        # SPLAT side: <splat_v6_cleaned>/<slug>_splat.ply
        src_splat = object_splat_dir / f"{slug}_splat.ply"
        dst_splat = out_splat_dir / f"{slug}.ply"
        _copy(_require(src_splat, f"object splat for {slug}"), dst_splat)
        rec["splat_ply"] = str(dst_splat)
        rec["splat_mb"] = _mb(dst_splat)
        _log(f"  [{slug}] splat : {dst_splat.name}  ({rec['splat_mb']:.2f} MB)")
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Stage 4 helpers -- mesh / splat removal + patches
# ---------------------------------------------------------------------------

def _mesh_face_centroids(mesh) -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return verts[faces].mean(axis=1)


def _bake_vertex_colors_from_atlas(mesh):
    """Bake the V32 atlas into per-vertex RGBA so trimmed PLYs are self-contained.

    Falls back to white if the mesh has no UVs / texture. Mutates `mesh`.
    """
    import trimesh
    try:
        if hasattr(mesh.visual, "to_color"):
            color_visual = mesh.visual.to_color()
            vc = np.asarray(color_visual.vertex_colors)
        elif hasattr(mesh.visual, "vertex_colors"):
            vc = np.asarray(mesh.visual.vertex_colors)
        else:
            vc = None
        if vc is not None and vc.shape[0] == len(mesh.vertices):
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vc)
            _log(f"  baked per-vertex colours from atlas "
                 f"(mean RGB = {vc[:, :3].mean(axis=0).round(0).astype(int)})")
            return
        _log(f"  WARN: atlas->vertex bake returned unexpected shape "
             f"{None if vc is None else vc.shape}")
    except Exception as e:
        _log(f"  WARN: atlas->vertex bake failed ({e})")


def _build_object_distance_predicate(extracted_ply: Path,
                                       dist_threshold_m: float,
                                       y_ceiling_offset_m: float = 1.0) -> dict:
    """Build a 3D distance predicate: 'is a point within dist_threshold_m
    of the extracted object mesh?'.

    Returns a dict {scene, dist_threshold_m, y_ceiling_m} used by
    _test_within_object_distance. The Y ceiling caps the distance query
    at Y_min + y_ceiling_offset_m so distant ceiling / wall faces above
    the desk cannot pollute the crop volume.
    """
    import open3d as o3d
    import trimesh as _tm
    m = _tm.load(str(extracted_ply), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.uint32)
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(verts)
    tm.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)
    y_min = float(verts[:, 1].min())
    return {
        "scene": scene,
        "dist_threshold_m": float(dist_threshold_m),
        "y_min": y_min,
        "y_ceiling_m": y_min + float(y_ceiling_offset_m),
        "extracted_ply": str(extracted_ply),
    }


def _test_within_object_distance(positions_metric: np.ndarray,
                                   pred: dict) -> np.ndarray:
    """True if positions_metric[i] is within pred['dist_threshold_m'] of the
    predicate's extracted mesh AND below the y_ceiling.
    """
    import open3d as o3d
    pts32 = positions_metric.astype(np.float32)
    dists = pred["scene"].compute_distance(o3d.core.Tensor(pts32)).numpy()
    inside = dists <= pred["dist_threshold_m"]
    inside &= positions_metric[:, 1] <= pred["y_ceiling_m"]
    return inside


def _drop_faces_inside_any(scene_mesh, predicates: list):
    """Drop faces where ANY vertex is within the distance threshold of ANY
    object predicate (or, equivalently, keep only faces whose ALL three
    vertices are more than the threshold away from every object).

    Any-vertex test (vs centroid-only) fixes the "dangling stub" bug: a
    face straddling the object boundary with 1-2 verts inside but centroid
    just outside was surviving before, leaving jagged geometry above the
    removed object.

    Returns (new_trimesh, n_removed_faces). Preserves per-vertex colours.
    """
    import trimesh
    verts = np.asarray(scene_mesh.vertices, dtype=np.float64)
    faces = np.asarray(scene_mesh.faces, dtype=np.int64)
    vert_inside = np.zeros(len(verts), dtype=bool)
    for pred in predicates:
        vert_inside |= _test_within_object_distance(verts, pred)
    # A face is dropped if any of its 3 vertices is inside an object volume.
    face_touches = vert_inside[faces].any(axis=1)
    inside_any = face_touches
    keep = ~inside_any
    faces = np.asarray(scene_mesh.faces, dtype=np.int64)
    kept_faces = faces[keep]
    used = np.unique(kept_faces.flatten())
    remap = -np.ones(len(scene_mesh.vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    new_verts = np.asarray(scene_mesh.vertices, dtype=np.float64)[used]
    new_faces = remap[kept_faces]

    new_colors = None
    visual = getattr(scene_mesh, "visual", None)
    if visual is not None and hasattr(visual, "vertex_colors"):
        vc = np.asarray(visual.vertex_colors)
        if vc.shape[0] == len(scene_mesh.vertices):
            new_colors = vc[used]
    if new_colors is not None:
        trimmed = trimesh.Trimesh(vertices=new_verts, faces=new_faces,
                                  vertex_colors=new_colors, process=False)
    else:
        trimmed = trimesh.Trimesh(vertices=new_verts, faces=new_faces,
                                  process=False)
    return trimmed, int(inside_any.sum())


def _build_mesh_patch(grid_points_metric: np.ndarray, grid_shape: tuple,
                      rgb01: np.ndarray):
    """Triangulated planar grid on the desk surface with per-vertex RGB."""
    import trimesh
    nx, nz = grid_shape
    verts = grid_points_metric.astype(np.float64)
    tris = []
    for i in range(nx - 1):
        for j in range(nz - 1):
            a = i * nz + j
            b = (i + 1) * nz + j
            c = i * nz + (j + 1)
            d = (i + 1) * nz + (j + 1)
            tris.append([a, b, d])
            tris.append([a, d, c])
    faces = np.asarray(tris, dtype=np.int64)
    colors = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    if colors.shape[1] == 3:
        rgba = np.concatenate(
            [colors, np.full((len(colors), 1), 255, dtype=np.uint8)], axis=-1)
    else:
        rgba = colors
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba,
                           process=False)


def _clone_patch_rgb_from_scene(npz_path: Path, scene_mesh,
                                  predicates: list, k: int = 200,
                                  donor_shell_inner_m: float = 0.05,
                                  donor_shell_outer_m: float = 0.30) -> tuple:
    """K-NN clone patch RGB from surrounding scene-mesh vertices (donor pool),
    NOT from view-sampled photos. This mirrors the splat-side clone logic:
    for each NPZ grid point, take K nearest donor verts and use per-channel
    median RGB. Donors are scene-mesh verts that are (a) outside every
    object 3D crop (dist > donor_shell_inner_m to all object meshes) AND
    (b) within donor_shell_outer_m of at least one object.

    Returns (grid_points_metric, rgb01[N,3], grid_shape_xz).
    """
    from scipy.spatial import cKDTree
    data = np.load(npz_path)
    pts_m = data["grid_points"].astype(np.float64)
    shape_xz = tuple(int(x) for x in data["shape_xz"])

    # Build the donor mask over scene mesh vertices using the same predicates
    # semantics as the splat crop. Donors = verts outside all crops but
    # within the outer shell.
    verts = np.asarray(scene_mesh.vertices, dtype=np.float64)
    d_min = np.full(len(verts), np.inf, dtype=np.float64)
    for pred in predicates:
        # Reuse the RaycastingScene inside the predicate to get 3D distance
        import open3d as o3d
        pts_t = o3d.core.Tensor(verts.astype(np.float32))
        d = pred["scene"].compute_distance(pts_t).numpy().astype(np.float64)
        d_min = np.minimum(d_min, d)
    donor_mask = ((d_min > donor_shell_inner_m)
                  & (d_min < donor_shell_outer_m))
    donor_positions = verts[donor_mask]

    if len(donor_positions) == 0:
        raise RuntimeError(
            f"No donor scene-mesh verts in shell [{donor_shell_inner_m*100:.1f} "
            f"cm, {donor_shell_outer_m*100:.1f} cm]; check predicates.")

    vc = np.asarray(scene_mesh.visual.vertex_colors)
    donor_rgb = vc[donor_mask, :3].astype(np.float64)

    tree = cKDTree(donor_positions)
    k_eff = min(k, len(donor_positions))
    _, nn_idx = tree.query(pts_m, k=k_eff)
    if k_eff == 1:
        rgb01 = donor_rgb[nn_idx] / 255.0
    else:
        rgb01 = np.median(donor_rgb[nn_idx], axis=1) / 255.0
    return pts_m, rgb01.astype(np.float32), shape_xz


def _resample_patch_rgb(npz_path: Path, scene_mesh_path: Path,
                         view_source, sam3_cache: Path | None = None,
                         prompt_slug: str | None = None) -> tuple:
    """Re-derive (grid_points_metric, rgb01, grid_shape) so the mesh patch
    uses the same colour sampling pipeline as the splat patch.

    Uses top-K views + per-channel median RGB (robust to SAM3 mask
    anti-aliasing edge bleed) and, when a SAM3 cache is provided, drops
    views whose mask marks the object at the target pixel (so object
    colour cannot bake into the patch)."""
    from _seed_desk_patch import (_topk_views_per_point,
                                   _sample_color_topk,
                                   _load_sam3_masks,
                                   _nn_fill_missing)
    data = np.load(npz_path)
    pts_m = data["grid_points"].astype(np.float32)
    seen = data["seen_count"]

    # Use the auto-desk normal when the NPZ has it; boosts scoring for
    # top-K view selection.
    desk_normal_metric = None
    if "desk_normal_metric" in data.files:
        desk_normal_metric = np.asarray(
            data["desk_normal_metric"], dtype=np.float64)

    sam3_masks = None
    if sam3_cache is not None and prompt_slug is not None:
        view_names = [v.name for v in view_source.all_views()]
        sam3_masks = _load_sam3_masks(sam3_cache, prompt_slug,
                                       view_names, dilate_px=8)

    _log(f"    resampling RGB for {len(pts_m)} grid points "
         f"({int((seen==0).sum())} truly-unseen); "
         f"sam3_guard={'on' if sam3_masks else 'off'}, "
         f"desk_normal={'set' if desk_normal_metric is not None else 'default'}")

    topk_view, topk_uv = _topk_views_per_point(
        view_source, scene_mesh_path, pts_m, seen,
        sam3_masks=sam3_masks,
        desk_normal_metric=desk_normal_metric,
        k=5,
    )
    rgb = _sample_color_topk(view_source, topk_view, topk_uv)
    rgb = _nn_fill_missing(pts_m.astype(np.float64), rgb)
    shape_xz = tuple(int(x) for x in data["shape_xz"])
    return pts_m.astype(np.float64), rgb.astype(np.float32), shape_xz


# ---------------------------------------------------------------------------
# Stage 4 helpers -- PyMeshFix + K-NN vertex-colour clone-fill
# ---------------------------------------------------------------------------

def _pymeshfix_fill(vertices: np.ndarray, faces: np.ndarray,
                    *, refine: bool = True,
                    max_boundary_edges: int = 0):
    """Run PyMeshFix.fill_holes on (vertices, faces).

    Returns (v_out, f_out, is_new_mask) where `is_new_mask[i] == True` iff
    v_out[i] was not present in the input vertex set (matched via KD-tree
    with 1e-9 m tolerance).
    """
    import pymeshfix
    from scipy.spatial import cKDTree
    mfix = pymeshfix.MeshFix(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )
    n_before = int(mfix.n_boundaries)
    try:
        # pymeshfix 0.16+ signature
        n_filled = mfix.fill_holes(nbe=max_boundary_edges, refine=refine)
    except TypeError:
        # Older builds use keyword `n_edges`
        try:
            n_filled = mfix.fill_holes(n_edges=max_boundary_edges,
                                        refine=refine)
        except TypeError:
            n_filled = mfix.fill_holes()
    n_after = int(mfix.n_boundaries)
    v_out = np.asarray(mfix.points, dtype=np.float64)
    f_out = np.asarray(mfix.faces, dtype=np.int64)

    tree_in = cKDTree(np.asarray(vertices, dtype=np.float64))
    d, _ = tree_in.query(v_out, k=1, distance_upper_bound=1e-7)
    is_new = ~np.isfinite(d)
    _log(f"    MeshFix: boundaries {n_before} -> {n_after}, "
         f"new_verts={int(is_new.sum())}, "
         f"n_verts {len(vertices)} -> {len(v_out)}, "
         f"n_faces {len(faces)} -> {len(f_out)}")
    return v_out, f_out, is_new


def _detect_hole_ring_indices(trimmed_mesh):
    """Return (boundary_vert_idx_by_loop, ring_ref_rgb_by_loop).

    - boundary_vert_idx_by_loop: list of arrays, one per boundary loop, of
      vertex indices ON the boundary.
    - ring_ref_rgb_by_loop: (n_loops, 3) uint8 RGB median per loop.
    """
    import trimesh
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    faces = np.asarray(trimmed_mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return [], np.zeros((0, 3), dtype=np.uint8)

    # Undirected edges of each face
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
                       axis=0)
    e_sorted = np.sort(e, axis=1)
    # Count each undirected edge; boundary edges appear exactly once
    edges_view = np.ascontiguousarray(e_sorted).view(
        np.dtype((np.void, e_sorted.dtype.itemsize * 2)))
    _, inv, counts = np.unique(edges_view, return_inverse=True,
                                return_counts=True)
    boundary_edge_mask = counts[inv] == 1
    boundary_edges = e_sorted[boundary_edge_mask]
    if len(boundary_edges) == 0:
        return [], np.zeros((0, 3), dtype=np.uint8)

    boundary_verts = np.unique(boundary_edges)
    n_verts = len(trimmed_mesh.vertices)
    # Build adjacency graph over boundary vertices only
    rows = np.concatenate([boundary_edges[:, 0], boundary_edges[:, 1]])
    cols = np.concatenate([boundary_edges[:, 1], boundary_edges[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
    n_comp, labels = connected_components(adj, directed=False)

    # Only keep components that actually contain any boundary vertex
    vc = np.asarray(trimmed_mesh.visual.vertex_colors)
    loops = []
    ring_rgbs = []
    for cid in np.unique(labels[boundary_verts]):
        idx = np.where((labels == cid))[0]
        idx = np.intersect1d(idx, boundary_verts)
        if len(idx) == 0:
            continue
        loops.append(idx)
        if vc.shape[0] == n_verts:
            ring_rgbs.append(np.median(vc[idx, :3], axis=0).astype(np.uint8))
        else:
            ring_rgbs.append(np.array([128, 128, 128], dtype=np.uint8))
    ring_ref_rgb = np.asarray(ring_rgbs, dtype=np.uint8) if ring_rgbs \
        else np.zeros((0, 3), dtype=np.uint8)
    return loops, ring_ref_rgb


def _assign_new_verts_to_loops(v_out: np.ndarray, is_new: np.ndarray,
                               trimmed_mesh, loops: list) -> np.ndarray:
    """Return per-new-vertex loop id (int, in [0, n_loops)) by nearest ring
    centroid. If loops is empty, returns all zeros of length n_new.
    """
    n_new = int(is_new.sum())
    if n_new == 0 or len(loops) == 0:
        return np.zeros(n_new, dtype=np.int64)
    tv = np.asarray(trimmed_mesh.vertices, dtype=np.float64)
    centroids = np.stack([tv[loop].mean(axis=0) for loop in loops], axis=0)
    from scipy.spatial import cKDTree
    tree = cKDTree(centroids)
    _, hole_id = tree.query(v_out[is_new], k=1)
    return hole_id.astype(np.int64)


def _build_donor_kdtree(scene_mesh, predicates: list,
                        *, donor_shell_m: float, donor_shell_inner_m: float,
                        hole_ring_positions: np.ndarray):
    """Return (kdtree, donor_positions, donor_rgba) over scene_mesh vertices
    that are:
      - outside every predicate (no crop-volume hits)
      - within [inner, outer] shell of the nearest hole-ring point
      - have vertex colours (rgba)
    Returns (None, None, None) if no donors.
    """
    from scipy.spatial import cKDTree
    verts = np.asarray(scene_mesh.vertices, dtype=np.float64)
    if len(verts) == 0:
        return None, None, None
    vc = getattr(scene_mesh.visual, "vertex_colors", None)
    if vc is None:
        return None, None, None
    vc = np.asarray(vc)
    if vc.shape[0] != len(verts):
        return None, None, None

    inside_any = np.zeros(len(verts), dtype=bool)
    for pred in predicates:
        inside_any |= _test_within_object_distance(verts, pred)
    outside = ~inside_any

    # Drop pure-white / fallback verts (luminance > 250 and ~R=G=B)
    rgb = vc[:, :3].astype(np.int32)
    lum = (0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2])
    is_white = (lum > 250) & (np.abs(rgb[:, 0] - rgb[:, 1]) < 6) \
                & (np.abs(rgb[:, 1] - rgb[:, 2]) < 6)
    valid = outside & ~is_white

    if hole_ring_positions is None or len(hole_ring_positions) == 0:
        return None, None, None

    ring_tree = cKDTree(np.asarray(hole_ring_positions, dtype=np.float64))
    d_ring, _ = ring_tree.query(verts, k=1)
    in_shell = (d_ring >= donor_shell_inner_m) & (d_ring <= donor_shell_m)

    donor_mask = valid & in_shell
    n_donors = int(donor_mask.sum())
    if n_donors == 0:
        return None, None, None
    donor_pos = verts[donor_mask]
    donor_rgba = vc[donor_mask].astype(np.uint8)
    kdtree = cKDTree(donor_pos)
    return kdtree, donor_pos, donor_rgba


def _clone_colors_knn(new_verts: np.ndarray, kdtree, donor_positions,
                       donor_rgba, *, k: int, outlier_reject_sigma: float,
                       ring_ref_rgb: np.ndarray) -> np.ndarray:
    """Per new_vert: K-NN over donor_positions, optional MAD outlier reject
    vs `ring_ref_rgb[i]`, then per-channel MEDIAN of survivors. Returns
    (N_new, 4) uint8 RGBA with A=255.
    """
    n_new = new_verts.shape[0]
    k_eff = min(k, len(donor_positions))
    dists, nn = kdtree.query(new_verts, k=k_eff)
    if k_eff == 1:
        nn = nn[:, None]
        dists = dists[:, None]
    donor_rgb = donor_rgba[:, :3].astype(np.float32)
    out = np.zeros((n_new, 4), dtype=np.uint8)
    out[:, 3] = 255

    ring_ref_rgb = np.asarray(ring_ref_rgb, dtype=np.float32)
    if ring_ref_rgb.shape != (n_new, 3):
        # ring_ref_rgb may be a single (3,) or (n_holes, 3) — normalise upstream
        ring_ref_rgb = np.broadcast_to(
            ring_ref_rgb.reshape(-1, 3)[0][None, :], (n_new, 3)).copy()

    for i in range(n_new):
        idx = nn[i]
        rgb_k = donor_rgb[idx]                    # (K, 3)
        if outlier_reject_sigma > 0 and k_eff >= 3:
            ref = ring_ref_rgb[i]
            deltas = np.abs(rgb_k - ref)
            mad = np.median(deltas, axis=0) + 1.0
            reject_band = outlier_reject_sigma * 1.4826 * mad
            keep = np.all(deltas <= reject_band, axis=1)
            if keep.sum() >= 3:
                rgb_k = rgb_k[keep]
        rgb_med = np.median(rgb_k, axis=0)
        out[i, :3] = np.clip(rgb_med, 0, 255).astype(np.uint8)
    return out


def _clone_fill_mesh_hole(*, trimmed_mesh, scene_mesh, predicates,
                          donor_shell_m: float = 0.30,
                          donor_shell_inner_m: float = 0.00,
                          k: int = 8,
                          outlier_reject_sigma: float = 2.0,
                          y_tolerance_m: tuple = (-0.01, 0.05),
                          fallback_planar_patches=(),
                          project_root: Path,
                          scene_name: str,
                          vs=None,
                          scene_mesh_path: Path):
    """Fill each hole in `trimmed_mesh` with PyMeshFix, then K-NN clone-stamp
    RGB from `scene_mesh` vertices onto every new vertex.

    Returns (combined_mesh, n_new_verts, n_new_faces, n_holes_closed).
    Falls back to planar-patch mode (existing behaviour) if PyMeshFix fails
    or adds 0 verts.
    """
    import trimesh
    from scipy.spatial import cKDTree

    def _planar_fallback(reason: str):
        _log(f"    clone-fill: FALLBACK to planar patches ({reason})")
        chunks = [trimmed_mesh]
        n_patch_faces = 0
        for npz, slug in fallback_planar_patches:
            if not Path(npz).exists() or vs is None:
                _log(f"    [{slug}] no NPZ at {npz}; skipping planar patch")
                continue
            sam3_cache = (project_root / "output"
                          / f"segmented_{scene_name}" / "masks")
            pts_m, rgb01, shape_xz = _resample_patch_rgb(
                Path(npz), scene_mesh_path, vs,
                sam3_cache=sam3_cache, prompt_slug=slug)
            patch_mesh = _build_mesh_patch(pts_m, shape_xz, rgb01)
            chunks.append(patch_mesh)
            n_patch_faces += len(patch_mesh.faces)
        if len(chunks) > 1:
            try:
                combined = trimesh.util.concatenate(chunks)
            except Exception as e:
                _log(f"    WARN planar concat failed ({e}); trimmed only")
                combined = trimmed_mesh
        else:
            combined = trimmed_mesh
        return combined, 0, n_patch_faces, 0

    n_faces_before = len(trimmed_mesh.faces)

    # (1) PyMeshFix.
    try:
        v_out, f_out, is_new = _pymeshfix_fill(
            trimmed_mesh.vertices, trimmed_mesh.faces,
            refine=True, max_boundary_edges=0)
    except Exception as e:
        return _planar_fallback(f"MeshFix raised {e!r}")

    n_new = int(is_new.sum())
    if n_new == 0:
        return _planar_fallback("MeshFix added 0 verts")

    # (2) Identify per-loop ring reference RGB.
    loops, ring_ref_rgb_by_hole = _detect_hole_ring_indices(trimmed_mesh)
    if len(loops) == 0:
        return _planar_fallback("no boundary loops detected")
    hole_id_of_new = _assign_new_verts_to_loops(
        v_out, is_new, trimmed_mesh, loops)

    # (3) Build donor set (widen once if too sparse).
    hole_ring_positions = v_out[is_new]
    kdtree, donor_pos, donor_rgba = _build_donor_kdtree(
        scene_mesh, predicates,
        donor_shell_m=donor_shell_m,
        donor_shell_inner_m=donor_shell_inner_m,
        hole_ring_positions=hole_ring_positions)
    if kdtree is None or len(donor_pos) < k:
        _log(f"    donor shell too sparse ({0 if kdtree is None else len(donor_pos)} < K={k}); "
             f"widening to 0.50 m")
        kdtree, donor_pos, donor_rgba = _build_donor_kdtree(
            scene_mesh, predicates,
            donor_shell_m=0.50, donor_shell_inner_m=0.00,
            hole_ring_positions=hole_ring_positions)
    if kdtree is None:
        return _planar_fallback("no donors available even at 0.50 m shell")

    # (4) K-NN clone.
    ring_ref_per_new = ring_ref_rgb_by_hole[hole_id_of_new] \
        if len(ring_ref_rgb_by_hole) > 0 \
        else np.full((n_new, 3), 128, dtype=np.uint8)
    new_rgba = _clone_colors_knn(
        hole_ring_positions, kdtree, donor_pos, donor_rgba,
        k=k, outlier_reject_sigma=outlier_reject_sigma,
        ring_ref_rgb=ring_ref_per_new)

    # (5) Assemble the full RGBA array (old verts + new verts). MeshFix
    # preserves input vertex ordering: rows corresponding to input verts
    # can be identified via KDTree membership on the input.
    tree_in = cKDTree(np.asarray(trimmed_mesh.vertices, dtype=np.float64))
    d, nn_in = tree_in.query(v_out, k=1, distance_upper_bound=1e-7)
    old_vc = np.asarray(trimmed_mesh.visual.vertex_colors, dtype=np.uint8)
    if old_vc.shape[0] != len(trimmed_mesh.vertices):
        old_vc = np.full((len(trimmed_mesh.vertices), 4), 200, dtype=np.uint8)
    full_rgba = np.zeros((len(v_out), 4), dtype=np.uint8)
    # For matched (~is_new) rows, copy from old_vc; for new rows, use new_rgba
    matched = ~is_new
    if matched.any():
        # Guard against distance_upper_bound sentinel (nn_in==len(v_in))
        nn_safe = np.where(nn_in[matched] < len(old_vc),
                           nn_in[matched], 0)
        full_rgba[matched] = old_vc[nn_safe]
    full_rgba[is_new] = new_rgba

    # (6) Y-sanity clamp of new verts vs their assigned loop's Y range.
    tv = np.asarray(trimmed_mesh.vertices, dtype=np.float64)
    n_clamped = 0
    y_lo_tol, y_hi_tol = y_tolerance_m
    v_out_mut = v_out.copy()
    new_idx = np.where(is_new)[0]
    for hid, loop in enumerate(loops):
        loop_y = tv[loop, 1]
        y_lo = float(loop_y.min()) + y_lo_tol
        y_hi = float(loop_y.max()) + y_hi_tol
        sel = new_idx[hole_id_of_new == hid]
        if len(sel) == 0:
            continue
        ys = v_out_mut[sel, 1]
        too_lo = ys < y_lo
        too_hi = ys > y_hi
        if too_lo.any():
            v_out_mut[sel[too_lo], 1] = y_lo
            n_clamped += int(too_lo.sum())
        if too_hi.any():
            v_out_mut[sel[too_hi], 1] = y_hi
            n_clamped += int(too_hi.sum())
    if n_clamped:
        _log(f"    Y-clamped {n_clamped}/{n_new} new verts to ring Y range")

    # Sanity: too many Y-clamps means MeshFix over-refined -> planar fallback
    if n_clamped > 0.5 * max(1, n_new):
        return _planar_fallback(
            f"MeshFix over-refined ({n_clamped}/{n_new} clamped)")

    # (7) Build final mesh; do NOT weld verts (would drop colours).
    filled = trimesh.Trimesh(vertices=v_out_mut, faces=f_out,
                              vertex_colors=full_rgba, process=False)
    try:
        if getattr(trimmed_mesh, "is_winding_consistent", False):
            filled.fix_normals()
    except Exception as e:
        _log(f"    WARN fix_normals skipped ({e})")

    n_new_faces = len(f_out) - n_faces_before
    n_holes_closed = len(loops)  # MeshFix closes them all with fill_holes(0)
    _log(f"    clone-fill: added {n_new:,} verts, {n_new_faces:,} faces; "
         f"donors used {len(donor_pos):,} in shell")
    return filled, n_new, n_new_faces, n_holes_closed


# ---------------------------------------------------------------------------
# Stage 4 -- scene without all objects (mesh + splat sides)
# ---------------------------------------------------------------------------

def stage_scene_without_objects(*, scene_name: str, project_root: Path,
                                 prompts: list,
                                 scene_mesh_path: Path,
                                 cleaned_splat_path: Path,
                                 object_mesh_dir: Path,
                                 patch_splat_dir: Path,
                                 unseen_dir: Path,
                                 dp_json: Path, bounds_json: Path,
                                 out_mesh_path: Path,
                                 out_splat_path: Path) -> dict:
    _log("[stage 4] scene without objects (mesh + splat, combined)")

    # ----- Build the spatial-crop predicate per prompt -----
    # 3D distance-to-extracted-mesh crop: replaces the earlier XZ silhouette
    # approach. For each object, drop mesh vertices / splat Gaussians
    # within a metric distance threshold (default 5 cm) of the extracted
    # object mesh, up to a Y ceiling (Y_min + 1 m) to prevent the query
    # picking up distant ceiling/wall geometry above the desk.
    #
    # This subsumes the "XZ dilation + Y range" approach and handles:
    #   - Objects with complex shapes (bottle neck vs base) where the XZ
    #     silhouette can't follow the 3D volume.
    #   - Stretched anisotropic Gaussians whose CENTER sits 3-5 cm outside
    #     the tight silhouette but whose ellipsoid extends into the
    #     object (visible as the "blue vertical stripe" artifact above
    #     the blue box in the v2 output — 682 anisotropy>5x Gaussians in
    #     the 3-5 cm ring alone).
    #   - Overhangs above the extracted mesh (translucent bottle tops
    #     missed by SAM3).
    #
    # V32 objects are 10-30 cm apart so a 5 cm radius does not intrude
    # on neighbours; if objects are tighter this can be lowered via
    # --crop-distance-m (default 0.05).
    predicates = []
    for slug in prompts:
        ply = _require(object_mesh_dir / slug / f"{slug}_extracted.ply",
                       f"extracted obj mesh for {slug}")
        pred = _build_object_distance_predicate(
            ply, dist_threshold_m=0.05, y_ceiling_offset_m=1.0)
        _log(f"  [{slug}] object predicate: 3D dist <= 5.0 cm, "
             f"y_ceiling = {pred['y_ceiling_m']:.3f} m")
        predicates.append(pred)

    # ===== SPLAT side =====
    _log("  --- splat side ---")
    R, t, dp_scale = _load_dataparser_transform(dp_json)
    bounds = json.loads(bounds_json.read_text())
    colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])
    cleaned_data, _ = _read_ply(cleaned_splat_path)
    n_cleaned = len(cleaned_data)
    pos_splat = np.stack(
        [cleaned_data["x"], cleaned_data["y"], cleaned_data["z"]],
        axis=-1).astype(np.float64)
    pos_metric = _splat_to_metric(pos_splat, R, t, dp_scale, colmap_to_metric)
    inside_any_splat = np.zeros(n_cleaned, dtype=bool)
    for pred, slug in zip(predicates, prompts):
        ins = _test_within_object_distance(pos_metric, pred)
        _log(f"    [{slug}] splat within 5 cm of object mesh: {int(ins.sum()):,}")
        inside_any_splat |= ins
    keep_splat = ~inside_any_splat
    _log(f"    cleaned splat: {n_cleaned:,} -> {int(keep_splat.sum()):,} "
         f"(removed {int(inside_any_splat.sum()):,})")
    scene_minus = cleaned_data[keep_splat]

    # Concatenate desk patches
    chunks = [scene_minus]
    for slug in prompts:
        patch_p = patch_splat_dir / f"desk_patch_{slug}.ply"
        if not patch_p.exists():
            _log(f"    WARN no desk patch for {slug}: {patch_p}")
            continue
        patch_data, _ = _read_ply(patch_p)
        if patch_data.dtype != scene_minus.dtype:
            _log(f"    WARN dtype mismatch for {slug} desk patch; "
                 f"reconciling field-by-field")
            # Field-by-field copy into scene dtype to avoid silent corruption.
            buf = np.zeros(len(patch_data), dtype=scene_minus.dtype)
            for name in scene_minus.dtype.names:
                if name in patch_data.dtype.names:
                    buf[name] = patch_data[name]
            patch_data = buf
        chunks.append(patch_data)
        _log(f"    + desk patch {slug}: +{len(patch_data):,} gaussians")
    merged_splat = np.concatenate(chunks)
    _write_ply(merged_splat, out_splat_path)
    _log(f"    wrote {out_splat_path}  "
         f"({_mb(out_splat_path):.1f} MB, {len(merged_splat):,} gaussians)")

    # ===== MESH side =====
    _log("  --- mesh side ---")
    import trimesh
    scene_mesh = trimesh.load(str(scene_mesh_path), force="mesh",
                              process=False)
    _log(f"    loaded scene mesh: {len(scene_mesh.vertices):,} verts, "
         f"{len(scene_mesh.faces):,} faces")
    _bake_vertex_colors_from_atlas(scene_mesh)

    trimmed_mesh, n_removed_faces = _drop_faces_inside_any(scene_mesh,
                                                            predicates)
    _log(f"    trimmed mesh: removed {n_removed_faces:,} faces, "
         f"{len(trimmed_mesh.faces):,} remain")

    # View source for colour resampling (mesh-patch RGB)
    try:
        from scene_segmenter.views import V32ViewSource
        vs = V32ViewSource(project_root)
    except Exception as e:
        _log(f"    WARN: V32ViewSource init failed ({e}); "
             f"mesh patches will be skipped")
        vs = None

    # SAM3 mask cache for the mesh-patch RGB sampler (prevents object
    # colour from baking into the patch when a candidate view still shows
    # the object). Absent-directory case is handled inside
    # _resample_patch_rgb -> _load_sam3_masks (prints WARN, sam3_guard=off).
    sam3_cache = project_root / "output" / f"segmented_{scene_name}" / "masks"
    _log(f"    sam3_cache = {sam3_cache}  (exists={sam3_cache.is_dir()})")

    # ---- Mesh fill: per-object planar patch (built from NPZ grid points) +
    # K-NN clone RGB from surrounding scene-mesh vertices (same clone
    # semantics as the splat side). The MeshFix.fill_holes(refine=True)
    # approach was tried but filled every one of ~23,000 boundary loops in
    # the scene mesh (not just the 3 object holes), adding 513k vertices
    # scene-wide -> chaotic moth-eaten desk. This approach only touches
    # the 3 patch regions.
    import trimesh as _tm
    mesh_chunks = [trimmed_mesh]
    n_patch_faces_total = 0
    for slug in prompts:
        npz = unseen_dir / f"unseen_core_{slug}.npz"
        if not npz.exists():
            _log(f"    [{slug}] no NPZ at {npz}; skipping mesh patch")
            continue
        pts_m, rgb01, shape_xz = _clone_patch_rgb_from_scene(
            npz, scene_mesh, predicates,
            k=200, donor_shell_inner_m=0.05, donor_shell_outer_m=0.30,
        )
        patch_mesh = _build_mesh_patch(pts_m, shape_xz, rgb01)
        n_patch_faces_total += len(patch_mesh.faces)
        mesh_chunks.append(patch_mesh)
        _log(f"    [{slug}] mesh patch: {len(patch_mesh.faces):,} faces, "
             f"{len(patch_mesh.vertices):,} verts, "
             f"mean RGB=({int(rgb01[:,0].mean()*255)},"
             f"{int(rgb01[:,1].mean()*255)},{int(rgb01[:,2].mean()*255)})")

    if len(mesh_chunks) > 1:
        try:
            combined = _tm.util.concatenate(mesh_chunks)
        except Exception as e:
            _log(f"    WARN concat failed ({e}); saving trimmed only")
            combined = trimmed_mesh
    else:
        combined = trimmed_mesh

    out_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(out_mesh_path))
    _log(f"    wrote {out_mesh_path}  "
         f"({_mb(out_mesh_path):.1f} MB, {len(combined.faces):,} faces, "
         f"{len(combined.vertices):,} verts)")

    return {
        "splat_n_cleaned_in": n_cleaned,
        "splat_n_removed": int(inside_any_splat.sum()),
        "splat_n_out": int(len(merged_splat)),
        "mesh_n_removed_faces": int(n_removed_faces),
        "mesh_n_patch_faces": int(n_patch_faces_total),
        "mesh_n_out_faces": int(len(combined.faces)),
    }


# ---------------------------------------------------------------------------
# Stage 5 -- scene_full copies (raw deliverables)
# ---------------------------------------------------------------------------

def stage_scene_full(*, splat_in: Path, mesh_in: Path,
                      out_mesh: Path, out_splat: Path,
                      texture_sidecar: Path | None):
    _log("[stage 5] copy scene_full assets")
    _copy(splat_in, out_splat)
    _log(f"  splat -> {out_splat}  ({_mb(out_splat):.1f} MB)")
    _copy(mesh_in, out_mesh)
    _log(f"  mesh  -> {out_mesh}  ({_mb(out_mesh):.1f} MB)")
    if texture_sidecar is not None and texture_sidecar.exists():
        dst = out_mesh.parent / texture_sidecar.name
        _copy(texture_sidecar, dst)
        _log(f"  tex   -> {dst}  ({_mb(dst):.2f} MB)")


# ---------------------------------------------------------------------------
# README emitter
# ---------------------------------------------------------------------------

def _write_readme(out_root: Path, scene_name: str, prompts: list,
                  stage2: dict, stage4: dict,
                  per_object: list):
    lines = [
        f"# Pipeline output: {scene_name}",
        "",
        "Generated by `_pipeline_full.py`.",
        "",
        "## Layout",
        "",
        "```",
        f"{out_root.name}/",
        "  mesh/",
        "    scene_full.ply",
        "    scene_textured0.png",
        "    scene_without_objects.ply",
        "    objects/",
    ]
    for slug in prompts:
        lines.append(f"      {slug}.ply")
    lines += [
        "  splat/",
        "    scene_full.ply",
        "    scene_cleaned.ply",
        "    scene_without_objects.ply",
        "    objects/",
    ]
    for slug in prompts:
        lines.append(f"      {slug}.ply")
    lines += [
        "```",
        "",
        "## Splat cleanup",
        "",
        f"- method: {stage2['method']} (mesh-distance smoothstep)",
        f"- mode: {stage2['mode']}  "
        f"(both = scale-shrink AND opacity-fade)",
        f"- inner edge: {stage2['inner_m']*100:.1f} cm  "
        f"(distance where multiplier = 1.0)",
        f"- outer edge: {stage2['outer_m']*100:.1f} cm  "
        f"(distance where multiplier = 0.0)",
        f"- gaussians kept: {stage2['n_kept']:,}/{stage2['n_in']:,}  "
        f"({stage2['pct_kept']:.2f}%)",
        "",
        "## Scene without objects",
        "",
        f"- splat: {stage4['splat_n_out']:,} gaussians "
        f"(removed {stage4['splat_n_removed']:,} object gaussians, "
        f"added 3 desk patches)",
        f"- mesh : {stage4['mesh_n_out_faces']:,} faces "
        f"(removed {stage4['mesh_n_removed_faces']:,} object faces, "
        f"added {stage4['mesh_n_patch_faces']:,} patch faces)",
        "",
        "## Per-object",
        "",
    ]
    for rec in per_object:
        lines.append(
            f"- **{rec['slug']}**: mesh {rec['mesh_mb']:.2f} MB, "
            f"splat {rec['splat_mb']:.2f} MB"
        )
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    _log(f"  wrote {out_root / 'README.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", type=Path, default=_PROJECT_ROOT,
                    help="(default: directory containing this script)")
    ap.add_argument("--scene-name", default="v32_data3")
    ap.add_argument("--skip-reconstruction", action="store_true",
                    help="reuse existing reconstruction outputs without running "
                         "reconstruct_realityscan.py")
    ap.add_argument("--prompts",
                    default="white_water_bottle,blue_box,red_lobster_figurine",
                    help="comma-separated object slugs (must match segmenter "
                         "output dir names)")
    ap.add_argument("--cleanup-mode",
                    choices=("scale", "opacity", "both"), default="both",
                    help="v11 splat-cleanup mode: 'scale' shrinks the "
                         "Gaussian scale axes, 'opacity' fades the sigmoid "
                         "opacity, 'both' does both (default)")
    ap.add_argument("--cleanup-inner-m", type=float, default=0.02,
                    help="v11 inner edge in metres: mesh-distance below "
                         "this leaves Gaussians untouched (default 0.02 "
                         "= 2 cm)")
    ap.add_argument("--cleanup-outer-m", type=float, default=0.20,
                    help="v11 outer edge in metres: mesh-distance beyond "
                         "this fully suppresses Gaussians (default 0.20 "
                         "= 20 cm)")
    ap.add_argument("--out-root", type=Path,
                    default=Path("output/pipeline_v32_test_run"))
    # Optional input overrides (sensible defaults wired for V32)
    ap.add_argument("--scene-splat", type=Path, default=None,
                    help="default: output/splat_<scene>_noinit_pruned.ply")
    ap.add_argument("--scene-mesh", type=Path, default=None,
                    help="default: output/mesh_<scene>/mesh_<scene>_openmvs.ply")
    ap.add_argument("--scene-mesh-texture", type=Path, default=None,
                    help="default: output/mesh_<scene>/scene_textured0.png")
    ap.add_argument("--object-mesh-dir", type=Path, default=None,
                    help="default: output/segmented_<scene>_v9a_fp_v2")
    ap.add_argument("--object-splat-dir", type=Path, default=None,
                    help="default: output/segmented_<scene>_v9a_fp_v2_splat_v6_cleaned")
    ap.add_argument("--patch-splat-dir", type=Path, default=None,
                    help="default: output/desk_patch_<scene>")
    ap.add_argument("--unseen-dir", type=Path, default=None,
                    help="default: output/unseen_core_<scene>")
    ap.add_argument("--dataparser", type=Path,
                    default=Path("nerfstudio/dense/splatfacto/2026-06-07_203807/"
                                 "dataparser_transforms.json"))
    ap.add_argument("--bounds-json", type=Path,
                    default=Path("colmap/dense/tof_bounds.json"))
    args = ap.parse_args()

    scene = args.scene_name
    project_root = args.project_root.resolve()
    out_root = (project_root / args.out_root) if not args.out_root.is_absolute() \
        else args.out_root
    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]

    # Resolve defaults that depend on scene name
    scene_splat = args.scene_splat or (
        project_root / "output" / f"splat_{scene}_noinit_pruned.ply")
    scene_mesh = args.scene_mesh or (
        project_root / "output" / f"mesh_{scene}" / f"mesh_{scene}_openmvs.ply")
    scene_mesh_tex = args.scene_mesh_texture or (
        project_root / "output" / f"mesh_{scene}" / "scene_textured0.png")
    object_mesh_dir = args.object_mesh_dir or (
        project_root / "output" / f"segmented_{scene}_v9a_fp_v2")
    object_splat_dir = args.object_splat_dir or (
        project_root / "output" / f"segmented_{scene}_v9a_fp_v2_splat_v6_cleaned")
    patch_splat_dir = args.patch_splat_dir or (
        project_root / "output" / f"desk_patch_{scene}")
    unseen_dir = args.unseen_dir or (
        project_root / "output" / f"unseen_core_{scene}")
    dp_json = args.dataparser if args.dataparser.is_absolute() \
        else project_root / args.dataparser
    bounds_json = args.bounds_json if args.bounds_json.is_absolute() \
        else project_root / args.bounds_json

    # ---- Step 0: validate inputs ----
    _section(f"validating inputs for scene = {scene}")
    if not args.skip_reconstruction:
        # Reconstruction may produce these; only require them after stage 1.
        pass
    # The downstream artefacts must already exist (per-object segmenter outputs).
    for slug in prompts:
        _require(object_mesh_dir / slug / f"{slug}_extracted.ply",
                 f"object mesh for {slug}")
        _require(object_splat_dir / f"{slug}_splat.ply",
                 f"object splat for {slug}")
    _require(dp_json, "dataparser_transforms.json")
    _require(bounds_json, "tof_bounds.json")
    _log("  all preexisting artefacts found")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "mesh" / "objects").mkdir(parents=True, exist_ok=True)
    (out_root / "splat" / "objects").mkdir(parents=True, exist_ok=True)
    _log(f"  out_root = {out_root}")

    # ---- Stage 1: reconstruction (optional) ----
    _section("stage 1 -- reconstruction")
    stage_reconstruction(
        project_root=project_root, scene_name=scene,
        skip=args.skip_reconstruction,
        splat_path=scene_splat, mesh_path=scene_mesh,
    )
    _require(scene_splat, "scene splat (post-reconstruction)")
    _require(scene_mesh, "scene mesh (post-reconstruction)")

    # ---- Stage 2: splat cleanup (v11 smoothstep, mode=both by default) ----
    _section(f"stage 2 -- splat cleanup "
             f"({args.cleanup_mode}, {args.cleanup_inner_m*100:.1f}->"
             f"{args.cleanup_outer_m*100:.1f} cm)")
    cleaned_splat = out_root / "splat" / "scene_cleaned.ply"
    stage2 = stage_splat_cleanup(
        splat_in=scene_splat, mesh_in=scene_mesh,
        dp_json=dp_json, bounds_json=bounds_json,
        inner_m=args.cleanup_inner_m,
        outer_m=args.cleanup_outer_m,
        mode=args.cleanup_mode,
        out_ply=cleaned_splat,
        project_root=project_root,
    )

    # ---- Stage 3: per-object outputs ----
    _section("stage 3 -- per-object outputs")
    per_object = stage_per_object(
        prompts=prompts,
        object_mesh_dir=object_mesh_dir,
        object_splat_dir=object_splat_dir,
        out_mesh_dir=out_root / "mesh" / "objects",
        out_splat_dir=out_root / "splat" / "objects",
    )

    # ---- Stage 4: scene without all objects ----
    _section("stage 4 -- scene without all objects")
    out_mesh_no_obj = out_root / "mesh" / "scene_without_objects.ply"
    out_splat_no_obj = out_root / "splat" / "scene_without_objects.ply"
    stage4 = stage_scene_without_objects(
        scene_name=scene, project_root=project_root,
        prompts=prompts,
        scene_mesh_path=scene_mesh,
        cleaned_splat_path=cleaned_splat,
        object_mesh_dir=object_mesh_dir,
        patch_splat_dir=patch_splat_dir,
        unseen_dir=unseen_dir,
        dp_json=dp_json, bounds_json=bounds_json,
        out_mesh_path=out_mesh_no_obj,
        out_splat_path=out_splat_no_obj,
    )

    # ---- Stage 5: copy scene_full ----
    _section("stage 5 -- scene_full copies")
    stage_scene_full(
        splat_in=scene_splat, mesh_in=scene_mesh,
        out_mesh=out_root / "mesh" / "scene_full.ply",
        out_splat=out_root / "splat" / "scene_full.ply",
        texture_sidecar=scene_mesh_tex if scene_mesh_tex.exists() else None,
    )

    # ---- README ----
    _section("writing README")
    _write_readme(out_root, scene, prompts,
                  stage2, stage4, per_object)

    # ---- Stage 6: summary ----
    _section("summary")
    files = [
        ("mesh/scene_full.ply",            out_root / "mesh" / "scene_full.ply"),
        ("mesh/scene_without_objects.ply", out_mesh_no_obj),
        ("splat/scene_full.ply",           out_root / "splat" / "scene_full.ply"),
        ("splat/scene_cleaned.ply",        cleaned_splat),
        ("splat/scene_without_objects.ply", out_splat_no_obj),
    ]
    for slug in prompts:
        files.append((f"mesh/objects/{slug}.ply",
                      out_root / "mesh" / "objects" / f"{slug}.ply"))
        files.append((f"splat/objects/{slug}.ply",
                      out_root / "splat" / "objects" / f"{slug}.ply"))
    _log(f"  {'file':<46s}  {'size (MB)':>10s}  {'exists':>6s}")
    _log(f"  {'-'*46}  {'-'*10}  {'-'*6}")
    for rel, p in files:
        _log(f"  {rel:<46s}  {_mb(p):>10.2f}  {str(p.exists()):>6s}")

    summary = {
        "scene_name": scene,
        "out_root": str(out_root),
        "prompts": prompts,
        "cleanup_mode": args.cleanup_mode,
        "cleanup_inner_m": args.cleanup_inner_m,
        "cleanup_outer_m": args.cleanup_outer_m,
        "stage2_splat_cleanup": stage2,
        "stage3_per_object": per_object,
        "stage4_scene_without_objects": stage4,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"\nwrote {out_root / 'summary.json'}")
    _log(f"\nDONE. Final deliverable: {out_root}")


if __name__ == "__main__":
    main()
