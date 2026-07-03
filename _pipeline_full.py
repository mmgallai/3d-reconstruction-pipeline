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
     - MESH: load V32 mesh, bake per-vertex atlas colours, drop faces whose
             centroid lies inside ANY of the prompt footprints (XZ + Y), then
             add one planar mesh patch per prompt (from the precomputed
             unseen_core NPZs + RGB resampling). Concatenate -> scene mesh.
     - SPLAT: start from stage-2 output, drop Gaussians inside ANY prompt
              footprint, concatenate the 3 desk patches from
              output/desk_patch_<scene>/desk_patch_<slug>.ply.

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
    _build_footprint_from_extracted_mesh,
    _test_inside_footprint,
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


def _drop_faces_inside_any(scene_mesh, footprints: list):
    """Drop faces whose centroid is inside ANY of the supplied footprints.

    Returns (new_trimesh, n_removed). Preserves per-vertex colours.
    """
    import trimesh
    centroids = _mesh_face_centroids(scene_mesh)
    inside_any = np.zeros(len(centroids), dtype=bool)
    for fp in footprints:
        inside_any |= _test_inside_footprint(centroids, fp)
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


def _resample_patch_rgb(npz_path: Path, scene_mesh_path: Path,
                         view_source) -> tuple:
    """Re-derive (grid_points_metric, rgb01, grid_shape) so the mesh patch
    uses the same colour sampling pipeline as the splat patch."""
    from _seed_desk_patch import (_best_view_per_point,
                                   _sample_color_for_points,
                                   _nn_fill_missing)
    data = np.load(npz_path)
    pts_m = data["grid_points"].astype(np.float32)
    seen = data["seen_count"]
    _log(f"    resampling RGB for {len(pts_m)} grid points "
         f"({int((seen==0).sum())} truly-unseen)")
    best_view, best_uv = _best_view_per_point(view_source, scene_mesh_path,
                                              pts_m, seen)
    rgb = _sample_color_for_points(view_source, best_view, best_uv)
    rgb = _nn_fill_missing(pts_m.astype(np.float64), rgb)
    shape_xz = tuple(int(x) for x in data["shape_xz"])
    return pts_m.astype(np.float64), rgb.astype(np.float32), shape_xz


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
    footprints = []
    for slug in prompts:
        ply = _require(object_mesh_dir / slug / f"{slug}_extracted.ply",
                       f"extracted obj mesh for {slug}")
        fp = _build_footprint_from_extracted_mesh(
            ply, xz_resolution=0.005, xz_dilate_cm=0.5, y_margin_cm=0.5)
        area_cm2 = float(fp["fp_mask"].sum()) * (fp["resolution"] * 100.0) ** 2
        _log(f"  [{slug}] footprint area = {area_cm2:.0f} cm^2, "
             f"Y in [{fp['y_min']:+.3f}, {fp['y_max']:+.3f}] m")
        footprints.append(fp)

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
    for fp, slug in zip(footprints, prompts):
        ins = _test_inside_footprint(pos_metric, fp)
        _log(f"    [{slug}] splat inside footprint: {int(ins.sum()):,}")
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
                                                            footprints)
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

    mesh_chunks = [trimmed_mesh]
    n_patch_faces_total = 0
    for slug in prompts:
        npz = unseen_dir / f"unseen_core_{slug}.npz"
        if not npz.exists() or vs is None:
            _log(f"    [{slug}] no NPZ at {npz}; skipping mesh patch")
            continue
        pts_m, rgb01, shape_xz = _resample_patch_rgb(npz, scene_mesh_path, vs)
        patch_mesh = _build_mesh_patch(pts_m, shape_xz, rgb01)
        n_patch_faces_total += len(patch_mesh.faces)
        mesh_chunks.append(patch_mesh)
        _log(f"    [{slug}] mesh patch: {len(patch_mesh.faces):,} faces, "
             f"{len(patch_mesh.vertices):,} verts")

    if len(mesh_chunks) > 1:
        try:
            combined = trimesh.util.concatenate(mesh_chunks)
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
