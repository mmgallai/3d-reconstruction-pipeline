"""Phase: trim + inpaint -- produces the apples-to-apples test artifacts
for both the SPLAT side and the MESH side of the pipeline.

Layout produced:
    output/phase_trim_inpaint_<scene>/
    ├── splat/
    │   ├── scene_full.ply
    │   └── per_object/<slug>/
    │       ├── object_only.ply                   (the trimmed object splat)
    │       ├── desk_patch_only.ply               (the NN-copy patch alone)
    │       └── scene_minus_object_patched.ply   (scene with object gone + patch)
    └── mesh/
        ├── scene_full.ply
        └── per_object/<slug>/
            ├── object_only.ply                   (the trimmed object mesh)
            ├── desk_patch_only.ply               (planar mesh patch alone)
            └── scene_minus_object_patched.ply   (scene mesh with object faces gone + patch)

The MESH side mirrors the SPLAT side stage-for-stage so a side-by-side
visual A/B is unambiguous: identical XZ footprint, identical desk plane,
identical RGB sampling, identical NN-copy.

Inputs (V32 defaults):
    - scene splat:   output/splat_v32_data3_noinit_pruned.ply
    - scene mesh:    output/mesh_v32_data3/mesh_v32_data3_openmvs.ply
    - object splats: output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned/<slug>_splat.ply
    - object meshes: output/segmented_v32_data3_v9a_fp_v2/<slug>/<slug>_extracted.ply
    - desk patches:  output/desk_patch_v32_data3/desk_patch_<slug>.ply
    - unseen NPZ:    output/unseen_core_v32_data3/unseen_core_<slug>.npz
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).parent))
from scene_segmenter.views import V32ViewSource  # noqa: E402
from _spatial_crop_splat import (              # noqa: E402
    _load_dataparser_transform,
    _splat_to_metric,
    _build_footprint_from_extracted_mesh,
    _test_inside_footprint,
)
from _seed_desk_patch import (                  # noqa: E402
    _best_view_per_point,
    _sample_color_for_points,
    _nn_fill_missing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_ply_vertex(p: Path):
    pd = PlyData.read(str(p))
    v = pd["vertex"]
    fields = [x.name for x in v.properties]
    return v.data, fields


def _slug_dir(out_root: Path, side: str, slug: str) -> Path:
    d = out_root / side / "per_object" / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def _copy_ply(src: Path, dst: Path):
    if not src.exists():
        print(f"  WARN missing source {src}")
        return False
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return True


# ---------------------------------------------------------------------------
# SPLAT side
# ---------------------------------------------------------------------------

def _splat_scene_minus_object(scene_v, scene_fields, scene_metric: np.ndarray,
                              fp: dict, patch_ply: Path | None,
                              out_ply: Path) -> dict:
    inside = _test_inside_footprint(scene_metric, fp)
    kept = ~inside
    kept_rows = scene_v[kept]
    if patch_ply is not None and patch_ply.exists():
        patch_v, patch_fields = _read_ply_vertex(patch_ply)
        if patch_fields != scene_fields:
            print(f"  WARN patch/scene field mismatch")
        merged = np.concatenate([kept_rows, patch_v])
        n_patch = len(patch_v)
    else:
        merged = kept_rows
        n_patch = 0
    el = PlyElement.describe(merged, "vertex")
    PlyData([el]).write(str(out_ply))
    return {
        "n_kept_from_scene": int(kept.sum()),
        "n_removed_from_scene": int(inside.sum()),
        "n_patch": n_patch,
        "n_total_out": int(len(merged)),
        "size_mb": out_ply.stat().st_size / 1_048_576,
    }


# ---------------------------------------------------------------------------
# MESH side
# ---------------------------------------------------------------------------

def _mesh_face_centroids(mesh) -> np.ndarray:
    """(n_faces, 3) metric centroid per face."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = verts[faces]            # (n_faces, 3, 3)
    return tri.mean(axis=1)


def _trim_mesh_by_footprint(mesh, fp: dict) -> tuple:
    """Drop faces whose centroid lies inside the XZ footprint AND Y range.

    Returns (trimmed_mesh, n_removed_faces).
    """
    import trimesh
    centroids = _mesh_face_centroids(mesh)
    inside = _test_inside_footprint(centroids, fp)
    keep = ~inside
    faces = np.asarray(mesh.faces, dtype=np.int64)
    kept_faces = faces[keep]
    # Drop unused vertices to keep file size sane
    used = np.unique(kept_faces.flatten())
    remap = -np.ones(len(mesh.vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    new_verts = np.asarray(mesh.vertices, dtype=np.float64)[used]
    new_faces = remap[kept_faces]
    # Preserve vertex colors if present
    visual = getattr(mesh, "visual", None)
    new_colors = None
    if visual is not None and hasattr(visual, "vertex_colors"):
        vc = np.asarray(visual.vertex_colors)
        if vc.shape[0] == len(mesh.vertices):
            new_colors = vc[used]
    if new_colors is not None:
        trimmed = trimesh.Trimesh(vertices=new_verts, faces=new_faces,
                                  vertex_colors=new_colors, process=False)
    else:
        trimmed = trimesh.Trimesh(vertices=new_verts, faces=new_faces,
                                  process=False)
    return trimmed, int(inside.sum())


def _build_mesh_patch(grid_points_metric: np.ndarray, grid_shape: tuple,
                      rgb01: np.ndarray) -> "trimesh.Trimesh":
    """Triangulated planar grid on the desk surface, with per-vertex RGB.

    `grid_points_metric` shape (N, 3); each point becomes a vertex.
    `grid_shape` = (nx, nz). `rgb01` shape (N, 3) in [0, 1].

    Triangulation: each (i, j)-(i+1, j+1) cell -> 2 tris.
        v_a = (i,   j  )
        v_b = (i+1, j  )
        v_c = (i,   j+1)
        v_d = (i+1, j+1)
        tri1 = (v_a, v_b, v_d)
        tri2 = (v_a, v_d, v_c)
    """
    import trimesh
    nx, nz = grid_shape
    verts = grid_points_metric.astype(np.float64)
    # build face list
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
    # trimesh wants RGBA
    if colors.shape[1] == 3:
        rgba = np.concatenate([colors, np.full((len(colors), 1), 255, dtype=np.uint8)], axis=-1)
    else:
        rgba = colors
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba,
                           process=False)


def _save_mesh(mesh, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use PLY (per-vertex colors). For larger interop later we can also
    # export OBJ+MTL+PNG but PLY is the apples-to-apples comparison format
    # for the .ply splats on the other side.
    mesh.export(str(out_path))


def _resample_patch_rgb(slug: str, npz_path: Path,
                        view_source, scene_mesh_path: Path
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive the same (grid_points_metric, rgb01) the splat patch used.

    Lets the mesh patch use identical colors so a splat-vs-mesh A/B isolates
    representation, not color sampling.
    """
    data = np.load(npz_path)
    pts_m = data["grid_points"].astype(np.float32)
    seen = data["seen_count"]
    print(f"  resampling RGB for {len(pts_m)} grid points "
          f"({int((seen==0).sum())} truly-unseen) ...")
    best_view, best_uv = _best_view_per_point(
        view_source, scene_mesh_path, pts_m, seen)
    rgb = _sample_color_for_points(view_source, best_view, best_uv)
    rgb = _nn_fill_missing(pts_m.astype(np.float64), rgb)
    shape_xz = tuple(int(x) for x in data["shape_xz"])
    return pts_m.astype(np.float64), rgb.astype(np.float32), shape_xz


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--scene-name", default="v32_data3")
    ap.add_argument("--scene-splat", type=Path,
                    default=Path("output/splat_v32_data3_noinit_pruned.ply"))
    ap.add_argument("--scene-mesh", type=Path,
                    default=Path("output/mesh_v32_data3/mesh_v32_data3_openmvs.ply"))
    ap.add_argument("--object-splat-dir", type=Path,
                    default=Path("output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned"))
    ap.add_argument("--object-mesh-dir", type=Path,
                    default=Path("output/segmented_v32_data3_v9a_fp_v2"))
    ap.add_argument("--patch-splat-dir", type=Path,
                    default=Path("output/desk_patch_v32_data3"))
    ap.add_argument("--unseen-dir", type=Path,
                    default=Path("output/unseen_core_v32_data3"))
    ap.add_argument("--dataparser", type=Path,
                    default=Path("nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json"))
    ap.add_argument("--bounds-json", type=Path,
                    default=Path("colmap/dense/tof_bounds.json"))
    ap.add_argument("--prompts",
                    default="white_water_bottle,blue_box,red_lobster_figurine")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="default: output/phase_trim_inpaint_<scene>")
    ap.add_argument("--xz-dilate-cm", type=float, default=0.5)
    ap.add_argument("--y-margin-cm", type=float, default=0.5)
    args = ap.parse_args()

    out_root = args.out_root or Path(f"output/phase_trim_inpaint_{args.scene_name}")
    (out_root / "splat").mkdir(parents=True, exist_ok=True)
    (out_root / "mesh").mkdir(parents=True, exist_ok=True)

    print(f"[init] writing into {out_root}")

    # === SCENE FULL: copy both sides ===
    print("\n=== scene_full ===")
    splat_full = out_root / "splat" / "scene_full.ply"
    mesh_full = out_root / "mesh" / "scene_full.ply"
    if _copy_ply(args.scene_splat, splat_full):
        print(f"  splat full -> {splat_full} ({splat_full.stat().st_size/1_048_576:.1f} MB)")
    if _copy_ply(args.scene_mesh, mesh_full):
        print(f"  mesh full  -> {mesh_full} ({mesh_full.stat().st_size/1_048_576:.1f} MB)")

    # === Load scene splat once for the splat-side trim/patch ===
    scene_v, scene_fields = _read_ply_vertex(args.scene_splat)
    bounds = json.loads(args.bounds_json.read_text())
    colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])
    dp_R, dp_t, dp_scale = _load_dataparser_transform(args.dataparser)
    splat_positions = np.stack([scene_v["x"], scene_v["y"], scene_v["z"]],
                               axis=-1).astype(np.float64)
    scene_metric = _splat_to_metric(splat_positions, dp_R, dp_t, dp_scale,
                                    colmap_to_metric)

    # === Load scene mesh once for the mesh-side trim ===
    # Bake per-vertex colors from the V32 atlas now, so trimmed outputs are
    # self-contained and don't depend on `scene_textured0.png` sidecars.
    import trimesh
    print("\n[init] loading scene mesh for mesh-side trim ...")
    scene_mesh = trimesh.load(str(args.scene_mesh), force="mesh", process=False)
    print(f"  {len(scene_mesh.vertices):,} verts, {len(scene_mesh.faces):,} faces")

    # Sample the atlas at every vertex's UV to get per-vertex RGBA.
    # `scene_mesh.visual.to_color()` returns a ColorVisuals with vertex_colors
    # if the mesh has UV+texture; falls through to white if it doesn't.
    try:
        if hasattr(scene_mesh.visual, "to_color"):
            color_visual = scene_mesh.visual.to_color()
            vc = np.asarray(color_visual.vertex_colors)
        elif hasattr(scene_mesh.visual, "vertex_colors"):
            vc = np.asarray(scene_mesh.visual.vertex_colors)
        else:
            vc = None
        if vc is not None and vc.shape[0] == len(scene_mesh.vertices):
            scene_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=scene_mesh, vertex_colors=vc)
            print(f"  baked per-vertex colors from atlas "
                  f"(mean RGB = {vc[:, :3].mean(axis=0).round(0).astype(int)})")
        else:
            print(f"  WARN: atlas->vertex-color bake produced unexpected shape "
                  f"{None if vc is None else vc.shape}; trimmed PLYs may be uncolored")
    except Exception as e:
        print(f"  WARN: atlas->vertex-color bake failed ({e}); "
              f"trimmed PLYs may be uncolored")

    # === Load view source once for mesh-patch color resampling ===
    vs = V32ViewSource(args.project_root)

    # === Per object ===
    summary = []
    for slug in args.prompts.split(","):
        slug = slug.strip()
        print(f"\n=== {slug} ===")
        # Footprint from the extracted mesh (same one v9a_fp_v2 produced)
        obj_mesh_ply = args.object_mesh_dir / slug / f"{slug}_extracted.ply"
        if not obj_mesh_ply.exists():
            print(f"  WARN missing extracted mesh: {obj_mesh_ply}")
            continue
        fp = _build_footprint_from_extracted_mesh(
            obj_mesh_ply, xz_resolution=0.005,
            xz_dilate_cm=args.xz_dilate_cm,
            y_margin_cm=args.y_margin_cm)

        splat_dir = _slug_dir(out_root, "splat", slug)
        mesh_dir = _slug_dir(out_root, "mesh", slug)
        rec = {"object": slug}

        # ----- SPLAT SIDE -----
        # object_only: v6_cleaned per-object splat (the trimmed object)
        obj_splat_src = args.object_splat_dir / f"{slug}_splat.ply"
        obj_splat_dst = splat_dir / "object_only.ply"
        if _copy_ply(obj_splat_src, obj_splat_dst):
            v_obj, _ = _read_ply_vertex(obj_splat_dst)
            rec["splat_object_n"] = int(len(v_obj))
            print(f"  splat object_only  : {len(v_obj):>7,d} gaussians")

        # desk_patch_only
        patch_src = args.patch_splat_dir / f"desk_patch_{slug}.ply"
        patch_dst = splat_dir / "desk_patch_only.ply"
        if _copy_ply(patch_src, patch_dst):
            v_patch, _ = _read_ply_vertex(patch_dst)
            rec["splat_patch_n"] = int(len(v_patch))
            print(f"  splat patch_only   : {len(v_patch):>7,d} gaussians")

        # scene_minus_object_patched
        scene_patched_dst = splat_dir / "scene_minus_object_patched.ply"
        s = _splat_scene_minus_object(
            scene_v, scene_fields, scene_metric, fp,
            patch_dst if patch_dst.exists() else None,
            scene_patched_dst)
        rec["splat_scene_patched_n"] = s["n_total_out"]
        rec["splat_scene_patched_removed"] = s["n_removed_from_scene"]
        print(f"  splat scene_patched: {s['n_total_out']:>7,d} gaussians "
              f"({s['size_mb']:.1f} MB, "
              f"removed {s['n_removed_from_scene']:,}, "
              f"patched {s['n_patch']:,})")

        # ----- MESH SIDE -----
        # object_only mesh: v9a_fp_v2 extracted (PLY + textured-PLY sidecars +
        # the OBJ/MTL/PNG triplet, all in one tidy per-object subdir)
        obj_mesh_dst = mesh_dir / "object_only.ply"
        if _copy_ply(obj_mesh_ply, obj_mesh_dst):
            # also copy any sibling texture + OBJ/MTL so the PLY/OBJ load
            # textured in MeshLab / Unity
            for sibling in obj_mesh_ply.parent.iterdir():
                if sibling.suffix.lower() in (".png", ".mtl", ".obj",
                                              ".glb", ".jpg", ".jpeg") \
                        and sibling.is_file():
                    dst = mesh_dir / sibling.name
                    if not dst.exists():
                        shutil.copy2(sibling, dst)
            mm = trimesh.load(str(obj_mesh_dst), force="mesh", process=False)
            rec["mesh_object_faces"] = int(len(mm.faces))
            print(f"  mesh object_only   : {len(mm.faces):>7,d} faces, "
                  f"{len(mm.vertices):,} verts")

        # mesh patch: triangulated planar grid + per-vertex RGB
        npz = args.unseen_dir / f"unseen_core_{slug}.npz"
        if npz.exists():
            pts_m, rgb01, shape_xz = _resample_patch_rgb(
                slug, npz, vs, args.scene_mesh)
            mesh_patch = _build_mesh_patch(pts_m, shape_xz, rgb01)
            patch_mesh_dst = mesh_dir / "desk_patch_only.ply"
            _save_mesh(mesh_patch, patch_mesh_dst)
            rec["mesh_patch_faces"] = int(len(mesh_patch.faces))
            print(f"  mesh patch_only    : {len(mesh_patch.faces):>7,d} faces, "
                  f"{len(mesh_patch.vertices):,} verts "
                  f"({patch_mesh_dst.stat().st_size/1024:.0f} KB)")
        else:
            mesh_patch = None
            print(f"  WARN no NPZ at {npz}; skipping mesh patch")

        # scene minus object + mesh patch
        trimmed_mesh, n_removed = _trim_mesh_by_footprint(scene_mesh, fp)
        print(f"  mesh trim          : removed {n_removed:>7,d} faces from scene")
        if mesh_patch is not None:
            try:
                combined = trimesh.util.concatenate([trimmed_mesh, mesh_patch])
            except Exception as e:
                print(f"  WARN combine fail: {e}; saving trimmed only")
                combined = trimmed_mesh
        else:
            combined = trimmed_mesh
        scene_patched_mesh_dst = mesh_dir / "scene_minus_object_patched.ply"
        _save_mesh(combined, scene_patched_mesh_dst)
        rec["mesh_scene_patched_faces"] = int(len(combined.faces))
        rec["mesh_scene_patched_removed"] = n_removed
        print(f"  mesh scene_patched : {len(combined.faces):>7,d} faces "
              f"({scene_patched_mesh_dst.stat().st_size/1_048_576:.1f} MB)")

        summary.append(rec)

    # Summary file
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_root / 'summary.json'}")
    print(f"\nLAYOUT: {out_root}")
    for side in ("splat", "mesh"):
        print(f"  {side}/")
        print(f"    scene_full.ply")
        for r in summary:
            print(f"    per_object/{r['object']}/")
            print(f"      object_only.ply")
            print(f"      desk_patch_only.ply")
            print(f"      scene_minus_object_patched.ply")


if __name__ == "__main__":
    main()
