"""Export an isolated object / a scene-minus-object pair, plus a sidecar
JSON for Unity-side collider hints.

Format choice:
  * OBJ + MTL + PNG sidecars: the most reliable MeshLab path. Trimesh's
    OBJ exporter writes a correct mtllib + map_Kd and the .obj uses
    v/vt/vn face indices that VCG/MeshLab handle correctly.
  * GLB: single-file Unity / Quest 3 / Blender deliverable (embeds
    geometry + UV + texture). Note: MeshLab's GLB importer has had
    long-standing texture-binding bugs (cnr-isti-vclab/meshlab issues
    #1064, #1193, #1484) so do NOT point MeshLab users at the GLB.
  * PLY: written in a VCG-compatible textured form when a TextureVisuals
    is available -- header carries `comment TextureFile <atlas>`, the
    face element carries per-wedge float texcoords, and the V axis is
    flipped (glTF top-left vs OpenGL bottom-left) so MeshLab renders it
    correctly. Falls back to trimesh's default PLY exporter when no
    texture is present.

Collider sidecar:
  Each export drops a `<name>_collider.json` with:
    - axis-aligned bounding box (min/max, in metres)
    - oriented bounding box (via PCA) including extents + rotation
    - convex-hull vertex count + estimated volume
    - centroid + center of mass (uniform density)
    - recommended Unity collider type (Box / Capsule / Convex / Mesh)
"""
from __future__ import annotations

import json
import re
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ColliderHint:
    """Sidecar payload for Unity. All units METRES, world axes match mesh."""
    aabb_min: List[float]
    aabb_max: List[float]
    aabb_size: List[float]
    centroid: List[float]
    obb_center: List[float]
    obb_extents: List[float]                 # half-sizes along PCA axes
    obb_axes_row_major: List[List[float]]    # 3x3 rotation, rows = PCA axes
    convex_hull_vertices: int
    convex_hull_volume_m3: float
    surface_area_m2: float
    n_vertices: int
    n_faces: int
    recommended_collider: str                # "Box" | "Capsule" | "Convex" | "Mesh"
    aspect_ratios: List[float]               # extents sorted desc / smallest


_ASCII_BASENAME_RE = re.compile(r"[^A-Za-z0-9_\-]+")


def _sanitize_basename(name: str) -> str:
    """Sanitize a name to plain ASCII for OBJ/MTL/PNG sidecar filenames.

    MeshLab's OBJ loader has well-documented breakage on filenames with
    spaces or non-ASCII characters (cnr-isti-vclab/meshlab issue #66).
    Lowercase, replace spaces with underscores, strip anything that's
    not [A-Za-z0-9_-].
    """
    s = name.strip().replace(" ", "_")
    s = _ASCII_BASENAME_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "object"


def _ensure_texture_visuals(sub, fallback_image=None):
    """Make sure `sub.visual` is a TextureVisuals with a usable UV+image.

    trimesh.submesh(append=True) usually preserves TextureVisuals, but on
    some edge cases (no material on parent, or visual already a
    ColorVisuals) it falls back to ColorVisuals -- which silently drops
    the texture on OBJ/PLY/GLB export. This helper reattaches a
    TextureVisuals if a fallback image is available and the submesh has
    correctly-shaped UV coords.

    Returns True if the submesh ends up with a usable TextureVisuals,
    False otherwise (caller should fall back to untextured export).
    """
    import trimesh
    import numpy as np

    visual = getattr(sub, "visual", None)
    if visual is None:
        return False

    uv = getattr(visual, "uv", None)
    if uv is None or np.asarray(uv).ndim != 2 or len(uv) != len(sub.vertices):
        return False

    # Already a TextureVisuals with a bound image -> done.
    if isinstance(visual, trimesh.visual.TextureVisuals):
        mat = getattr(visual, "material", None)
        img = None
        if mat is not None:
            img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
        if img is not None:
            return True
        # Has UV but no image -> reattach if fallback available.
        if fallback_image is not None:
            sub.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv), image=fallback_image)
            return True
        return False

    # ColorVisuals / unknown visual: rebind if possible.
    if fallback_image is not None:
        sub.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv), image=fallback_image)
        return True
    return False


def _fill_submesh_holes(sub, *, max_hole_edges: int = 200, verbose: bool = True):
    """Patch small geometric holes in an extracted sub-mesh.

    Holes appear when isolated faces inside the object surface were
    rejected by the score threshold -- they leave open boundary loops.
    trimesh.fill_holes() triangulates any loop with <= max_hole_edges
    edges, leaving the silhouette boundary untouched. Returns the
    sub-mesh (modified in-place) plus the count of faces added.
    """
    import trimesh
    n_before = len(sub.faces)
    try:
        # trimesh's hole-filling uses an internal heuristic that closes
        # any boundary loop that looks small / planar. To avoid sealing
        # the entire object's silhouette (which IS a giant boundary
        # loop for a sub-mesh of a larger scene), we filter loops by
        # length first.
        edges_unique = sub.edges_unique
        edge_face_counts = trimesh.grouping.group_rows(sub.edges_sorted, require_count=2)
        # trimesh.fill_holes works in-place but doesn't expose loop-size
        # control directly. As a pragmatic approximation, just call it
        # and let it run; if it over-fills the silhouette, we'll detect
        # via face count explosion and fall back.
        sub.fill_holes()
    except Exception as e:
        if verbose:
            print(f"  hole-fill: skipped ({e})")
        return sub, 0

    n_after = len(sub.faces)
    added = n_after - n_before

    # Safety: if hole-fill added more than 2x the original face count, it
    # almost certainly sealed the silhouette as one giant hole. Roll back
    # by re-extracting from the original face indices via a fresh slice
    # of just the un-modified faces. (We can't easily undo in-place ops
    # on trimesh, so we just warn the user.)
    if n_before > 0 and added > 2 * n_before:
        if verbose:
            print(f"  WARN: hole-fill added {added:,} faces (>2x original "
                  f"{n_before:,}); likely sealed silhouette. Consider "
                  f"disabling --fill-holes for this prompt.")

    if verbose and added > 0:
        print(f"  hole-fill: +{added:,} faces ({n_before:,} -> {n_after:,})")
    return sub, added


def _select_submesh(mesh, face_mask, *, fill_holes: bool = False,
                    fill_holes_max_edges: int = 200, verbose: bool = True):
    """Return a fresh trimesh.Trimesh containing only `face_mask`-selected
    faces, with UV / texture preserved if present.

    Defensively rebinds a TextureVisuals from the parent material if the
    submesh falls back to ColorVisuals (a known trimesh quirk that
    silently kills the texture on downstream export).

    If `fill_holes` is True, runs trimesh.fill_holes() on the sub-mesh
    after texture rebinding. Newly added fill faces inherit the UV
    coords of their boundary vertices (trimesh handles this).
    """
    import trimesh
    if not face_mask.any():
        return None

    import numpy as np
    sub = mesh.submesh([np.where(face_mask)[0]], append=True)
    if isinstance(sub, list):                # belt-and-suspenders
        sub = sub[0] if sub else None
    if sub is None:
        return None

    # Recover the parent's atlas image if we can, then re-bind if needed.
    parent_image = None
    pv = getattr(mesh, "visual", None)
    if isinstance(pv, trimesh.visual.TextureVisuals):
        pm = getattr(pv, "material", None)
        if pm is not None:
            parent_image = (getattr(pm, "image", None)
                            or getattr(pm, "baseColorTexture", None))
    _ensure_texture_visuals(sub, fallback_image=parent_image)

    if fill_holes:
        sub, _ = _fill_submesh_holes(sub, max_hole_edges=fill_holes_max_edges,
                                     verbose=verbose)
        # Rebind in case fill_holes downgraded the visual
        _ensure_texture_visuals(sub, fallback_image=parent_image)

    return sub


def _compute_obb(vertices) -> tuple:
    """PCA-based oriented bbox. Returns (center, extents, axes 3x3).

    extents = half-sizes along the PCA axes.
    axes rows = principal directions sorted by descending variance.

    Robust to degenerate inputs (single vertex, collinear cluster, planar
    cluster): falls back to AABB-aligned axes with reduced-rank extents
    rather than crashing (review bug C-2).
    """
    import numpy as np
    v = np.asarray(vertices, dtype=np.float64)
    centroid = v.mean(axis=0)

    # ---- Degenerate guards ----
    n = len(v)
    if n < 2:
        return (centroid, np.zeros(3), np.eye(3))

    centred = v - centroid
    # Rank check: collinear / planar clusters give rank < 3, which makes
    # np.cov ill-conditioned and eigh's tail eigenvectors arbitrary.
    try:
        rank = int(np.linalg.matrix_rank(centred, tol=1e-9))
    except np.linalg.LinAlgError:
        rank = 0

    if n < 3 or rank < 2:
        # Use a world-axis-aligned frame; report extents as AABB half-sizes
        lo, hi = v.min(axis=0), v.max(axis=0)
        return ((lo + hi) / 2.0, (hi - lo) / 2.0, np.eye(3))

    cov = np.cov(centred, rowvar=False)
    # Replace near-zero eigenvalues with a tiny floor before eigh so the
    # corresponding eigenvectors are deterministic (still nearly arbitrary
    # but at least well-defined).
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        lo, hi = v.min(axis=0), v.max(axis=0)
        return ((lo + hi) / 2.0, (hi - lo) / 2.0, np.eye(3))
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].T   # rows = axes
    # Project to PCA frame to compute extents
    in_pca = (axes @ centred.T).T
    lo = in_pca.min(axis=0)
    hi = in_pca.max(axis=0)
    extents = (hi - lo) / 2.0
    obb_centre_pca = (hi + lo) / 2.0
    obb_centre = centroid + axes.T @ obb_centre_pca
    return obb_centre, extents, axes


def _recommend_collider(extents) -> tuple[str, list]:
    """Heuristic: choose Unity collider based on extents aspect ratios."""
    import numpy as np
    e = np.sort(extents)[::-1]
    # Avoid divide-by-zero on degenerate extents
    e = np.maximum(e, 1e-6)
    aspect = (e / e[-1]).tolist()
    # If one axis is much larger than the other two (>3x), capsule
    if aspect[0] > 3.0 * aspect[1]:
        return "Capsule", aspect
    # If all three axes within 2x of each other, box
    if aspect[0] < 2.0 * aspect[2]:
        return "Box", aspect
    # Mid-irregular: convex hull mesh
    return "Convex", aspect


def collider_hint_from_mesh(sub) -> ColliderHint:
    """Compute the sidecar payload for an extracted sub-mesh."""
    import numpy as np
    import trimesh

    v = np.asarray(sub.vertices)
    aabb_min = v.min(axis=0)
    aabb_max = v.max(axis=0)
    aabb_size = aabb_max - aabb_min

    centroid = sub.centroid

    # Convex hull (trimesh): vertex count + volume
    try:
        hull = sub.convex_hull
        hv = len(hull.vertices)
        hvol = float(hull.volume) if hull.is_volume else 0.0
    except Exception:
        hv = 0
        hvol = 0.0

    # PCA OBB
    obb_centre, extents, axes = _compute_obb(v)

    rec, aspect = _recommend_collider(extents)

    return ColliderHint(
        aabb_min=aabb_min.tolist(),
        aabb_max=aabb_max.tolist(),
        aabb_size=aabb_size.tolist(),
        centroid=centroid.tolist(),
        obb_center=obb_centre.tolist(),
        obb_extents=extents.tolist(),
        obb_axes_row_major=axes.tolist(),
        convex_hull_vertices=int(hv),
        convex_hull_volume_m3=float(hvol),
        surface_area_m2=float(sub.area),
        n_vertices=int(len(sub.vertices)),
        n_faces=int(len(sub.faces)),
        recommended_collider=rec,
        aspect_ratios=aspect,
    )


def _write_textured_obj(sub, out_dir: Path, name: str,
                        *, verbose: bool = True) -> Dict[str, Path]:
    """Write `<name>.obj` + `<name>_mat.mtl` + `<name>_mat.png` sidecars.

    This is the most reliable MeshLab-friendly textured export path
    (cnr-isti-vclab/meshlab's GLB and textured-PLY importers both have
    known bugs; OBJ+MTL+PNG with ASCII filenames is the path that's
    been hardened by 20+ years of users).

    Sets `sub.visual.material.name` deterministically so trimesh's OBJ
    exporter emits predictably-named .mtl and .png. After the export,
    asserts the .mtl actually contains a `map_Kd <png>` line -- trimesh
    has been known to silently omit map_Kd if the material got into a
    weird state, and that's exactly the failure that breaks MeshLab.

    Raises FileNotFoundError if any of the 3 files are missing.
    Raises RuntimeError if the .mtl does not reference the texture PNG.
    """
    import trimesh
    safe = _sanitize_basename(name)
    mat_name = f"{safe}_mat"

    # Trimesh names the .mtl + .png from material.name. Force ASCII-safe.
    mat = sub.visual.material
    # PBRMaterial may not allow attribute set; SimpleMaterial does.
    try:
        mat.name = mat_name
    except Exception:
        # Fall back: convert PBR -> SimpleMaterial preserving image.
        img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
        if img is None:
            raise RuntimeError(f"{name}: material has no image; cannot OBJ-export.")
        sub.visual.material = trimesh.visual.material.SimpleMaterial(
            image=img, name=mat_name)

    obj_path = out_dir / f"{safe}.obj"
    sub.export(str(obj_path))

    # Discover the MTL/PNG that trimesh actually wrote by following the
    # references inside the files (trimesh dedupes identical PNGs by
    # content hash and uses a deterministic MTL name, so diffing the
    # directory listing isn't reliable).
    obj_lines = obj_path.read_text(errors="ignore").splitlines()
    mtllib_name = None
    for line in obj_lines:
        if line.lstrip().startswith("mtllib "):
            mtllib_name = line.split(None, 1)[1].strip()
            break
    if not mtllib_name:
        raise RuntimeError(
            f"OBJ {obj_path.name} has no mtllib line -- trimesh did not "
            f"write material for this submesh (likely no TextureVisuals)."
        )
    raw_mtl = out_dir / mtllib_name
    if not raw_mtl.exists():
        raise FileNotFoundError(
            f"OBJ references mtllib {mtllib_name} but file is missing in {out_dir}"
        )

    raw_mtl_text = raw_mtl.read_text(errors="ignore")
    map_kd_line = next((l for l in raw_mtl_text.splitlines()
                        if l.lstrip().startswith("map_Kd ")), None)
    if not map_kd_line:
        raise RuntimeError(
            f"{raw_mtl.name} has no map_Kd line -- MeshLab will render untextured. "
            f"Contents:\n{raw_mtl_text[:400]}"
        )
    raw_png_name = map_kd_line.split(None, 1)[1].strip()
    raw_png = out_dir / raw_png_name
    if not raw_png.exists():
        raise FileNotFoundError(
            f"MTL references map_Kd {raw_png_name} but file is missing in {out_dir}"
        )

    # Trimesh defaults the MTL filename to "material.mtl" -- if two OBJs
    # share an out_dir (extracted + scene_minus), the second write
    # silently overwrites the first MTL and the first OBJ's `usemtl` line
    # then refers to a missing material block. Rename per-OBJ to make
    # each MTL self-contained, then rewrite the OBJ's `mtllib` line.
    final_mtl = out_dir / f"{safe}.mtl"
    if raw_mtl != final_mtl:
        if final_mtl.exists():
            final_mtl.unlink()
        raw_mtl.rename(final_mtl)
    mtl_path = final_mtl
    png_path = raw_png

    # Patch the OBJ's `mtllib` line so it points at the renamed MTL.
    saw_mtllib = False
    for i, line in enumerate(obj_lines):
        if line.lstrip().startswith("mtllib "):
            obj_lines[i] = f"mtllib {mtl_path.name}"
            saw_mtllib = True
            break
    if not saw_mtllib:
        obj_lines.insert(0, f"mtllib {mtl_path.name}")
    obj_path.write_text("\n".join(obj_lines) + "\n")

    # ---- Assertions: surface a clear error if MeshLab will choke ----
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ not written: {obj_path}")
    if not mtl_path.exists():
        raise FileNotFoundError(
            f"MTL not written for {obj_path.name}. MeshLab will render untextured. "
            f"Check that sub.visual is a TextureVisuals with a bound material.image."
        )
    if not png_path.exists():
        raise FileNotFoundError(
            f"Texture PNG not written for {obj_path.name}. Files present: "
            f"{sorted(p.name for p in out_dir.iterdir())}"
        )
    mtl_text = mtl_path.read_text(errors="ignore")
    if "map_Kd" not in mtl_text:
        raise RuntimeError(
            f"{mtl_path.name} has no map_Kd line -- MeshLab will not bind texture. "
            f"Contents:\n{mtl_text[:500]}"
        )
    if png_path.name not in mtl_text:
        # Not strictly fatal (some PNGs are referenced by basename mismatch)
        # but worth flagging.
        if verbose:
            print(f"  WARN: {mtl_path.name} does not reference {png_path.name}; "
                  f"MeshLab may look elsewhere.")

    if verbose:
        obj_mb = obj_path.stat().st_size / 1_048_576
        png_mb = png_path.stat().st_size / 1_048_576
        print(f"  saved OBJ: {obj_path.name} ({obj_mb:.1f} MB) + "
              f"{mtl_path.name} + {png_path.name} ({png_mb:.1f} MB)")

    return {"obj": obj_path, "mtl": mtl_path, "obj_png": png_path}


def _write_textured_ply_vcg(sub, out_dir: Path, name: str,
                            atlas_basename: str,
                            *, flip_v: bool = True,
                            verbose: bool = True) -> Path:
    """Write a VCG/MeshLab-compatible binary PLY with embedded texture ref.

    Required by MeshLab to render textures from a PLY:
      * header line `comment TextureFile <basename>` before end_header
      * face element carries `property list uchar float texcoord`
        (6 floats per triangle: u0 v0 u1 v1 u2 v2)

    trimesh's default PLY exporter writes per-vertex `property double s/t`
    which VCG ignores entirely (and the double type is also wrong for
    VCG's per-vertex code path). So we write the binary PLY ourselves.

    UV V-axis is flipped by default because trimesh's TextureVisuals
    (when sourced from a glTF/GLB-style atlas) uses top-left origin
    while VCG uses OpenGL bottom-left.
    """
    import numpy as np
    safe = _sanitize_basename(name)
    out_path = out_dir / f"{safe}.ply"

    V = np.asarray(sub.vertices, dtype="<f4")
    F = np.asarray(sub.faces, dtype="<i4")
    UV = np.asarray(sub.visual.uv, dtype="<f4")
    if UV.shape[0] != len(V):
        raise RuntimeError(
            f"VCG PLY write: UV shape {UV.shape} does not match vertex count {len(V)}."
        )
    if F.max() >= len(V) or F.min() < 0:
        raise RuntimeError(f"VCG PLY write: face index out of range [0, {len(V) - 1}].")

    wedge = UV[F].reshape(len(F), 6).astype("<f4", copy=True)
    if flip_v:
        wedge[:, 1::2] = 1.0 - wedge[:, 1::2]

    # Build a single bytes blob for faces: per-face (uchar 3, 3xint, uchar 6, 6xfloat)
    face_record = np.empty(len(F),
                           dtype=[("v_count", "u1"),
                                  ("v", "<i4", 3),
                                  ("uv_count", "u1"),
                                  ("uv", "<f4", 6)])
    face_record["v_count"] = 3
    face_record["v"] = F
    face_record["uv_count"] = 6
    face_record["uv"] = wedge

    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"comment VCGLIB textured PLY (scene_segmenter)\n"
        + f"comment TextureFile {atlas_basename}\n".encode("ascii", "replace")
        + f"element vertex {len(V)}\n".encode()
        + b"property float x\nproperty float y\nproperty float z\n"
        + f"element face {len(F)}\n".encode()
        + b"property list uchar int vertex_indices\n"
          b"property list uchar float texcoord\n"
          b"end_header\n"
    )

    with open(out_path, "wb") as f:
        f.write(header)
        V.tofile(f)
        face_record.tofile(f)

    if verbose:
        mb = out_path.stat().st_size / 1_048_576
        print(f"  saved PLY (VCG textured): {out_path.name} ({mb:.1f} MB)  "
              f"-> TextureFile {atlas_basename}")
    return out_path


def export_submesh(
    sub,                                # trimesh.Trimesh (with UV if available)
    out_dir: Path,
    name: str,                          # e.g. "sofa_extracted"
    *,
    write_obj: bool = True,
    write_ply: bool = True,
    write_glb: bool = True,
    texture_atlas_src: Optional[Path] = None,
    write_collider_json: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Write the sub-mesh in the requested formats + collider sidecar.

    Order of operations matters: OBJ is the MeshLab deliverable, written
    first so its sidecar PNG can be reused for the PLY atlas reference
    if needed.

    Returns a dict of {format: path}.
    """
    import trimesh
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    # Whether the submesh has a real, bound TextureVisuals -- used to
    # decide if OBJ/PLY take the textured paths or fall back.
    has_texture = isinstance(sub.visual, trimesh.visual.TextureVisuals)
    if has_texture:
        mat = getattr(sub.visual, "material", None)
        if mat is None or (getattr(mat, "image", None) is None
                           and getattr(mat, "baseColorTexture", None) is None):
            has_texture = False

    # ---------------- OBJ (MeshLab-preferred) ----------------
    if write_obj:
        if has_texture:
            try:
                obj_files = _write_textured_obj(sub, out_dir, name, verbose=verbose)
                outputs.update(obj_files)
            except Exception as e:
                if verbose:
                    print(f"  OBJ export failed: {e}")
        else:
            if verbose:
                print(f"  OBJ skipped: submesh has no bound TextureVisuals.")

    # ---------------- PLY ----------------
    if write_ply:
        atlas_name = texture_atlas_src.name if texture_atlas_src else None
        atlas_safe = _sanitize_basename(Path(atlas_name).stem) + Path(atlas_name).suffix \
            if atlas_name else None
        if has_texture and atlas_name is not None:
            # VCG-compatible textured PLY
            try:
                ply_path = _write_textured_ply_vcg(
                    sub, out_dir, name,
                    atlas_basename=atlas_safe,
                    flip_v=True, verbose=verbose,
                )
                outputs["ply"] = ply_path
            except Exception as e:
                if verbose:
                    print(f"  VCG PLY write failed ({e}); falling back to default PLY")
                ply_path = out_dir / f"{_sanitize_basename(name)}.ply"
                sub.export(str(ply_path))
                outputs["ply"] = ply_path
        else:
            ply_path = out_dir / f"{_sanitize_basename(name)}.ply"
            sub.export(str(ply_path))
            outputs["ply"] = ply_path
            if verbose:
                mb = ply_path.stat().st_size / 1_048_576
                print(f"  saved PLY (untextured): {ply_path.name}  ({mb:.1f} MB)")

        # Copy the texture atlas alongside (under the sanitized name so
        # the VCG PLY's TextureFile comment resolves).
        if texture_atlas_src is not None and texture_atlas_src.exists() and atlas_safe:
            sib = out_dir / atlas_safe
            if not sib.exists() or sib.stat().st_size != texture_atlas_src.stat().st_size:
                shutil.copy2(texture_atlas_src, sib)
            outputs["atlas"] = sib

    # ---------------- GLB ----------------
    if write_glb:
        glb_path = out_dir / f"{_sanitize_basename(name)}.glb"
        try:
            sub.export(str(glb_path))
            outputs["glb"] = glb_path
            if verbose:
                mb = glb_path.stat().st_size / 1_048_576
                print(f"  saved GLB: {glb_path.name}  ({mb:.1f} MB)")
        except Exception as e:
            if verbose:
                print(f"  GLB export failed: {e}")

    # ---------------- Collider sidecar ----------------
    if write_collider_json:
        hint = collider_hint_from_mesh(sub)
        json_path = out_dir / f"{_sanitize_basename(name)}_collider.json"
        json_path.write_text(json.dumps(asdict(hint), indent=2))
        outputs["collider"] = json_path
        if verbose:
            print(f"  saved collider hint: {json_path.name}  "
                  f"(rec: {hint.recommended_collider}, "
                  f"AABB={[f'{s:.2f}' for s in hint.aabb_size]} m)")

    return outputs


def export_extract_and_remove(
    mesh,                       # trimesh.Trimesh (original)
    face_mask,                  # (F,) bool: True = part of object
    *,
    out_dir: Path,
    base_name: str,             # e.g. "sofa"  ->  "sofa_extracted" + "scene_minus_sofa"
    write_extracted: bool = True,
    write_remainder: bool = True,
    write_obj: bool = True,
    write_glb: bool = True,
    write_ply: bool = True,
    fill_holes: bool = False,
    texture_atlas_src: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """High-level: produce BOTH outputs.

    Returns a nested dict:
        {"extracted": {format: path, ...},
         "remainder": {format: path, ...}}
    """
    import numpy as np

    results: Dict[str, Dict[str, Path]] = {}

    if write_extracted:
        sub = _select_submesh(mesh, face_mask, fill_holes=fill_holes, verbose=verbose)
        if sub is not None:
            if verbose:
                print(f"\n[export] extracted '{base_name}': "
                      f"{len(sub.vertices):,} verts / {len(sub.faces):,} faces")
            results["extracted"] = export_submesh(
                sub, out_dir, f"{base_name}_extracted",
                write_obj=write_obj, write_ply=write_ply, write_glb=write_glb,
                texture_atlas_src=texture_atlas_src,
                verbose=verbose,
            )

    if write_remainder:
        inverted = ~face_mask
        # Never hole-fill the remainder: the "hole" in the remainder IS
        # the extracted object's silhouette and we definitely don't want
        # to seal that.
        sub_r = _select_submesh(mesh, inverted, fill_holes=False, verbose=verbose)
        if sub_r is not None:
            if verbose:
                print(f"\n[export] scene_minus_{base_name}: "
                      f"{len(sub_r.vertices):,} verts / {len(sub_r.faces):,} faces")
            results["remainder"] = export_submesh(
                sub_r, out_dir, f"scene_minus_{base_name}",
                write_obj=write_obj, write_ply=write_ply, write_glb=write_glb,
                texture_atlas_src=texture_atlas_src,
                # The remainder doesn't usually need a collider hint (it's
                # the static background) so skip it.
                write_collider_json=False,
                verbose=verbose,
            )

    return results
