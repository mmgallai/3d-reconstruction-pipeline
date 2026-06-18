"""Spatial cropping: pick mesh triangles by their 3D centroid being inside
a metric AABB, not by 2D mask voting. Eliminates swiss-cheese holes
entirely (cropping by location has no "maybe this face" mode).

Mode A (--mode a):
  Run rasterized SAM3 voting -> faces with score >= 0.7 are SEEDS.
  Compute the seeds' AABB, expand by --margin %, then crop ALL faces
  in the original mesh whose centroid is inside the box.

Mode B (--mode b):
  Read a prior run's <prompt>_extracted_collider.json (which already
  has aabb_min/aabb_max), expand by --margin %, crop ALL faces inside.
  Skips voting entirely.

Both modes write to output/segmented_v32_data3_v9{a,b}/<prompt>/ in
the same OBJ + MTL + PNG + PLY + GLB layout as v7/v8.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from scene_segmenter.views import V32ViewSource
from scene_segmenter.extract_faces import accumulate_face_scores_rasterized
from scene_segmenter.sam3_segment import _slugify
from scene_segmenter.exporter import export_extract_and_remove


MESH_PATH = Path("output/mesh_v32_data3/mesh_v32_data3_openmvs.ply")
ATLAS_PATH = Path("output/mesh_v32_data3/scene_textured0.png")
PROMPTS = ["white water bottle", "blue box", "red lobster figurine"]


def _bind_atlas(mesh, atlas_path):
    img = Image.open(atlas_path).convert("RGB")
    uv = getattr(mesh.visual, "uv", None)
    if uv is not None and len(uv) == len(mesh.vertices):
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=img)
    return mesh


def _expand(aabb_min, aabb_max, margin):
    size = aabb_max - aabb_min
    return aabb_min - size * margin, aabb_max + size * margin


def _spatial_mask(mesh, aabb_min, aabb_max):
    """Per-face: True if centroid is inside the AABB."""
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    return ((centroids >= aabb_min) & (centroids <= aabb_max)).all(axis=1)


# ---- Footprint-based cropping (replaces rectangular AABB) ------------------
#
# AABB cropping pulls in too much desk for objects whose top-down outline is
# not a rectangle (the lobster's spread claws are the worst offender; the
# bottle's round base is moderate). The fix: build a 2D occupancy mask on
# the horizontal X-Z plane from the seed faces' centroids -- a top-down
# "shadow" of the object -- and crop by that shadow + a vertical Y range.
# Desk pixels outside the shadow are NOT in the crop.

def _build_xz_footprint(seed_xz: np.ndarray,
                        resolution: float = 0.005,
                        dilate_cells: int = 2):
    """Build a 2D X-Z occupancy mask from a set of (x, z) points.

    resolution    : metres per grid cell (5 mm default)
    dilate_cells  : grow the mask by this many cells using PIL.MaxFilter
                    (2 cells * 5 mm = 1 cm safety margin around seeds)

    Returns (mask, xz_min, shape, resolution) for later membership tests.
    """
    from PIL import Image as _PILImage, ImageFilter as _IF
    margin = resolution * (dilate_cells + 1)
    xz_min = seed_xz.min(axis=0) - margin
    xz_max = seed_xz.max(axis=0) + margin
    shape = np.ceil((xz_max - xz_min) / resolution).astype(int) + 1
    grid = np.zeros((int(shape[0]), int(shape[1])), dtype=np.uint8)
    cells = ((seed_xz - xz_min) / resolution).astype(int)
    cells = np.clip(cells, 0, shape - 1)
    grid[cells[:, 0], cells[:, 1]] = 255
    if dilate_cells > 0:
        pil = _PILImage.fromarray(grid)
        pil = pil.filter(_IF.MaxFilter(2 * dilate_cells + 1))
        grid = np.array(pil)
    mask = grid > 128
    return mask, xz_min, shape, resolution


def _xz_in_footprint(centroids_xz: np.ndarray,
                     mask: np.ndarray, xz_min: np.ndarray,
                     shape: np.ndarray, resolution: float) -> np.ndarray:
    """Per-face: True if (x, z) lands in a True cell of the footprint mask."""
    cells = ((centroids_xz - xz_min) / resolution).astype(int)
    in_bounds = ((cells >= 0) & (cells < shape)).all(axis=1)
    cells_safe = np.clip(cells, 0, shape - 1)
    inside = mask[cells_safe[:, 0], cells_safe[:, 1]]
    return inside & in_bounds


def _largest_seed_component(mesh, seed_face_idx: np.ndarray,
                             voxel_size_m: float = 0.01) -> np.ndarray:
    """Keep only seeds in the largest SPATIAL cluster.

    We voxelize the seed centroids into a 3D grid at `voxel_size_m`
    resolution and find connected components in that grid (26-connectivity).
    Seeds in the largest connected voxel-component are kept; the rest are
    presumed SAM3 false positives sitting elsewhere in the scene.

    Why spatial (voxel) and not mesh-edge adjacency: OpenMVS meshes
    routinely have non-manifold edges / T-junctions that break the
    face-adjacency graph -- a contiguous patch of the object can come
    out as dozens of disconnected components. Voxel adjacency in
    metric world space sidesteps that topology mess.
    """
    if len(seed_face_idx) < 2:
        return seed_face_idx
    from scipy.ndimage import label as _scipy_label

    centroids = mesh.vertices[mesh.faces[seed_face_idx]].mean(axis=1)
    margin = voxel_size_m
    vox_min = centroids.min(axis=0) - margin
    vox_max = centroids.max(axis=0) + margin
    shape = np.ceil((vox_max - vox_min) / voxel_size_m).astype(int) + 1
    grid = np.zeros(tuple(int(s) for s in shape), dtype=bool)
    cells = ((centroids - vox_min) / voxel_size_m).astype(int)
    cells = np.clip(cells, 0, shape - 1)
    grid[cells[:, 0], cells[:, 1], cells[:, 2]] = True

    # 26-connectivity (all 3x3x3 neighbors)
    struct = np.ones((3, 3, 3), dtype=bool)
    labels, n = _scipy_label(grid, structure=struct)
    if n == 0:
        return seed_face_idx

    sizes = np.bincount(labels.ravel())[1:]  # skip background label 0
    largest_label = int(np.argmax(sizes)) + 1
    seed_labels = labels[cells[:, 0], cells[:, 1], cells[:, 2]]
    keep = seed_labels == largest_label
    return seed_face_idx[keep]


def _footprint_crop(mesh, seed_face_idx: np.ndarray,
                    *, xz_dilate_cells: int = 2,
                    xz_resolution: float = 0.005,
                    y_margin: float = 0.01,
                    cc_filter_seeds: bool = True):
    """Footprint-based spatial crop.

    seed_face_idx : (N,) face indices considered confident object hits.
    xz_dilate_cells : safety margin in cells (default 2 = 1 cm at 5 mm res)
    y_margin : metres of expansion above/below the seeds' Y extent

    Returns (face_mask, info_dict).
    """
    n_seed_input = len(seed_face_idx)
    if cc_filter_seeds:
        seed_face_idx = _largest_seed_component(mesh, seed_face_idx)
    n_seed_cleaned = len(seed_face_idx)

    all_centroids = mesh.vertices[mesh.faces].mean(axis=1)
    seed_centroids = all_centroids[seed_face_idx]

    y_min = float(seed_centroids[:, 1].min()) - y_margin
    y_max = float(seed_centroids[:, 1].max()) + y_margin

    fp_mask, xz_min, shape, res = _build_xz_footprint(
        seed_centroids[:, [0, 2]],
        resolution=xz_resolution,
        dilate_cells=xz_dilate_cells,
    )

    in_xz = _xz_in_footprint(all_centroids[:, [0, 2]],
                             fp_mask, xz_min, shape, res)
    in_y = (all_centroids[:, 1] >= y_min) & (all_centroids[:, 1] <= y_max)
    face_mask = in_xz & in_y

    fp_area_cm2 = float(fp_mask.sum()) * (res * 100.0) ** 2
    aabb_area_cm2 = float(shape[0] * shape[1]) * (res * 100.0) ** 2
    info = {
        "y_range_m": (y_min, y_max),
        "y_span_m": y_max - y_min,
        "footprint_area_cm2": fp_area_cm2,
        "aabb_footprint_cm2": aabb_area_cm2,
        "footprint_fraction": fp_area_cm2 / aabb_area_cm2,
        "n_seed_input": n_seed_input,
        "n_seed_cleaned": n_seed_cleaned,
    }
    return face_mask, info


def _load_masks(mask_dir, view_names, prompt_slug):
    out = {}
    for vn in view_names:
        mp = mask_dir / f"{vn}__{prompt_slug}.png"
        if mp.exists():
            m = np.array(Image.open(mp).convert("L")) > 128
            out[vn] = m
    return out


def run_mode_a(output_dir, mask_dir, margin, seed_threshold,
               write_remainder, *,
               crop_style: str = "footprint",
               xz_dilate_cm: float = 1.0,
               y_margin_cm: float = 1.0,
               cc_filter_seeds: bool = True):
    style_desc = (f"2D X-Z footprint + {xz_dilate_cm:.1f}cm dilation + "
                  f"{y_margin_cm:.1f}cm vertical margin"
                  if crop_style == "footprint"
                  else f"3D AABB + {margin*100:.0f}% expansion")
    print(f"\n=== Mode A: SAM3 votes -> seed faces -> {style_desc} ===")
    print(f"  output:       {output_dir}")
    print(f"  masks from:   {mask_dir}")
    print(f"  seed thresh:  {seed_threshold}")

    print(f"\n[load] mesh + atlas ...")
    t0 = time.time()
    mesh = trimesh.load(str(MESH_PATH), force="mesh", process=False)
    _bind_atlas(mesh, ATLAS_PATH)
    src = V32ViewSource(".")
    views = src.all_views()
    view_names = [v.name for v in views]
    print(f"  loaded in {time.time()-t0:.1f}s "
          f"({len(mesh.faces):,} faces, {len(views)} views)")

    xz_resolution = 0.005                            # 5 mm cells
    dilate_cells = int(round(xz_dilate_cm / 100.0 / xz_resolution))
    y_margin_m = y_margin_cm / 100.0

    for prompt in PROMPTS:
        slug = _slugify(prompt)
        prompt_dir = output_dir / slug
        prompt_dir.mkdir(parents=True, exist_ok=True)
        masks = _load_masks(mask_dir, view_names, slug)
        if not masks:
            print(f"\n[{prompt}] no masks found in {mask_dir} -- skipping")
            continue

        print(f"\n[{prompt}] rasterized voting ({len(masks)} masks) ...")
        t0 = time.time()
        score, voted, n_used = accumulate_face_scores_rasterized(
            mesh, views, masks, min_pixel_count=8, mask_margin_px=3, verbose=False,
        )
        print(f"  voted in {time.time()-t0:.1f}s, {n_used} views contributed")

        seeds = score >= seed_threshold
        n_seeds = int(seeds.sum())
        print(f"  seeds (score>={seed_threshold}): {n_seeds:,} faces")
        if n_seeds < 50:
            print(f"  too few seeds -- skipping")
            continue

        if crop_style == "footprint":
            face_mask, info = _footprint_crop(
                mesh, np.where(seeds)[0],
                xz_dilate_cells=dilate_cells,
                xz_resolution=xz_resolution,
                y_margin=y_margin_m,
                cc_filter_seeds=cc_filter_seeds,
            )
            if cc_filter_seeds:
                print(f"  seed CC filter: {info['n_seed_input']:,} -> "
                      f"{info['n_seed_cleaned']:,} faces (largest connected component)")
            print(f"  footprint area: {info['footprint_area_cm2']:.0f} cm^2 "
                  f"(vs AABB {info['aabb_footprint_cm2']:.0f} cm^2, "
                  f"{info['footprint_fraction']*100:.0f}% tight)")
            print(f"  Y range: {info['y_range_m'][0]:.3f} -> {info['y_range_m'][1]:.3f} m  "
                  f"(span {info['y_span_m']*100:.1f}cm)")
        else:
            seed_verts = mesh.vertices[mesh.faces[seeds].flatten()]
            aabb_min = seed_verts.min(axis=0)
            aabb_max = seed_verts.max(axis=0)
            size_pre = aabb_max - aabb_min
            aabb_min, aabb_max = _expand(aabb_min, aabb_max, margin)
            print(f"  seed AABB size={size_pre[0]:.2f}x{size_pre[1]:.2f}x{size_pre[2]:.2f}m")
            face_mask = _spatial_mask(mesh, aabb_min, aabb_max)

        print(f"  spatial crop: {int(face_mask.sum()):,} faces")

        export_extract_and_remove(
            mesh, face_mask,
            out_dir=prompt_dir,
            base_name=slug,
            write_remainder=write_remainder,
            texture_atlas_src=ATLAS_PATH,
            verbose=True,
        )


def run_mode_b(output_dir, json_source_dir, margin, write_remainder):
    print(f"\n=== Mode B: prior _collider.json -> AABB -> spatial crop ===")
    print(f"  output:        {output_dir}")
    print(f"  JSON source:   {json_source_dir}")
    print(f"  margin:        {margin*100:.0f}%")

    print(f"\n[load] mesh + atlas ...")
    mesh = trimesh.load(str(MESH_PATH), force="mesh", process=False)
    _bind_atlas(mesh, ATLAS_PATH)

    for prompt in PROMPTS:
        slug = _slugify(prompt)
        prompt_dir = output_dir / slug
        prompt_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_source_dir / slug / f"{slug}_extracted_collider.json"
        if not json_path.exists():
            print(f"\n[{prompt}] no JSON at {json_path} -- skipping")
            continue

        meta = json.loads(json_path.read_text())
        aabb_min = np.array(meta["aabb_min"], dtype=np.float64)
        aabb_max = np.array(meta["aabb_max"], dtype=np.float64)
        size_pre = aabb_max - aabb_min
        aabb_min, aabb_max = _expand(aabb_min, aabb_max, margin)
        print(f"\n[{prompt}] AABB from {json_path.name}: "
              f"size {size_pre[0]:.2f}x{size_pre[1]:.2f}x{size_pre[2]:.2f}m "
              f"(expanded {margin*100:.0f}%)")

        face_mask = _spatial_mask(mesh, aabb_min, aabb_max)
        print(f"  spatial crop: {int(face_mask.sum()):,} faces")

        export_extract_and_remove(
            mesh, face_mask,
            out_dir=prompt_dir,
            base_name=slug,
            write_remainder=write_remainder,
            texture_atlas_src=ATLAS_PATH,
            verbose=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["a", "b"])
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--mask-dir", type=Path,
                   default=Path("output/segmented_v32_data3_v7/masks"),
                   help="(mode a) directory holding cached SAM3 masks")
    p.add_argument("--json-source-dir", type=Path,
                   default=Path("output/segmented_v32_data3_v7"),
                   help="(mode b) directory holding prior _collider.json files")
    p.add_argument("--margin", type=float, default=0.05,
                   help="Fraction to expand AABB in each direction (0.05 = 5%%)")
    p.add_argument("--seed-threshold", type=float, default=0.7,
                   help="(mode a) face score required to be a seed")
    p.add_argument("--no-remainder", action="store_true",
                   help="Skip the scene_minus_* output (saves disk + time)")
    p.add_argument("--crop-style", default="footprint",
                   choices=["footprint", "aabb"],
                   help="(mode a) footprint = 2D X-Z occupancy mask "
                        "(tight to object outline). aabb = rectangular box.")
    p.add_argument("--xz-dilate-cm", type=float, default=1.0,
                   help="(footprint) safety dilation of the X-Z mask in cm")
    p.add_argument("--y-margin-cm", type=float, default=1.0,
                   help="(footprint) Y range expansion above/below in cm")
    p.add_argument("--no-cc-filter-seeds", action="store_true",
                   help="(footprint) skip the largest-component seed filter "
                        "(useful if your object has genuinely disconnected "
                        "parts in the mesh, e.g. a chair leg poking through)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "a":
        run_mode_a(args.output_dir, args.mask_dir, args.margin,
                   args.seed_threshold, not args.no_remainder,
                   crop_style=args.crop_style,
                   xz_dilate_cm=args.xz_dilate_cm,
                   y_margin_cm=args.y_margin_cm,
                   cc_filter_seeds=not args.no_cc_filter_seeds)
    else:
        run_mode_b(args.output_dir, args.json_source_dir, args.margin,
                   not args.no_remainder)


if __name__ == "__main__":
    main()
