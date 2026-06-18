"""Spatial crop on a nerfstudio Gaussian splat.

Reads a per-object AABB (from a prior segmenter run's _collider.json)
in METRES, transforms it into the splat's coordinate frame via the
chain:
    metric -> COLMAP units  (*= scale_factor_da3_to_colmap)
    COLMAP -> splat normalized  (apply dataparser_transform, * dataparser_scale)
and keeps every Gaussian whose centre is inside the resulting region.

Faster path: instead of transforming the AABB (which doesn't transform
cleanly under rotation), we transform the Gaussian positions INVERSELY
back to mesh-metric space and test against the original AABB. Same
result, cleaner math.

Output is a standard nerfstudio-format .ply with the same per-Gaussian
properties as the input (positions, normals, SH, opacity, scales, rotations),
loadable by SuperSplat, nerfstudio's viewer, Aras-P Unity splat plugin,
Three.js gsplat viewers, and Babylon's GaussianSplattingMesh.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def _load_dataparser_transform(dp_path: Path):
    """Return (R 3x3, t 3, scale) from a nerfstudio dataparser_transforms.json.
    Forward transform: splat_pos = scale * (R @ original_pos + t)
    """
    d = json.loads(dp_path.read_text())
    T = np.asarray(d["transform"], dtype=np.float64)            # (3, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    s = float(d["scale"])
    return R, t, s


def _splat_to_metric(positions_splat: np.ndarray,
                     R: np.ndarray, t: np.ndarray, dp_scale: float,
                     colmap_to_metric: float) -> np.ndarray:
    """Inverse of the chain. Returns Gaussian positions in metric metres.

    Forward chain (how the splat was made):
        colmap_pos = original COLMAP-unit position from transforms.json
        splat_pos  = dp_scale * (R @ colmap_pos + t)

    Inverse:
        colmap_pos  = R.T @ (splat_pos / dp_scale - t)
        metric_pos  = colmap_pos * colmap_to_metric          # 1 / scale_factor
    """
    colmap_pos = (R.T @ (positions_splat / dp_scale - t).T).T
    return colmap_pos * colmap_to_metric


def _build_footprint_from_extracted_mesh(extracted_obj_path: Path,
                                         xz_resolution: float = 0.005,
                                         xz_dilate_cm: float = 0.5,
                                         y_margin_cm: float = 0.5):
    """Compute a 2D X-Z occupancy mask + Y range directly from an extracted
    object mesh's vertices.

    The extracted mesh from the v9a_fp_v2 pipeline IS the tight object
    outline. Projecting its vertices onto the horizontal plane gives the
    exact 2D footprint to use for splat cropping -- no extra voting
    needed, and tighter than the AABB.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from _spatial_crop import _build_xz_footprint

    import trimesh
    m = trimesh.load(str(extracted_obj_path), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float64)
    if len(verts) == 0:
        raise ValueError(f"{extracted_obj_path} has no vertices")
    dilate_cells = max(0, int(round(xz_dilate_cm / 100.0 / xz_resolution)))
    fp_mask, xz_min, shape, res = _build_xz_footprint(
        verts[:, [0, 2]],
        resolution=xz_resolution,
        dilate_cells=dilate_cells,
    )
    y_min = float(verts[:, 1].min()) - y_margin_cm / 100.0
    y_max = float(verts[:, 1].max()) + y_margin_cm / 100.0
    return {
        "fp_mask": fp_mask, "xz_min": xz_min, "shape": shape,
        "resolution": res, "y_min": y_min, "y_max": y_max,
    }


def _test_inside_footprint(positions_metric: np.ndarray, fp) -> np.ndarray:
    """Per Gaussian: True if (x, z) lands in the footprint mask AND y in range."""
    cells = ((positions_metric[:, [0, 2]] - fp["xz_min"]) / fp["resolution"]).astype(int)
    in_bounds = ((cells >= 0) & (cells < fp["shape"])).all(axis=1)
    cells_safe = np.clip(cells, 0, fp["shape"] - 1)
    in_xz = fp["fp_mask"][cells_safe[:, 0], cells_safe[:, 1]] & in_bounds
    in_y = ((positions_metric[:, 1] >= fp["y_min"])
            & (positions_metric[:, 1] <= fp["y_max"]))
    return in_xz & in_y


_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = np.array([
    1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
    -1.0925484305920792, 0.5462742152960396,
], dtype=np.float32)
_SH_C3 = np.array([
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
    0.3731763325901154, -0.4570457994644658, 1.445305721320277,
    -0.5900435899266435,
], dtype=np.float32)


def _per_gaussian_rgb01(elem) -> np.ndarray:
    """DC-only RGB in [0, 1] (clipped). Cheap fallback / legacy path."""
    f_dc = np.stack([
        np.asarray(elem["f_dc_0"], dtype=np.float32),
        np.asarray(elem["f_dc_1"], dtype=np.float32),
        np.asarray(elem["f_dc_2"], dtype=np.float32),
    ], axis=-1)
    rgb = f_dc * _SH_C0 + 0.5
    return np.clip(rgb, 0.0, 1.0)


def _per_gaussian_viewport_rgb01(elem, view_dir: np.ndarray) -> np.ndarray:
    """Evaluate the per-Gaussian SH (degree 3) at `view_dir` -> rendered
    base RGB in [0, 1] (clipped). This is what SuperSplat / Quest 3
    actually displays, modulo per-pixel blending. Far better than DC
    for color-based filtering.

    The 45 f_rest_* properties are stored in nerfstudio's gsplat-style
    order: 15 SH coefficients per channel (bands 1, 2, 3), with the R
    channel first (f_rest_0..14), then G (15..29), then B (30..44).
    """
    v = np.asarray(view_dir, dtype=np.float32)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z

    n = len(elem)
    dc = np.stack([
        np.asarray(elem["f_dc_0"], dtype=np.float32),
        np.asarray(elem["f_dc_1"], dtype=np.float32),
        np.asarray(elem["f_dc_2"], dtype=np.float32),
    ], axis=-1)                                                      # (n, 3)

    # Load f_rest into (n, 3, 15) -- channel-contiguous storage.
    rest = np.empty((n, 3, 15), dtype=np.float32)
    for ch in range(3):
        for k in range(15):
            rest[:, ch, k] = np.asarray(elem[f"f_rest_{ch * 15 + k}"],
                                        dtype=np.float32)

    out = _SH_C0 * dc                                                # band 0

    # Band 1 (m = -1, 0, +1)
    out -= _SH_C1 * y * rest[:, :, 0]
    out += _SH_C1 * z * rest[:, :, 1]
    out -= _SH_C1 * x * rest[:, :, 2]

    # Band 2 (m = -2..+2)
    out += _SH_C2[0] * xy * rest[:, :, 3]
    out += _SH_C2[1] * yz * rest[:, :, 4]
    out += _SH_C2[2] * (2 * zz - xx - yy) * rest[:, :, 5]
    out += _SH_C2[3] * xz * rest[:, :, 6]
    out += _SH_C2[4] * (xx - yy) * rest[:, :, 7]

    # Band 3 (m = -3..+3)
    out += _SH_C3[0] * y * (3 * xx - yy) * rest[:, :, 8]
    out += _SH_C3[1] * xy * z * rest[:, :, 9]
    out += _SH_C3[2] * y * (4 * zz - xx - yy) * rest[:, :, 10]
    out += _SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * rest[:, :, 11]
    out += _SH_C3[4] * x * (4 * zz - xx - yy) * rest[:, :, 12]
    out += _SH_C3[5] * z * (xx - yy) * rest[:, :, 13]
    out += _SH_C3[6] * x * (xx - 3 * yy) * rest[:, :, 14]

    out = out + 0.5
    return np.clip(out, 0.0, 1.0)


def _per_gaussian_scales(elem) -> np.ndarray:
    """Return per-Gaussian scales in SPLAT units (= exp(stored log-scale)).
    Shape (N, 3)."""
    s = np.stack([
        np.asarray(elem["scale_0"], dtype=np.float32),
        np.asarray(elem["scale_1"], dtype=np.float32),
        np.asarray(elem["scale_2"], dtype=np.float32),
    ], axis=-1)
    return np.exp(s)


def _splat_unit_to_metric(dp_scale: float, colmap_to_metric: float) -> float:
    """Conversion factor: splat-unit * factor -> metric metres."""
    return colmap_to_metric / dp_scale


def _filter_by_exclude_color(elem, exclude_rgb_255, tolerance_255: float,
                              view_dir: np.ndarray = None) -> np.ndarray:
    """Mask: True where the Gaussian's color is FARTHER from
    `exclude_rgb_255` than `tolerance_255`.

    If `view_dir` is given, color = full SH evaluation at that direction
    (matches what the splat actually renders to the screen).
    If `view_dir` is None, color = DC band only (cheap, less reliable).
    """
    if exclude_rgb_255 is None:
        return np.ones(len(elem), dtype=bool)
    if view_dir is None:
        rgb_255 = _per_gaussian_rgb01(elem) * 255.0
    else:
        rgb_255 = _per_gaussian_viewport_rgb01(elem, view_dir) * 255.0
    target = np.asarray(exclude_rgb_255, dtype=np.float32)
    dist = np.linalg.norm(rgb_255 - target, axis=-1)
    return dist > tolerance_255


def _filter_by_max_scale(elem, max_scale_cm: float,
                          splat_to_metric: float,
                          band_y_max: float = None,
                          pos_metric: np.ndarray = None,
                          max_scale_cm_base: float = None) -> np.ndarray:
    """Mask: True where the Gaussian passes the max-scale cut.

    Supports three modes:
      1. Global single threshold: pass `max_scale_cm` only. Drops
         Gaussians anywhere whose largest axis exceeds the threshold.
      2. Band-aware single threshold: also pass `band_y_max` +
         `pos_metric`. Only Gaussians at Y <= band_y_max (the base
         band) are subject to the cut; above the band -- the object
         body -- pass unconditionally.
      3. Two-tier (recommended for best quality + cleanup): also pass
         `max_scale_cm_base`. Base band uses `max_scale_cm_base`
         (tight, e.g. 0.6 cm); body uses `max_scale_cm` (loose, e.g.
         2.0 cm). Object body keeps v5-level quality, base gets
         aggressive halo cleanup. Set `max_scale_cm <= 0` to disable
         the body cut entirely.
    """
    has_band = band_y_max is not None and pos_metric is not None
    body_active = max_scale_cm > 0
    base_active = max_scale_cm_base is not None and max_scale_cm_base > 0
    if not body_active and not base_active:
        return np.ones(len(elem), dtype=bool)

    scales_splat = _per_gaussian_scales(elem)
    max_scale_cm_actual = scales_splat.max(axis=-1) * splat_to_metric * 100.0

    if not has_band:
        return max_scale_cm_actual <= max_scale_cm

    in_band = pos_metric[:, 1] <= band_y_max
    body_pass = (max_scale_cm_actual <= max_scale_cm) if body_active else np.ones_like(in_band, dtype=bool)
    base_thresh = max_scale_cm_base if base_active else max_scale_cm
    base_pass = (max_scale_cm_actual <= base_thresh) if base_active else np.ones_like(in_band, dtype=bool)
    return (in_band & base_pass) | (~in_band & body_pass)


def _filter_by_anisotropy(elem, max_ratio: float) -> np.ndarray:
    """Mask: True where max_scale / min_scale <= max_ratio.

    Very flat / elongated Gaussians (huge anisotropy) are often
    surface-bridging fluff or floaters that don't represent crisp
    geometry. max_ratio ~ 10 is a reasonable cut.
    """
    if max_ratio <= 0:
        return np.ones(len(elem), dtype=bool)
    scales = _per_gaussian_scales(elem)
    smax = scales.max(axis=-1)
    smin = scales.min(axis=-1)
    return smax / np.maximum(smin, 1e-9) <= max_ratio


def _filter_by_opacity(elem, opacity_threshold: float) -> np.ndarray:
    """Filter mask True where sigmoid(opacity_logit) > threshold.

    Nerfstudio stores opacity as a pre-sigmoid logit; the rendered alpha
    is sigmoid(opacity). Very low alpha Gaussians are usually "filler"
    that contribute almost nothing visually but bloat per-object .plys.
    """
    if opacity_threshold <= 0.0:
        return np.ones(len(elem), dtype=bool)
    op_logit = np.asarray(elem["opacity"], dtype=np.float32)
    alpha = 1.0 / (1.0 + np.exp(-op_logit))
    return alpha > opacity_threshold


def crop_splat(splat_path: Path, dataparser_json: Path,
               output_path: Path, *,
               crop_style: str,
               colmap_to_metric: float,
               collider_json: Path = None,
               extracted_obj: Path = None,
               margin: float = 0.05,
               xz_dilate_cm: float = 0.5,
               y_margin_cm: float = 0.5,
               opacity_threshold: float = 0.0,
               exclude_color: tuple = None,
               color_tolerance: float = 50.0,
               view_dir: np.ndarray = None,
               max_scale_cm: float = 0.0,
               max_scale_cm_base: float = 0.0,
               scale_filter_band_cm: float = 0.0,
               max_anisotropy: float = 0.0) -> int:
    """Crop a Gaussian splat by either AABB or 2D X-Z footprint.

    crop_style = 'aabb'      -> uses collider_json's aabb_{min,max}.
    crop_style = 'footprint' -> uses extracted_obj's vertices to build a
                                tight 2D X-Z mask + Y range. Recommended
                                for clean per-object splats.
    """
    print(f"  splat:    {splat_path.name}")
    R, t, dp_scale = _load_dataparser_transform(dataparser_json)
    print(f"  dataparser: scale={dp_scale:.4f}, R det={np.linalg.det(R):.4f}")

    ply = PlyData.read(str(splat_path))
    elem = ply["vertex"]
    n_in = len(elem)
    pos_splat = np.stack([elem["x"], elem["y"], elem["z"]], axis=-1)
    pos_metric = _splat_to_metric(pos_splat, R, t, dp_scale, colmap_to_metric)

    fp_y_min = None                                 # bottom of object Y, for band-aware filters
    if crop_style == "footprint":
        if extracted_obj is None or not extracted_obj.exists():
            raise FileNotFoundError(
                f"footprint mode needs --extracted-obj or a "
                f"<prompt>_extracted.obj sitting next to the collider JSON; "
                f"got {extracted_obj}"
            )
        print(f"  source:   {extracted_obj.name} (vertex footprint)")
        fp = _build_footprint_from_extracted_mesh(
            extracted_obj,
            xz_dilate_cm=xz_dilate_cm,
            y_margin_cm=y_margin_cm,
        )
        fp_area_cm2 = float(fp["fp_mask"].sum()) * (fp["resolution"] * 100.0) ** 2
        print(f"  footprint area:  {fp_area_cm2:.0f} cm^2  "
              f"(Y span {(fp['y_max']-fp['y_min'])*100:.1f} cm, "
              f"dilate {xz_dilate_cm:.1f}cm)")
        inside = _test_inside_footprint(pos_metric, fp)
        fp_y_min = fp["y_min"]
    else:                                                # aabb mode
        if collider_json is None or not collider_json.exists():
            raise FileNotFoundError(f"aabb mode needs --collider-json; got {collider_json}")
        print(f"  source:   {collider_json.name} (AABB)")
        cj = json.loads(collider_json.read_text())
        aabb_min = np.array(cj["aabb_min"], dtype=np.float64)
        aabb_max = np.array(cj["aabb_max"], dtype=np.float64)
        size = aabb_max - aabb_min
        aabb_min -= size * margin
        aabb_max += size * margin
        print(f"  AABB size: {(aabb_max-aabb_min)[0]*100:.1f} x "
              f"{(aabb_max-aabb_min)[1]*100:.1f} x "
              f"{(aabb_max-aabb_min)[2]*100:.1f} cm  "
              f"(+{margin*100:.0f}% margin)")
        inside = ((pos_metric >= aabb_min) & (pos_metric <= aabb_max)).all(axis=-1)
        fp_y_min = float(aabb_min[1])

    n_after_spatial = int(inside.sum())
    print(f"  spatial filter: {n_in:,} -> {n_after_spatial:,} "
          f"({100*n_after_spatial/n_in:.1f}%)")

    if opacity_threshold > 0:
        op_mask = _filter_by_opacity(elem, opacity_threshold)
        before = int(inside.sum())
        inside = inside & op_mask
        print(f"  opacity filter (alpha > {opacity_threshold:.2f}): "
              f"{before:,} -> {int(inside.sum()):,} "
              f"({100*int(inside.sum())/max(before,1):.1f}%)")

    if exclude_color is not None:
        col_mask = _filter_by_exclude_color(elem, exclude_color, color_tolerance,
                                             view_dir=view_dir)
        before = int(inside.sum())
        inside = inside & col_mask
        space = "SH-viewport" if view_dir is not None else "DC"
        print(f"  {space} color filter (exclude RGB{tuple(exclude_color)} "
              f"+/- {color_tolerance:.0f}): {before:,} -> "
              f"{int(inside.sum()):,} "
              f"({100*int(inside.sum())/max(before,1):.1f}%)")

    has_body_cut = max_scale_cm > 0
    has_base_cut = max_scale_cm_base > 0
    if has_body_cut or has_base_cut:
        splat_to_metric = colmap_to_metric / dp_scale
        use_band = scale_filter_band_cm > 0 and fp_y_min is not None
        if use_band:
            band_y_max = fp_y_min + scale_filter_band_cm / 100.0
        else:
            band_y_max = None
        sc_mask = _filter_by_max_scale(
            elem,
            max_scale_cm=max_scale_cm,
            splat_to_metric=splat_to_metric,
            band_y_max=band_y_max if use_band else None,
            pos_metric=pos_metric if use_band else None,
            max_scale_cm_base=(max_scale_cm_base if has_base_cut else None),
        )
        if use_band and has_base_cut and has_body_cut:
            label = (f"two-tier (body <={max_scale_cm:.1f} cm, "
                     f"base[0..{scale_filter_band_cm:.0f}cm] <={max_scale_cm_base:.1f} cm)")
        elif use_band and has_base_cut:
            label = (f"base-only (bottom {scale_filter_band_cm:.0f} cm "
                     f"<={max_scale_cm_base:.1f} cm; body unrestricted)")
        elif use_band:
            label = (f"band-aware (bottom {scale_filter_band_cm:.0f} cm "
                     f"<={max_scale_cm:.1f} cm)")
        else:
            label = f"global <= {max_scale_cm:.1f} cm"
        before = int(inside.sum())
        inside = inside & sc_mask
        print(f"  max-scale filter ({label}): "
              f"{before:,} -> {int(inside.sum()):,} "
              f"({100*int(inside.sum())/max(before,1):.1f}%)")

    if max_anisotropy > 0:
        an_mask = _filter_by_anisotropy(elem, max_anisotropy)
        before = int(inside.sum())
        inside = inside & an_mask
        print(f"  anisotropy filter (max/min <= {max_anisotropy:.0f}): "
              f"{before:,} -> {int(inside.sum()):,} "
              f"({100*int(inside.sum())/max(before,1):.1f}%)")

    n_kept = int(inside.sum())
    if n_kept == 0:
        print(f"  WARNING: empty crop; check dataparser_transforms.json + scale.")
        return 0

    new_data = elem.data[inside]
    new_elem = PlyElement.describe(new_data, "vertex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([new_elem], text=False).write(str(output_path))
    out_mb = output_path.stat().st_size / 1_048_576
    print(f"  wrote {output_path.name}  ({out_mb:.1f} MB)")
    return n_kept


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splat", type=Path, required=True,
                   help="nerfstudio splatfacto .ply (e.g. splat_v32_data3_noinit_pruned.ply)")
    p.add_argument("--dataparser", type=Path, required=True,
                   help="nerfstudio dataparser_transforms.json (sits next to "
                        "the splatfacto run's config.yml)")
    p.add_argument("--tof-bounds", type=Path,
                   default=Path("colmap/dense/tof_bounds.json"),
                   help="tof_bounds.json containing scale_factor_da3_to_colmap")
    p.add_argument("--segmented-dir", type=Path, required=True,
                   help="prior segmenter output dir holding the "
                        "<prompt>/<prompt>_extracted_collider.json files")
    p.add_argument("--prompts", required=True,
                   help='comma-separated prompts (must match earlier run, '
                        'e.g. "white water bottle,blue box,red lobster figurine")')
    p.add_argument("--output-dir", type=Path, required=True,
                   help="where to write the per-object splat .plys")
    p.add_argument("--margin", type=float, default=0.05,
                   help="(aabb mode) fraction to expand the AABB (0.05 = 5%%)")
    p.add_argument("--crop-style", default="footprint",
                   choices=["aabb", "footprint"],
                   help="aabb = rectangular box from collider JSON. "
                        "footprint = tight 2D X-Z mask from the extracted "
                        "mesh's vertices (default, tighter).")
    p.add_argument("--xz-dilate-cm", type=float, default=0.5,
                   help="(footprint) safety dilation of the X-Z mask in cm")
    p.add_argument("--y-margin-cm", type=float, default=0.5,
                   help="(footprint) Y range expansion above/below in cm")
    p.add_argument("--opacity-threshold", type=float, default=0.0,
                   help="If > 0, drop Gaussians whose sigmoid(opacity_logit) "
                        "is below this (e.g. 0.05 kills the wispy fluff "
                        "halo around objects). 0 disables.")
    p.add_argument("--exclude-color", default=None,
                   help='Drop Gaussians whose base RGB is within '
                        '--color-tolerance of this color. Format: "R,G,B" '
                        'in 0-255. Example for the cream-coloured desk: '
                        '"200,180,140".')
    p.add_argument("--color-tolerance", type=float, default=50.0,
                   help="Euclidean RGB distance (0-255 scale) for color "
                        "exclusion. ~50 = strict, ~80 = lenient.")
    p.add_argument("--sh-view-dir", default=None,
                   help='If set, evaluates full SH at this view direction '
                        '(matches what SuperSplat/Quest renders) instead of '
                        'using DC alone. Format "vx,vy,vz". Common picks: '
                        '"0,0,1" front, "0,-1,0" top-down. Strongly '
                        'recommended when using --exclude-color.')
    p.add_argument("--max-scale-cm", type=float, default=0.0,
                   help="If > 0, drop Gaussians whose largest axis scale "
                        "exceeds this in metric cm. Fluff/floater Gaussians "
                        "are typically >1.5cm; tight surface Gaussians "
                        "are mm-scale.")
    p.add_argument("--scale-filter-band-cm", type=float, default=0.0,
                   help="If > 0, the --max-scale-cm cut applies ONLY to "
                        "Gaussians at Y in [object_bottom, object_bottom + "
                        "this many cm]. Above the band -- the object body "
                        "-- all sizes are kept. ~2cm preserves object detail "
                        "while still clearing the desk halo at the base.")
    p.add_argument("--max-scale-cm-base", type=float, default=0.0,
                   help="Two-tier scale filter: tighter threshold (cm) for "
                        "Gaussians inside the base band defined by "
                        "--scale-filter-band-cm. Best of both worlds -- pair "
                        "with --max-scale-cm 2.0 (lenient body) and this at "
                        "0.6 cm (aggressive base) to preserve object detail "
                        "AND clean the desk halo aggressively.")
    p.add_argument("--max-anisotropy", type=float, default=0.0,
                   help="If > 0, drop Gaussians whose max-scale / min-scale "
                        "ratio exceeds this. ~10 cuts very flat fluff.")
    args = p.parse_args()

    view_dir = None
    if args.sh_view_dir:
        parts = [float(x.strip()) for x in args.sh_view_dir.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"--sh-view-dir must be 'vx,vy,vz' (got {parts})")
        view_dir = np.array(parts, dtype=np.float32)

    exclude_color = None
    if args.exclude_color:
        parts = [int(x.strip()) for x in args.exclude_color.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"--exclude-color must be 'R,G,B' (got {parts})")
        exclude_color = tuple(parts)

    bj = json.loads(args.tof_bounds.read_text())
    scale_factor = float(bj["scale_factor_da3_to_colmap"])
    colmap_to_metric = 1.0 / scale_factor
    print(f"scale_factor_da3_to_colmap = {scale_factor}  "
          f"(colmap_to_metric = {colmap_to_metric:.6f})")

    import re
    def _slug(s: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", s.strip().lower())
        return re.sub(r"_+", "_", s).strip("_")[:40]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    for prompt in prompts:
        slug = _slug(prompt)
        collider = args.segmented_dir / slug / f"{slug}_extracted_collider.json"
        extracted_obj = args.segmented_dir / slug / f"{slug}_extracted.obj"
        if args.crop_style == "footprint" and not extracted_obj.exists():
            print(f"\n[{prompt}] no extracted mesh at {extracted_obj} -- skipping")
            continue
        if args.crop_style == "aabb" and not collider.exists():
            print(f"\n[{prompt}] no collider JSON at {collider} -- skipping")
            continue
        print(f"\n=== {prompt} ({slug}) -- {args.crop_style} ===")
        crop_splat(
            args.splat, args.dataparser,
            args.output_dir / f"{slug}_splat.ply",
            crop_style=args.crop_style,
            colmap_to_metric=colmap_to_metric,
            collider_json=collider,
            extracted_obj=extracted_obj,
            margin=args.margin,
            xz_dilate_cm=args.xz_dilate_cm,
            y_margin_cm=args.y_margin_cm,
            opacity_threshold=args.opacity_threshold,
            exclude_color=exclude_color,
            color_tolerance=args.color_tolerance,
            view_dir=view_dir,
            max_scale_cm=args.max_scale_cm,
            max_scale_cm_base=args.max_scale_cm_base,
            scale_filter_band_cm=args.scale_filter_band_cm,
            max_anisotropy=args.max_anisotropy,
        )


if __name__ == "__main__":
    main()
