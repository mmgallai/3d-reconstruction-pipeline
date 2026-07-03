"""Track 2a deliverable: produce a 'patched' V32 scene splat = original
scene splat MINUS object Gaussians PLUS the desk-patch Gaussians.

This is the runtime state when the user lifts an object in VR.

For each object:
  1. Compute the spatial-crop predicate from the v9a_fp_v2 extracted-mesh
     footprint (XZ mask + Y range), in metric metres.
  2. Transform existing scene-splat Gaussian positions back to metric,
     drop those inside the footprint (the object's Gaussians).
  3. Append the per-object desk_patch Gaussians (already in splat space).

Output: one merged `<scene>_patched_<slug>.ply`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).parent))
from _spatial_crop_splat import (         # type: ignore[import]
    _load_dataparser_transform,
    _splat_to_metric,
    _build_footprint_from_extracted_mesh,
    _test_inside_footprint,
)


def _read_ply_vertex(p: Path) -> tuple[np.ndarray, list[str]]:
    pd = PlyData.read(str(p))
    v = pd["vertex"]
    fields = [x.name for x in v.properties]
    return v.data, fields


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-splat", type=Path,
                    default=Path("output/splat_v32_data3_noinit_pruned.ply"))
    ap.add_argument("--dataparser", type=Path,
                    default=Path("nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json"))
    ap.add_argument("--bounds-json", type=Path,
                    default=Path("colmap/dense/tof_bounds.json"))
    ap.add_argument("--segmented-dir", type=Path,
                    default=Path("output/segmented_v32_data3_v9a_fp_v2"))
    ap.add_argument("--patch-dir", type=Path,
                    default=Path("output/desk_patch_v32_data3"))
    ap.add_argument("--prompts",
                    default="white_water_bottle,blue_box,red_lobster_figurine")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("output/scene_patched_v32_data3"))
    ap.add_argument("--xz-dilate-cm", type=float, default=0.5,
                    help="match _spatial_crop_splat.py production value")
    ap.add_argument("--y-margin-cm", type=float, default=0.5,
                    help="match _spatial_crop_splat.py production value")
    ap.add_argument("--combined", action="store_true",
                    help="also produce a single splat with ALL objects removed "
                         "and ALL patches inserted (room with no grabbable objects)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] loading scene splat: {args.scene_splat}")
    scene_v, scene_fields = _read_ply_vertex(args.scene_splat)
    n_scene = len(scene_v)
    print(f"[init]   {n_scene:,} Gaussians, {len(scene_fields)} fields")

    # Load dataparser to invert splat positions back to metric.
    import json
    bounds = json.loads(args.bounds_json.read_text())
    colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])
    dp_R, dp_t, dp_scale = _load_dataparser_transform(args.dataparser)

    # Scene Gaussian positions in metric metres
    splat_positions = np.stack([scene_v["x"], scene_v["y"], scene_v["z"]],
                               axis=-1).astype(np.float64)
    metric_positions = _splat_to_metric(splat_positions,
                                        dp_R, dp_t, dp_scale, colmap_to_metric)
    print(f"[init] scene metric range: "
          f"x[{metric_positions[:,0].min():.2f},{metric_positions[:,0].max():.2f}]  "
          f"y[{metric_positions[:,1].min():.2f},{metric_positions[:,1].max():.2f}]  "
          f"z[{metric_positions[:,2].min():.2f},{metric_positions[:,2].max():.2f}]")

    # Build per-object footprints and per-object Gaussian-inside masks.
    inside_any = np.zeros(n_scene, dtype=bool)
    inside_per_obj: dict[str, np.ndarray] = {}
    patches: dict[str, np.ndarray] = {}

    for slug in args.prompts.split(","):
        slug = slug.strip()
        extracted_obj = args.segmented_dir / slug / f"{slug}_extracted.obj"
        if not extracted_obj.exists():
            extracted_obj = args.segmented_dir / slug / f"{slug}_extracted.ply"
        if not extracted_obj.exists():
            print(f"[{slug}] missing extracted mesh; skipping")
            continue
        fp = _build_footprint_from_extracted_mesh(
            extracted_obj, xz_resolution=0.005,
            xz_dilate_cm=args.xz_dilate_cm,
            y_margin_cm=args.y_margin_cm)
        inside = _test_inside_footprint(metric_positions, fp)
        inside_per_obj[slug] = inside
        inside_any |= inside
        n_inside = int(inside.sum())
        print(f"[{slug}] {n_inside:,} scene Gaussians inside footprint "
              f"(will be removed); Y range "
              f"[{fp['y_min']:.3f}, {fp['y_max']:.3f}]")

        # Load patch
        patch_path = args.patch_dir / f"desk_patch_{slug}.ply"
        if not patch_path.exists():
            print(f"[{slug}]   patch missing: {patch_path}")
            continue
        patch_v, patch_fields = _read_ply_vertex(patch_path)
        if patch_fields != scene_fields:
            missing_in_patch = set(scene_fields) - set(patch_fields)
            extra_in_patch = set(patch_fields) - set(scene_fields)
            print(f"[{slug}]   WARN field mismatch -- patch missing "
                  f"{sorted(missing_in_patch)[:5]}, "
                  f"extra {sorted(extra_in_patch)[:5]}")
        patches[slug] = patch_v

    # Per-object outputs
    for slug, inside in inside_per_obj.items():
        kept = ~inside
        kept_rows = scene_v[kept]
        patch_v = patches.get(slug)
        if patch_v is None:
            merged = kept_rows
        else:
            merged = np.concatenate([kept_rows, patch_v])
        out = args.out_dir / f"scene_patched_{slug}.ply"
        el = PlyElement.describe(merged, "vertex")
        PlyData([el]).write(str(out))
        sz = out.stat().st_size / 1_048_576
        print(f"[{slug}] -> {out.name} : "
              f"{len(merged):,} gaussians ({sz:.1f} MB)  "
              f"= {int(kept.sum()):,} kept + {len(patch_v) if patch_v is not None else 0:,} patch")

    if args.combined:
        kept = ~inside_any
        kept_rows = scene_v[kept]
        all_patches = [p for p in patches.values()]
        if all_patches:
            merged = np.concatenate([kept_rows, *all_patches])
        else:
            merged = kept_rows
        out = args.out_dir / "scene_patched_ALL.ply"
        el = PlyElement.describe(merged, "vertex")
        PlyData([el]).write(str(out))
        sz = out.stat().st_size / 1_048_576
        print(f"[ALL] -> {out.name} : "
              f"{len(merged):,} gaussians ({sz:.1f} MB)  "
              f"= {int(kept.sum()):,} kept + "
              f"{sum(len(p) for p in all_patches):,} patch")


if __name__ == "__main__":
    main()
