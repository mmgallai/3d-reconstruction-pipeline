"""Clean-GS pruning stage on the v5_shcolor per-object splats.

This is the **v6_cleaned** stage of the segmenter splat pipeline. Place
this AFTER `_spatial_crop_splat.py` (which produces v5_shcolor):

    SAM3 masks (cached)
      |
      v
    pipeline.py + _spatial_crop.py        --> v9a_fp_v2 mesh
      |
      v
    _spatial_crop_splat.py                --> v5_shcolor per-object splats
      |
      v
    _clean_splat.py  (this script)        --> v6_cleaned per-object splats
      |
      v
    Unity / Quest 3 deliverable

What it does
------------
For each prompt, takes the v5_shcolor `<prompt>_splat.ply` (already
spatially cropped to the object's 3D footprint via v9a_fp_v2) and runs
Clean-GS's `colour-validate + k-NN outlier prune` step inside that
crop. Because the input is already 3D-tight, the 2D-mask depth-ambiguity
that produces halos when Clean-GS is fed the full scene splat is gone --
this only ever prunes WITHIN the existing object envelope.

Empirically (V32 data3): removes 12-19% of Gaussians, keeps the
bounding box unchanged, and slightly raises per-Gaussian opacity
confidence -- i.e. the prunes are real floaters, not surface erosion.

Inputs
------
- v5_shcolor splat dir (`<prompt>_splat.ply` per prompt)
- A Clean-GS-style `cameras.json` for the dataset (per-view K/R/T,
  metric-metres translations, image_name matching the canonical
  SAM3 cache filenames)
- The canonical SAM3 mask cache `<segmenter-output>/masks/` containing
  `IMG_<view>__<prompt_slug>.png` files
- One or more prompts (slug form, e.g. `white_water_bottle`)

Output: `<output-dir>/<prompt>_splat.ply` per prompt (same naming +
schema as v5_shcolor, so existing tooling Just Works).

Heaviness
---------
Pure CPU; uses Clean-GS's `multiprocessing.Pool` over views. Per object:
~5-15 s on a recent CPU when masks are already cached, ~1 MB in / ~1 MB
out. SAM3 mask generation (5 min for 140 views x N prompts) is NOT done
here -- those are reused from `pipeline.py`'s output. No GPU touched.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _slug(prompt: str) -> str:
    return prompt.strip().lower().replace(" ", "_")


def _stage_masks_for_prompt(sam3_cache: Path, prompt_slug: str,
                            staging_dir: Path, max_views: int,
                            photo_dir: Path) -> int:
    """Build `masked_images` for Clean-GS: source photos with non-object
    pixels blacked out.

    Reads:
      - binary SAM3 mask from `<sam3_cache>/IMG_*__<prompt_slug>.png`
      - matching RGB photo from `<photo_dir>/IMG_*.jpg`
    Writes `<staging_dir>/IMG_*.png` = photo * (mask > 0).

    Clean-GS's `--masked_images` expects masked PHOTOS, not binary masks,
    so it can do RGB-distance colour validation against each Gaussian's
    SH-decoded colour. Feeding raw binary masks here causes the colour-
    validation step to reject 95+% of Gaussians (their colour disagrees
    with the mostly-black mask image).

    If `max_views > 0` and there are more masks than that, evenly subsample
    along the view sequence. Subsampling matters: Clean-GS's colour-
    validation step gets monotonically stricter with more views (one
    disagreement is enough to drop a Gaussian), so feeding all 140 cached
    masks over-prunes catastrophically. ~20 views matches the settings
    under which v2 produced its sweet-spot 12-19% pruning.

    Returns the number of mask files staged.
    """
    import cv2
    import numpy as _np

    staging_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale files from a previous run to keep things deterministic
    for old in staging_dir.glob("*.png"):
        old.unlink()

    sources = sorted(sam3_cache.glob(f"*__{prompt_slug}.png"))
    if max_views > 0 and len(sources) > max_views:
        idx = _np.linspace(0, len(sources) - 1, max_views).round().astype(int)
        sources = [sources[i] for i in sorted(set(idx.tolist()))]

    n_written = 0
    for src in sources:
        # `IMG_femto_0042__white_water_bottle.png` -> stem `IMG_femto_0042`
        view_stem = src.name.split(f"__{prompt_slug}")[0]
        photo_path = photo_dir / f"{view_stem}.jpg"
        if not photo_path.is_file():
            # try .png photo as fallback
            alt = photo_dir / f"{view_stem}.png"
            if alt.is_file():
                photo_path = alt
            else:
                print(f"    skip {view_stem}: no source photo at {photo_path}")
                continue

        photo = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if photo is None or mask is None:
            print(f"    skip {view_stem}: failed to load photo or mask")
            continue
        if mask.shape[:2] != photo.shape[:2]:
            mask = cv2.resize(mask, (photo.shape[1], photo.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        masked = photo.copy()
        masked[mask == 0] = 0
        cv2.imwrite(str(staging_dir / f"{view_stem}.png"), masked)
        n_written += 1

    return n_written


def _run_clean_gs(clean_gs_script: Path, python_exe: Path,
                  in_ply: Path, cameras_json: Path, masks_dir: Path,
                  out_ply: Path, outlier_mode: str, color_threshold: float,
                  extra_args: list[str]) -> int:
    """Invoke clean-gs.py in a subprocess.

    `PYTHONIOENCODING=utf-8` is forced so clean-gs.py's final
    `print("✓ Saved ...")` doesn't crash on the Windows cp1252 stdout
    encoding (which otherwise raises UnicodeEncodeError AFTER the .ply has
    already been written, producing a misleading rc=1).
    """
    import os as _os
    cmd = [
        str(python_exe),
        str(clean_gs_script),
        "--ply", str(in_ply),
        "--cameras", str(cameras_json),
        "--masked_images", str(masks_dir),
        "--output", str(out_ply),
        "--mode", outlier_mode,
        "--color_threshold", str(color_threshold),
        *extra_args,
    ]
    env = {**_os.environ, "PYTHONIOENCODING": "utf-8"}
    print(f"  > {' '.join(cmd)}")
    return subprocess.run(cmd, check=False, env=env).returncode


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--splat-dir", type=Path,
                   default=Path("output/segmented_v32_data3_v9a_fp_v2_splat_v5_shcolor"),
                   help="Directory containing the v5_shcolor `<prompt>_splat.ply` files")
    p.add_argument("--cameras", type=Path,
                   default=Path("output/clean_gs_v32_data3/cameras.json"),
                   help="Clean-GS-format cameras.json for this dataset")
    p.add_argument("--sam3-cache", type=Path,
                   default=Path("output/segmented_v32_data3/masks"),
                   help="Canonical SAM3 binary mask cache (`IMG_*__<prompt>.png`)")
    p.add_argument("--photo-dir", type=Path,
                   default=Path("nerfstudio_data/images"),
                   help="Source RGB photos to mask. Filename stems must match the SAM3 cache "
                        "view stems (e.g. `IMG_femto_0042.jpg`).")
    p.add_argument("--prompts", default="white_water_bottle,blue_box,red_lobster_figurine",
                   help="Comma-separated prompt SLUGS (snake_case)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned"),
                   help="Destination for cleaned `<prompt>_splat.ply` files")
    p.add_argument("--clean-gs-script", type=Path,
                   default=Path(r"C:\Users\mgallai\Downloads\clean-gs\clean-gs.py"),
                   help="Path to the Clean-GS clean-gs.py script")
    p.add_argument("--python-exe", type=Path,
                   default=Path(r"C:\Users\mgallai\AppData\Local\miniconda3\envs\sam3\python.exe"),
                   help="Python exe to invoke Clean-GS with (env must have plyfile + cv2 + numpy)")
    p.add_argument("--outlier-mode", default="neighbor",
                   choices=["none", "spatial", "neighbor", "multiview", "combined"],
                   help="Clean-GS outlier-removal stage. `neighbor` is the production pick.")
    p.add_argument("--color-threshold", type=float, default=0.40,
                   help="Clean-GS RGB-distance threshold for the colour-validation step (0-1)")
    p.add_argument("--max-views", type=int, default=20,
                   help="Evenly subsample to at most this many cached SAM3 masks per object. "
                        "Clean-GS over-prunes catastrophically with all 140; ~20 reproduces the "
                        "production v2 sweet spot. Pass 0 to disable subsampling.")
    p.add_argument("--keep-staged-masks", action="store_true",
                   help="Don't delete the staged per-prompt mask dirs on exit (debug)")
    args, extra = p.parse_known_args()

    if not args.splat_dir.is_dir():
        sys.exit(f"missing splat dir {args.splat_dir} -- run _spatial_crop_splat.py first")
    if not args.cameras.is_file():
        sys.exit(f"missing cameras.json {args.cameras}")
    if not args.sam3_cache.is_dir():
        sys.exit(f"missing SAM3 mask cache {args.sam3_cache} -- run pipeline.py first")
    if not args.clean_gs_script.is_file():
        sys.exit(f"missing clean-gs.py at {args.clean_gs_script}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    staging_root = args.output_dir / "_mask_staging"

    summary: list[tuple[str, str]] = []
    failed = 0
    for raw in args.prompts.split(","):
        slug = _slug(raw)
        in_ply = args.splat_dir / f"{slug}_splat.ply"
        out_ply = args.output_dir / f"{slug}_splat.ply"
        masks_dir = staging_root / slug

        if not in_ply.is_file():
            print(f"[{slug}] missing input {in_ply} -- skipping")
            summary.append((slug, "skipped (no v5 splat)"))
            failed += 1
            continue

        n_masks = _stage_masks_for_prompt(args.sam3_cache, slug, masks_dir,
                                          args.max_views, args.photo_dir)
        if n_masks == 0:
            print(f"[{slug}] no masks `*__{slug}.png` in {args.sam3_cache} -- skipping")
            summary.append((slug, "skipped (no masks)"))
            failed += 1
            continue

        sz_in_mb = in_ply.stat().st_size / 1_048_576
        print(f"\n=== {slug} ===")
        print(f"  input splat : {in_ply}  ({sz_in_mb:.2f} MB)")
        print(f"  masks staged: {masks_dir}  ({n_masks} files)")
        print(f"  cameras     : {args.cameras}")

        rc = _run_clean_gs(args.clean_gs_script, args.python_exe,
                           in_ply, args.cameras, masks_dir, out_ply,
                           args.outlier_mode, args.color_threshold, extra)
        if rc != 0 or not out_ply.is_file():
            print(f"[{slug}] clean-gs returned {rc}, output missing -- failure")
            summary.append((slug, f"FAILED (rc={rc})"))
            failed += 1
            continue

        sz_out_mb = out_ply.stat().st_size / 1_048_576
        print(f"  -> {out_ply}  ({sz_out_mb:.2f} MB, "
              f"{100 * sz_out_mb / sz_in_mb:.0f}% of v5 size)")
        summary.append((slug, f"{sz_in_mb:.2f} MB -> {sz_out_mb:.2f} MB"))

    if not args.keep_staged_masks and staging_root.is_dir():
        shutil.rmtree(staging_root, ignore_errors=True)

    print("\n--- v6_cleaned summary ---")
    for slug, msg in summary:
        print(f"  {slug:30s}  {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
