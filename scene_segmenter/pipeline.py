"""scene_segmenter.pipeline -- end-to-end orchestration.

Inputs:
  --mesh        : path to the source mesh (PLY with UV + texture atlas, OR GLB)
  --project-root: project root containing nerfstudio_data/ + colmap/dense/
  --prompts     : comma-separated SAM3 text prompts
  --output-dir  : where to write extracted_* / scene_minus_* / *_collider.json

Optional:
  --max-views   : downsample views (default = use all, deterministic spaced)
  --score-threshold, --min-component-fraction, --erode-px, etc.

Two-phase execution:
  Phase 1: load views + run SAM3 multi-prompt -> save masks PNGs
  Phase 2: per prompt, accumulate face votes -> CC filter -> export
            extracted + scene-minus pair + collider sidecar

Phase 2 can run independently from saved masks (skip --rerun-sam3),
useful for tuning the score threshold without re-running SAM3 each time.

Typical usage:
  conda run -n sam3 python -m scene_segmenter.pipeline \
      --mesh   output/mesh_v32_data3/mesh_v32_data3_openmvs.ply \
      --atlas  output/mesh_v32_data3/scene_textured0.png \
      --project-root . \
      --prompts "monitor,keyboard,mouse,book,lamp,mug,plant" \
      --output-dir output/segmented_v32_data3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Allow running both as a module and as a script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scene_segmenter.views import V32ViewSource
    from scene_segmenter.sam3_segment import segment_multi_prompt, _slugify
    from scene_segmenter.extract_faces import extract_object
    from scene_segmenter.exporter import export_extract_and_remove
else:
    from .views import V32ViewSource
    from .sam3_segment import segment_multi_prompt, _slugify
    from .extract_faces import extract_object
    from .exporter import export_extract_and_remove


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True, type=Path,
                   help="Source mesh (.ply or .glb). Should have UV mapping for "
                        "textured output.")
    p.add_argument("--atlas", type=Path, default=None,
                   help="Texture atlas PNG. Copied alongside the extracted PLY. "
                        "Defaults to <mesh_dir>/scene_textured0.png if present.")
    p.add_argument("--project-root", type=Path, default=Path("."),
                   help="Project root containing nerfstudio_data/ + colmap/dense/")
    p.add_argument("--prompts", required=True,
                   help="Comma-separated SAM3 prompts, e.g. 'monitor,keyboard,lamp'")
    p.add_argument("--output-dir", required=True, type=Path)

    # View sampling
    p.add_argument("--max-views", type=int, default=0,
                   help="If >0, sample this many views evenly. 0 = use all.")

    # SAM3 knobs
    p.add_argument("--erode-px", type=int, default=2,
                   help="Mask erosion in pixels (helps drop floor/edge bleed).")
    p.add_argument("--sam3-mode", default="union", choices=["union", "best"])
    p.add_argument("--sam3-threshold", type=float, default=0.5)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    # Extraction knobs
    p.add_argument("--score-threshold", type=float, default=0.40,
                   help="Faces below this score are not part of the object.")
    p.add_argument("--min-component-fraction", type=float, default=0.05,
                   help="In the CC filter, drop components smaller than this "
                        "fraction of the largest. 0 = keep all.")
    p.add_argument("--no-cc-filter", action="store_true",
                   help="Disable connected-component filtering (debug).")
    p.add_argument("--z-buffer-tol", type=float, default=0.02)
    p.add_argument("--min-view-angle-cos", type=float, default=0.10)
    p.add_argument("--samples-per-face", type=int, default=7, choices=[3, 7],
                   help="LEGACY sample-point voting only. Ignored when "
                        "--rasterize is on (the default). 3 = vertices only, "
                        "7 = vertices + edge midpoints + centroid.")
    p.add_argument("--no-rasterize", action="store_true",
                   help="Disable Open3D ray-cast FaceID rasterization. "
                        "Falls back to legacy sample-point voting (worse).")
    p.add_argument("--min-pixel-count", type=int, default=8,
                   help="When rasterizing, faces with fewer than this many "
                        "rendered pixels across all views get score 0 "
                        "(low-sample bias guard).")
    p.add_argument("--mask-margin-px", type=int, default=3,
                   help="Erode the SAM3 mask by this many pixels for "
                        "POSITIVE evidence; dilate by the same for "
                        "NEGATIVE evidence. The band in between is "
                        "treated as uncertain (no vote). Larger = more "
                        "conservative, fewer boundary artifacts. "
                        "0 disables 3-way evidence (legacy in/out).")
    p.add_argument("--smooth-iterations", type=int, default=2,
                   help="Laplacian-smooth per-face scores N times before "
                        "thresholding (fills swiss-cheese holes from noisy "
                        "per-face scoring). 0 disables.")
    p.add_argument("--smooth-alpha", type=float, default=0.5,
                   help="Smoothing strength in [0,1]; higher = more averaging "
                        "with neighbors.")
    p.add_argument("--dilate-iterations", type=int, default=0,
                   help="After thresholding, dilate the face mask N rings "
                        "via mesh adjacency. Grows the silhouette and fills "
                        "interior gaps; 1-3 is typical.")
    p.add_argument("--dilate-min-neighbors", type=int, default=1,
                   help="A False face becomes True only if it has at least "
                        "this many True neighbors. 1 = aggressive, 2 = "
                        "conservative.")
    p.add_argument("--fill-holes", action="store_true",
                   help="Run trimesh.fill_holes() on each extracted sub-mesh "
                        "to patch interior holes geometrically.")

    # Export knobs
    p.add_argument("--no-obj", action="store_true",
                   help="Skip OBJ+MTL+PNG sidecars (the MeshLab-friendly default).")
    p.add_argument("--no-glb", action="store_true")
    p.add_argument("--no-ply", action="store_true")
    p.add_argument("--no-remainder", action="store_true",
                   help="Skip the scene-minus-object output.")

    # Speed
    p.add_argument("--rerun-sam3", action="store_true",
                   help="Force SAM3 re-run even if masks PNGs exist.")
    p.add_argument("--scale-override", type=float, default=None,
                   help="Manually set COLMAP-to-metric scale (debug).")
    return p.parse_args()


def _existing_masks(masks_dir: Path, view_names: List[str], prompts: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Reload previously-saved masks from disk. Returns {prompt: {view: bool}}."""
    from PIL import Image
    out: Dict[str, Dict[str, np.ndarray]] = {p: {} for p in prompts}
    for prompt in prompts:
        slug = _slugify(prompt)
        for view_name in view_names:
            mp = masks_dir / f"{view_name}__{slug}.png"
            if mp.exists():
                arr = np.array(Image.open(mp).convert("L"))
                out[prompt][view_name] = arr > 128
    return out


def main():
    args = _parse_args()
    t_start = time.time()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        print("ERROR: no prompts given", file=sys.stderr)
        return 2

    # Atlas auto-detect
    atlas_path = args.atlas
    if atlas_path is None:
        cand = args.mesh.parent / "scene_textured0.png"
        if cand.exists():
            atlas_path = cand

    print(f"=== scene_segmenter ===")
    print(f"  mesh:         {args.mesh}")
    print(f"  atlas:        {atlas_path}")
    print(f"  project root: {args.project_root.resolve()}")
    print(f"  prompts:      {prompts}")
    print(f"  output:       {out_dir}")

    # -------------------- Phase 0: load views + mesh --------------------
    print("\n[phase 0] Loading views + mesh ...")
    src = V32ViewSource(args.project_root, scale_override=args.scale_override)
    print(f"  views available: {len(src)}")
    views = src.sample(args.max_views) if args.max_views > 0 else src.all_views()
    print(f"  views to use:    {len(views)}")
    print(f"  scale (COLMAP-> metric): {src.colmap_to_metric:.6f}  "
          f"(transforms.json translations divided by this)")

    import trimesh
    print(f"  loading mesh {args.mesh.name} ...")
    mesh = trimesh.load(str(args.mesh), force="mesh", process=False)
    print(f"  mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

    # ---- Bind texture atlas as a TextureVisuals -----------------------
    # trimesh.load on a PLY loads the UV coords but does NOT attach the
    # sibling PNG as a material image. Without this binding, sub-mesh
    # exports lose all texture (the user reported plain/untextured GLBs
    # on the first run). Re-wrap the visual with the atlas image so the
    # material flows through submesh + GLB embedding cleanly.
    if atlas_path is not None and atlas_path.exists():
        try:
            from PIL import Image as _PILImage
            atlas_img = _PILImage.open(atlas_path).convert("RGB")
            uv = None
            if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
                uv = mesh.visual.uv
            elif hasattr(mesh, "vertex_attributes") and "uv" in (mesh.vertex_attributes or {}):
                uv = mesh.vertex_attributes["uv"]
            if uv is not None and len(uv) == len(mesh.vertices):
                mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=atlas_img)
                print(f"  bound atlas {atlas_path.name} ({atlas_img.size[0]}x{atlas_img.size[1]}) "
                      f"to mesh as TextureVisuals")
            else:
                print(f"  WARN: atlas present but mesh has no usable UV coords "
                      f"(uv={uv is not None and len(uv) if uv is not None else 'None'}, "
                      f"verts={len(mesh.vertices)}). Output will be untextured.")
        except Exception as e:
            print(f"  WARN: failed to bind atlas: {e}")
    else:
        print(f"  no atlas bound -> outputs will be untextured")

    # -------------------- Phase 1: SAM3 segmentation --------------------
    image_paths = [v.image_path for v in views]
    view_names = [v.name for v in views]

    need_sam3 = args.rerun_sam3
    if not need_sam3 and not masks_dir.exists():
        need_sam3 = True
    if not need_sam3:
        # Verify all expected mask files exist
        for prompt in prompts:
            slug = _slugify(prompt)
            for vn in view_names:
                if not (masks_dir / f"{vn}__{slug}.png").exists():
                    need_sam3 = True
                    break
            if need_sam3:
                break

    if need_sam3:
        print(f"\n[phase 1] SAM3 segmentation ({len(image_paths)} views x {len(prompts)} prompts) ...")
        t0 = time.time()
        results = segment_multi_prompt(
            image_paths=image_paths,
            prompts=prompts,
            masks_dir=masks_dir,
            mode=args.sam3_mode,
            threshold=args.sam3_threshold,
            erode_px=args.erode_px,
            device=args.device,
            verbose=True,
        )
        print(f"[phase 1] done in {time.time()-t0:.1f}s")
    else:
        print(f"\n[phase 1] reloading {len(prompts) * len(view_names)} existing masks from {masks_dir} ...")
        results = _existing_masks(masks_dir, view_names, prompts)

    # -------------------- Phase 2: extract + export per prompt --------------------
    summary: Dict[str, Dict] = {}
    for prompt in prompts:
        masks_by_view = results.get(prompt, {})
        if not masks_by_view:
            print(f"\n[phase 2] '{prompt}': no masks - skipping")
            continue

        ext_res = extract_object(
            mesh, views, masks_by_view,
            prompt=prompt,
            score_threshold=args.score_threshold,
            min_component_fraction=args.min_component_fraction,
            z_buffer_tol=args.z_buffer_tol,
            min_view_angle_cos=args.min_view_angle_cos,
            smooth_iterations=args.smooth_iterations,
            smooth_alpha=args.smooth_alpha,
            dilate_iterations=args.dilate_iterations,
            dilate_min_neighbors=args.dilate_min_neighbors,
            samples_per_face=args.samples_per_face,
            use_rasterizer=not args.no_rasterize,
            min_pixel_count=args.min_pixel_count,
            mask_margin_px=args.mask_margin_px,
            apply_cc_filter=not args.no_cc_filter,
            verbose=True,
        )

        if not ext_res.face_mask.any():
            print(f"  '{prompt}': zero faces matched - skipping export")
            summary[prompt] = {"status": "no faces", "n_views": ext_res.n_views_contributing}
            continue

        prompt_slug = _slugify(prompt)
        prompt_dir = out_dir / prompt_slug
        files = export_extract_and_remove(
            mesh, ext_res.face_mask,
            out_dir=prompt_dir,
            base_name=prompt_slug,
            write_remainder=not args.no_remainder,
            write_obj=not args.no_obj,
            write_glb=not args.no_glb,
            write_ply=not args.no_ply,
            fill_holes=args.fill_holes,
            texture_atlas_src=atlas_path,
            verbose=True,
        )

        summary[prompt] = {
            "status": "ok",
            "n_views_used": ext_res.n_views_contributing,
            "n_components_before_cc": ext_res.n_components_before_cc,
            "n_components_after_cc":  ext_res.n_components_after_cc,
            "n_faces_total": ext_res.n_faces_total,
            "n_faces_extracted": int(ext_res.face_mask.sum()),
            "fraction": float(ext_res.face_mask.mean()),
            "outputs": {k: {fmt: str(p) for fmt, p in d.items()} for k, d in files.items()},
        }

    # -------------------- Phase 3: summary --------------------
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Done in {time.time()-t_start:.1f}s ===")
    print(f"  summary: {summary_path}")
    for prompt, info in summary.items():
        if info.get("status") == "ok":
            print(f"  {prompt:>16s}: {info['n_faces_extracted']:,} faces "
                  f"({info['fraction']*100:.1f}% of mesh), "
                  f"{info['n_components_after_cc']} component(s)")
        else:
            print(f"  {prompt:>16s}: {info.get('status', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
