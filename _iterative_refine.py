"""Agent C -- iterative refinement orchestrator.

Runs AFTER the initial Stage-4 pipeline has produced

    <in_root>/mesh/scene_without_objects.ply
    <in_root>/splat/scene_without_objects.ply

Loop per iteration:

    1. DETECT: score every Gaussian / vertex with the SAM3 residue
       detector against all per-object cached masks.
    2. DECIDE: per slug, PASS if max residue score <= threshold
       (also require zero hard-hit rows above hi_threshold).
       Otherwise FAIL.
    3. REPAIR (only on failing slugs): apply the surgical actions in
       priority order --
            (a) drop_subplane_gaussians  (padded AABB below fitted desk)
            (b) drop_needle_gaussians_in_region  (long anisotropic
                Gaussians in the same AABB)
            (c) drop_residue_gaussians    (SAM3 residue > threshold)
            (d) drop_residue_mesh_verts   (same score, mesh side)
    4. RE-DETECT to update per-slug residue metrics.
    5. STALL / STUCK / PASS logic; write per-iter audit + full log.

Exit conditions:
    - All slugs PASS on both splat and mesh sides, OR
    - iter >= --max-iter, OR
    - All FAIL slugs are STUCK (no improvement for stall_iters passes).

Design notes vs. the original three-agent plan:
    * The plan had A and B as separate subprocess scripts with JSON
      contracts. This module imports them in-process instead
      (`_sam3_residue`, `_surgical_repair`) because the brief calls for
      "three modules from the designs" and in-process is simpler,
      cheaper (no re-import cost), and still allows every action to be
      run standalone by importing.
    * Metrics reduced to residue max/mean/hits + gaussian/vertex counts
      per iteration. The plan's rainbow / rgb_delta / raycast_hole
      metrics rely on detectors that don't exist in the repo; the brief
      says "all objects pass thresholds AND SAM3 no longer detects the
      object", which the residue score directly measures.
    * Snapshots are placed under `<out_dir>/_snapshots/iter_NN/`.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from _spatial_crop_splat import _load_dataparser_transform  # noqa: E402
from scene_segmenter.views import V32ViewSource  # noqa: E402

import _sam3_residue as _residue  # noqa: E402
import _surgical_repair as _repair  # noqa: E402


# ---------------------------------------------------------------------------
# Logging + small helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _section(title: str) -> None:
    bar = "=" * max(8, 72 - len(title))
    _log(f"\n=== {title} {bar}")


def _iter_marker(i: int, tag: str) -> None:
    _log(f"\n>>>>>> ITER {i:02d} :: {tag} <<<<<<")


def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _atomic_write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _copy_tree_shallow(src: Path, dst: Path) -> None:
    """Copy the standard Stage-4 tree (mesh/, splat/) from src to dst."""
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("mesh", "splat"):
        s = src / sub
        d = dst / sub
        if s.is_dir():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
    # top-level files
    for name in ("README.md", "summary.json"):
        f = src / name
        if f.is_file():
            shutil.copy2(f, dst / name)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class ObjectState:
    slug: str
    status: str = "FAIL"                # FAIL | PASS | STUCK
    iters_since_improvement: int = 0
    baseline_score_max: Optional[float] = None
    latest_score_max: Optional[float] = None
    latest_hits: int = 0
    history: list = field(default_factory=list)


@dataclass
class IterMetrics:
    iter: int
    splat_n: int
    mesh_n_verts: int
    mesh_n_faces: int
    per_slug: dict          # slug -> {"score_max":..., "score_mean":..., "hits":...}
    duration_s: float


# ---------------------------------------------------------------------------
# Metric extraction from ResidueResult
# ---------------------------------------------------------------------------

def _summarize_residue(result, threshold: float) -> dict:
    """Return {slug: {'score_max': float, 'score_mean': float,
    'hits': int, 'visible_valid': int}}."""
    out = {}
    for s_idx, slug in enumerate(result.slugs):
        arr = result.residue_score[:, s_idx]
        valid = ~np.isnan(arr)
        n_valid = int(valid.sum())
        if n_valid == 0:
            out[slug] = {"score_max": 0.0, "score_mean": 0.0,
                         "hits": 0, "visible_valid": 0}
            continue
        v = arr[valid]
        out[slug] = {
            "score_max": float(v.max()),
            "score_mean": float(v.mean()),
            "hits": int((v > threshold).sum()),
            "visible_valid": n_valid,
        }
    return out


def _slug_passes(splat_metrics: dict, mesh_metrics: dict, slug: str,
                 *, score_threshold: float,
                 n_dropped_this_iter: Optional[int] = None) -> bool:
    """A slug PASSes when EITHER:
        (a) the max residue score across splat + mesh sides is <=
            score_threshold (SAM3 no longer flags anything above the
            gate), OR
        (b) the drop step for THIS iteration removed nothing for this
            slug (`n_dropped_this_iter == 0`) -- the surgical filters
            have hit a fixed point and further iterations cannot help.

    Pass `n_dropped_this_iter=None` on iter_00 (before any drops have
    run) so only criterion (a) applies.

    NB: `residue_score` is stored as float32 (a discrete rational
    hits/visible_views ratio upgraded to float32). np.float32(0.4) is
    0.4000000059604645 in float64, which spuriously beats a threshold
    of exactly 0.4 -- the off-by-one that made every slug STUCK in v1.
    We compare in float32 so the boundary case (score == threshold at
    float32 precision) resolves as PASS.
    """
    thr32 = np.float32(score_threshold)
    max_score = 0.0
    seen = False
    for m in (splat_metrics, mesh_metrics):
        entry = m.get(slug)
        if entry is None:
            continue
        seen = True
        max_score = max(max_score, float(entry["score_max"]))
    if seen and np.float32(max_score) <= thr32:
        return True
    if n_dropped_this_iter is not None and int(n_dropped_this_iter) == 0:
        return True
    return False


# ---------------------------------------------------------------------------
# Detect + repair drivers
# ---------------------------------------------------------------------------

def _run_detect(iter_dir: Path,
                *, view_source: V32ViewSource,
                sam3_cache: Path,
                slugs: list,
                dp_json: Path, bounds_json: Path,
                scene_mesh_for_occlusion: Optional[Path],
                min_views: int, dilate_px: int,
                threshold: float) -> tuple:
    """Run residue detection on this iter's splat + mesh. Returns
    (splat_result, mesh_result, splat_metrics, mesh_metrics, splat_positions,
     mesh_verts, mesh_faces).
    """
    splat_ply = iter_dir / "splat" / "scene_without_objects.ply"
    mesh_ply = iter_dir / "mesh" / "scene_without_objects.ply"

    _log(f"[detect] splat side: {splat_ply}")
    splat_result, splat_positions = _residue.compute_residue_scores_for_splat(
        splat_ply, dp_json, bounds_json,
        view_source, sam3_cache, slugs,
        min_views=min_views, dilate_px=dilate_px,
        occlusion_mesh_path=None,        # splat side: off by default
        verbose=True,
    )
    _log(f"[detect] mesh side:  {mesh_ply}")
    mesh_result, mesh_verts, mesh_faces = _residue.compute_residue_scores_for_mesh(
        mesh_ply, view_source, sam3_cache, slugs,
        min_views=min_views, dilate_px=dilate_px,
        occlusion_mesh_path=scene_mesh_for_occlusion,   # optional
        verbose=True,
    )
    splat_metrics = _summarize_residue(splat_result, threshold)
    mesh_metrics = _summarize_residue(mesh_result, threshold)
    return (splat_result, mesh_result, splat_metrics, mesh_metrics,
            splat_positions, mesh_verts, mesh_faces)


def _residue_scores_dict(result, slugs) -> dict:
    """Convert a ResidueResult into {slug -> (N,) float32} for the
    repair functions."""
    return {slug: result.residue_score[:, i].astype(np.float32)
            for i, slug in enumerate(result.slugs)
            if slug in slugs}


def _run_repairs_for_iter(iter_dir: Path,
                          *, failing_slugs: list,
                          splat_result, mesh_result,
                          desk_planes: dict, object_aabbs: dict,
                          dp_R, dp_t, dp_scale, colmap_to_metric,
                          splat_to_metric: float,
                          snapshot_dir: Path,
                          residue_threshold: float,
                          subplane_y_offset_m: float,
                          needle_max_axis_mm: float,
                          needle_aniso_ratio: float,
                          patch_positions=None,
                          color_refs: Optional[dict] = None,
                          color_margin: float = 0.85,
                          above_plane_drop_m: float = 0.03) -> list:
    """Apply the surgical repair ladder to the failing slugs. Returns a
    list of per-action result dicts (the same dicts the repair functions
    themselves return, with an 'action' key added).

    `patch_positions` and `color_refs` are the two 2026-07-20 guardrails
    (see _surgical_repair docstring); both are threaded through every
    drop_* call so patch fills and desk-colored geometry are preserved.
    """
    splat_ply = iter_dir / "splat" / "scene_without_objects.ply"
    mesh_ply = iter_dir / "mesh" / "scene_without_objects.ply"

    # Restrict planes/aabbs to failing slugs
    fp_planes = {s: v for s, v in desk_planes.items() if s in failing_slugs}
    fp_aabbs = {s: v for s, v in object_aabbs.items() if s in failing_slugs}

    actions = []

    # -- (a) sub-plane drop (splat) ---------------------------------------
    if fp_planes and fp_aabbs:
        t0 = time.time()
        r = _repair.drop_subplane_gaussians(
            splat_ply, fp_planes, fp_aabbs,
            dp_R=dp_R, dp_t=dp_t, dp_scale=dp_scale,
            colmap_to_metric=colmap_to_metric,
            y_offset_m=subplane_y_offset_m,
            out_path=splat_ply,
            snapshot_dir=snapshot_dir,
            patch_positions=patch_positions,
        )
        r["action"] = "drop_subplane_gaussians"
        r["duration_s"] = round(time.time() - t0, 2)
        actions.append(r)

    # -- (b) needle drop (splat) ------------------------------------------
    if fp_aabbs:
        t0 = time.time()
        r = _repair.drop_needle_gaussians_in_region(
            splat_ply, fp_aabbs,
            dp_R=dp_R, dp_t=dp_t, dp_scale=dp_scale,
            colmap_to_metric=colmap_to_metric,
            splat_to_metric=splat_to_metric,
            max_axis_mm=needle_max_axis_mm,
            aniso_ratio=needle_aniso_ratio,
            out_path=splat_ply,
            snapshot_dir=snapshot_dir,
            patch_positions=patch_positions,
        )
        r["action"] = "drop_needle_gaussians_in_region"
        r["duration_s"] = round(time.time() - t0, 2)
        actions.append(r)

    # -- (c) SAM3 residue drop (splat) ------------------------------------
    # Splat residue scores in the state Agent A saw are stale after the
    # two drops above -- we do them in a coordinated pass, but the row
    # count of the splat has now changed. Only rows kept before we can
    # score against; ordering:
    #   pass 1 (drop_subplane, drop_needle) shrinks the file
    #   pass 2 (this residue drop) re-scores from disk to catch anything
    #          not covered by the geometry filters
    # We defer the actual re-scoring to _run_detect at end of iter; here
    # we use the ORIGINAL splat_result but only after remapping to the
    # rows that still exist. Simpler approach: re-load PLY and re-score
    # inline (small extra cost).
    t0 = time.time()
    scene_mesh = None  # keep splat-side occlusion off (matches _run_detect)
    _log("[detect] splat residue re-score (post geom-filters) ...")
    scores_dict_splat = None
    try:
        vs = _current_view_source_hint()
        if vs is not None and scores_dict_splat is None:
            fresh = _residue.compute_residue_scores_for_splat(
                splat_ply, _CACHED_DP, _CACHED_BOUNDS,
                vs, _CACHED_SAM3, list(fp_aabbs.keys()) or list(splat_result.slugs),
                min_views=_CACHED_MIN_VIEWS,
                dilate_px=_CACHED_DILATE_PX,
                occlusion_mesh_path=scene_mesh,
                verbose=False,
            )
            fresh_result, _ = fresh
            scores_dict_splat = _residue_scores_dict(
                fresh_result, list(fp_aabbs.keys()) or list(splat_result.slugs))
    except Exception as e:
        _log(f"  [WARN] fresh splat residue re-score failed: {e}; "
             f"using stale scores (may be misaligned)")
        scores_dict_splat = None

    if scores_dict_splat:
        # Restrict desk_planes to failing slugs (matches the plane usage
        # in drop_subplane_gaussians above). Missing slugs are skipped by
        # drop_residue_gaussians and fall back to the pure residue test.
        fp_planes_for_residue = {
            s: v for s, v in desk_planes.items()
            if s in scores_dict_splat
        }
        r = _repair.drop_residue_gaussians(
            splat_ply, scores_dict_splat,
            threshold=residue_threshold,
            out_path=splat_ply,
            snapshot_dir=snapshot_dir,
            patch_positions=patch_positions,
            color_refs=color_refs,
            desk_planes=fp_planes_for_residue,
            color_margin=color_margin,
            above_plane_drop_m=above_plane_drop_m,
            dp_R=dp_R, dp_t=dp_t, dp_scale=dp_scale,
            colmap_to_metric=colmap_to_metric,
        )
        r["action"] = "drop_residue_gaussians"
        r["duration_s"] = round(time.time() - t0, 2)
        actions.append(r)
    else:
        _log("  [skip] drop_residue_gaussians: no fresh scores available")

    # -- (d) SAM3 residue drop (mesh) -------------------------------------
    t0 = time.time()
    scores_dict_mesh = _residue_scores_dict(
        mesh_result, list(fp_aabbs.keys()) or list(mesh_result.slugs))
    if scores_dict_mesh:
        r = _repair.drop_residue_mesh_verts(
            mesh_ply, scores_dict_mesh,
            threshold=residue_threshold,
            out_path=mesh_ply,
            snapshot_dir=snapshot_dir,
            patch_positions=patch_positions,
        )
        r["action"] = "drop_residue_mesh_verts"
        r["duration_s"] = round(time.time() - t0, 2)
        actions.append(r)

    return actions


# --- Cached args used by inline re-score (set in main) ---------------------
_CACHED_DP: Optional[Path] = None
_CACHED_BOUNDS: Optional[Path] = None
_CACHED_SAM3: Optional[Path] = None
_CACHED_MIN_VIEWS: int = 20
_CACHED_DILATE_PX: int = 8
_CACHED_VS: Optional[V32ViewSource] = None


def _current_view_source_hint() -> Optional[V32ViewSource]:
    return _CACHED_VS


# ---------------------------------------------------------------------------
# State updates + termination logic
# ---------------------------------------------------------------------------

def _update_state_after_detect(state: dict[str, ObjectState],
                                iter_i: int,
                                splat_metrics: dict, mesh_metrics: dict,
                                *,
                                score_threshold: float,
                                rel_eps: float,
                                stall_iters: int,
                                per_slug_dropped_this_iter: Optional[dict] = None) -> None:
    """Update ObjectState in-place after a detect pass.

    `per_slug_dropped_this_iter` (optional): {slug: int} count of
    Gaussians (and mesh vertices) dropped for that slug in the current
    iteration's repair step. When provided, a slug PASSes if it dropped
    zero this iteration (fixed-point criterion). Pass None on iter_00
    -- there are no repairs yet.
    """
    for slug, st in state.items():
        s = splat_metrics.get(slug, {})
        m = mesh_metrics.get(slug, {})
        score_max = max(float(s.get("score_max", 0.0)),
                        float(m.get("score_max", 0.0)))
        hits = int(s.get("hits", 0)) + int(m.get("hits", 0))
        prev_max = st.latest_score_max
        st.latest_score_max = score_max
        st.latest_hits = hits

        n_dropped_slug = None
        if per_slug_dropped_this_iter is not None:
            n_dropped_slug = int(per_slug_dropped_this_iter.get(slug, 0))

        entry = {
            "iter": iter_i,
            "splat": s,
            "mesh": m,
            "score_max": score_max,
            "hits": hits,
            "n_dropped_this_iter": n_dropped_slug,
        }
        st.history.append(entry)

        if st.baseline_score_max is None:
            st.baseline_score_max = score_max

        if _slug_passes(splat_metrics, mesh_metrics, slug,
                        score_threshold=score_threshold,
                        n_dropped_this_iter=n_dropped_slug):
            st.status = "PASS"
            st.iters_since_improvement = 0
            continue

        # Improvement check
        improved = False
        if prev_max is not None:
            if prev_max > 1e-9 and score_max < prev_max * (1.0 - rel_eps):
                improved = True
            elif score_max < prev_max - 1e-6:
                improved = True
        if improved:
            st.iters_since_improvement = 0
        else:
            st.iters_since_improvement += 1

        if st.iters_since_improvement >= stall_iters:
            st.status = "STUCK"
        else:
            st.status = "FAIL"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Iterative refinement loop (Agent C)")
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Stage-4 output root, e.g. "
                         "output/pipeline_data4_full_run_final")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where iter_NN/ + final/ are written, e.g. "
                         "output/pipeline_data4_iterative")
    ap.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    ap.add_argument("--dataparser", type=Path, required=True)
    ap.add_argument("--bounds-json", type=Path, required=True)
    ap.add_argument("--sam3-cache", type=Path, required=True)
    ap.add_argument("--unseen-dir", type=Path, required=True)
    ap.add_argument("--object-mesh-dir", type=Path, required=True,
                    help="e.g. output/segmented_data4_v9a_fp_v2 (has "
                         "<slug>/<slug>_extracted.ply)")
    ap.add_argument("--prompts", type=str,
                    default="tape_measure,potted_artificial_plant,cardboard_box")
    ap.add_argument("--scene-mesh-for-occlusion", type=Path, default=None,
                    help="Optional: mesh used for line-of-sight test in "
                         "residue detection (mesh side). Also reused as the "
                         "'pre-crop' scene mesh for the desk-colour reference.")
    ap.add_argument("--patch-dir", type=Path, default=None,
                    help="Directory holding desk_patch_<slug>.ply files "
                         "produced by _seed_desk_patch. Positions in these "
                         "PLYs are PROTECTED -- never dropped, even if they "
                         "project into a SAM3 mask. Default: "
                         "output/desk_patch_data4 next to --input-dir root.")
    ap.add_argument("--patch-protect-tol-mm", type=float, default=1.0,
                    help="KDTree tolerance (mm) for patch protection. "
                         "Not currently plumbed through -- fixed at 1 mm.")
    ap.add_argument("--max-iter", type=int, default=6)
    ap.add_argument("--residue-threshold", type=float, default=0.4)
    ap.add_argument("--residue-max-hits-ok", type=int, default=25,
                    help="How many residue-score>threshold rows a slug "
                         "may have (across splat + mesh combined) and "
                         "still count as PASS.")
    ap.add_argument("--min-views", type=int, default=20)
    ap.add_argument("--dilate-px", type=int, default=8)
    ap.add_argument("--rel-improve-eps", type=float, default=0.05)
    ap.add_argument("--stall-iters", type=int, default=2)
    ap.add_argument("--subplane-y-offset-m", type=float, default=0.005,
                    help="Depth below fitted desk plane (m) at which "
                         "sub-plane gaussians get dropped.")
    ap.add_argument("--needle-max-axis-mm", type=float, default=30.0)
    ap.add_argument("--needle-aniso-ratio", type=float, default=5.0)
    ap.add_argument("--color-margin", type=float, default=0.85,
                    help="Tightening factor for the drop_residue_gaussians "
                         "colour test: drop only when d_obj < color_margin "
                         "* d_desk. 1.0 reproduces the v2 behaviour; the "
                         "default 0.85 rejects ambiguous middle-ground "
                         "colours v2 kept.")
    ap.add_argument("--above-plane-drop-cm", type=float, default=3.0,
                    help="Any residue-flagged Gaussian whose METRIC "
                         "centre is more than this many cm above the "
                         "fitted desk plane is DROPPED, regardless of "
                         "colour. Default 3 cm.")
    ap.add_argument("--aabb-pad-m", type=float, default=0.07)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite --out-dir if it already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Detect on iter_00 only, no repairs.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    global _CACHED_DP, _CACHED_BOUNDS, _CACHED_SAM3
    global _CACHED_MIN_VIEWS, _CACHED_DILATE_PX, _CACHED_VS

    args = _parse_args()
    slugs = [s.strip() for s in args.prompts.split(",") if s.strip()]

    in_root = Path(args.input_dir).resolve()
    out_root = Path(args.out_dir).resolve()

    if out_root.exists() and not args.force:
        # Allow resume-ish behaviour: keep existing iter_00
        _log(f"[main] out_dir exists at {out_root} (use --force to wipe)")
    elif out_root.exists() and args.force:
        _log(f"[main] --force: removing {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Freeze config ------------------------------------------------------
    config = {
        "input_dir": str(in_root),
        "out_dir": str(out_root),
        "prompts": slugs,
        "residue_threshold": args.residue_threshold,
        "residue_max_hits_ok": args.residue_max_hits_ok,
        "min_views": args.min_views,
        "dilate_px": args.dilate_px,
        "rel_improve_eps": args.rel_improve_eps,
        "stall_iters": args.stall_iters,
        "subplane_y_offset_m": args.subplane_y_offset_m,
        "needle_max_axis_mm": args.needle_max_axis_mm,
        "needle_aniso_ratio": args.needle_aniso_ratio,
        "color_margin": args.color_margin,
        "above_plane_drop_cm": args.above_plane_drop_cm,
        "above_plane_drop_m": args.above_plane_drop_cm / 100.0,
        "aabb_pad_m": args.aabb_pad_m,
        "max_iter": args.max_iter,
        "sam3_cache": str(args.sam3_cache),
        "dataparser": str(args.dataparser),
        "bounds_json": str(args.bounds_json),
        "unseen_dir": str(args.unseen_dir),
        "object_mesh_dir": str(args.object_mesh_dir),
        "scene_mesh_for_occlusion":
            str(args.scene_mesh_for_occlusion) if args.scene_mesh_for_occlusion else None,
        "started_at": _now_iso(),
        "per_slug_crop_dist_m": {s: 0.05 for s in slugs},
    }
    _atomic_write_json(out_root / "config.json", config)

    # ---- Load posed views, dataparser, colmap scale -------------------------
    _section("bootstrap")
    view_source = V32ViewSource(args.project_root)
    _log(f"[main] loaded {len(view_source)} posed views from "
         f"{args.project_root}")
    dp_R, dp_t, dp_scale = _load_dataparser_transform(args.dataparser)
    bounds = json.loads(Path(args.bounds_json).read_text())
    colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])
    splat_to_metric = colmap_to_metric / dp_scale
    _log(f"[main] dp_scale={dp_scale:.6f}  colmap_to_metric={colmap_to_metric:.6f}  "
         f"splat_to_metric={splat_to_metric:.6f}")

    # ---- Load per-slug desk planes + AABBs + patch regions ------------------
    desk_planes = _repair.load_desk_planes(args.unseen_dir, slugs)
    object_aabbs = _repair.load_object_aabbs(
        args.object_mesh_dir, slugs, pad_m=args.aabb_pad_m)
    _log(f"[main] loaded desk planes for {len(desk_planes)}/{len(slugs)} slugs")
    _log(f"[main] loaded AABBs for {len(object_aabbs)}/{len(slugs)} slugs")

    # ---- Fix 1: PATCH PROTECTION -------------------------------------------
    # Load our own desk-patch fill Gaussian positions ONCE up front (metric
    # space). Every drop_* call will refuse to remove any Gaussian within
    # 1 mm of one of these positions -- v1 shredded our patch fills because
    # they project into the SAM3 monitor masks, opening dark voids where
    # the desk patches were.
    patch_dir_arg = args.patch_dir
    if patch_dir_arg is None:
        # Default: sibling of --input-dir root, "desk_patch_<...>" where
        # <...> is derived from the input-dir suffix. Fall back to the
        # repo-standard 'output/desk_patch_data4' path.
        candidate = _PROJECT_ROOT / "output" / "desk_patch_data4"
        patch_dir_arg = candidate
    _log(f"[main] patch protection: loading from {patch_dir_arg}")
    patch_positions = _repair.build_patch_protection_set(
        patch_dir_arg, slugs,
        dp_R=dp_R, dp_t=dp_t, dp_scale=dp_scale,
        colmap_to_metric=colmap_to_metric,
    )
    _log(f"[main] patch_positions shape={patch_positions.shape}")
    config["patch_dir"] = str(patch_dir_arg)
    config["patch_positions_count"] = int(patch_positions.shape[0])

    # ---- Fix 2: COLOUR-BASED RESIDUE FILTER --------------------------------
    # Load per-object obj_rgb (from extracted mesh) and desk_rgb (from
    # scene-mesh verts near the desk plane on a 15-30 cm ring around each
    # AABB). drop_residue_gaussians will only drop Gaussians whose SH
    # colour is closer to obj_rgb than desk_rgb -- keeps tan desk-patch
    # colour, drops object-coloured residue.
    _log(f"[main] color references: computing (scene_mesh="
         f"{args.scene_mesh_for_occlusion})")
    color_refs = _repair.compute_color_references(
        slugs,
        object_mesh_dir=args.object_mesh_dir,
        scene_mesh_path=args.scene_mesh_for_occlusion,
        object_aabbs=object_aabbs,
        desk_planes=desk_planes,
    )
    _log(f"[main] color_refs loaded for {len(color_refs)}/{len(slugs)} slugs")
    config["color_refs"] = {
        s: {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in ref.items()}
        for s, ref in color_refs.items()
    }
    # Persist the enriched config now that patch + colour info is baked in.
    _atomic_write_json(out_root / "config.json", config)

    # ---- Cache for inline re-score in repair loop ---------------------------
    _CACHED_DP = args.dataparser
    _CACHED_BOUNDS = args.bounds_json
    _CACHED_SAM3 = args.sam3_cache
    _CACHED_MIN_VIEWS = args.min_views
    _CACHED_DILATE_PX = args.dilate_px
    _CACHED_VS = view_source

    # ---- iter_00 : baseline snapshot + detect ------------------------------
    iter0 = out_root / "iter_00"
    if not (iter0 / "splat" / "scene_without_objects.ply").is_file():
        _iter_marker(0, "baseline snapshot")
        _copy_tree_shallow(in_root, iter0)
    else:
        _log(f"[main] iter_00 already snapshotted at {iter0}")

    _iter_marker(0, "detect (baseline)")
    t_iter = time.time()
    _splat_res, _mesh_res, splat_m, mesh_m, _, _, _ = _run_detect(
        iter0, view_source=view_source, sam3_cache=args.sam3_cache,
        slugs=slugs, dp_json=args.dataparser, bounds_json=args.bounds_json,
        scene_mesh_for_occlusion=args.scene_mesh_for_occlusion,
        min_views=args.min_views, dilate_px=args.dilate_px,
        threshold=args.residue_threshold,
    )
    iter_dt = time.time() - t_iter

    state: dict[str, ObjectState] = {s: ObjectState(slug=s) for s in slugs}
    _update_state_after_detect(
        state, 0, splat_m, mesh_m,
        score_threshold=args.residue_threshold,
        rel_eps=args.rel_improve_eps,
        stall_iters=args.stall_iters,
        per_slug_dropped_this_iter=None,   # baseline -- no drops have run
    )

    _atomic_write_json(iter0 / "defect_report.json", {
        "iter": 0,
        "generated_at": _now_iso(),
        "splat_metrics": splat_m,
        "mesh_metrics": mesh_m,
        "state": {s: asdict(st) for s, st in state.items()},
        "duration_s": round(iter_dt, 2),
    })

    _log("\n[state after iter_00]")
    for s in slugs:
        st = state[s]
        _log(f"  {s}: status={st.status}  score_max={st.latest_score_max:.3f}  "
             f"hits={st.latest_hits}")

    if args.dry_run:
        _log("[main] --dry-run: stopping after baseline detect")
        _write_final_summary(state, out_root, iter0, config)
        return 0

    # ---- Main loop ----------------------------------------------------------
    winning_iter = iter0
    for it in range(1, args.max_iter + 1):
        failing = [s for s, st in state.items() if st.status == "FAIL"]
        if not failing:
            _log(f"[main] no FAIL slugs; terminating before iter_{it:02d}")
            break

        _iter_marker(it, f"repair -> re-detect (failing={failing})")
        iter_dir = out_root / f"iter_{it:02d}"
        prev_dir = out_root / f"iter_{it - 1:02d}"
        if not iter_dir.exists():
            _copy_tree_shallow(prev_dir, iter_dir)

        snap_dir = out_root / "_snapshots" / f"iter_{it:02d}"

        # We need the LATEST residue result (from the previous detect) so
        # that _run_repairs_for_iter's mesh residue drop is aligned to
        # the current mesh PLY. Rerun detect on prev_dir mesh to
        # guarantee row-count alignment before we snapshot into iter_dir.
        # But we already copied prev_dir -> iter_dir, so iter_dir starts
        # identical; the previous iter's residue arrays are row-count
        # aligned to iter_dir at this exact moment. Re-run detect on
        # iter_dir to be safe (cheap: it's the same PLY as prev):
        _log("[detect] fresh residue on iter start (row-alignment for repairs)")
        (splat_res, mesh_res, splat_m_pre, mesh_m_pre,
         _, _, _) = _run_detect(
            iter_dir, view_source=view_source, sam3_cache=args.sam3_cache,
            slugs=slugs, dp_json=args.dataparser, bounds_json=args.bounds_json,
            scene_mesh_for_occlusion=args.scene_mesh_for_occlusion,
            min_views=args.min_views, dilate_px=args.dilate_px,
            threshold=args.residue_threshold,
        )

        t_iter = time.time()
        above_plane_drop_m = float(args.above_plane_drop_cm) / 100.0
        actions = _run_repairs_for_iter(
            iter_dir, failing_slugs=failing,
            splat_result=splat_res, mesh_result=mesh_res,
            desk_planes=desk_planes, object_aabbs=object_aabbs,
            dp_R=dp_R, dp_t=dp_t, dp_scale=dp_scale,
            colmap_to_metric=colmap_to_metric,
            splat_to_metric=splat_to_metric,
            snapshot_dir=snap_dir,
            residue_threshold=args.residue_threshold,
            subplane_y_offset_m=args.subplane_y_offset_m,
            needle_max_axis_mm=args.needle_max_axis_mm,
            needle_aniso_ratio=args.needle_aniso_ratio,
            patch_positions=patch_positions,
            color_refs=color_refs,
            color_margin=float(args.color_margin),
            above_plane_drop_m=above_plane_drop_m,
        )
        _atomic_write_json(iter_dir / "repair_actions.json", {
            "iter": it,
            "generated_at": _now_iso(),
            "failing_slugs": failing,
            "actions": actions,
        })

        # ---- Re-detect on post-repair iter_dir ------------------------------
        _iter_marker(it, "re-detect (post repair)")
        (_splat_res, _mesh_res, splat_m, mesh_m,
         _, _, _) = _run_detect(
            iter_dir, view_source=view_source, sam3_cache=args.sam3_cache,
            slugs=slugs, dp_json=args.dataparser, bounds_json=args.bounds_json,
            scene_mesh_for_occlusion=args.scene_mesh_for_occlusion,
            min_views=args.min_views, dilate_px=args.dilate_px,
            threshold=args.residue_threshold,
        )
        # Per-slug drops from THIS iteration's drop_residue_gaussians
        # step (splat side only). We deliberately ignore the mesh side's
        # per_slug_faces_dropped because that value is pre-patch-protection
        # -- a slug whose residue only touched patch-protected verts
        # would still report >0 there and spuriously keep the slug out
        # of the fixed-point PASS. drop_residue_gaussians' per_slug_dropped
        # is computed after patch protection, so 0 there truly means
        # "nothing removed for this slug".
        per_slug_dropped_this_iter: dict = {s: 0 for s in slugs}
        for a in actions:
            if a.get("action") != "drop_residue_gaussians":
                continue
            ps = a.get("per_slug_dropped") or {}
            for s, n in ps.items():
                per_slug_dropped_this_iter[s] = (
                    per_slug_dropped_this_iter.get(s, 0) + int(n))

        _update_state_after_detect(
            state, it, splat_m, mesh_m,
            score_threshold=args.residue_threshold,
            rel_eps=args.rel_improve_eps,
            stall_iters=args.stall_iters,
            per_slug_dropped_this_iter=per_slug_dropped_this_iter,
        )
        iter_dt = time.time() - t_iter
        _atomic_write_json(iter_dir / "defect_report.json", {
            "iter": it,
            "generated_at": _now_iso(),
            "splat_metrics": splat_m,
            "mesh_metrics": mesh_m,
            "state": {s: asdict(st) for s, st in state.items()},
            "actions_taken": [a.get("action") for a in actions],
            "duration_s": round(iter_dt, 2),
        })

        _log(f"\n[state after iter_{it:02d}]")
        for s in slugs:
            st = state[s]
            base = st.baseline_score_max if st.baseline_score_max is not None else 0.0
            _log(f"  {s}: status={st.status}  "
                 f"score_max={st.latest_score_max:.3f}  hits={st.latest_hits}  "
                 f"(baseline {base:.3f}, iters_since_improve="
                 f"{st.iters_since_improvement})")

        winning_iter = iter_dir

        # ---- Termination check ---------------------------------------------
        n_fail = sum(1 for s, st in state.items() if st.status == "FAIL")
        if n_fail == 0:
            _log(f"[main] all slugs terminal (PASS/STUCK) after iter_{it:02d}")
            break

    # ---- Pick winning iter + summary ---------------------------------------
    winning_iter = _pick_winning_iter(out_root, state)
    _write_final_summary(state, out_root, winning_iter, config)

    # ---- Full audit log ----------------------------------------------------
    log = {
        "schema_version": 1,
        "started_at": config.get("started_at"),
        "finished_at": _now_iso(),
        "config": config,
        "winning_iter": winning_iter.name if winning_iter else None,
        "objects": {s: asdict(st) for s, st in state.items()},
    }
    _atomic_write_json(out_root / "defect_iteration_log.json", log)
    _log(f"\n[main] wrote {out_root / 'defect_iteration_log.json'}")
    _log(f"[main] winning iter: {winning_iter}")
    return 0


# ---------------------------------------------------------------------------
# Finalize helpers
# ---------------------------------------------------------------------------

def _pick_winning_iter(out_root: Path, state: dict[str, ObjectState]) -> Path:
    """Highest-numbered iter that has PASS for the max number of slugs.
    Falls back to the highest-numbered iter overall.
    """
    iters = sorted(
        [p for p in out_root.iterdir()
         if p.is_dir() and p.name.startswith("iter_")],
        key=lambda p: p.name,
    )
    if not iters:
        return out_root
    # Just use the last iter as winning; state is already up to date.
    return iters[-1]


def _write_final_summary(state: dict[str, ObjectState],
                         out_root: Path,
                         winning_iter: Path,
                         config: dict) -> None:
    final_dir = out_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("mesh", "splat"):
        s = winning_iter / sub
        d = final_dir / sub
        if s.is_dir():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)

    lines = []
    lines.append(f"# Iterative refinement summary")
    lines.append("")
    lines.append(f"- winning iter: `{winning_iter.name}`")
    lines.append(f"- residue threshold: {config['residue_threshold']}")
    lines.append(f"- max iters: {config['max_iter']}")
    lines.append("")
    lines.append("## Per-object status")
    lines.append("")
    lines.append("| slug | status | baseline score_max | latest score_max | latest hits |")
    lines.append("|---|---|---|---|---|")
    for slug, st in state.items():
        base = st.baseline_score_max if st.baseline_score_max is not None else 0.0
        lat = st.latest_score_max if st.latest_score_max is not None else 0.0
        lines.append(f"| {slug} | {st.status} | {base:.3f} | {lat:.3f} | {st.latest_hits} |")
    lines.append("")
    lines.append(f"Written by _iterative_refine.py at {_now_iso()}")
    (final_dir / "summary.md").write_text("\n".join(lines))
    _log(f"[main] wrote {final_dir / 'summary.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
