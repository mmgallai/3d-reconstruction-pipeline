"""Direct projection-based object extraction from a Gaussian splat.

Bypasses the existing mesh-based extraction pipeline (which has coordinate-
transform issues with this splatfacto config). Uses the same camera setup as
the working renderer (_render_splat_views.py) — dp_transform + dp_scale
applied to camera translations — to project each Gaussian center through
every view and check against cached SAM3 masks.

Per-Gaussian classification:
  For each Gaussian i:
    inside_count[i,obj] = # views where projection(i) lands inside SAM3 mask for obj
    visible_count[i]    = # views where Gaussian passes near-plane + in-frustum
    score[i,obj]        = inside_count[i,obj] / max(visible_count[i], 1)

  Gaussian belongs to object OBJ if score[i,OBJ] >= keep_thresh
  Scene-without-objects = full splat MINUS { Gaussians assigned to ANY object }

Runs on GPU via torch batched projection. ~1 sec per (view, all Gaussians) pair.

Usage:
    python _extract_by_projection.py \
        --splat output/splat_v37_da3_pruned.ply \
        --transforms colmap/dense/transforms.json \
        --dataparser nerfstudio/dense/splatfacto/2026-07-27_001106/dataparser_transforms.json \
        --masks-dir output/segmented_room_v9a_fp_v2/masks \
        --prompts blue_armchair,black_subwoofer,wooden_coffee_table \
        --out-dir output/july/room_v2 \
        --min-views 5 --keep-thresh 0.30
"""
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement


def _slugify(s):
    s = re.sub(r"[^a-z0-9]+", "_", s.strip().lower())
    return re.sub(r"_+", "_", s).strip("_")[:40]


def load_ply(path):
    print(f"[load] {path}")
    ply = PlyData.read(path)
    v = ply["vertex"]
    N = len(v)
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    return v, xyz, N


def load_poses_all(transforms_json, dataparser_json):
    tj = json.loads(Path(transforms_json).read_text())
    dj = json.loads(Path(dataparser_json).read_text())
    dp_T = np.array(dj["transform"], dtype=np.float64)
    if dp_T.shape == (3, 4):
        dp_T = np.vstack([dp_T, [0.0, 0.0, 0.0, 1.0]])
    dp_scale = float(dj["scale"])

    W = int(tj.get("w"))
    H = int(tj.get("h"))
    fx = float(tj.get("fl_x"))
    fy = float(tj.get("fl_y"))
    cx = float(tj.get("cx"))
    cy = float(tj.get("cy"))

    frames = tj["frames"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    viewmats, names = [], []
    for fr in frames:
        c2w_gl = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w_new = dp_T @ c2w_gl
        c2w_new[:3, 3] *= dp_scale
        c2w_cv = c2w_new.copy()
        c2w_cv[:3, 1] *= -1
        c2w_cv[:3, 2] *= -1
        w2c = np.linalg.inv(c2w_cv).astype(np.float32)
        viewmats.append(w2c)
        names.append(Path(fr.get("file_path", "")).stem)
    return np.stack(viewmats), K, W, H, names


def project_batch(means_gpu, viewmats_gpu, K_gpu, W, H, near=0.01):
    """Project Gaussian centers into all views.

    Returns:
        inside_frustum: (N, V) bool — Gaussian is in front of camera and in frame
        pix_uv: (N, V, 2) int32 — pixel coords (u, v) rounded
    """
    N = means_gpu.shape[0]
    V = viewmats_gpu.shape[0]
    ones = torch.ones((N, 1), dtype=torch.float32, device=means_gpu.device)
    means_h = torch.cat([means_gpu, ones], dim=1)  # (N, 4)

    # For each view, transform points: cam = viewmat @ world; then project
    # viewmats: (V, 4, 4). Do this loop-per-view to avoid huge memory.
    inside_frustum = torch.zeros((N, V), dtype=torch.bool, device=means_gpu.device)
    pix_u = torch.zeros((N, V), dtype=torch.int32, device=means_gpu.device)
    pix_v = torch.zeros((N, V), dtype=torch.int32, device=means_gpu.device)

    for vi in range(V):
        # (4,4) @ (N,4)^T -> (4, N) -> (N, 4)
        cam = (viewmats_gpu[vi] @ means_h.T).T  # (N, 4)
        z = cam[:, 2]
        in_front = z > near
        x_cam = cam[:, 0]
        y_cam = cam[:, 1]
        # Perspective projection
        u = K_gpu[0, 0] * (x_cam / z.clamp(min=1e-6)) + K_gpu[0, 2]
        v_ = K_gpu[1, 1] * (y_cam / z.clamp(min=1e-6)) + K_gpu[1, 2]
        in_frame = (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H)
        keep = in_front & in_frame
        inside_frustum[:, vi] = keep
        pix_u[:, vi] = torch.round(u).to(torch.int32).clamp(0, W - 1)
        pix_v[:, vi] = torch.round(v_).to(torch.int32).clamp(0, H - 1)
    return inside_frustum, pix_u, pix_v


def score_against_masks(inside_frustum, pix_u, pix_v, mask_stack, names_in_frames):
    """mask_stack: (P, V, H_m, W_m) bool tensor on GPU.
       Returns hit_count: (N, P) int, visible_count: (N,) int.
    """
    N, V = inside_frustum.shape
    P, _, H_m, W_m = mask_stack.shape

    hit_count = torch.zeros((N, P), dtype=torch.int32, device=inside_frustum.device)
    visible_count = inside_frustum.sum(dim=1).to(torch.int32)  # (N,)

    for vi in range(V):
        keep = inside_frustum[:, vi]
        if not keep.any():
            continue
        u = pix_u[:, vi]
        v = pix_v[:, vi]
        # Look up mask value for each Gaussian for each prompt at this view
        # mask_stack[:, vi, v, u] : (P, N) — advanced indexing
        for pi in range(P):
            hits = mask_stack[pi, vi, v, u] & keep
            hit_count[:, pi] += hits.to(torch.int32)
    return hit_count, visible_count


def write_ply_subset(orig_vertex, keep_mask, out_path):
    """Write PLY with all original vertex attributes, subset by keep_mask."""
    keep_idx = np.where(keep_mask)[0]
    if len(keep_idx) == 0:
        print(f"  WARN: 0 vertices after filter — writing empty PLY {out_path}")
    sub = orig_vertex.data[keep_idx]
    el = PlyElement.describe(sub, "vertex")
    PlyData([el], byte_order="<").write(str(out_path))
    print(f"  wrote {out_path}  N={len(keep_idx):,}  ({keep_idx.size/len(orig_vertex.data)*100:.1f}% of full)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splat", required=True, type=Path)
    ap.add_argument("--transforms", required=True, type=Path)
    ap.add_argument("--dataparser", required=True, type=Path)
    ap.add_argument("--masks-dir", required=True, type=Path)
    ap.add_argument("--prompts", required=True,
                    help="comma-sep object specs. Each spec is either a single slug or "
                         "'slug1+slug2+...' to UNION multiple SAM3 masks into one object. "
                         "Object name = first slug (or use 'name:slug1+slug2' to override). "
                         "Example: 'blue_armchair+teddy_bear,black_subwoofer,wooden_coffee_table'")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--min-views", type=int, default=5,
                    help="Gaussian must be visible in >= this many views (else undecided)")
    ap.add_argument("--keep-thresh", type=float, default=0.30,
                    help="Gaussian belongs to object if hit_count/visible_count >= this")
    ap.add_argument("--drop-anisotropy", type=float, default=0.0,
                    help="If > 0, drop Gaussians whose (max_scale/min_scale) exceeds this "
                         "(reduces spike artifacts on cropped objects). 30-50 is a reasonable range.")
    ap.add_argument("--drop-max-scale-m", type=float, default=0.0,
                    help="If > 0, drop Gaussians whose max scale (in splat units) exceeds this "
                         "(kills huge floater Gaussians that cover the whole scene).")
    ap.add_argument("--keep-largest-cc", type=float, default=0.0,
                    help="If > 0, per object keep only the LARGEST connected component in K-NN graph "
                         "with edge distance <= this (splat units). "
                         "e.g. 0.05 = 5cm. Removes isolated floaters/artifacts.")
    ap.add_argument("--fill-holes", action="store_true",
                    help="After removing object Gaussians, K-NN clone remaining Gaussians "
                         "to fill the vacated bounding region of each removed object. "
                         "Cloned copies are placed at each removed Gaussian's position "
                         "but with properties (color/scale/opacity) from K nearest remainings.")
    ap.add_argument("--fill-k", type=int, default=8, help="K for K-NN cloning")
    ap.add_argument("--fill-mode", choices=["nearest", "blend"], default="blend",
                    help="'nearest'=clone the single closest donor (crisp, may show single-donor color); "
                         "'blend'=median of K donors (smoother but can darken).")
    ap.add_argument("--fill-shell-inner-m", type=float, default=0.02,
                    help="Donor Gaussians must be >= this distance from removed set (splat units)")
    ap.add_argument("--fill-shell-outer-m", type=float, default=0.30,
                    help="Donor Gaussians must be <= this distance from removed set (splat units)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-gaussians", type=int, default=200_000)
    args = ap.parse_args()

    # Parse object specs: each is 'name:slug1+slug2' or 'slug1+slug2' or 'slug'
    prompts = []       # display names (object output filename stem)
    prompt_slugs = []  # list-of-lists of slugs to UNION for each object
    for spec in args.prompts.split(","):
        spec = spec.strip()
        if not spec: continue
        if ":" in spec:
            name, slug_part = spec.split(":", 1)
        else:
            slug_part = spec
            name = spec.split("+")[0]  # first slug is default name
        slugs = [s.strip() for s in slug_part.split("+") if s.strip()]
        prompts.append(name)
        prompt_slugs.append(slugs)
    print(f"[cfg] objects={list(zip(prompts, prompt_slugs))}  min_views={args.min_views}  keep_thresh={args.keep_thresh}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "splat").mkdir(exist_ok=True)
    (args.out_dir / "splat" / "objects").mkdir(exist_ok=True)

    # 1. Load splat
    orig_v, xyz, N = load_ply(args.splat)
    print(f"[splat] N={N}")

    # 2. Load poses
    viewmats, K_np, W, H, names = load_poses_all(args.transforms, args.dataparser)
    V = viewmats.shape[0]
    print(f"[poses] V={V}  W={W}  H={H}")

    # 3. Discover masks per object (UNION multiple slugs if configured)
    print("[masks] discovering...")
    P = len(prompts)
    mask_slabs = np.zeros((P, V, H, W), dtype=bool)
    n_found = np.zeros(P, dtype=int)
    for vi, view_name in enumerate(names):
        for pi, slugs in enumerate(prompt_slugs):
            combined = None
            for slug in slugs:
                mp = args.masks_dir / f"{view_name}__{slug}.png"
                if not mp.exists(): continue
                m = np.array(Image.open(mp)) > 127
                if m.shape != (H, W):
                    from PIL import Image as _I
                    m = np.array(_I.fromarray(m.astype(np.uint8) * 255).resize((W, H), _I.NEAREST)) > 127
                combined = m if combined is None else (combined | m)
            if combined is not None:
                mask_slabs[pi, vi] = combined
                n_found[pi] += 1
    for pi, prompt in enumerate(prompts):
        cov = mask_slabs[pi].mean(axis=(1, 2))
        print(f"  {prompt} (union of {prompt_slugs[pi]}): {n_found[pi]}/{V} views, mean coverage {cov.mean()*100:.1f}%")

    # 4. GPU tensors
    device = args.device
    means = torch.from_numpy(xyz).to(device)
    viewmats_t = torch.from_numpy(viewmats).to(device)
    K_t = torch.from_numpy(K_np).to(device)
    mask_stack = torch.from_numpy(mask_slabs).to(device)  # (P, V, H, W)

    # 5. Project + score in Gaussian-batches
    hit_count_total = torch.zeros((N, P), dtype=torch.int32, device=device)
    visible_count_total = torch.zeros(N, dtype=torch.int32, device=device)
    B = args.batch_gaussians
    n_batches = (N + B - 1) // B
    print(f"[project+score] {N:,} gaussians × {V} views × {P} prompts in {n_batches} batches")
    for bi in range(n_batches):
        s, e = bi * B, min((bi + 1) * B, N)
        means_b = means[s:e]
        inside, pu, pv = project_batch(means_b, viewmats_t, K_t, W, H)
        hits, vis = score_against_masks(inside, pu, pv, mask_stack, None)
        hit_count_total[s:e] = hits
        visible_count_total[s:e] = vis
        if (bi + 1) % max(1, n_batches // 5) == 0:
            print(f"  batch {bi+1}/{n_batches} done")

    hit_count = hit_count_total.cpu().numpy()
    visible_count = visible_count_total.cpu().numpy()

    # 6. Classify Gaussians
    #    score[i, p] = hit_count[i, p] / max(visible_count[i], 1)
    scores = hit_count.astype(np.float32) / np.maximum(visible_count, 1).astype(np.float32)[:, None]

    # Gaussian is "assigned" to argmax prompt IF:
    #   - visible in >= min_views
    #   - argmax score >= keep_thresh
    #   - argmax score > 2nd best (unambiguous)
    top_score = scores.max(axis=1)
    top_prompt = scores.argmax(axis=1)
    if P > 1:
        sorted_scores = np.sort(scores, axis=1)
        second_score = sorted_scores[:, -2]
        unambiguous = top_score > second_score + 0.05
    else:
        unambiguous = np.ones(N, dtype=bool)

    enough_views = visible_count >= args.min_views
    passes_thresh = top_score >= args.keep_thresh

    # Spike/floater filter (applies only to per-OBJECT outputs; scene_without keeps everything)
    scale_reject = np.zeros(N, dtype=bool)
    if args.drop_anisotropy > 0 or args.drop_max_scale_m > 0:
        sx = np.exp(orig_v["scale_0"])
        sy = np.exp(orig_v["scale_1"])
        sz = np.exp(orig_v["scale_2"])
        scale_arr = np.stack([sx, sy, sz], -1)
        maxs = scale_arr.max(-1)
        mins = scale_arr.min(-1) + 1e-9
        aniso = maxs / mins
        if args.drop_anisotropy > 0:
            scale_reject |= aniso > args.drop_anisotropy
        if args.drop_max_scale_m > 0:
            scale_reject |= maxs > args.drop_max_scale_m
        print(f"[filter] scale reject: {scale_reject.sum():,}/{N:,} ({scale_reject.mean()*100:.1f}%)")

    # Per-object masks
    print("[classify]")
    all_assigned = np.zeros(N, dtype=bool)
    for pi, prompt in enumerate(prompts):
        obj_mask = enough_views & passes_thresh & unambiguous & (top_prompt == pi)
        # For assignment / removal from scene: use FULL obj_mask (isolated Gaussians still removed)
        all_assigned |= obj_mask
        # For per-object output: drop spikes + optional largest-CC coherence
        obj_mask_clean = obj_mask & ~scale_reject
        if args.keep_largest_cc > 0 and obj_mask_clean.sum() > 100:
            from scipy.spatial import cKDTree
            obj_idx = np.where(obj_mask_clean)[0]
            pts = xyz[obj_idx]
            tree = cKDTree(pts)
            pairs = tree.query_pairs(args.keep_largest_cc)
            # Union-find over pairs
            n_obj = len(pts)
            parent = np.arange(n_obj)
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for a, b in pairs:
                ra, rb = find(a), find(b)
                if ra != rb: parent[ra] = rb
            roots = np.array([find(i) for i in range(n_obj)])
            _, cc_counts = np.unique(roots, return_counts=True)
            largest_root = np.unique(roots)[np.argmax(cc_counts)]
            keep_in_cc = roots == largest_root
            print(f"  [{prompt}] largest-CC keeps {keep_in_cc.sum():,}/{n_obj:,} (drop {n_obj - keep_in_cc.sum():,} floater(s))")
            new_mask = np.zeros(N, dtype=bool)
            new_mask[obj_idx[keep_in_cc]] = True
            obj_mask_clean = new_mask
        out_path = args.out_dir / "splat" / "objects" / f"{prompt}.ply"
        write_ply_subset(orig_v, obj_mask_clean, out_path)

    # Scene without objects = full splat minus all_assigned
    scene_without = ~all_assigned
    write_ply_subset(orig_v, scene_without,
                     args.out_dir / "splat" / "scene_without_objects.ply")

    # Optional hole-fill: K-NN clone remaining Gaussians into removed positions
    if args.fill_holes and all_assigned.any():
        print(f"[fill] K-NN cloning {all_assigned.sum():,} removed positions from remaining donors")
        from scipy.spatial import cKDTree
        removed_xyz = xyz[all_assigned]
        remain_xyz = xyz[scene_without]
        tree = cKDTree(remain_xyz)
        d, idx = tree.query(removed_xyz, k=args.fill_k)  # (M, K)
        # Only fill positions whose k-th neighbor distance is within outer shell
        if args.fill_k == 1:
            d = d[:, None]
            idx = idx[:, None]
        fill_ok = (d.min(axis=1) >= args.fill_shell_inner_m) & (d.max(axis=1) <= args.fill_shell_outer_m)
        n_fill = int(fill_ok.sum())
        print(f"  fillable positions: {n_fill:,}/{len(removed_xyz):,}")
        if n_fill > 0:
            remain_indices = np.where(scene_without)[0]
            fill_pos_idx = np.where(all_assigned)[0][fill_ok]
            donor_local_idx_k = idx[fill_ok]  # (M, K)
            donor_orig_idx_k = remain_indices[donor_local_idx_k]
            if args.fill_mode == "nearest":
                base_donor = orig_v.data[donor_orig_idx_k[:, 0]].copy()
            else:
                base_donor = orig_v.data[donor_orig_idx_k[:, 0]].copy()
                for f in [n for n in base_donor.dtype.names if n not in ("x", "y", "z")]:
                    gathered = np.stack([orig_v.data[donor_orig_idx_k[:, k]][f]
                                         for k in range(args.fill_k)], axis=1)
                    if f.startswith("rot_"):
                        base_donor[f] = gathered[:, 0]
                    else:
                        base_donor[f] = np.median(gathered, axis=1)
            target_xyz = xyz[fill_pos_idx]
            base_donor["x"] = target_xyz[:, 0]
            base_donor["y"] = target_xyz[:, 1]
            base_donor["z"] = target_xyz[:, 2]
            kept_rows = orig_v.data[scene_without]
            combined = np.concatenate([kept_rows, base_donor], axis=0)
            el = PlyElement.describe(combined, "vertex")
            filled_path = args.out_dir / "splat" / "scene_without_objects_filled.ply"
            PlyData([el], byte_order="<").write(str(filled_path))
            print(f"  wrote {filled_path}  N={len(combined):,}  (+{n_fill:,} {args.fill_mode}-fill clones)")

    # Also copy full scene
    import shutil
    shutil.copy2(args.splat, args.out_dir / "splat" / "scene_full.ply")
    print(f"[copy] {args.splat} -> {args.out_dir / 'splat' / 'scene_full.ply'}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Total Gaussians:      {N:,}")
    print(f"Enough views (>={args.min_views}): {enough_views.sum():,} ({enough_views.mean()*100:.1f}%)")
    print(f"Assigned to any obj:  {all_assigned.sum():,} ({all_assigned.mean()*100:.1f}%)")
    print(f"Scene without:        {scene_without.sum():,}")
    for pi, prompt in enumerate(prompts):
        n = (enough_views & passes_thresh & unambiguous & (top_prompt == pi)).sum()
        print(f"  {prompt}: {n:,} gaussians")


if __name__ == "__main__":
    main()
