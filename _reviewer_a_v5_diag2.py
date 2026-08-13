"""Additional diagnostics for the splat-patch-darker finding and the
216 remaining small holes in object XZ areas.

- Look at the size and per-slug distribution of the remaining small loops
- Verify splat patch darkness by inspecting per-Gaussian RGB distributions
- Also inspect the desk-patch PLY that _seed_desk_patch_clone.py wrote.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import open3d as o3d
from plyfile import PlyData

_ROOT = Path(r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(_ROOT))
from _spatial_crop_splat import _load_dataparser_transform, _splat_to_metric  # noqa

V5 = _ROOT / "output" / "pipeline_v32_v11_patches_v5_clone"
SEG = _ROOT / "output" / "segmented_v32_data3_v9a_fp_v2"
DP_JSON = _ROOT / "nerfstudio" / "dense" / "splatfacto" / "2026-06-07_203807" / "dataparser_transforms.json"
BOUNDS_JSON = _ROOT / "colmap" / "dense" / "tof_bounds.json"

_SH_C0 = 0.28209479177387814
SLUGS = ["white_water_bottle", "blue_box", "red_lobster_figurine"]


def load_splat(path):
    R, t, dp_scale = _load_dataparser_transform(DP_JSON)
    bounds = json.loads(BOUNDS_JSON.read_text())
    colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])
    pd = PlyData.read(str(path))
    v = pd["vertex"].data
    pos = _splat_to_metric(
        np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64),
        R, t, dp_scale, colmap_to_metric)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], -1).astype(np.float32)
    rgb = np.clip(dc * _SH_C0 + 0.5, 0, 1)
    return dict(pos=pos, rgb=rgb, n=int(len(pos)))


def object_dist_field(obj_ply: Path):
    m = trimesh.load(str(obj_ply), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.uint32)
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(verts)
    tm.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)
    def dist(p):
        p32 = np.ascontiguousarray(p, dtype=np.float32)
        return scene.compute_distance(o3d.core.Tensor(p32)).numpy()
    return dist


def main():
    # 1. small-hole size distribution per slug in object XZ AABBs
    mesh_path = V5 / "mesh" / "scene_without_objects.ply"
    m = trimesh.load(str(mesh_path), force="mesh", process=False)
    outline = m.outline()

    # object AABBs
    bboxes = {}
    for slug in SLUGS:
        v = np.asarray(trimesh.load(str(SEG / slug / f"{slug}_extracted.ply"),
                                    force="mesh", process=False).vertices)
        bboxes[slug] = dict(x_min=v[:, 0].min(), x_max=v[:, 0].max(),
                            z_min=v[:, 2].min(), z_max=v[:, 2].max())

    peris_by_slug = {slug: [] for slug in SLUGS}
    for entity in outline.entities:
        try:
            pts = outline.vertices[entity.points]
        except Exception:
            continue
        if len(pts) < 3:
            continue
        seg_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        if not np.allclose(pts[0], pts[-1]):
            seg_len += float(np.linalg.norm(pts[-1] - pts[0]))
        cx, _cy, cz = pts.mean(axis=0)
        for slug, bb in bboxes.items():
            if bb["x_min"] <= cx <= bb["x_max"] and bb["z_min"] <= cz <= bb["z_max"]:
                peris_by_slug[slug].append(seg_len)

    print("=== small-hole size distribution inside each object AABB ===")
    for slug in SLUGS:
        arr = np.array(peris_by_slug[slug])
        n = len(arr)
        n_small = int((arr < 0.30).sum())
        n_tiny  = int((arr < 0.05).sum())  # <5 cm perimeter -> micro/pinhole
        print(f"  {slug}: n_loops={n}, n_small(<30cm)={n_small}, n_tiny(<5cm)={n_tiny}")
        if n:
            print(f"    perim percentiles [50,75,90,95,99,max] cm: "
                  f"{(np.percentile(arr,[50,75,90,95,99])*100).round(2).tolist()} "
                  f"max={arr.max()*100:.2f}")

    # 2. splat-patch darkness check
    splat_path = V5 / "splat" / "scene_without_objects.ply"
    s = load_splat(splat_path)
    print(f"\n=== splat patch darkness check ===")
    print(f"scene splat n={s['n']:,}")

    for slug in SLUGS:
        dist_fn = object_dist_field(SEG / slug / f"{slug}_extracted.ply")
        N = s["pos"].shape[0]
        d = np.empty(N, dtype=np.float32)
        step = 200_000
        for i in range(0, N, step):
            d[i:i + step] = dist_fn(s["pos"][i:i + step])
        ring_mask = (d > 0.05) & (d <= 0.15)
        desk_y = float(np.median(s["pos"][ring_mask, 1])) if ring_mask.any() else 0.0
        patch = (d <= 0.05) & (np.abs(s["pos"][:, 1] - desk_y) <= 0.08)
        ring = (d > 0.05) & (d <= 0.15) & (np.abs(s["pos"][:, 1] - desk_y) <= 0.08)
        p_rgb = (s["rgb"][patch] * 255)
        r_rgb = (s["rgb"][ring] * 255)
        print(f"\n  {slug} (desk_y estimate = {desk_y:.3f}):")
        print(f"    patch n={patch.sum()}, ring n={ring.sum()}")
        if patch.sum():
            print(f"    patch  RGB percentiles [10,50,90]: "
                  f"R={np.percentile(p_rgb[:,0],[10,50,90]).round(1).tolist()}, "
                  f"G={np.percentile(p_rgb[:,1],[10,50,90]).round(1).tolist()}, "
                  f"B={np.percentile(p_rgb[:,2],[10,50,90]).round(1).tolist()}")
        if ring.sum():
            print(f"    ring   RGB percentiles [10,50,90]: "
                  f"R={np.percentile(r_rgb[:,0],[10,50,90]).round(1).tolist()}, "
                  f"G={np.percentile(r_rgb[:,1],[10,50,90]).round(1).tolist()}, "
                  f"B={np.percentile(r_rgb[:,2],[10,50,90]).round(1).tolist()}")

        # Is the patch dominated by donor clones with low RGB? Check if it
        # correlates with a "cloned Gaussian" property (they should have
        # identical positions on a grid; count unique unique-y-values)
        if patch.sum() and ring.sum():
            # try to distinguish: cloned patch positions likely lie on a
            # single Y plane -> stdev < 1 cm
            print(f"    patch Y stdev = {s['pos'][patch,1].std():.4f} m; "
                  f"ring Y stdev = {s['pos'][ring,1].std():.4f} m")


if __name__ == "__main__":
    main()
