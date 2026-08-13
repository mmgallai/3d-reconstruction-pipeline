"""Reviewer D — final: inspect the surviving red splat + summarize deltas."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
import open3d as o3d
import trimesh

_ROOT = Path(r"C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
sys.path.insert(0, str(_ROOT))
from _spatial_crop_splat import _load_dataparser_transform, _splat_to_metric  # noqa

V4 = _ROOT / "output" / "pipeline_v32_v11_patches_v4"
V2 = _ROOT / "output" / "pipeline_v32_v11_patches_v2"
SEG = _ROOT / "output" / "segmented_v32_data3_v9a_fp_v2"
DP_JSON = _ROOT / "nerfstudio" / "dense" / "splatfacto" / "2026-06-07_203807" / "dataparser_transforms.json"
BOUNDS_JSON = _ROOT / "colmap" / "dense" / "tof_bounds.json"

_SH_C0 = 0.28209479177387814


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
    return dict(pos=pos, rgb=rgb)


def dist_fn(mesh_ply):
    m = trimesh.load(str(mesh_ply), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.uint32)
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(verts)
    tm.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)
    def f(p): return scene.compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy()
    y_min = float(verts[:, 1].min())
    y_max = float(verts[:, 1].max())
    return f, y_min, y_max


def main():
    s_v4 = load_splat(V4 / "splat" / "scene_without_objects.ply")
    s_v2 = load_splat(V2 / "splat" / "scene_without_objects.ply")
    lob_f, y_min, y_max = dist_fn(SEG / "red_lobster_figurine" / "red_lobster_figurine_extracted.ply")

    # Locate the "1 pure red" survivor in v4
    d = lob_f(s_v4["pos"])
    within = d <= 0.05
    rgb = s_v4["rgb"]
    red = (rgb[:, 0] > 0.6) & (rgb[:, 1] < 0.4) & (rgb[:, 2] < 0.4)
    idx = np.where(within & red)[0]
    print(f"[v4] pure-red-within-5cm indices: {idx.tolist()}")
    for i in idx:
        print(f"  idx {i}: pos={s_v4['pos'][i].round(4).tolist()}, "
              f"rgb={s_v4['rgb'][i].round(3).tolist()}, dist={d[i]:.4f} m")
        print(f"    y vs lobster_y_min={y_min:.3f}, lobster_y_max={y_max:.3f}  "
              f"y_over_top={s_v4['pos'][i, 1] - y_max:.3f} m")

    # Same for v2
    d2 = lob_f(s_v2["pos"])
    within2 = d2 <= 0.05
    rgb2 = s_v2["rgb"]
    red2 = (rgb2[:, 0] > 0.6) & (rgb2[:, 1] < 0.4) & (rgb2[:, 2] < 0.4)
    idx2 = np.where(within2 & red2)[0]
    print(f"\n[v2] pure-red-within-5cm indices ({len(idx2)}):")
    for i in idx2[:15]:
        print(f"  idx {i}: pos={s_v2['pos'][i].round(4).tolist()}, "
              f"rgb={s_v2['rgb'][i].round(3).tolist()}, dist={d2[i]:.4f} m, "
              f"y_over_top={s_v2['pos'][i, 1] - y_max:.3f} m")

    # Distance distribution of red splats near lobster
    print(f"\n[v2] loose-red splats within 5cm distance percentiles:")
    red_loose2 = (rgb2[:, 0] > 0.4) & (rgb2[:, 0] > rgb2[:, 1] + 0.1) & (rgb2[:, 0] > rgb2[:, 2] + 0.1)
    d2_red = d2[red_loose2]
    print(f"  n_loose_red_all={red_loose2.sum()}, within_5cm={((red_loose2)&(d2<=0.05)).sum()}, "
          f"within_3cm={((red_loose2)&(d2<=0.03)).sum()}")

    print(f"\n[v4] loose-red splats within 5cm distance percentiles:")
    red_loose4 = (rgb[:, 0] > 0.4) & (rgb[:, 0] > rgb[:, 1] + 0.1) & (rgb[:, 0] > rgb[:, 2] + 0.1)
    d4_red = d[red_loose4]
    print(f"  n_loose_red_all={red_loose4.sum()}, within_5cm={((red_loose4)&(d<=0.05)).sum()}, "
          f"within_3cm={((red_loose4)&(d<=0.03)).sum()}")

    # summary deltas
    print("\n=== SUMMARY DELTAS v2 -> v4 ===")
    n_within_v2 = int(within2.sum())
    n_within_v4 = int(within.sum())
    print(f"Lobster (within 5cm splats):    v2={n_within_v2:,}  v4={n_within_v4:,}  "
          f"delta={n_within_v4-n_within_v2:+d} ({100*(n_within_v4-n_within_v2)/max(n_within_v2,1):+.1f}%)")


if __name__ == "__main__":
    main()
