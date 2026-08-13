"""Diagnostic — figure out white_water_bottle patch-emptiness and the
small-hole story."""
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
    return dist, verts


def main():
    mesh_path = V5 / "mesh" / "scene_without_objects.ply"
    m = trimesh.load(str(mesh_path), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float64)
    print(f"scene verts: {verts.shape[0]:,}")

    for slug in SLUGS:
        obj_ply = SEG / slug / f"{slug}_extracted.ply"
        dist_fn, ov = object_dist_field(obj_ply)

        N = verts.shape[0]
        d = np.empty(N, dtype=np.float32)
        step = 200_000
        for i in range(0, N, step):
            d[i:i + step] = dist_fn(verts[i:i + step])

        y_bot_obj = float(np.percentile(ov[:, 1], 5))
        y_med_obj = float(np.median(ov[:, 1]))
        y_max_obj = float(ov[:, 1].max())
        print(f"\n=== {slug} =========================================")
        print(f"  object mesh Y: min={ov[:,1].min():.3f}, p5={y_bot_obj:.3f}, "
              f"median={y_med_obj:.3f}, max={y_max_obj:.3f}")

        # look at scene verts within 5 cm of object mesh, split by Y band
        near = verts[d <= 0.05]
        print(f"  scene verts within 5 cm of object mesh: {len(near):,}")
        if len(near):
            print(f"  their Y percentiles [5,25,50,75,95]: "
                  f"{np.percentile(near[:,1],[5,25,50,75,95]).round(3).tolist()}")
            # count in progressively larger Y bands relative to y_bot_obj
            for band in (0.05, 0.10, 0.15, 0.25, 0.5):
                n = int(np.sum(np.abs(near[:, 1] - y_bot_obj) <= band))
                print(f"    |Y - obj_p5| <= {band*100:.0f} cm: n={n:,}")

        # ring
        ring = verts[(d > 0.05) & (d <= 0.15)]
        print(f"  ring (5-15 cm from obj): {len(ring):,}")
        if len(ring):
            print(f"  ring Y percentiles [5,25,50,75,95]: "
                  f"{np.percentile(ring[:,1],[5,25,50,75,95]).round(3).tolist()}")


if __name__ == "__main__":
    main()
