"""
Bake per-vertex ambient occlusion (AO) onto a mesh using Open3D's raycasting.

For each vertex:
  - Sample N rays in the hemisphere oriented along the vertex normal
  - Count how many rays hit the mesh within a max distance
  - AO = 1 - hit_ratio  (1.0 = fully exposed, 0.0 = fully occluded)

The AO values are stored as the R=G=B vertex color (greyscale) so the
output PLY renders correctly in any viewer that respects vertex colors,
including MeshLab, Blender, and most web viewers.

Usage:
    python lib/mesh_ao_bake.py <input.ply> <output.ply> [--rays 64 --max-dist 0.5]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def hemisphere_dirs(normal: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n unit vectors in the hemisphere oriented by `normal`."""
    # Cosine-weighted sampling: bias rays toward the normal direction
    # (matches the cos(θ) term in standard AO integral).
    u = rng.random(n)
    v = rng.random(n)
    r = np.sqrt(u)
    theta = 2 * np.pi * v
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(np.maximum(0.0, 1.0 - u))
    local = np.stack([x, y, z], axis=1)  # n × 3, hemisphere around +Z

    # Build orthonormal basis around `normal`
    n_unit = normal / (np.linalg.norm(normal) + 1e-12)
    if abs(n_unit[2]) < 0.9:
        tangent = np.cross(n_unit, [0.0, 0.0, 1.0])
    else:
        tangent = np.cross(n_unit, [1.0, 0.0, 0.0])
    tangent /= np.linalg.norm(tangent) + 1e-12
    bitangent = np.cross(n_unit, tangent)
    rot = np.stack([tangent, bitangent, n_unit], axis=1)  # 3 × 3
    return local @ rot.T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--rays", type=int, default=64,
                    help="rays per vertex (more = smoother AO, slower)")
    ap.add_argument("--max-dist", type=float, default=0.5,
                    help="max ray distance in mesh units (0.5 m default)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    print(f"Loading {in_path} ...", flush=True)
    mesh = o3d.io.read_triangle_mesh(str(in_path))
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    print(f"  {len(verts):,} verts / {len(faces):,} faces", flush=True)

    # Build raycasting scene
    print("Building raycasting scene ...", flush=True)
    rt_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(rt_mesh)

    # AO baking: process vertices in chunks of ~10k to avoid building one giant
    # ray array (memory) but still benefit from batched raycasting.
    chunk = 10_000
    n_rays = args.rays
    max_dist = args.max_dist
    rng = np.random.default_rng(42)

    ao = np.zeros(len(verts), dtype=np.float32)
    t0 = time.time()
    print(f"Casting {len(verts) * n_rays:,} rays ({n_rays} per vertex) ...", flush=True)

    for start in range(0, len(verts), chunk):
        end = min(start + chunk, len(verts))
        c_verts = verts[start:end]
        c_normals = normals[start:end]
        c_size = end - start

        # Build ray tensor: shape (chunk * n_rays, 6)  [origin xyz, dir xyz]
        rays = np.empty((c_size * n_rays, 6), dtype=np.float32)
        # Tiny offset along normal so rays don't hit the surface they came from
        epsilon = 1e-4
        for i in range(c_size):
            dirs = hemisphere_dirs(c_normals[i], n_rays, rng)
            origin = c_verts[i] + c_normals[i] * epsilon
            rays[i * n_rays:(i + 1) * n_rays, 0:3] = origin
            rays[i * n_rays:(i + 1) * n_rays, 3:6] = dirs

        ray_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        result = scene.cast_rays(ray_t)
        t_hit = result["t_hit"].numpy()

        # A vertex is "occluded" by a ray if the ray hit something within max_dist
        hits = (t_hit < max_dist).reshape(c_size, n_rays)
        occlusion_ratio = hits.mean(axis=1)
        ao[start:end] = 1.0 - occlusion_ratio

        if start % (chunk * 5) == 0:
            elapsed = time.time() - t0
            rate = end / elapsed if elapsed > 0 else 0
            eta = (len(verts) - end) / rate if rate > 0 else 0
            print(f"  {end:,}/{len(verts):,} verts  ({rate:,.0f}/sec, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  done: {elapsed:.1f}s, AO range = [{ao.min():.3f}, {ao.max():.3f}], mean = {ao.mean():.3f}", flush=True)

    # Write back as vertex-colored PLY
    print(f"Writing {out_path} ...", flush=True)
    rgb = (np.clip(ao, 0.0, 1.0) * 255).astype(np.uint8)
    rgb_arr = np.stack([rgb, rgb, rgb], axis=1)  # greyscale R=G=B

    # Compose binary little-endian PLY
    n_v = len(verts)
    n_f = len(faces)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_v}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {n_f}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    vert_dt = np.dtype([
        ("xyz", "<f4", 3),
        ("rgb", "u1", 3),
    ])
    vert_arr = np.empty(n_v, dtype=vert_dt)
    vert_arr["xyz"] = verts
    vert_arr["rgb"] = rgb_arr

    face_dt = np.dtype([("count", "u1"), ("vidx", "<i4", 3)])
    face_arr = np.empty(n_f, dtype=face_dt)
    face_arr["count"] = 3
    face_arr["vidx"] = faces

    with open(out_path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(vert_arr.tobytes())
        fh.write(face_arr.tobytes())

    print(f"  saved: {out_path.name}  ({out_path.stat().st_size/1024/1024:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
