"""
Screened Poisson surface reconstruction (Kazhdan 2013) via Open3D.

Replacement for OpenMVS's `ReconstructMesh` (Delaunay graph-cut) which caps
mesh density at the scene's surface complexity regardless of input point
density. Poisson with octree-depth control can produce arbitrarily dense
meshes from the same dense cloud.

Inputs:
  scene_dense.ply  — point cloud with per-point normals
                      (produced by DensifyPointCloud --estimate-normals 2)
Outputs:
  scene_mesh.ply   — watertight triangle mesh
                      (replaces what OpenMVS's ReconstructMesh would have written)

Octree depth recommendations:
  depth=8  →  ~100K-500K verts   (low detail)
  depth=10 →  ~1M-3M verts       (medium)
  depth=11 →  ~3M-8M verts       (high — comparable to OpenMVS output)
  depth=12 →  ~8M-20M verts      (very high — aim for RealityScan parity)
  depth=13 →  may OOM on 32 GB RAM with 5M+ point clouds

`density_threshold` (0..1) drops vertices that Poisson placed in
low-confidence regions (i.e. far from any input point) — these are
extrapolations the algorithm guesses to keep the mesh watertight, but
they're often spurious balloons in empty space.

Usage:
    python lib/poisson_mesh.py <input_dense.ply> <output_mesh.ply> [depth=11] [density=0.05]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def main(in_path: Path, out_path: Path,
         depth: int = 11,
         density_threshold: float = 0.05,
         linear_fit: bool = False) -> int:
    if not in_path.exists():
        print(f"  ERROR: {in_path} not found", flush=True)
        return 2

    print(f"Loading point cloud: {in_path} ...", flush=True)
    pcd = o3d.io.read_point_cloud(str(in_path))
    n_pts = len(pcd.points)
    has_normals = pcd.has_normals()
    print(f"  {n_pts:,} points, has_normals={has_normals}, has_colors={pcd.has_colors()}", flush=True)
    if n_pts == 0:
        return 3

    if not has_normals:
        # Estimate normals from local neighbourhoods (fallback)
        print("  WARN: input cloud has no normals — estimating from KNN", flush=True)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        # Need to orient them consistently — Poisson is sensitive to flipped normals
        pcd.orient_normals_consistent_tangent_plane(k=30)

    # Run Poisson reconstruction
    print(f"\nRunning Screened Poisson surface reconstruction (depth={depth}) ...", flush=True)
    print(f"  (this may take 5-30 min on {n_pts/1e6:.1f}M points; uses up to ~{(n_pts/1e6)*2:.0f} GB RAM)", flush=True)
    t0 = time.time()
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=linear_fit
    )
    elapsed = time.time() - t0
    n_v0, n_f0 = len(mesh.vertices), len(mesh.triangles)
    density = np.asarray(density)
    print(f"  Poisson done in {elapsed:.1f}s: {n_v0:,} verts / {n_f0:,} faces", flush=True)
    print(f"  density stats: min={density.min():.3f}, p25={np.percentile(density, 25):.3f}, "
          f"median={np.percentile(density, 50):.3f}, p75={np.percentile(density, 75):.3f}, max={density.max():.3f}", flush=True)

    # Drop low-density (extrapolated) regions to remove spurious surfaces
    if density_threshold > 0:
        threshold = np.quantile(density, density_threshold)
        print(f"\nDropping vertices with density below {threshold:.3f} (bottom {density_threshold*100:.0f}% — likely extrapolated) ...", flush=True)
        verts_to_remove = density < threshold
        mesh.remove_vertices_by_mask(verts_to_remove)
        n_v1, n_f1 = len(mesh.vertices), len(mesh.triangles)
        print(f"  after density filter: {n_v1:,} verts / {n_f1:,} faces  "
              f"(Δ {n_v1-n_v0:+,} verts, {n_f1-n_f0:+,} faces)", flush=True)
    else:
        n_v1, n_f1 = n_v0, n_f0

    # Light cleanup
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    n_v2, n_f2 = len(mesh.vertices), len(mesh.triangles)
    if (n_v2, n_f2) != (n_v1, n_f1):
        print(f"  after cleanup: {n_v2:,} verts / {n_f2:,} faces  (Δ {n_v2-n_v1:+,} verts, {n_f2-n_f1:+,} faces)", flush=True)

    # Save
    print(f"\nSaving {out_path} ...", flush=True)
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False, compressed=False)
    print(f"  saved: {out_path.name}  ({out_path.stat().st_size/1024/1024:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="dense point cloud PLY (with normals)")
    ap.add_argument("output", help="output mesh PLY path")
    ap.add_argument("--depth", type=int, default=11, help="octree depth (8-13). Higher = denser, slower.")
    ap.add_argument("--density-threshold", type=float, default=0.05,
                    help="quantile [0..1] below which to drop low-confidence vertices")
    ap.add_argument("--linear-fit", action="store_true",
                    help="use linear (instead of cubic) interpolation — slightly faster, less smooth")
    args = ap.parse_args()

    sys.exit(main(Path(args.input), Path(args.output),
                  depth=args.depth,
                  density_threshold=args.density_threshold,
                  linear_fit=args.linear_fit))
