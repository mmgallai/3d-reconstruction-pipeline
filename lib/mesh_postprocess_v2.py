"""
Tier-4 mesh post-processor — pymeshlab-based for arbitrary hole closing and
proper manifold winding. Replaces the trimesh-only Tier-1 post-process.

Pipeline:
  1. Drop tiny components (< min_component_faces)
  2. Re-orient face winding consistently using Open3D's orient_triangles_consistent
     (works on whole multi-shell meshes, unlike trimesh.fix_normals)
  3. Close holes up to N edges using pymeshlab's `meshing_close_holes` filter
     (trimesh's fill_holes only handles 3- and 4-edge holes)
  4. Save the cleaned mesh

Usage:
    python lib/mesh_postprocess_v2.py <in.ply> <out.ply> [min_faces=100] [max_hole_edges=300]
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def _open3d_to_pymeshlab(verts: np.ndarray, faces: np.ndarray):
    """Build a pymeshlab MeshSet from numpy arrays (avoids round-tripping disk)."""
    import pymeshlab as ml
    ms = ml.MeshSet()
    m = ml.Mesh(vertex_matrix=verts.astype(np.float64),
                face_matrix=faces.astype(np.int32))
    ms.add_mesh(m, "scene")
    return ms


def main(in_path: Path, out_path: Path,
         min_component_faces: int = 100,
         max_hole_edges: int = 300) -> int:
    if not in_path.exists():
        print(f"  ERROR: {in_path} not found", flush=True)
        return 2

    print(f"Loading {in_path} ...", flush=True)
    m = o3d.io.read_triangle_mesh(str(in_path))
    v0, f0 = len(m.vertices), len(m.triangles)
    print(f"  loaded: {v0:,} verts / {f0:,} faces", flush=True)
    if f0 == 0:
        return 3

    # 1) Component cleanup — Open3D version (preserves manifoldness better than trimesh.split)
    print("Removing tiny components ...", flush=True)
    triangle_clusters, cluster_n_triangles, _ = m.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    big_clusters = np.where(cluster_n_triangles >= min_component_faces)[0]
    print(f"  components: {len(cluster_n_triangles):,} total, {len(big_clusters):,} kept (>= {min_component_faces} faces)")
    if len(big_clusters) == 0:
        print("  ERROR: every component below threshold")
        return 4
    keep_mask = np.isin(triangle_clusters, big_clusters)
    drop_idx = np.where(~keep_mask)[0]
    m.remove_triangles_by_index(drop_idx)
    m.remove_unreferenced_vertices()
    v1, f1 = len(m.vertices), len(m.triangles)
    print(f"  after component filter: {v1:,} verts / {f1:,} faces  (delta {v1-v0:+,} verts, {f1-f0:+,} faces)", flush=True)

    # 2) Consistent face winding (replaces trimesh.fix_normals — works on multi-shell)
    print("Orienting triangles consistently ...", flush=True)
    m.orient_triangles()
    print("  done", flush=True)

    # 3) Hole closing via pymeshlab (handles arbitrary hole sizes)
    print(f"Closing holes (max {max_hole_edges} edges) ...", flush=True)
    verts = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.triangles, dtype=np.int32)
    ms = _open3d_to_pymeshlab(verts, faces)

    try:
        ms.apply_filter("meshing_close_holes", maxholesize=max_hole_edges, newfaceselected=False)
        out = ms.current_mesh()
        v_after = out.vertex_matrix().astype(np.float32)
        f_after = out.face_matrix().astype(np.int32)
        v2, f2 = len(v_after), len(f_after)
        print(f"  after close_holes: {v2:,} verts / {f2:,} faces  (delta {v2-v1:+,} verts, {f2-f1:+,} faces)", flush=True)
    except Exception as e:
        print(f"  WARN: close_holes failed: {e}; using mesh without hole closing", flush=True)
        v_after = verts.astype(np.float32)
        f_after = faces

    # 4) Save as binary PLY (geometry only — UVs/textures regenerated downstream by TextureMesh)
    print(f"Saving {out_path} ...", flush=True)
    out_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v_after.astype(np.float64)),
        o3d.utility.Vector3iVector(f_after),
    )
    out_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_path), out_mesh, write_ascii=False, compressed=False)
    print(f"  saved: {out_path.name}  ({out_path.stat().st_size/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: mesh_postprocess_v2.py <in.ply> <out.ply> [min_faces=100] [max_hole_edges=300]", file=sys.stderr)
        sys.exit(1)
    in_p = Path(sys.argv[1])
    out_p = Path(sys.argv[2])
    min_f = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    max_h = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    sys.exit(main(in_p, out_p, min_f, max_h))
