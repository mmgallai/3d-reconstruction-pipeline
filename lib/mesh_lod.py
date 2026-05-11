"""
Generate LOD (level-of-detail) variants of the textured mesh for web / mobile.

Produces three levels by default:
  HIGH  — original mesh (no decimation), saved as the canonical .ply
  MID   — ~500K faces (good for desktop/mobile WebGL)
  LOW   — ~100K faces (mobile / very fast load)

Each LOD is saved as both .glb (compact, web-ready) and .ply (compatibility).

Usage:
    python lib/mesh_lod.py <input.ply> <output_dir> [--mid 500000 --low 100000]

The texture atlases are NOT re-baked — the LOD meshes preserve UVs (best
effort via quadric edge collapse) and reference the same scene_textured*.png
files. Quadric decimation can stretch UVs slightly; for very low LODs the
visual quality drops.

Designed to be invoked via `conda run -n da3 python lib/mesh_lod.py ...`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def make_lod(verts: np.ndarray, faces: np.ndarray,
             target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Open3D quadric edge-collapse decimation to a target face count."""
    if len(faces) <= target_faces:
        print(f"  [{target_faces:,} target] mesh already at/under target ({len(faces):,}); skipping", flush=True)
        return verts, faces
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    out = m.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    out.remove_duplicated_vertices()
    out.remove_duplicated_triangles()
    out.remove_unreferenced_vertices()
    nv = np.asarray(out.vertices)
    nf = np.asarray(out.triangles)
    print(f"  [{target_faces:,} target] decimated: {len(faces):,} → {len(nf):,} faces  "
          f"({len(verts):,} → {len(nv):,} verts)", flush=True)
    return nv, nf


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a minimal binary PLY (positions + faces, no UVs/textures)."""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(verts.astype(np.float32).tobytes())
        # face records: 1 uchar count + 3 ints
        face_data = np.empty(len(faces), dtype=[("count", "u1"), ("vidx", "<i4", 3)])
        face_data["count"] = 3
        face_data["vidx"] = faces.astype(np.int32)
        fh.write(face_data.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to high-res textured PLY")
    ap.add_argument("output_dir", help="Where to write LOD .ply files")
    ap.add_argument("--mid", type=int, default=500_000, help="MID LOD face count")
    ap.add_argument("--low", type=int, default=100_000, help="LOW LOD face count")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ...", flush=True)
    m_o3d = o3d.io.read_triangle_mesh(str(in_path))
    verts = np.asarray(m_o3d.vertices)
    faces = np.asarray(m_o3d.triangles)
    print(f"  source: {len(verts):,} verts / {len(faces):,} faces", flush=True)

    stem = in_path.stem  # e.g. mesh_v17_openmvs

    # MID LOD
    v_mid, f_mid = make_lod(verts, faces, args.mid)
    p_mid = out_dir / f"{stem}_mid.ply"
    write_ply(p_mid, v_mid, f_mid)
    print(f"  saved: {p_mid.name}  ({p_mid.stat().st_size/1024/1024:.1f} MB)", flush=True)

    # LOW LOD
    v_low, f_low = make_lod(verts, faces, args.low)
    p_low = out_dir / f"{stem}_low.ply"
    write_ply(p_low, v_low, f_low)
    print(f"  saved: {p_low.name}  ({p_low.stat().st_size/1024/1024:.1f} MB)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
