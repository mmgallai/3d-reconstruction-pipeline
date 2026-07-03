"""
Generate LOD (level-of-detail) variants of the textured mesh for web / mobile.

Produces three levels by default:
  HIGH  — original mesh (no decimation), saved as the canonical .ply
  MID   — ~500K faces (good for desktop/mobile WebGL)
  LOW   — ~100K faces (mobile / very fast load)

Each LOD is saved as a textured binary PLY preserving UVs.

Designed to be invoked via `conda run -n da3 python lib/mesh_lod.py ...` or direct python.
"""
import argparse
import sys
import shutil
from pathlib import Path
import pymeshlab as ml


def make_lod(in_path: Path, out_path: Path, target_faces: int) -> None:
    """Decimate the mesh using PyMeshLab's SOTA quadric edge-collapse simplification, preserving UVs."""
    print(f"Decimating to {target_faces:,} faces ...", flush=True)
    ms = ml.MeshSet()
    ms.load_new_mesh(str(in_path))
    
    m_in = ms.current_mesh()
    face_count = m_in.face_number()
    if face_count <= target_faces:
        print(f"  [{target_faces:,} target] mesh already at/under target ({face_count:,}); copying", flush=True)
        shutil.copy2(in_path, out_path)
        return
        
    # Simplify using PyMeshLab's quadric edge collapse decimation filter with texture.
    # This ensures the face-varying (wedge) UV atlas coordinates are correctly simplified.
    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse_with_texture",
        targetfacenum=target_faces,
        preservenormal=True
    )
    
    ms.save_current_mesh(str(out_path), binary=True)
    m_out = ms.current_mesh()
    print(f"  [{target_faces:,} target] decimated: {face_count:,} -> {m_out.face_number():,} faces  "
          f"({m_in.vertex_number():,} -> {m_out.vertex_number():,} verts)", flush=True)


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

    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        return 1

    stem = in_path.stem  # e.g. mesh_v32_openmvs

    # MID LOD
    p_mid = out_dir / f"{stem}_mid.ply"
    make_lod(in_path, p_mid, args.mid)
    if p_mid.exists():
        print(f"  saved: {p_mid.name}  ({p_mid.stat().st_size/1024/1024:.1f} MB)", flush=True)

    # LOW LOD
    p_low = out_dir / f"{stem}_low.ply"
    make_lod(in_path, p_low, args.low)
    if p_low.exists():
        print(f"  saved: {p_low.name}  ({p_low.stat().st_size/1024/1024:.1f} MB)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
