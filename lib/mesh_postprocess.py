"""
Standalone mesh post-processor — invoked via `conda run -n da3 python lib/mesh_postprocess.py <in.ply> <out.ply>`.

Splits the input mesh into connected components, drops tiny ones, fixes
winding, fills small triangle/quad holes with trimesh, and writes a clean
PLY ready for TextureMesh.
"""
import sys
from pathlib import Path

import trimesh


def main(in_path: Path, out_path: Path, min_component_faces: int = 100) -> int:
    if not in_path.exists():
        print(f"  ERROR: input mesh not found: {in_path}", flush=True)
        return 2

    m = trimesh.load(str(in_path), force="mesh", process=False)
    v0, f0 = len(m.vertices), len(m.faces)
    print(f"  loaded: {v0:,} verts / {f0:,} faces", flush=True)

    if f0 == 0:
        print("  ERROR: input mesh has zero faces", flush=True)
        return 3

    components = m.split(only_watertight=False)
    big = [c for c in components if len(c.faces) >= min_component_faces]
    print(f"  components: {len(components):,} total, {len(big):,} kept (>= {min_component_faces} faces)", flush=True)
    if not big:
        print("  ERROR: every component dropped — mesh too fragmented", flush=True)
        return 4
    m = trimesh.util.concatenate(big)

    m.fix_normals()
    print("  winding fixed", flush=True)

    filled = trimesh.repair.fill_holes(m)
    print(f"  fill_holes returned: {filled}", flush=True)

    m.export(str(out_path))
    v1, f1 = len(m.vertices), len(m.faces)
    print(f"  saved: {v1:,} verts / {f1:,} faces  (Δ {v1-v0:+,} verts, {f1-f0:+,} faces)", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: mesh_postprocess.py <input.ply> <output.ply> [min_component_faces=100]", file=sys.stderr)
        sys.exit(1)
    in_p = Path(sys.argv[1])
    out_p = Path(sys.argv[2])
    min_f = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    sys.exit(main(in_p, out_p, min_f))
