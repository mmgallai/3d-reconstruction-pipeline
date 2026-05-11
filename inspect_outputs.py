"""
Deep mesh-quality comparison.

Compares:
  • RealityScan (textured mesh, vertex-colored)
  • Our V14 (Tier-0 baseline) — both raw and textured
  • Our V15 (Tier-1: DensifyPointCloud + post-processing) — both raw and textured

Reports per mesh:
  - geometry: vertices, faces, bounding box, surface area, volume
  - topology: watertight?, holes, non-manifold edges, components
  - color/texture: vertex colors? UV mapped? texture count
"""
from pathlib import Path
import numpy as np
import trimesh

ROOT = Path(r"C:\Users\mgallai\Projects\3d_automated\reconstruction_project")

# Tuples of (label, path, kind)
TARGETS = [
    ("RealityScan",        ROOT / "output" / "other apps" / "reality_scan.ply",     "textured"),
    ("V14 textured",       ROOT / "output" / "mesh_v14"  / "mesh_v14_openmvs.ply",  "textured"),
    ("V17 textured+scaled",ROOT / "output" / "mesh_v17"  / "mesh_v17_openmvs.ply",  "textured"),
    ("V18 textured+scaled",ROOT / "output" / "mesh_v18"  / "mesh_v18_openmvs.ply",  "textured"),
    ("V20 vert-colored",   ROOT / "output" / "mesh_v20"  / "mesh_v20_openmvs.ply",  "vc"),
]


def fmt_dim(v):
    return f"({v[0]:7.3f}, {v[1]:7.3f}, {v[2]:7.3f})"


def analyse(name: str, path: Path) -> dict | None:
    print(f"\n{'='*78}\n{name}  —  {path.name}  ({path.stat().st_size/1_048_576:.1f} MB)\n{'='*78}")
    m = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        print("  not a triangle mesh, skipping")
        return None

    v = m.vertices; f = m.faces
    bb = v.max(0) - v.min(0)
    boundary = m.edges[trimesh.grouping.group_rows(m.edges_sorted, require_count=1)]
    edges_ct = np.unique(m.edges_sorted, axis=0, return_counts=True)[1]
    nm = int((edges_ct > 2).sum())
    cc = trimesh.graph.connected_components(m.face_adjacency, min_len=1)
    largest = max(len(c) for c in cc)

    has_uv = hasattr(m.visual, "uv") and m.visual.uv is not None
    try:
        has_vc = (m.visual is not None
                  and hasattr(m.visual, "vertex_colors")
                  and m.visual.vertex_colors is not None
                  and len(m.visual.vertex_colors) > 0)
    except Exception:
        has_vc = False

    print(f"  Vertices              : {len(v):>12,}")
    print(f"  Faces                 : {len(f):>12,}")
    print(f"  Bounding box          : {fmt_dim(bb)}")
    print(f"  BB diagonal           : {np.linalg.norm(bb):.3f}")
    print(f"  Surface area          : {m.area:.2f}")
    try:
        print(f"  Volume                : {m.volume:.2f}")
    except Exception:
        print(f"  Volume                : (not closed)")
    print(f"  Watertight            : {m.is_watertight}")
    print(f"  Boundary edges (holes): {len(boundary):>12,}")
    print(f"  Non-manifold edges    : {nm:>12,}")
    print(f"  Connected components  : {len(cc):>12,}    (largest = {largest:,} = {largest/len(f)*100:.2f}%)")
    print(f"  Vertex colors         : {has_vc}")
    print(f"  UV mapped             : {has_uv}")

    return {
        "name": name, "path": path,
        "verts": len(v), "faces": len(f),
        "area": m.area,
        "watertight": m.is_watertight,
        "holes": len(boundary),
        "components": len(cc),
        "largest_pct": largest / len(f) * 100,
        "bb": bb,
    }


def main():
    results = []
    for label, path, _kind in TARGETS:
        if path is None or not path.exists():
            print(f"\n{label:18}: not found, skipping")
            continue
        r = analyse(label, path)
        if r:
            results.append(r)

    print(f"\n\n{'='*78}\n{'SUMMARY':^78}\n{'='*78}\n")
    headers = ["Mesh", "Verts", "Faces", "Holes", "Comps", "Largest%", "Watertight"]
    print(f"  {headers[0]:<20} {headers[1]:>12} {headers[2]:>12} {headers[3]:>10} {headers[4]:>7} {headers[5]:>10} {headers[6]:>11}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*7} {'-'*10} {'-'*11}")
    for r in results:
        print(f"  {r['name']:<20} {r['verts']:>12,} {r['faces']:>12,} {r['holes']:>10,} {r['components']:>7,} {r['largest_pct']:>9.2f}% {str(r['watertight']):>11}")


if __name__ == "__main__":
    main()
