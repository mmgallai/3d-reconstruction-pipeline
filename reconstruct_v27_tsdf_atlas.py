"""
V27 — TSDF mesh + OpenMVS UV-atlas texturing (best-of-both hybrid).

Combines:
  - V26 TSDF mesh geometry (better coverage, better connectivity)
  - V23 OpenMVS UV-atlas texturing (sharp 8K texture vs vertex colors)

Pipeline:
  1. Run tsdf_fusion_femto.py to produce a metric-meter mesh.
  2. Scale the TSDF mesh UP by Femto scale (to put it in COLMAP units so
     it matches the cameras in openmvs/scene.mvs).
  3. Hand it to OpenMVS TextureMesh (re-uses existing scene.mvs).
  4. Apply scale_calibrate (back to metres).
  5. LOD decimation + ambient occlusion bake (re-uses pipeline utilities).
  6. Package outputs in output/mesh_v27_tsdf_atlas/.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import openmvs_pipeline      # noqa: E402
from lib import mesh_scale_calibrate  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=None)
    p.add_argument("--tsdf-mesh", default="output/mesh_v26_tsdf/mesh_v26_tsdf.ply",
                   help="Pre-computed TSDF mesh in metres (from V26 script).")
    p.add_argument("--regen-tsdf", action="store_true",
                   help="Re-run tsdf_fusion_femto.py instead of reusing cached output.")
    p.add_argument("--bounds-json", default="colmap/dense/tof_bounds.json")
    p.add_argument("--output-name", default="mesh_v27_tsdf_atlas",
                   help="Folder under output/ for the deliverable.")
    return p.parse_args()


def _scale_ply_vertices(in_ply: Path, out_ply: Path, scale: float) -> None:
    """Multiply x,y,z of a binary-little-endian PLY by `scale`. Preserves all
    other vertex / face data byte-for-byte."""
    data = in_ply.read_bytes()
    end_marker = b"end_header\n"
    header_len = data.find(end_marker) + len(end_marker)
    header = data[:header_len].decode("ascii", errors="replace").splitlines()

    fmt_chars = {
        "char": "b", "uchar": "B", "uint8": "B", "int8": "b",
        "short": "h", "ushort": "H", "int": "i", "uint": "I",
        "float": "f", "float32": "f", "double": "d", "float64": "d",
    }
    type_size = {k: {"b": 1, "B": 1, "h": 2, "H": 2,
                     "i": 4, "I": 4, "f": 4, "d": 8}[v]
                 for k, v in fmt_chars.items()}

    # Parse vertex element + offsets.
    n_verts, vertex_props = None, []
    in_vertex = False
    for line in header:
        if line.startswith("element vertex "):
            n_verts = int(line.split()[2])
            in_vertex = True
        elif line.startswith("element ") and in_vertex:
            in_vertex = False
        elif line.startswith("property ") and in_vertex:
            parts = line.split()
            vertex_props.append((parts[1], parts[2]))

    record_size = sum(type_size[t] for t, _ in vertex_props)
    offsets = {}
    off = 0
    for t, name in vertex_props:
        if name in ("x", "y", "z"):
            offsets[name] = (off, t)
        off += type_size[t]

    is_double = offsets["x"][1] in ("double", "float64")
    np_dtype  = np.float64 if is_double else np.float32
    coord_size = 8 if is_double else 4

    raw = bytearray(data[header_len:header_len + n_verts * record_size])
    vbuf = np.frombuffer(raw, dtype=np.uint8).reshape(n_verts, record_size).copy()
    for k in ("x", "y", "z"):
        off, _ = offsets[k]
        vals  = np.frombuffer(vbuf[:, off:off + coord_size].tobytes(), dtype=np_dtype)
        scaled = (vals * scale).astype(np_dtype, copy=False)
        vbuf[:, off:off + coord_size] = np.frombuffer(
            scaled.tobytes(), dtype=np.uint8).reshape(n_verts, coord_size)

    out_bytes = bytearray(data)
    out_bytes[header_len:header_len + n_verts * record_size] = vbuf.tobytes()
    out_ply.write_bytes(bytes(out_bytes))


def main():
    args = _parse_args()
    root = Path(args.project_root or __file__).resolve()
    if root.is_file():
        root = root.parent

    tsdf_metric = root / args.tsdf_mesh
    bounds_p    = root / args.bounds_json
    out_dir     = root / "output" / args.output_name
    omvs_dir    = root / "openmvs"

    print(f"=== V27: TSDF mesh + OpenMVS UV-atlas ===")
    print(f"  TSDF mesh  : {tsdf_metric}")
    print(f"  bounds     : {bounds_p}")
    print(f"  output     : {out_dir}")

    # 1) Get TSDF mesh (regen or reuse)
    if args.regen_tsdf or not tsdf_metric.exists():
        print("  step 1: running tsdf_fusion_femto.py ...")
        subprocess.run([
            "conda", "run", "--no-capture-output", "-n", "da3",
            "python", str(root / "tsdf_fusion_femto.py"),
        ], check=True)
    else:
        print(f"  step 1: reusing {tsdf_metric.name}")

    # 2) Clean the TSDF mesh first.  TSDF marching cubes produces non-manifold
    # edges and tiny degenerate triangles that OpenMVS's atlas-packer crashes
    # on (SIGSEGV at TextureMesh's MapTextureColorChannel).  Use Open3D's
    # built-in cleanup which removes degenerate triangles, duplicates,
    # non-manifold edges, and unreferenced vertices.
    print("  step 2a: cleaning TSDF mesh (strip colors, decimate, manifold cleanup) ...")
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(tsdf_metric))
    before_v, before_f = len(mesh.vertices), len(mesh.triangles)
    # Strip vertex colors + normals - TextureMesh's atlas packer chokes on them
    mesh.vertex_colors = o3d.utility.Vector3dVector()
    mesh.vertex_normals = o3d.utility.Vector3dVector()
    # Decimate to ~500K faces - removes the tiny near-degenerate triangles
    # that the OpenMVS atlas-packer crashes on.
    target_faces = 500_000
    if len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    after_v, after_f = len(mesh.vertices), len(mesh.triangles)
    cleaned_tsdf = tsdf_metric.parent / f"{tsdf_metric.stem}_clean.ply"
    o3d.io.write_triangle_mesh(str(cleaned_tsdf), mesh, write_vertex_colors=False,
                               write_vertex_normals=False)
    print(f"    {before_v:,}v/{before_f:,}f -> {after_v:,}v/{after_f:,}f (decimated + cleaned)")

    # 2b) Scale TSDF mesh metres -> COLMAP units (matches openmvs/scene.mvs cameras)
    bounds = json.loads(bounds_p.read_text())
    colmap_scale = float(bounds["scale_factor_da3_to_colmap"])
    print(f"  step 2b: scaling cleaned TSDF mesh by {colmap_scale:.4f} (m -> COLMAP units)")

    scene_mesh_input = omvs_dir / "scene_mesh_v27_tsdf_colmap.ply"
    omvs_dir.mkdir(parents=True, exist_ok=True)
    _scale_ply_vertices(cleaned_tsdf, scene_mesh_input, colmap_scale)
    print(f"    wrote: {scene_mesh_input.name}  ({scene_mesh_input.stat().st_size/1024/1024:.1f} MB)")

    # 3) OpenMVS TextureMesh (uses scene.mvs cameras+images, applies UV atlas).
    # Resolution-level 1 (half-res images, 4K atlas) is more robust than 0
    # for irregular marching-cubes geometry - full-res atlas-packer hits
    # std::out_of_range on the TSDF mesh.
    print("  step 3: OpenMVS TextureMesh - applying 4K UV atlas (res-level 1 for stability) ...")
    ok = openmvs_pipeline.texture_mesh(
        project_root=root,
        mesh_ply=scene_mesh_input.name,
        resolution_level=1,
    )
    if not ok:
        print("  ERROR: TextureMesh failed", file=sys.stderr)
        return 2

    # 4) Move + rename outputs into out_dir, then scale-calibrate to metres
    out_dir.mkdir(parents=True, exist_ok=True)
    src_textured = omvs_dir / "scene_textured.ply"
    dst_textured = out_dir / f"{args.output_name}.ply"
    shutil.copy2(src_textured, dst_textured)
    # OpenMVS may also have written a .mtl / texture .png
    for ext in (".mtl", "0.png", "_material_0.png"):
        for src in omvs_dir.glob(f"scene_textured{ext}"):
            shutil.copy2(src, out_dir / src.name)

    print("  step 4: scale_calibrate — COLMAP units -> metres ...")
    rc = mesh_scale_calibrate.main(dst_textured, dst_textured, bounds_p)
    if rc != 0:
        print("  ERROR: scale_calibrate failed", file=sys.stderr)
        return rc

    print(f"\n=== V27 done ===")
    print(f"  mesh + atlas: {dst_textured}  ({dst_textured.stat().st_size/1024/1024:.1f} MB)")
    print(f"  (Run lod + AO via prune_splat / inspect_outputs if needed.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
