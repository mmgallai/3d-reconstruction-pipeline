"""One-off: bake atlas -> per-vertex RGB on per-object PLYs.

Loads each per-object mesh from the segmenter's OBJ (whose UVs are correctly
oriented relative to scene_textured0.png), samples the atlas at each vertex's
UV to get per-vertex RGBA, and overwrites the destination PLY at
output/pipeline_v32_test_run/mesh/objects/<slug>.ply with a self-contained
vertex-coloured PLY (no TextureFile reference).

We load from the OBJ (not the VCG-PLY) because the VCG PLY stores UV in
bottom-up convention while the atlas image is top-left origin; trimesh
honours the OBJ's UVs verbatim and produces correct colours, whereas the
VCG PLY UVs sampled directly hit the orange (255,127,39) atlas padding
because of the V flip.

After all 3 PLYs are baked, deletes the now-unused scene_textured0.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData


HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "output" / "segmented_v32_data3_v9a_fp_v2"
DST_DIR = HERE / "output" / "pipeline_v32_test_run" / "mesh" / "objects"
SLUGS = ["blue_box", "red_lobster_figurine", "white_water_bottle"]


def bake_one(slug: str) -> dict:
    obj_src = SRC_DIR / slug / f"{slug}_extracted.obj"
    dst_ply = DST_DIR / f"{slug}.ply"
    mesh = trimesh.load(str(obj_src), force="mesh", process=False)
    if not hasattr(mesh.visual, "to_color"):
        raise RuntimeError(f"{obj_src}: visual is "
                           f"{type(mesh.visual).__name__}, no to_color()")
    color_visual = mesh.visual.to_color()
    vc = np.asarray(color_visual.vertex_colors)
    if vc.shape[0] != len(mesh.vertices):
        raise RuntimeError(f"{obj_src}: vertex_colors {vc.shape} != verts "
                           f"{len(mesh.vertices)}")
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vc)
    mesh.export(str(dst_ply))
    # Verify
    pd = PlyData.read(str(dst_ply))
    v = pd["vertex"]
    props = {p.name for p in v.properties}
    for need in ("red", "green", "blue"):
        if need not in props:
            raise RuntimeError(f"{dst_ply}: saved PLY missing '{need}' "
                               f"property; have {sorted(props)}")
    mean_rgb = tuple(int(x) for x in vc[:, :3].mean(axis=0).round(0))
    size_mb = dst_ply.stat().st_size / 1_048_576
    return {
        "path": str(dst_ply.resolve()).replace("\\", "/"),
        "n_verts": int(len(mesh.vertices)),
        "mean_rgb": f"({mean_rgb[0]}, {mean_rgb[1]}, {mean_rgb[2]})",
        "size_mb": round(size_mb, 4),
    }


def main():
    results = []
    for slug in SLUGS:
        print(f"baking {slug} ...", flush=True)
        rec = bake_one(slug)
        print(f"  -> n_verts={rec['n_verts']} mean_rgb={rec['mean_rgb']} "
              f"size={rec['size_mb']} MB", flush=True)
        results.append(rec)
    atlas = DST_DIR / "scene_textured0.png"
    if atlas.exists():
        atlas.unlink()
        print(f"deleted {atlas}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
