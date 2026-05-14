"""
V31 - Custom UV-atlas baker for the V26 TSDF mesh.

Bypasses OpenMVS TextureMesh (which SIGSEGVs on TSDF marching-cubes
topology). Uses xatlas to compute a UV unwrap of the TSDF mesh, then
projects each face into its best-view camera (COLMAP poses) and samples
RGB to bake an atlas PNG. Result: TSDF geometry coverage + RGB-photo
texture sharpness.

Output:
  output/mesh_v31_tsdf_atlas/mesh_v31_tsdf_atlas.obj
  output/mesh_v31_tsdf_atlas/mesh_v31_tsdf_atlas.png   (texture)
  output/mesh_v31_tsdf_atlas/mesh_v31_tsdf_atlas.mtl
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import xatlas

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_da3_depths as g_da3  # noqa


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=None)
    p.add_argument("--tsdf-mesh", default="output/mesh_v26_tsdf/mesh_v26_tsdf.ply")
    p.add_argument("--colmap-sparse", default="colmap/dense/sparse/0")
    p.add_argument("--images-dir",    default="colmap/dense/images")
    p.add_argument("--bounds-json",   default="colmap/dense/tof_bounds.json")
    p.add_argument("--output-name",   default="mesh_v31_tsdf_atlas")
    p.add_argument("--atlas-size",    type=int, default=4096,
                   help="Texture atlas resolution (default 4096, i.e. 4K)")
    p.add_argument("--decimate-faces", type=int, default=500_000,
                   help="Decimate TSDF mesh to this many faces first (0 = off)")
    return p.parse_args()


def _project_face_centers(verts, faces, w2c, K):
    """For each triangle, compute the (u, v, z) of its centroid in image
    coordinates. Returns (centroids_uv, centroids_z, in_frame_mask)."""
    centroids = verts[faces].mean(axis=1)  # (F, 3)
    ones = np.ones((centroids.shape[0], 1))
    homog = np.hstack([centroids, ones])
    cam   = (w2c @ homog.T).T[:, :3]
    z     = cam[:, 2]
    uv    = (K @ cam.T).T
    uv    = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
    return uv, z, centroids


def main():
    args = _parse_args()
    root = Path(args.project_root or __file__).resolve()
    if root.is_file():
        root = root.parent

    tsdf_path = root / args.tsdf_mesh
    sparse    = root / args.colmap_sparse
    images_d  = root / args.images_dir
    bounds_p  = root / args.bounds_json
    out_dir   = root / "output" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== V31: TSDF + custom xatlas UV baker ===")
    bounds = json.loads(bounds_p.read_text())
    colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])

    # 1) Load + decimate TSDF mesh
    print(f"  loading {tsdf_path} ...")
    mesh = o3d.io.read_triangle_mesh(str(tsdf_path))
    mesh.vertex_colors = o3d.utility.Vector3dVector()
    if args.decimate_faces and len(mesh.triangles) > args.decimate_faces:
        before = len(mesh.triangles)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.decimate_faces)
        print(f"  decimated: {before:,} -> {len(mesh.triangles):,} faces")
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    verts = np.asarray(mesh.vertices, dtype=np.float64)   # in metres
    faces = np.asarray(mesh.triangles, dtype=np.uint32)
    print(f"  mesh: {len(verts):,} verts, {len(faces):,} faces")

    # 2) UV unwrap with xatlas (0.0.11 API: build Atlas, add mesh, generate)
    print("  unwrapping UVs (xatlas) ...")
    atlas = xatlas.Atlas()
    atlas.add_mesh(verts.astype(np.float32), faces.astype(np.uint32))
    chart_options = xatlas.ChartOptions()
    pack_options  = xatlas.PackOptions()
    pack_options.resolution = args.atlas_size
    pack_options.bilinear   = True
    atlas.generate(chart_options=chart_options, pack_options=pack_options, verbose=False)
    vmapping, indices, uvs = atlas[0]
    # vmapping: (N_unique_uv,) maps each new vertex idx back into the original
    # indices:  (F, 3)  new vertex indices into vmapping/uvs
    # uvs:      (N_unique_uv, 2)  per-vertex UV in [0,1]
    n_uv_verts = vmapping.shape[0]
    print(f"  unwrapped: {n_uv_verts:,} UV vertices, {indices.shape[0]:,} faces")

    # 3) Project each face into every camera, find best-view per face
    print("  reading COLMAP poses ...")
    cameras = g_da3._read_cameras(sparse / "cameras.bin")
    images  = sorted(g_da3._read_images(sparse / "images.bin"),
                     key=lambda i: i["name"])

    print("  finding best view per face (highest centroid normal-dot, in-frame, in-front) ...")
    # Compute face normals
    p0 = verts[faces[:, 0]]; p1 = verts[faces[:, 1]]; p2 = verts[faces[:, 2]]
    f_normals = np.cross(p1 - p0, p2 - p0)
    f_normals_norm = np.linalg.norm(f_normals, axis=1, keepdims=True) + 1e-8
    f_normals = f_normals / f_normals_norm

    best_view  = np.full(len(faces), -1, dtype=np.int32)
    best_score = np.full(len(faces), -np.inf, dtype=np.float32)
    best_uvz   = [None] * len(faces)   # (uv, z) per face from the chosen view

    for ci, img in enumerate(images):
        cam = cameras[img["camera_id"]]
        K   = g_da3._intrinsics_from_camera(cam)
        w2c = g_da3._w2c_from_image(img).astype(np.float64)
        w2c[:3, 3] *= colmap_to_metric
        H, W = int(cam["height"]), int(cam["width"])

        uv, z, centroids = _project_face_centers(verts, faces, w2c, K)
        in_front = z > 0.05
        in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        valid    = in_front & in_frame

        # View direction (camera centre to face centre, in world). For score
        # use dot product with face normal (front-facing is +ve, big = good).
        cam_centre_world = -w2c[:3, :3].T @ w2c[:3, 3]  # c2w translation
        view_dirs = centroids - cam_centre_world
        view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8)
        # We want the camera to be looking AT the face: dot(face_normal, -view_dir) > 0
        dot = -(f_normals * view_dirs).sum(axis=1)

        score = np.where(valid, dot, -np.inf)
        better = score > best_score
        best_score = np.where(better, score, best_score)
        best_view  = np.where(better, ci, best_view)
        for fi in np.where(better)[0]:
            best_uvz[fi] = (uv[fi], z[fi])

        if (ci + 1) % 20 == 0:
            print(f"    cam {ci+1}/{len(images)}")

    covered = best_view >= 0
    print(f"  faces with view: {int(covered.sum()):,} / {len(faces):,}  ({covered.mean()*100:.1f}%)")
    if covered.sum() == 0:
        print("  ERROR: no faces could be projected. Aborting.", file=sys.stderr)
        return 2

    # 4) Pre-load camera images per used view, then bake atlas
    print(f"  baking {args.atlas_size}x{args.atlas_size} atlas ...")
    used_views = np.unique(best_view[covered])
    cache = {}
    for vi in used_views:
        path = images_d / images[int(vi)]["name"]
        bgr  = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        cache[int(vi)] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    atlas = np.zeros((args.atlas_size, args.atlas_size, 3), dtype=np.uint8)
    atlas_mask = np.zeros((args.atlas_size, args.atlas_size), dtype=bool)
    px = uvs[:, 0] * (args.atlas_size - 1)
    py = (1.0 - uvs[:, 1]) * (args.atlas_size - 1)
    px = px.astype(np.float32); py = py.astype(np.float32)
    # For each face, splat its texels into atlas (barycentric rasterization).
    # Use cv2.fillConvexPoly for simplicity (one solid colour per face: the
    # centroid colour from the best view). Cheap & cheerful - not seam-aware.
    n_baked, n_failed = 0, 0
    for fi in range(len(faces)):
        vi = int(best_view[fi])
        if vi < 0 or vi not in cache:
            n_failed += 1
            continue
        img_rgb = cache[vi]
        uv_face, z_face = best_uvz[fi]  # uv: pixel coords in source image, len 3 per face
        # Sample centroid pixel from source image
        u_centre = float(uv_face[0]); v_centre = float(uv_face[1])
        u_idx = int(np.clip(u_centre, 0, img_rgb.shape[1] - 1))
        v_idx = int(np.clip(v_centre, 0, img_rgb.shape[0] - 1))
        colour = img_rgb[v_idx, u_idx]

        # Fill the triangle in atlas with that colour
        tri = np.array([
            [px[indices[fi, 0]], py[indices[fi, 0]]],
            [px[indices[fi, 1]], py[indices[fi, 1]]],
            [px[indices[fi, 2]], py[indices[fi, 2]]],
        ], dtype=np.int32)
        cv2.fillConvexPoly(atlas, tri, color=tuple(int(c) for c in colour))
        cv2.fillConvexPoly(atlas_mask.view(np.uint8).reshape(args.atlas_size, args.atlas_size),
                           tri, color=1)
        n_baked += 1

    print(f"  baked {n_baked:,} faces ({n_failed:,} skipped)")
    print(f"  atlas coverage: {atlas_mask.mean()*100:.1f}% of texels")

    # 5) Write outputs (OBJ + MTL + PNG)
    obj_path = out_dir / f"{args.output_name}.obj"
    mtl_path = out_dir / f"{args.output_name}.mtl"
    png_path = out_dir / f"{args.output_name}.png"
    cv2.imwrite(str(png_path), cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR))
    print(f"  saved: {png_path.name}")

    # OBJ with UV
    new_verts = verts[vmapping]   # (N_uv, 3) - vertices in the order the UVs expect
    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("usemtl atlas\n")
        for v in new_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for u in uvs:
            f.write(f"vt {u[0]:.6f} {u[1]:.6f}\n")
        for tri in indices:
            # OBJ is 1-indexed; same idx for v and vt since we packed them aligned
            f.write(f"f {tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1} {tri[2]+1}/{tri[2]+1}\n")
    print(f"  saved: {obj_path.name}")

    with open(mtl_path, "w") as f:
        f.write("newmtl atlas\n")
        f.write("Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 10\n")
        f.write(f"map_Kd {png_path.name}\n")
    print(f"  saved: {mtl_path.name}")

    print(f"\n=== V31 done ===")
    print(f"  geometry: {len(new_verts):,} verts / {len(indices):,} faces")
    print(f"  atlas:    {args.atlas_size}x{args.atlas_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
