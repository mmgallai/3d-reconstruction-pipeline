"""
Paradigm C — TSDF volumetric fusion of Femto Mega RGB-D frames.

KinectFusion-style: skip COLMAP MVS / splatting entirely. Integrate every
ToF depth frame into an Open3D ScalableTSDFVolume using the COLMAP poses
(scaled to meters), then marching-cubes a colored triangle mesh.

The mesh is metric (meters) from the volume up. No splat involved, no
photogrammetric MVS. Pure depth-camera fusion.

Output: output/mesh_v26_tsdf/mesh_v26_tsdf.ply
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_da3_depths as g_da3  # noqa


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=None)
    p.add_argument("--colmap-sparse", default="colmap/dense/sparse/0")
    p.add_argument("--images-dir",    default="colmap/dense/images")
    p.add_argument("--depths-dir",    default="nerfstudio_data/depths_femto")
    p.add_argument("--bounds-json",   default="colmap/dense/tof_bounds.json",
                   help="Provides COLMAP→metric scale_factor_da3_to_colmap")
    p.add_argument("--output-dir",    default="output/mesh_v26_tsdf")
    p.add_argument("--voxel-size",    type=float, default=0.005,
                   help="TSDF voxel size in metres (default 5 mm)")
    p.add_argument("--sdf-trunc",     type=float, default=0.02,
                   help="SDF truncation distance in metres (default 2 cm)")
    p.add_argument("--min-depth",     type=float, default=0.25)
    p.add_argument("--max-depth",     type=float, default=5.3)
    return p.parse_args()


def main():
    args = _parse_args()
    root = Path(args.project_root or __file__).resolve()
    if root.is_file():
        root = root.parent

    sparse_dir = root / args.colmap_sparse
    images_dir = root / args.images_dir
    depths_dir = root / args.depths_dir
    out_dir    = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== TSDF fusion (Femto Mega RGB-D) ===")
    print(f"  COLMAP : {sparse_dir}")
    print(f"  images : {images_dir}")
    print(f"  depths : {depths_dir}")
    print(f"  output : {out_dir}")

    # ── Scale factor (COLMAP arbitrary → metres) ───────────────────────────
    bounds_p = root / args.bounds_json
    if not bounds_p.exists():
        print(f"  ERROR: bounds JSON not found: {bounds_p}", file=sys.stderr)
        return 2
    bounds = json.loads(bounds_p.read_text())
    colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])
    print(f"  COLMAP→metric scale: 1/{1/colmap_to_metric:.4f} = {colmap_to_metric:.6f}")

    # ── Read COLMAP poses + intrinsics ─────────────────────────────────────
    cameras = g_da3._read_cameras(sparse_dir / "cameras.bin")
    images  = g_da3._read_images(sparse_dir / "images.bin")
    print(f"  COLMAP: {len(cameras)} camera(s), {len(images)} registered image(s)")

    # ── TSDF volume ────────────────────────────────────────────────────────
    print(f"  TSDF: voxel={args.voxel_size*1000:.1f} mm, sdf_trunc={args.sdf_trunc*1000:.1f} mm")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # ── Integrate frames ───────────────────────────────────────────────────
    n_integrated, n_skipped = 0, 0
    for img in sorted(images, key=lambda i: i["name"]):
        img_name = img["name"]
        stem     = Path(img_name).stem
        depth_p  = depths_dir / f"{stem}.npy"
        conf_p   = depths_dir / f"{stem}_conf.npy"
        rgb_p    = images_dir / img_name
        if not depth_p.exists() or not rgb_p.exists():
            n_skipped += 1
            continue

        depth_m = np.load(depth_p).astype(np.float32)
        conf    = np.load(conf_p).astype(bool) if conf_p.exists() \
                  else np.ones_like(depth_m, dtype=bool)

        # Apply ToF validity (range + confidence). Invalidate as 0 (TSDF skips).
        valid = conf & (depth_m >= args.min_depth) & (depth_m <= args.max_depth)
        depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)

        # RGB
        rgb_bgr = o3d.io.read_image(str(rgb_p))
        rgb_np  = np.asarray(rgb_bgr)
        h_rgb, w_rgb = rgb_np.shape[:2]
        if depth_m.shape != (h_rgb, w_rgb):
            # Resize depth to match RGB (Femto's HW D2C should already match)
            import cv2
            depth_m = cv2.resize(depth_m, (w_rgb, h_rgb),
                                 interpolation=cv2.INTER_NEAREST)

        # Build Open3D images
        color_img = o3d.geometry.Image(rgb_np)
        depth_img = o3d.geometry.Image(depth_m)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=1.0,                 # depth already in metres
            depth_trunc=args.max_depth + 0.1,
            convert_rgb_to_intensity=False,
        )

        # Intrinsics from COLMAP camera
        cam = cameras[img["camera_id"]]
        K = g_da3._intrinsics_from_camera(cam)
        intr = o3d.camera.PinholeCameraIntrinsic(
            width=int(cam["width"]), height=int(cam["height"]),
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
        )

        # COLMAP w2c (world-to-camera) — Open3D wants the same convention.
        # Scale only the translation: COLMAP units → metres.
        w2c = g_da3._w2c_from_image(img).astype(np.float64)
        w2c[:3, 3] *= colmap_to_metric  # rotation stays; translation rescaled

        volume.integrate(rgbd, intr, w2c)
        n_integrated += 1
        if n_integrated % 20 == 0:
            print(f"    integrated {n_integrated}/{len(images)}")

    print(f"  integrated {n_integrated} frames ({n_skipped} skipped)")

    # ── Extract mesh ───────────────────────────────────────────────────────
    print("  extracting triangle mesh (marching cubes) ...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    n_v, n_f = len(mesh.vertices), len(mesh.triangles)
    print(f"    {n_v:,} verts, {n_f:,} faces")

    out_ply = out_dir / "mesh_v26_tsdf.ply"
    o3d.io.write_triangle_mesh(str(out_ply), mesh)
    print(f"  saved: {out_ply}  ({out_ply.stat().st_size/1024/1024:.1f} MB)")

    # Also write the dense point cloud (for inspection / diagnostics)
    pcd = volume.extract_point_cloud()
    out_pcd = out_dir / "mesh_v26_tsdf_points.ply"
    o3d.io.write_point_cloud(str(out_pcd), pcd)
    print(f"  saved: {out_pcd}  ({out_pcd.stat().st_size/1024/1024:.1f} MB)")

    print(f"\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
