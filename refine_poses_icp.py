"""
V29 helper - refine COLMAP poses using Femto ToF point-cloud ICP.

COLMAP poses are RGB-only and have feature-matching residual error. ICP
re-aligning each frame's depth cloud against its neighbours tightens
registration (typically 5-10% on indoor handheld scenes).

Approach: pairwise coloured ICP between adjacent frames; build a pose
graph with those edges plus a few "long-range" edges for loop closure;
globally optimise via Open3D, then write a new transforms.json.

Usage:
    conda run -n da3 python refine_poses_icp.py
        [--max-icp-distance 0.05]  [--neighbor-count 1]  [--loop-closure 0]

Output: colmap/dense/transforms_icp_refined.json
"""
import argparse
import json
import sys
from copy import deepcopy
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
    p.add_argument("--bounds-json",   default="colmap/dense/tof_bounds.json")
    p.add_argument("--in-transforms", default="colmap/dense/transforms.json")
    p.add_argument("--out-transforms", default="colmap/dense/transforms_icp_refined.json")
    p.add_argument("--max-icp-distance", type=float, default=0.05,
                   help="Coloured-ICP correspondence threshold in metres (default 5 cm)")
    p.add_argument("--neighbor-count", type=int, default=1,
                   help="Adjacent-frame edges per node (default 1 = i to i+1)")
    p.add_argument("--loop-closure", type=int, default=0,
                   help="Long-range edges between i and i+N for N in this many steps (default 0 = off)")
    p.add_argument("--voxel-size", type=float, default=0.01,
                   help="Point-cloud downsample voxel in metres (default 1 cm)")
    p.add_argument("--min-depth", type=float, default=0.25)
    p.add_argument("--max-depth", type=float, default=5.3)
    return p.parse_args()


def _build_pcd(depth_m: np.ndarray, conf: np.ndarray, rgb: np.ndarray,
               K: np.ndarray, min_d: float, max_d: float, voxel: float):
    H, W = depth_m.shape
    valid = conf & (depth_m >= min_d) & (depth_m <= max_d)
    vs, us = np.where(valid)
    if len(vs) == 0:
        return None
    d  = depth_m[vs, us].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (us - cx) / fx * d
    y_cam = (vs - cy) / fy * d
    z_cam = d
    pts = np.stack([x_cam, y_cam, z_cam], axis=1)

    # Per-point colour from RGB
    if rgb is not None and rgb.shape[:2] == depth_m.shape:
        c = rgb[vs, us].astype(np.float64) / 255.0
    else:
        c = np.full((len(vs), 3), 0.7)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(c)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    return pcd


def _icp_refine(src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud,
                init: np.ndarray, max_dist: float):
    """Coloured ICP. Returns (transform, fitness, rmse). transform takes
    src into tgt's frame."""
    result = o3d.pipelines.registration.registration_colored_icp(
        src, tgt, max_dist, init,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50,
        ),
    )
    return result.transformation, result.fitness, result.inlier_rmse


def main():
    args = _parse_args()
    root = Path(args.project_root or __file__).resolve()
    if root.is_file():
        root = root.parent

    sparse_dir   = root / args.colmap_sparse
    images_dir   = root / args.images_dir
    depths_dir   = root / args.depths_dir
    bounds_p     = root / args.bounds_json
    in_tj        = root / args.in_transforms
    out_tj       = root / args.out_transforms

    print("=== V29: ICP pose refinement ===")
    bounds = json.loads(bounds_p.read_text())
    colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])
    print(f"  COLMAP->metric: {colmap_to_metric:.6f}")

    cameras = g_da3._read_cameras(sparse_dir / "cameras.bin")
    images  = sorted(g_da3._read_images(sparse_dir / "images.bin"),
                     key=lambda i: i["name"])
    print(f"  {len(images)} frames, {len(cameras)} camera(s)")

    # Build per-frame point clouds (in metres, camera-local)
    print("  building per-frame point clouds ...")
    import cv2
    pcds = []
    metas = []
    for i, img in enumerate(images):
        stem = Path(img["name"]).stem
        dp   = depths_dir / f"{stem}.npy"
        cp   = depths_dir / f"{stem}_conf.npy"
        rp   = images_dir / img["name"]
        if not dp.exists():
            print(f"    skip {img['name']} (no depth)"); continue
        depth = np.load(dp).astype(np.float32)
        conf  = np.load(cp).astype(bool) if cp.exists() else np.ones_like(depth, dtype=bool)
        rgb   = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB) if rp.exists() else None
        if rgb is not None and rgb.shape[:2] != depth.shape:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
        cam = cameras[img["camera_id"]]
        K   = g_da3._intrinsics_from_camera(cam)
        pcd = _build_pcd(depth, conf, rgb, K,
                         args.min_depth, args.max_depth, args.voxel_size)
        if pcd is None:
            continue
        pcds.append(pcd)
        metas.append(img)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(images)}")

    n = len(pcds)
    print(f"  built {n} clouds")

    # Initial c2w in metres from COLMAP
    init_c2w = []
    for img in metas:
        w2c = g_da3._w2c_from_image(img).astype(np.float64)
        w2c[:3, 3] *= colmap_to_metric
        init_c2w.append(np.linalg.inv(w2c))

    # Build pose graph: nodes = c2w, edges = relative transforms via ICP
    print("  building pose graph + ICP edges ...")
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for c2w in init_c2w:
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(c2w))

    n_good_edges = 0
    n_failed_edges = 0
    # Adjacent edges
    edge_pairs = [(i, j) for i in range(n)
                  for j in range(i + 1, min(i + 1 + args.neighbor_count, n))]
    # Loop-closure edges
    if args.loop_closure > 0:
        step = max(1, n // (args.loop_closure + 1))
        for i in range(n):
            j = (i + step) % n
            if j > i and (i, j) not in edge_pairs:
                edge_pairs.append((i, j))

    for k, (i, j) in enumerate(edge_pairs):
        # Relative init: tgt_c2w^-1 @ src_c2w  (puts src into tgt frame)
        rel_init = np.linalg.inv(init_c2w[j]) @ init_c2w[i]
        try:
            T, fitness, rmse = _icp_refine(pcds[i], pcds[j], rel_init,
                                            args.max_icp_distance)
        except Exception as e:
            n_failed_edges += 1
            continue
        if fitness < 0.3:
            n_failed_edges += 1
            continue
        # Information matrix (6x6) for the edge
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            pcds[i], pcds[j], args.max_icp_distance, T,
        )
        uncertain = (j != i + 1)  # treat non-adjacent edges as loop closures
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i, j, T, info, uncertain=uncertain,
            )
        )
        n_good_edges += 1
        if (k + 1) % 20 == 0:
            print(f"    edges {k+1}/{len(edge_pairs)}  good={n_good_edges} bad={n_failed_edges}")

    print(f"  total edges: good={n_good_edges}  bad={n_failed_edges}")

    if n_good_edges == 0:
        print("  ERROR: no good ICP edges - aborting", file=sys.stderr)
        return 2

    # Optimise pose graph
    print("  optimising pose graph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=args.max_icp_distance,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning):
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

    # Diff: average translation change vs init (in metres)
    deltas = []
    for i, node in enumerate(pose_graph.nodes):
        t_init  = init_c2w[i][:3, 3]
        t_after = node.pose[:3, 3]
        deltas.append(float(np.linalg.norm(t_after - t_init)))
    deltas = np.array(deltas)
    print(f"  pose translation change (m):  mean={deltas.mean()*1000:.1f} mm  "
          f"max={deltas.max()*1000:.1f} mm  median={np.median(deltas)*1000:.1f} mm")

    # Write refined transforms.json. Keep nerfstudio's c2w convention.
    # nerfstudio uses OpenGL/right-handed-Y-up; COLMAP uses OpenCV.  The
    # in_transforms file is already in nerfstudio convention, so just
    # replace each transform_matrix with the refined c2w but converted back
    # to nerfstudio convention.  Simplest: re-derive c2w_ns from refined
    # COLMAP w2c using the same helper colmap_to_ns uses.
    in_data = json.loads(in_tj.read_text())
    name_to_node = {meta["name"]: i for i, meta in enumerate(metas)}

    n_updated, n_unchanged = 0, 0
    for frame in in_data["frames"]:
        img_name = Path(frame["file_path"]).name
        if img_name in name_to_node:
            i = name_to_node[img_name]
            c2w_m = pose_graph.nodes[i].pose            # metres
            # Convert metric c2w back to COLMAP c2w (multiply translation by scale)
            c2w_colmap = c2w_m.copy()
            c2w_colmap[:3, 3] *= float(bounds["scale_factor_da3_to_colmap"])

            # nerfstudio expects c2w with +Y up, -Z forward (OpenGL).
            # colmap_to_ns.py applies a flip: c2w_ns = c2w_opencv * diag(1,-1,-1,1)
            flip = np.diag([1.0, -1.0, -1.0, 1.0])
            c2w_ns = c2w_colmap @ flip

            frame["transform_matrix"] = c2w_ns.tolist()
            n_updated += 1
        else:
            n_unchanged += 1

    print(f"  updated {n_updated} frames, {n_unchanged} unchanged")
    out_tj.parent.mkdir(parents=True, exist_ok=True)
    out_tj.write_text(json.dumps(in_data, indent=2))
    print(f"  saved: {out_tj}")
    print(f"\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
