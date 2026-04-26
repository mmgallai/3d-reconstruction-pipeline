"""
Convert a COLMAP sparse model to Nerfstudio transforms.json format.

Reads cameras.bin, images.bin (and optionally points3D.bin) from a COLMAP
sparse model directory and writes a transforms.json suitable for the
nerfstudio-data dataparser.

Coordinate convention
---------------------
COLMAP stores world-to-camera in OpenCV convention:
    p_cam = R_w2c @ p_world + t_w2c     (Y-down, Z-into-scene)

Nerfstudio transforms.json stores camera-to-world in OpenGL convention:
    p_world = R_c2w @ p_cam + t_c2w     (Y-up, Z-out-of-scene)

Conversion:
    1. Invert COLMAP w2c -> c2w:  R_c2w = R_w2c^T,  t_c2w = -R_c2w @ t_w2c
    2. Flip Y and Z columns of R_c2w to go from OpenCV to OpenGL axes.
"""

import json
import logging
import struct
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Number of floating-point parameters for each COLMAP camera model ID
_MODEL_NUM_PARAMS: dict[int, int] = {
    0:  3,   # SIMPLE_PINHOLE:      f, cx, cy
    1:  4,   # PINHOLE:             fx, fy, cx, cy
    2:  4,   # SIMPLE_RADIAL:       f, cx, cy, k1
    3:  5,   # RADIAL:              f, cx, cy, k1, k2
    4:  8,   # OPENCV:              fx, fy, cx, cy, k1, k2, p1, p2
    5: 12,   # OPENCV_FISHEYE:      fx, fy, cx, cy, k1..k4  (+ extras)
    6:  9,   # FULL_OPENCV:         fx, fy, cx, cy, k1, k2, p1, p2, k3
    7:  5,   # FOV:                 fx, fy, cx, cy, omega
    8:  4,   # SIMPLE_RADIAL_FISHEYE
    9:  5,   # RADIAL_FISHEYE
    10: 13,  # THIN_PRISM_FISHEYE
}


# ──────────────────────────────────────────────────────────────
# Binary readers
# ──────────────────────────────────────────────────────────────

def _read_cameras(path: Path) -> dict:
    """Return {camera_id: {model_id, width, height, params}} from cameras.bin."""
    cameras: dict = {}
    with open(path, "rb") as f:
        (num_cameras,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_cameras):
            (camera_id,) = struct.unpack("<I", f.read(4))
            (model_id,)  = struct.unpack("<i", f.read(4))
            (width,)     = struct.unpack("<Q", f.read(8))
            (height,)    = struct.unpack("<Q", f.read(8))
            n_params = _MODEL_NUM_PARAMS.get(model_id, 4)
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[camera_id] = dict(
                model_id=model_id,
                width=int(width),
                height=int(height),
                params=params,
            )
    return cameras


def _read_images(path: Path) -> list:
    """Return list of {image_id, qvec, tvec, camera_id, name} from images.bin."""
    images = []
    with open(path, "rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            (image_id,)  = struct.unpack("<I", f.read(4))
            qvec         = struct.unpack("<4d", f.read(32))   # qw, qx, qy, qz
            tvec         = struct.unpack("<3d", f.read(24))
            (camera_id,) = struct.unpack("<I", f.read(4))
            # Null-terminated filename
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8", errors="replace")
            (num_pts2d,) = struct.unpack("<Q", f.read(8))
            f.read(24 * num_pts2d)   # skip xys + point3D_ids
            images.append(dict(
                image_id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=name,
            ))
    return images


def _read_points3d(path: Path) -> tuple:
    """
    Return (xyz, rgb) ndarrays of shape (N, 3) from points3D.bin.
    xyz dtype float32, rgb dtype uint8.
    """
    xyz_list, rgb_list = [], []
    with open(path, "rb") as f:
        (num_pts,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_pts):
            f.read(8)                           # point3D_id  (uint64)
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            f.read(8)                           # reprojection error (double)
            (num_el,) = struct.unpack("<Q", f.read(8))
            f.read(8 * num_el)                  # image_id + point2D_idx pairs
            xyz_list.append(xyz)
            rgb_list.append(rgb)
    return (
        np.array(xyz_list, dtype=np.float32),
        np.array(rgb_list, dtype=np.uint8),
    )


# ──────────────────────────────────────────────────────────────
# Coordinate conversion helpers
# ──────────────────────────────────────────────────────────────

def _qvec_to_rotmat(qvec) -> np.ndarray:
    """COLMAP quaternion [qw, qx, qy, qz] → 3×3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def _colmap_to_nerf_c2w(qvec, tvec) -> np.ndarray:
    """
    Convert COLMAP world-to-camera (qvec, tvec) to a 4×4 Nerfstudio
    camera-to-world matrix in OpenGL convention.
    """
    R_w2c = _qvec_to_rotmat(qvec)
    t_w2c = np.array(tvec, dtype=np.float64)

    # Invert: camera-to-world in COLMAP (OpenCV) convention
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3]  = t_c2w

    # Flip Y and Z axes: OpenCV (Y-down, Z-into-scene) → OpenGL (Y-up, Z-out)
    c2w[:3, 1:3] *= -1
    return c2w


def _extract_intrinsics(cam: dict) -> dict:
    """
    Extract fl_x, fl_y, cx, cy, k1, k2, p1, p2 from a COLMAP camera record.
    After image_undistorter output, cameras are always PINHOLE (model_id=1).
    """
    p  = cam["params"]
    mid = cam["model_id"]
    zero = dict(k1=0., k2=0., p1=0., p2=0.)

    if mid == 0:                    # SIMPLE_PINHOLE: f, cx, cy
        return dict(fl_x=p[0], fl_y=p[0], cx=p[1], cy=p[2], **zero)
    elif mid == 1:                  # PINHOLE: fx, fy, cx, cy
        return dict(fl_x=p[0], fl_y=p[1], cx=p[2], cy=p[3], **zero)
    elif mid == 2:                  # SIMPLE_RADIAL: f, cx, cy, k1
        return dict(fl_x=p[0], fl_y=p[0], cx=p[1], cy=p[2],
                    k1=p[3], k2=0., p1=0., p2=0.)
    elif mid == 3:                  # RADIAL: f, cx, cy, k1, k2
        return dict(fl_x=p[0], fl_y=p[0], cx=p[1], cy=p[2],
                    k1=p[3], k2=p[4], p1=0., p2=0.)
    elif mid == 4:                  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        return dict(fl_x=p[0], fl_y=p[1], cx=p[2], cy=p[3],
                    k1=p[4], k2=p[5], p1=p[6], p2=p[7])
    else:
        logger.warning(f"Unsupported COLMAP camera model {mid}, assuming PINHOLE layout.")
        fx = p[0]
        fy = p[1] if len(p) > 1 else p[0]
        cx = p[2] if len(p) > 2 else cam["width"]  / 2
        cy = p[3] if len(p) > 3 else cam["height"] / 2
        return dict(fl_x=fx, fl_y=fy, cx=cx, cy=cy, **zero)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def convert(
    sparse_model_dir: Path,
    images_dir:       Path,
    output_json:      Path,
    ply_file_path:    Optional[str] = None,
    depths_dir:       Optional[Path] = None,
    da3_bounds:       Optional[dict] = None,
) -> int:
    """
    Convert a COLMAP sparse model directory to a Nerfstudio transforms.json.

    Parameters
    ----------
    sparse_model_dir : directory containing cameras.bin, images.bin, points3D.bin
    images_dir       : directory containing the actual image files (undistorted)
    output_json      : path to write transforms.json
    ply_file_path    : optional path to a PLY point cloud for Gaussian init,
                       relative to output_json's parent directory
    depths_dir       : optional directory containing per-frame .npy depth maps
                       (output of generate_da3_depths.py) for DN-Splatter supervision
    da3_bounds       : optional dict with x_min/max y_min/max z_min/max from DA3
                       used to embed scene bounds in the JSON

    Returns
    -------
    Number of frames written.
    """
    cameras = _read_cameras(sparse_model_dir / "cameras.bin")
    images  = _read_images(sparse_model_dir  / "images.bin")
    logger.info(f"COLMAP model: {len(cameras)} camera(s), {len(images)} registered image(s)")

    # Compute camera cluster stats for diagnostics
    _all_c2w = [_colmap_to_nerf_c2w(img["qvec"], img["tvec"]) for img in images]
    _cam_positions = np.array([c2w[:3, 3] for c2w in _all_c2w], dtype=np.float64)
    _cam_center = _cam_positions.mean(axis=0)
    _cam_dists  = np.linalg.norm(_cam_positions - _cam_center, axis=1)
    _cam_radius = float(np.percentile(_cam_dists, 90))
    logger.info(f"  Camera cluster: center={_cam_center.round(2).tolist()}  p90_radius={_cam_radius:.3f}")

    frames  = []
    skipped = 0
    for img in sorted(images, key=lambda x: x["name"]):
        img_path = images_dir / img["name"]
        if not img_path.exists():
            skipped += 1
            continue

        cam  = cameras[img["camera_id"]]
        intr = _extract_intrinsics(cam)
        c2w  = _colmap_to_nerf_c2w(img["qvec"], img["tvec"])

        frame: dict = {
            "file_path":        f"images/{img['name']}",
            "transform_matrix": c2w.tolist(),
            "fl_x": intr["fl_x"],
            "fl_y": intr["fl_y"],
            "cx":   intr["cx"],
            "cy":   intr["cy"],
            "w":    cam["width"],
            "h":    cam["height"],
        }

        # Add DA3 depth map path if available (used by DN-Splatter for depth supervision)
        if depths_dir is not None:
            stem       = Path(img["name"]).stem
            depth_file = depths_dir / f"{stem}.npy"
            if depth_file.exists():
                # Store relative to output_json's parent (transforms.json dir)
                try:
                    rel = depth_file.relative_to(output_json.parent)
                    frame["depth_file_path"] = rel.as_posix()
                except ValueError:
                    frame["depth_file_path"] = str(depth_file)
        # Only include distortion coefficients if they are non-zero
        # (they are zero after COLMAP image_undistorter)
        if any(intr[k] != 0. for k in ("k1", "k2", "p1", "p2")):
            frame.update(k1=intr["k1"], k2=intr["k2"],
                         p1=intr["p1"], p2=intr["p2"])
        frames.append(frame)

    if skipped:
        logger.warning(f"Skipped {skipped} images (file not found in {images_dir})")
    if not frames:
        raise RuntimeError("No valid frames converted — check images_dir path.")

    # Use first camera's intrinsics as global header defaults
    first_cam  = cameras[images[0]["camera_id"]]
    first_intr = _extract_intrinsics(first_cam)

    n_with_depth = sum(1 for fr in frames if "depth_file_path" in fr)
    if n_with_depth:
        logger.info(f"  Depth maps linked: {n_with_depth}/{len(frames)} frames")

    doc: dict = {
        "camera_model": "OPENCV",
        "fl_x": first_intr["fl_x"],
        "fl_y": first_intr["fl_y"],
        "cx":   first_intr["cx"],
        "cy":   first_intr["cy"],
        "w":    first_cam["width"],
        "h":    first_cam["height"],
        "frames": frames,
    }
    if ply_file_path:
        doc["ply_file_path"] = ply_file_path

    # NOTE: We intentionally do NOT inject scene_box here.
    # Nerfstudio's nerfstudio-data dataparser computes the scale from camera
    # poses. Injecting scene_box can fight that auto-normalization (and was
    # one of the V5 failure factors). The DA3 bounds are still saved to
    # da3_bounds.json for diagnostics but not passed to nerfstudio.
    if da3_bounds is not None:
        logger.info(f"  DA3 bounds present (not embedded): "
                    f"x=[{da3_bounds.get('x_min', 0):.2f},{da3_bounds.get('x_max', 0):.2f}] "
                    f"scale_factor={da3_bounds.get('scale_factor_da3_to_colmap', 1.0):.4f}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)

    logger.info(f"transforms.json: {len(frames)} frames → {output_json}")
    return len(frames)


def export_sparse_ply(sparse_model_dir: Path, output_ply: Path) -> int:
    """
    Export points3D.bin from a COLMAP sparse model as a binary PLY (RGB coloured).

    Returns the number of points written (0 if points3D.bin is absent/empty).
    """
    pts3d_bin = sparse_model_dir / "points3D.bin"
    if not pts3d_bin.exists():
        logger.warning(f"points3D.bin not found: {pts3d_bin}")
        return 0

    xyz, rgb = _read_points3d(pts3d_bin)
    n = len(xyz)
    if n == 0:
        logger.warning("Sparse model has no 3D points.")
        return 0

    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    ).encode("ascii")

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ply, "wb") as f:
        f.write(header)
        for i in range(n):
            f.write(struct.pack("<3f", float(xyz[i, 0]),
                                      float(xyz[i, 1]),
                                      float(xyz[i, 2])))
            f.write(struct.pack("<3B", int(rgb[i, 0]),
                                      int(rgb[i, 1]),
                                      int(rgb[i, 2])))

    logger.info(f"Sparse PLY: {n} points → {output_ply}")
    return n
