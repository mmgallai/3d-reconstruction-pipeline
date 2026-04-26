"""
RealityScan / CaptureReality XMP parser.

Reads xcr:Rotation, xcr:Position, xcr:FocalLength35mm, xcr:PrincipalPointU/V
from each .xmp sidecar and builds a Nerfstudio transforms.json.

Coordinate convention
---------------------
RealityScan stores the camera-to-world rotation R_wc (3×3, row-major) and
the camera centre in world space (Position).  The c2w matrix passed to
Nerfstudio is therefore:

    c2w = [R_wc | Position]
          [  0  |     1   ]

RealityScan / ARKit use a right-handed, Y-up system where the camera looks
down its -Z axis (OpenGL convention).  Nerfstudio's "nerfstudio" dataparser
expects OpenCV convention (camera looks down +Z, Y points down).  We therefore
flip the Y and Z columns of R_wc before writing:

    R_opencv = R_wc * diag(1, -1, -1)

If the resulting reconstruction looks upside-down or mirrored, toggle the
FLIP_YZ constant below.
"""

import json
import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Set to False if poses appear flipped in Nerfstudio viewer
FLIP_YZ = True

_XCR = "http://www.capturingreality.com/ns/xcr/1.1#"


def _parse_floats(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def parse_xmp(xmp_path: Path) -> Optional[dict]:
    """
    Parse one RealityScan XMP sidecar.
    Returns a dict with keys: rotation (list[9]), position (list[3]),
    focal_length_35mm (float), principal_point_u (float), principal_point_v (float).
    Returns None if parsing fails.
    """
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        # <rdf:Description ...> can appear as direct child or nested under <rdf:RDF>
        ns = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
        desc = root.find(".//rdf:Description", ns)
        if desc is None:
            logger.warning(f"No rdf:Description in {xmp_path.name}")
            return None

        fl35 = float(desc.get(f"{{{_XCR}}}FocalLength35mm", "0"))
        ppu  = float(desc.get(f"{{{_XCR}}}PrincipalPointU", "0"))
        ppv  = float(desc.get(f"{{{_XCR}}}PrincipalPointV", "0"))

        rot_el  = desc.find(f"{{{_XCR}}}Rotation")
        pos_el  = desc.find(f"{{{_XCR}}}Position")
        if rot_el is None or pos_el is None:
            logger.warning(f"Missing Rotation or Position in {xmp_path.name}")
            return None

        return {
            "rotation":          _parse_floats(rot_el.text.strip()),
            "position":          _parse_floats(pos_el.text.strip()),
            "focal_length_35mm": fl35,
            "principal_point_u": ppu,
            "principal_point_v": ppv,
        }
    except Exception as e:
        logger.warning(f"Failed to parse {xmp_path.name}: {e}")
        return None


def _build_c2w(rotation: list[float], position: list[float]) -> list[list[float]]:
    """
    Build a 4×4 camera-to-world matrix from xcr Rotation (row-major 3×3)
    and Position (world-space camera centre).

    xcr:Rotation is R_wc (camera→world).  We optionally flip Y/Z columns
    to convert from OpenGL camera convention to OpenCV.
    """
    r = rotation          # 9 values, row-major
    # R_wc as columns: col_j = [r[j], r[3+j], r[6+j]]
    R = [[r[0], r[1], r[2]],
         [r[3], r[4], r[5]],
         [r[6], r[7], r[8]]]

    if FLIP_YZ:
        # Flip Y and Z columns: camera in OpenGL looks down -Z (Y up)
        # OpenCV looks down +Z (Y down) — multiply columns 1 and 2 by -1
        for row in R:
            row[1] *= -1
            row[2] *= -1

    t = position  # [x, y, z] camera centre in world space
    return [
        [R[0][0], R[0][1], R[0][2], t[0]],
        [R[1][0], R[1][1], R[1][2], t[1]],
        [R[2][0], R[2][1], R[2][2], t[2]],
        [0.0,     0.0,     0.0,     1.0 ],
    ]


def generate_transforms_json(
    scan_dir:    Path,
    jpeg_dir:    Path,
    out_json:    Path,
    img_w:       int,
    img_h:       int,
    depth_dir:   Optional[Path] = None,   # folder of per-frame depth PNGs
    init_ply:    Optional[Path] = None,   # DA3 init point cloud for splatfacto
) -> tuple[int, list[dict]]:
    """
    Walk scan_dir for .xmp files, parse poses, and write Nerfstudio
    transforms.json to out_json.  jpeg_dir should contain the JPEG
    counterparts with the same stem (.jpg extension).

    depth_dir: if provided, adds depth_file_path entries pointing to
               per-frame depth PNGs (uint16 millimetres, scale=0.001).
    init_ply:  if provided, adds ply_file_path at the top level so
               Nerfstudio splatfacto uses it as Gaussian initialisation.

    Returns (n_frames_written, frames_list).
    """
    xmp_files = sorted(scan_dir.glob("*.xmp"))
    if not xmp_files:
        raise FileNotFoundError(f"No .xmp files found in {scan_dir}")

    frames = []
    skipped = 0
    fl_sum = 0.0

    for xmp in xmp_files:
        jpeg = jpeg_dir / (xmp.stem + ".jpg")
        if not jpeg.exists():
            logger.warning(f"  JPEG not found for {xmp.name}, skipping")
            skipped += 1
            continue

        data = parse_xmp(xmp)
        if data is None:
            skipped += 1
            continue

        fl35 = data["focal_length_35mm"]
        # fl_pixels: 35mm-equiv uses the sensor's long side = 36mm
        # For portrait 3024×4032 the physical sensor is landscape, so the
        # long-side pixel count is max(w, h).
        fl_px = fl35 * max(img_w, img_h) / 36.0
        fl_sum += fl_px

        # Principal point in pixels
        # PrincipalPointU/V are offsets from image centre, normalised to sensor width (= 1)
        cx = img_w * (0.5 + data["principal_point_u"])
        cy = img_h * (0.5 + data["principal_point_v"])

        c2w = _build_c2w(data["rotation"], data["position"])

        frame: dict = {
            "file_path": f"images/{jpeg.name}",
            "fl_x": fl_px,
            "fl_y": fl_px,
            "cx":   cx,
            "cy":   cy,
            "w":    img_w,
            "h":    img_h,
            "transform_matrix": c2w,
            # Store intrinsics as plain dict for later use by DA3 / pointcloud code
            "_intrinsics": {"fl_x": fl_px, "fl_y": fl_px, "cx": cx, "cy": cy},
        }

        # Optional depth supervision
        if depth_dir is not None:
            depth_png = depth_dir / (xmp.stem + ".png")
            if depth_png.exists():
                frame["depth_file_path"] = f"depth/{depth_png.name}"

        frames.append(frame)

    if not frames:
        raise RuntimeError("No valid frames parsed — check XMP files and JPEG folder.")

    fl_mean = fl_sum / len(frames)

    # Strip the internal _intrinsics key before writing JSON
    clean_frames = [{k: v for k, v in f.items() if k != "_intrinsics"} for f in frames]

    transforms: dict = {
        "camera_model": "OPENCV",
        "fl_x":  fl_mean,
        "fl_y":  fl_mean,
        "cx":    img_w / 2.0,
        "cy":    img_h / 2.0,
        "w":     img_w,
        "h":     img_h,
        "frames": clean_frames,
    }

    if init_ply is not None and init_ply.exists():
        # Relative path from the data directory (same folder as transforms.json)
        try:
            rel = init_ply.relative_to(out_json.parent)
        except ValueError:
            rel = init_ply   # fall back to absolute if not under data_dir
        transforms["ply_file_path"] = str(rel).replace("\\", "/")

    if any("depth_file_path" in f for f in clean_frames):
        transforms["depth_unit_scale_factor"] = 0.001   # mm → m

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(transforms, indent=2))
    n = len(frames)
    logger.info(f"transforms.json: {n} frames, {skipped} skipped → {out_json}")
    if init_ply and init_ply.exists():
        logger.info(f"  ply_file_path: {transforms.get('ply_file_path')}")
    return n, frames
