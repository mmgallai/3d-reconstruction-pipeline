"""
Depth Anything 3 (DA3) integration.

Installation (one-time):
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3
    cd Depth-Anything-3
    pip install -e .

    # For direct Gaussian Splatting output (--da3-gs mode):
    pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
    pip install -e ".[all]"

Recommended models:
    Depth only:  depth-anything/da3-large        (0.35B, fast, good quality)
    Metric:      depth-anything/da3metric-large  (metric depth in meters)
    Direct GS:   depth-anything/da3-giant        (1.15B, requires --da3-gs flag)
"""

from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def _check_installed() -> None:
    try:
        import depth_anything_3  # noqa: F401
    except ImportError:
        raise ImportError(
            "Depth Anything 3 is not installed.\n\n"
            "  git clone https://github.com/ByteDance-Seed/Depth-Anything-3\n"
            "  cd Depth-Anything-3\n"
            "  pip install -e .\n"
        )


def load_model(model_id: str = "depth-anything/da3-large", device: str = "cuda"):
    """Load DA3 model from HuggingFace."""
    _check_installed()
    import torch
    from depth_anything_3.api import DepthAnything3

    logger.info(f"Loading DA3 model: {model_id} on {device}")
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=torch.device(device))
    model.eval()
    logger.info("DA3 model ready.")
    return model


def _build_w2c_batch(c2w_list: list[list[list[float]]]) -> np.ndarray:
    """
    Convert list of c2w 4x4 matrices to w2c [N, 4, 4] numpy array.
    DA3 expects extrinsics as w2c (world-to-camera) with shape (N, 4, 4).
    """
    N = len(c2w_list)
    w2c = np.zeros((N, 4, 4), dtype=np.float32)
    for i, c2w in enumerate(c2w_list):
        c2w_np = np.array(c2w, dtype=np.float32)   # [4, 4]
        w2c[i] = np.linalg.inv(c2w_np)             # [4, 4]
    return w2c


def _build_K_batch(intrinsics_list: list[dict]) -> np.ndarray:
    """
    Build [N, 3, 3] intrinsics array from list of per-frame dicts
    with keys: fl_x, fl_y, cx, cy.
    """
    N = len(intrinsics_list)
    K = np.zeros((N, 3, 3), dtype=np.float32)
    for i, intr in enumerate(intrinsics_list):
        K[i] = [[intr["fl_x"],      0.0, intr["cx"]],
                [     0.0,    intr["fl_y"], intr["cy"]],
                [     0.0,         0.0,      1.0     ]]
    return K


def estimate_depth(
    model,
    jpeg_paths:       list[Path],
    c2w_list:         list[list[list[float]]],  # per-frame c2w [4][4]
    intrinsics_list:  list[dict],               # per-frame {fl_x, fl_y, cx, cy}
    batch_size:       int = 8,
    process_res:      int = 504,
    orig_w:           int = 3024,   # kept for API compat, no longer used here
    orig_h:           int = 4032,   # kept for API compat, no longer used here
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Run DA3 depth estimation in batches.

    Returns depth maps and confidence maps at process_res resolution
    (NOT upsampled to original resolution — saves ~8 GB RAM for 185 images).
    Use pointcloud_utils.save_depth_png(..., orig_w, orig_h) to upsample
    one-at-a-time when writing depth PNGs.

    Returns:
        depth_maps: list of [H_proc, W_proc] float32 arrays (meters)
        conf_maps:  list of [H_proc, W_proc] float32 arrays (0–1)
    """
    N = len(jpeg_paths)
    all_depth: list[np.ndarray] = []
    all_conf:  list[np.ndarray] = []

    n_batches = math.ceil(N / batch_size)
    logger.info(f"DA3 depth estimation: {N} images, {n_batches} batches of ≤{batch_size}")

    for b in range(n_batches):
        start = b * batch_size
        end   = min(start + batch_size, N)
        c2w_b  = c2w_list[start:end]
        intr_b = intrinsics_list[start:end]

        # Pre-resize to process_res before passing to DA3.
        # DA3 resizes internally anyway; pre-resizing avoids loading full-res
        # images in its multiprocessing pool (prevents RAM OOM on large images).
        imgs = _load_resized(jpeg_paths[start:end], process_res)

        # Scale intrinsics to match the resized dimensions
        h0, w0 = orig_h, orig_w
        h1, w1 = imgs[0].shape[:2]
        sx, sy = w1 / w0, h1 / h0
        intr_scaled = [
            {"fl_x": d["fl_x"] * sx, "fl_y": d["fl_y"] * sy,
             "cx":   d["cx"]   * sx, "cy":   d["cy"]   * sy}
            for d in intr_b
        ]

        w2c_np = _build_w2c_batch(c2w_b)          # [B, 4, 4]
        K_np   = _build_K_batch(intr_scaled)       # [B, 3, 3]

        real_count = len(imgs)

        # Umeyama alignment requires ≥3 DISTINCT poses (always runs when extrinsics
        # are provided). Padding with duplicates produces degenerate covariance.
        # For tiny tail batches just drop extrinsics — depth is still estimated,
        # just without metric scale alignment for those 1-2 frames.
        use_ext = real_count >= 3
        logger.info(f"  Batch {b+1}/{n_batches}: images {start}–{end-1}")
        pred = model.inference(
            image               = imgs,
            extrinsics          = w2c_np if use_ext else None,
            intrinsics          = K_np   if use_ext else None,
            align_to_input_ext_scale = True,
            process_res         = process_res,
        )

        depth_batch = pred.depth   # [B, H_proc, W_proc]
        conf_batch  = pred.conf    # [B, H_proc, W_proc]

        # Keep at process_res — do NOT upsample all maps to full resolution.
        # Full-res upsample (e.g. 4032x3024 float32) x 185 frames = ~8 GB RAM.
        # Callers that need full-res (e.g. depth PNG saving) upsample one at a time.
        # Only keep real_count results.
        for i in range(real_count):
            all_depth.append(depth_batch[i].copy())
            all_conf.append(conf_batch[i].copy())

    logger.info(f"DA3 depth estimation complete: {len(all_depth)} depth maps.")
    return all_depth, all_conf


def run_direct_gs(
    model,
    jpeg_paths:       list[Path],
    c2w_list:         list[list[list[float]]],
    intrinsics_list:  list[dict],
    output_dir:       Path,
    batch_size:       int = 8,
    process_res:      int = 504,
) -> Optional[Path]:
    """
    Run DA3 in feed-forward Gaussian Splatting mode.
    Requires da3-giant or da3nested-giant-large model.
    Returns path to exported splat.ply or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    N = len(jpeg_paths)
    n_batches = math.ceil(N / batch_size)
    logger.info(f"DA3 direct GS: {N} images, {n_batches} batches")

    # For direct GS, run all images together for best global consistency
    # (batch if needed for memory)
    all_plys: list[Path] = []
    for b in range(n_batches):
        start = b * batch_size
        end   = min(start + batch_size, N)
        imgs   = [str(p) for p in jpeg_paths[start:end]]
        w2c_np = _build_w2c_batch(c2w_list[start:end])
        K_np   = _build_K_batch(intrinsics_list[start:end])

        batch_dir = output_dir / f"batch_{b:04d}"
        logger.info(f"  Batch {b+1}/{n_batches} → {batch_dir}")
        pred = model.inference(
            image               = imgs,
            extrinsics          = w2c_np,
            intrinsics          = K_np,
            align_to_input_ext_scale = True,
            infer_gs            = True,
            process_res         = process_res,
            export_dir          = str(batch_dir),
            export_format       = "gs_ply",
        )
        ply = batch_dir / "gaussians.ply"
        if ply.exists():
            all_plys.append(ply)
        else:
            # Try finding any .ply in batch_dir
            found = list(batch_dir.glob("*.ply"))
            if found:
                all_plys.append(found[0])

    if not all_plys:
        logger.error("DA3 direct GS: no .ply files produced.")
        return None

    if len(all_plys) == 1:
        return all_plys[0]

    # Merge multiple batch PLYs
    return _merge_gs_plys(all_plys, output_dir / "splat_da3.ply")


# ─── Helpers ────────────────────────────────────────────────────────────────

def _load_resized(paths: list[Path], process_res: int) -> list[np.ndarray]:
    """
    Load images and downscale so the long side = process_res.
    Returns list of uint8 [H, W, 3] numpy arrays.
    Keeps images in memory at process_res size instead of full resolution,
    preventing RAM OOM when DA3's multiprocessing pool loads many large images.
    """
    out = []
    for p in paths:
        img = PILImage.open(p).convert("RGB")
        w, h = img.size
        scale = process_res / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), PILImage.LANCZOS)
        out.append(np.array(img))
    return out


def _resize_map(arr: np.ndarray, h: int, w: int, resample) -> np.ndarray:
    """Resize a 2D float map to (h, w)."""
    if arr.shape == (h, w):
        return arr.astype(np.float32)
    img = PILImage.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((w, h), resample=resample)
    return np.array(img, dtype=np.float32)


def _merge_gs_plys(ply_paths: list[Path], out: Path) -> Path:
    """Naively concatenate Gaussian Splat PLY files (vertex lists)."""
    import re
    all_headers: list[str] = []
    all_data: list[bytes] = []
    total_verts = 0

    for p in ply_paths:
        raw = p.read_bytes()
        end = raw.index(b"end_header\n") + len(b"end_header\n")
        header = raw[:end].decode("ascii")
        data   = raw[end:]
        m = re.search(r"element vertex (\d+)", header)
        if m:
            total_verts += int(m.group(1))
        if not all_headers:
            all_headers.append(header)
        all_data.append(data)

    if not all_headers:
        raise RuntimeError("No valid PLY headers found.")

    merged_header = re.sub(
        r"element vertex \d+",
        f"element vertex {total_verts}",
        all_headers[0],
    )
    out.write_bytes(merged_header.encode("ascii") + b"".join(all_data))
    logger.info(f"Merged {len(ply_paths)} GS PLYs → {out} ({total_verts} Gaussians)")
    return out
