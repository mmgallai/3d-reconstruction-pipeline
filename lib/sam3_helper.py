"""
SAM3 masking helper — loads model once and processes a directory of images.
Keeps all SAM3-specific logic isolated from the pipeline.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_sam3(sam3_repo: Path, checkpoint: Path, device: str):
    """Load SAM3 model and return a processor. Call once per run."""
    if str(sam3_repo) not in sys.path:
        sys.path.insert(0, str(sam3_repo))

    try:
        import torch
        import inspect
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        raise ImportError(f"SAM3 not found. Check SAM3_REPO_PATH in config.py.\n{e}")

    logger.info(f"Loading SAM3 on {device} from {checkpoint}")
    sig = inspect.signature(build_sam3_image_model)
    kwargs = {}
    if "device"          in sig.parameters: kwargs["device"]          = device
    if "eval_mode"       in sig.parameters: kwargs["eval_mode"]       = True
    if "checkpoint_path" in sig.parameters: kwargs["checkpoint_path"] = str(checkpoint)
    if "load_from_HF"    in sig.parameters: kwargs["load_from_HF"]    = False

    model = build_sam3_image_model(**kwargs)
    model.eval()
    return Sam3Processor(model)


def _normalize_masks(masks_np: np.ndarray) -> np.ndarray:
    """Squeeze extra dimensions → always returns (N, H, W) bool array."""
    while masks_np.ndim > 3:
        if   masks_np.shape[1] == 1: masks_np = masks_np[:, 0]
        elif masks_np.shape[-1] == 1: masks_np = masks_np[..., 0]
        else: break
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis]
    return masks_np


def segment_image(
    image_path: Path,
    prompt: str,
    processor,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run SAM3 on one image and return a (H, W) bool mask.
    The highest-scoring mask is returned when multiple candidates exist.
    """
    import torch

    pil_img = Image.open(image_path).convert("RGB")
    state   = processor.set_image(pil_img)

    with torch.inference_mode():
        output = processor.set_text_prompt(state=state, prompt=prompt)

    raw_masks = output.get("masks")
    scores    = output.get("scores")

    if raw_masks is None:
        logger.warning(f"SAM3 returned no mask for {image_path.name}")
        w, h = pil_img.size
        return np.zeros((h, w), dtype=bool)

    masks_np  = raw_masks.detach().cpu().numpy() if hasattr(raw_masks, "detach") else np.asarray(raw_masks)
    masks_np  = _normalize_masks(masks_np) > threshold   # (N, H, W) bool

    if scores is not None and len(scores) > 0:
        sc  = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
        idx = int(np.argmax(sc))
        return masks_np[idx] if idx < len(masks_np) else masks_np[0]

    # No scores — union all masks
    return np.any(masks_np, axis=0)


def mask_directory(
    images_dir: Path,
    masks_dir: Path,
    prompt: str,
    processor,
    threshold: float = 0.5,
    smooth_radius: int = 3,
) -> int:
    """
    Run SAM3 on every image in `images_dir`, save binary masks to `masks_dir`.
    Saves two filenames per image to satisfy both COLMAP and Nerfstudio:
      - <name>.jpg.png  → COLMAP mask_path convention
      - <name>.png      → Nerfstudio masks-path convention

    Returns number of images processed.
    """
    masks_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in VALID_EXTS)

    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    logger.info(f"Masking {len(image_files)} images with prompt='{prompt}'")

    for img_path in image_files:
        mask    = segment_image(img_path, prompt, processor, threshold)
        mask_u8 = (mask.astype(np.uint8) * 255)
        pil_mask = Image.fromarray(mask_u8)

        if smooth_radius > 1:
            pil_mask = pil_mask.filter(ImageFilter.MinFilter(smooth_radius))

        # COLMAP convention:     image.jpg  → mask = image.jpg.png
        pil_mask.save(masks_dir / f"{img_path.name}.png")
        # Nerfstudio convention: image.jpg  → mask = image.png
        pil_mask.save(masks_dir / f"{img_path.stem}.png")

        coverage = mask.sum() / mask.size * 100
        logger.info(f"  {img_path.name}: {coverage:.1f}% masked")

    return len(image_files)


def generate_blank_masks(images_dir: Path, masks_dir: Path) -> int:
    """
    For any image in images_dir that lacks a Nerfstudio-style mask (<stem>.png),
    generate a fully-black (no-mask) PNG. Prevents Nerfstudio crashes.
    """
    masks_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in VALID_EXTS:
            continue
        mask_path = masks_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            with Image.open(img_path) as img:
                Image.new("L", img.size, 0).save(mask_path)
            logger.info(f"  Blank mask generated: {mask_path.name}")
            generated += 1
    return generated
