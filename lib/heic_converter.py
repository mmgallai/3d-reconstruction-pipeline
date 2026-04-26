"""
HEIC → JPEG converter.
Requires: pip install pillow-heif
"""

import logging
from pathlib import Path
from PIL import Image
import pillow_heif

logger = logging.getLogger(__name__)

# Register HEIC support once at import time
pillow_heif.register_heif_opener()


def convert_directory(heic_dir: Path, jpeg_dir: Path, quality: int = 92) -> list[Path]:
    """
    Convert all HEIC files in heic_dir to JPEG in jpeg_dir.
    Skips files that already exist (idempotent).
    Returns list of output JPEG paths.
    """
    heic_files = sorted(
        f for f in heic_dir.iterdir()
        if f.suffix.lower() in {".heic", ".heif"}
    )
    if not heic_files:
        raise FileNotFoundError(f"No HEIC/HEIF files found in {heic_dir}")

    jpeg_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for heic in heic_files:
        dst = jpeg_dir / (heic.stem + ".jpg")
        if dst.exists():
            outputs.append(dst)
            continue
        img = Image.open(heic)
        img.save(dst, format="JPEG", quality=quality)
        outputs.append(dst)
        logger.debug(f"  Converted {heic.name} → {dst.name}")

    logger.info(f"HEIC conversion: {len(outputs)} images → {jpeg_dir}")
    return outputs
