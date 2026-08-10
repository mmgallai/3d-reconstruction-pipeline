#!/usr/bin/env python3
"""Inpaint chair+teddy from room images using IOPaint LaMa.

Usage:
    python _inpaint_chair_lama.py \
        --masks-dir  output/segmented_room_v9a_fp_v2/masks \
        --images-dir /path/to/images_2 \
        --out-images-dir out/lama_images \
        --out-masks-dir  out/lama_masks \
        --dilate-px 40
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

# IOPaint (heavyweight; imported lazily inside main() so `--help` works even
# when iopaint / torch aren't installed in the calling env).
#   from iopaint.model_manager import ModelManager
#   from iopaint.schema import HDStrategy, InpaintRequest, LDMSampler

LABELS = ("blue_armchair", "teddy_bear")
IMG_EXTS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
def _find_mask_file(masks_root: Path, stem: str, label: str) -> Path | None:
    """Locate a per-view mask.

    Supports two on-disk conventions:
      1. subdir  : ``<masks_root>/<stem>/<label>.png``   (scene_segmenter design)
      2. flat    : ``<masks_root>/<stem>__<label>.png``  (v9a_fp_v2 actual layout)
    Returns the first hit, or None.
    """
    subdir = masks_root / stem / f"{label}.png"
    if subdir.exists():
        return subdir
    flat = masks_root / f"{stem}__{label}.png"
    if flat.exists():
        return flat
    return None


def _load_union_mask(
    masks_root: Path, stem: str, hw: tuple[int, int], dilate_px: int
) -> tuple[np.ndarray, bool]:
    """Return (uint8 mask [255=fill / 0=keep], any_label_found)."""
    h, w = hw
    union = np.zeros((h, w), dtype=bool)
    found = False
    for label in LABELS:
        p = _find_mask_file(masks_root, stem, label)
        if p is None:
            continue
        m = np.array(Image.open(p).convert("L")) > 127
        if m.shape != (h, w):  # SAM3 sometimes writes at source res; guard anyway
            m = np.array(
                Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (w, h), Image.NEAREST
                )
            ) > 127
        union |= m
        found = True
    if not found:
        return np.zeros((h, w), dtype=np.uint8), False
    if dilate_px > 0 and union.any():
        union = binary_dilation(union, iterations=dilate_px)
    return (union.astype(np.uint8)) * 255, True


# ---------------------------------------------------------------------------
def inpaint_one(
    img_path: Path,
    masks_root: Path,
    out_img_dir: Path,
    out_mask_dir: Path,
    model: ModelManager,
    cfg: InpaintRequest,
    dilate_px: int,
) -> tuple[bool, bool, float]:
    """Inpaint a single image.

    Returns
    -------
    (did_fill, had_mask, elapsed_s)
        did_fill  : LaMa was actually invoked
        had_mask  : at least one SAM3 label existed on disk for this view
    """
    t0 = time.perf_counter()
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    h, w = img_rgb.shape[:2]

    mask, had_mask = _load_union_mask(masks_root, img_path.stem, (h, w), dilate_px)
    Image.fromarray(mask).save(out_mask_dir / f"{img_path.stem}.png")

    out_path = out_img_dir / img_path.name
    if mask.sum() == 0:  # nothing to fill — copy through
        Image.fromarray(img_rgb).save(out_path)
        return False, had_mask, time.perf_counter() - t0

    # IOPaint's ModelManager: takes BGR uint8 input, returns RGB uint8 output.
    # (Empirically verified 2026-08-03: swapping on both sides produces R<->B swap
    # in pixels LaMa preserved unchanged.) Mask is uint8 single-channel, 255=fill.
    img_bgr = img_rgb[:, :, ::-1].copy()
    result_rgb = model(img_bgr, mask, cfg)
    Image.fromarray(result_rgb).save(out_path)
    return True, had_mask, time.perf_counter() - t0


# ---------------------------------------------------------------------------
def _dir_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Batch-inpaint blue_armchair + teddy_bear from a scene "
        "using IOPaint LaMa. Absolute paths accepted for every path arg.",
    )
    ap.add_argument("--masks-dir",      type=Path, required=True,
                    help="SAM3 masks root (subdir or flat '<stem>__<label>.png').")
    ap.add_argument("--images-dir",     type=Path, required=True,
                    help="Source RGB images (jpg/jpeg/png).")
    ap.add_argument("--out-images-dir", type=Path, required=True,
                    help="Where to write inpainted RGB images.")
    ap.add_argument("--out-masks-dir",  type=Path, required=True,
                    help="Where to dump the dilated union masks (debug).")
    ap.add_argument("--dilate-px",      type=int,  default=40,
                    help="scipy binary_dilation iterations on the union mask.")
    ap.add_argument("--model",          choices=["lama", "migan"], default="lama",
                    help="IOPaint model name.")
    ap.add_argument("--hd-strategy",    choices=["Original", "Resize", "Crop"],
                    default="Original",
                    help="Full-res inpaint (Original) vs Resize/Crop tiling.")
    ap.add_argument("--limit",          type=int, default=None,
                    help="Only process the first N images (sorted).")
    args = ap.parse_args()

    # Absolute-path-safe: resolve every path so cwd doesn't matter.
    args.masks_dir      = args.masks_dir.resolve()
    args.images_dir     = args.images_dir.resolve()
    args.out_images_dir = args.out_images_dir.resolve()
    args.out_masks_dir  = args.out_masks_dir.resolve()

    args.out_images_dir.mkdir(parents=True, exist_ok=True)
    args.out_masks_dir.mkdir(parents=True, exist_ok=True)

    # Deferred imports — heavy, and not needed for --help.
    import torch
    from iopaint.model_manager import ModelManager
    from iopaint.schema import HDStrategy, InpaintRequest, LDMSampler

    images = sorted(
        p for p in args.images_dir.iterdir() if p.suffix.lower() in IMG_EXTS
    )
    if args.limit:
        images = images[: args.limit]
    print(
        f"[inpaint] {len(images)} images | dilate={args.dilate_px}px "
        f"| model={args.model} | hd={args.hd_strategy}"
    )
    print(f"[inpaint] masks : {args.masks_dir}")
    print(f"[inpaint] images: {args.images_dir}")
    print(f"[inpaint] out   : {args.out_images_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[inpaint] device: {device}")
    model = ModelManager(name=args.model, device=device)
    cfg = InpaintRequest(
        hd_strategy=HDStrategy(args.hd_strategy),
        hd_strategy_crop_trigger_size=2048,
        hd_strategy_crop_margin=196,
        hd_strategy_resize_limit=2048,
        ldm_steps=20,
        ldm_sampler=LDMSampler.plms,  # ignored by LaMa; harmless
    )

    times: list[float] = []
    filled = 0
    no_mask_warned = 0
    for i, img_path in enumerate(images):
        did, had_mask, dt = inpaint_one(
            img_path,
            args.masks_dir,
            args.out_images_dir,
            args.out_masks_dir,
            model,
            cfg,
            args.dilate_px,
        )
        times.append(dt)
        filled += int(did)
        if not had_mask:
            no_mask_warned += 1
            if no_mask_warned <= 3:
                print(f"  [warn] no SAM3 mask for {img_path.stem} — copied through")
        if (i + 1) % 20 == 0:
            print(
                f"  [{i+1:>4}/{len(images)}] "
                f"mean={np.mean(times):.2f}s  last={dt:.2f}s"
            )

    print("\n[inpaint] done")
    print(
        f"  processed : {len(images)}  "
        f"(filled={filled}, passthrough={len(images)-filled}, "
        f"no-mask={no_mask_warned})"
    )
    if times:
        print(
            f"  mean/img  : {np.mean(times):.2f}s   "
            f"p95={np.percentile(times, 95):.2f}s"
        )
        print(f"  wall      : {sum(times):.1f}s")
    print(
        f"  out imgs  : {_dir_size_mb(args.out_images_dir):.1f} MB "
        f"({args.out_images_dir})"
    )
    print(
        f"  out masks : {_dir_size_mb(args.out_masks_dir):.1f} MB "
        f"({args.out_masks_dir})"
    )


if __name__ == "__main__":
    main()
