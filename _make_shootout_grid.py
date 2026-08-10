"""Compose a 4-column x 3-row comparison grid PNG for the model shootout.

Layout:
  Row 1: [original]              [LaMa v40]          [Kontext-v2 remove]   [Kontext-v2 replace]
  Row 2: [Kontext-v1 remove]     [Kontext-v1 replace][FLUX Fill remove]    [SAM3 mask overlay]
  Row 3: [Qwen remove]           [Qwen replace]      [FireRed remove]      [FireRed replace]
"""
from __future__ import annotations

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")

# ---- Input paths ------------------------------------------------------------
P_ORIG        = ROOT / "colmap/dense/images.orig_with_chair/DSCF4907.jpg"
P_MASK        = ROOT / "output/segmented_room_v9a_fp_v2/masks/DSCF4907__blue_armchair.png"
P_LAMA_V40    = ROOT / "output/lama_chair_removal/native_2074/images/DSCF4907.jpg"
P_KTX_RM_V1   = ROOT / "output/model_shootout/flux_kontext/remove_style.jpg"
P_KTX_RP_V1   = ROOT / "output/model_shootout/flux_kontext/replace_plant.jpg"
P_KTX_RM_V2   = ROOT / "output/model_shootout/flux_kontext/remove_v2.jpg"
P_KTX_RP_V2   = ROOT / "output/model_shootout/flux_kontext/replace_v2.jpg"
P_QWEN_RM     = ROOT / "output/model_shootout/qwen_edit/remove.jpg"
P_QWEN_RP     = ROOT / "output/model_shootout/qwen_edit/replace.jpg"
P_FIRED_RM    = ROOT / "output/model_shootout/fired_edit/remove.jpg"
P_FIRED_RP    = ROOT / "output/model_shootout/fired_edit/replace.jpg"
P_FLUX_FILL   = ROOT / "output/model_shootout/flux_fill/remove.jpg"

OUT_PNG       = ROOT / "output/model_shootout/all_grid_v2.png"

# ---- Layout constants -------------------------------------------------------
TILE_LONG_EDGE = 640
TITLE_H        = 28
PAD            = 8
BG_COLOR       = (24, 24, 24)
TITLE_BG       = (40, 40, 40)
TITLE_FG       = (240, 240, 240)


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    """Try a handful of common Windows fonts, fall back to default."""
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT = _load_font(16)


def _fit_letterbox(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """Resize preserving aspect ratio, letterbox to (box_w, box_h) with black."""
    img = img.convert("RGB")
    src_w, src_h = img.size
    scale = min(box_w / src_w, box_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (box_w, box_h), (0, 0, 0))
    canvas.paste(resized, ((box_w - new_w) // 2, (box_h - new_h) // 2))
    return canvas


def _make_placeholder(box_w: int, box_h: int, text: str) -> Image.Image:
    canvas = Image.new("RGB", (box_w, box_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        bbox = draw.textbbox((0, 0), text, font=FONT)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=FONT)
    draw.text(((box_w - tw) / 2, (box_h - th) / 2), text, fill=(200, 200, 200), font=FONT)
    return canvas


def _make_tile(img_path: Path | None, label: str, box_w: int, box_h: int,
               *, missing_text: str | None = None) -> Image.Image:
    """Build a single tile: title bar on top, image below (letterboxed)."""
    if img_path is not None and img_path.exists():
        img = Image.open(img_path)
        tile_body = _fit_letterbox(img, box_w, box_h - TITLE_H)
    else:
        tile_body = _make_placeholder(
            box_w, box_h - TITLE_H, missing_text or "(missing)"
        )

    tile = Image.new("RGB", (box_w, box_h), BG_COLOR)
    # Title bar
    title_bar = Image.new("RGB", (box_w, TITLE_H), TITLE_BG)
    draw = ImageDraw.Draw(title_bar)
    try:
        bbox = draw.textbbox((0, 0), label, font=FONT)
        th = bbox[3] - bbox[1]
    except Exception:
        _tw, th = draw.textsize(label, font=FONT)
    draw.text((8, (TITLE_H - th) / 2 - 2), label, fill=TITLE_FG, font=FONT)
    tile.paste(title_bar, (0, 0))
    tile.paste(tile_body, (0, TITLE_H))
    return tile


def _make_mask_overlay(orig_path: Path, mask_path: Path,
                       box_w: int, box_h: int) -> Image.Image:
    """Blend SAM3 mask (red 50%) onto the original."""
    body_h = box_h - TITLE_H
    orig = Image.open(orig_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(orig.size, Image.NEAREST)

    red = Image.new("RGB", orig.size, (255, 40, 40))
    blended = Image.composite(
        Image.blend(orig, red, 0.45),
        orig,
        mask,
    )
    return _fit_letterbox(blended, box_w, body_h)


def _wrap_overlay_tile(label: str, box_w: int, box_h: int) -> Image.Image:
    body = _make_mask_overlay(P_ORIG, P_MASK, box_w, box_h)
    tile = Image.new("RGB", (box_w, box_h), BG_COLOR)
    title_bar = Image.new("RGB", (box_w, TITLE_H), TITLE_BG)
    draw = ImageDraw.Draw(title_bar)
    try:
        bbox = draw.textbbox((0, 0), label, font=FONT)
        th = bbox[3] - bbox[1]
    except Exception:
        _tw, th = draw.textsize(label, font=FONT)
    draw.text((8, (TITLE_H - th) / 2 - 2), label, fill=TITLE_FG, font=FONT)
    tile.paste(title_bar, (0, 0))
    tile.paste(body, (0, TITLE_H))
    return tile


def build_grid() -> Image.Image:
    # Compute tile dimensions from the original aspect ratio so the whole tile
    # (title + letterboxed body) has long-edge == TILE_LONG_EDGE.
    with Image.open(P_ORIG) as im:
        src_w, src_h = im.size
    if src_w >= src_h:
        tile_w = TILE_LONG_EDGE
        tile_h = int(round(TILE_LONG_EDGE * src_h / src_w)) + TITLE_H
    else:
        tile_h = TILE_LONG_EDGE + TITLE_H
        tile_w = int(round(TILE_LONG_EDGE * src_w / src_h))

    # Flux Fill tile: use if exists, else placeholder tile
    flux_fill_tile = _make_tile(
        P_FLUX_FILL if P_FLUX_FILL.exists() else None,
        "FLUX Fill: remove" if P_FLUX_FILL.exists() else "FLUX Fill: still running",
        tile_w, tile_h,
        missing_text="FLUX Fill: still running",
    )

    tiles: list[list[Image.Image]] = [
        # Row 1
        [
            _make_tile(P_ORIG,      "Original DSCF4907",         tile_w, tile_h),
            _make_tile(P_LAMA_V40,  "LaMa v40 (removal ref)",    tile_w, tile_h),
            _make_tile(P_KTX_RM_V2, "Kontext v2 remove (no-mask)", tile_w, tile_h),
            _make_tile(P_KTX_RP_V2, "Kontext v2 replace (no-mask)", tile_w, tile_h),
        ],
        # Row 2
        [
            _make_tile(P_KTX_RM_V1, "Kontext v1 remove (masked)",  tile_w, tile_h),
            _make_tile(P_KTX_RP_V1, "Kontext v1 replace (masked)", tile_w, tile_h),
            flux_fill_tile,
            _wrap_overlay_tile("SAM3 mask overlay", tile_w, tile_h),
        ],
        # Row 3
        [
            _make_tile(P_QWEN_RM,  "Qwen edit: remove",   tile_w, tile_h),
            _make_tile(P_QWEN_RP,  "Qwen edit: replace",  tile_w, tile_h),
            _make_tile(P_FIRED_RM, "FireRed edit: remove", tile_w, tile_h),
            _make_tile(P_FIRED_RP, "FireRed edit: replace", tile_w, tile_h),
        ],
    ]

    n_rows = len(tiles)
    n_cols = len(tiles[0])
    grid_w = n_cols * tile_w + (n_cols + 1) * PAD
    grid_h = n_rows * tile_h + (n_rows + 1) * PAD
    grid = Image.new("RGB", (grid_w, grid_h), BG_COLOR)

    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            x = PAD + c * (tile_w + PAD)
            y = PAD + r * (tile_h + PAD)
            grid.paste(tile, (x, y))
    return grid


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    grid = build_grid()
    grid.save(OUT_PNG, "PNG", optimize=True)
    w, h = grid.size
    print(f"Saved: {OUT_PNG}")
    print(f"Dimensions: {w} x {h} px")


if __name__ == "__main__":
    main()
