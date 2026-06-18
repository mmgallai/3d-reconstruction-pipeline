"""SAM3 segmentation: text-prompted, multi-prompt, model loaded once.

Improvements over the original automated_trimming sam3_segment:
  * Multi-prompt in a single SAM3 model load (the original re-loaded SAM3
    per prompt subprocess - the model takes ~5-10 sec to load).
  * Per-view, per-prompt mask cache keyed by (frame_stem, prompt_slug).
  * Erosion strength configurable; SAM3 thin-feature accuracy is the main
    failure mode at thresholds that work for blocky objects.
  * No subprocess - importable. Caller chooses the env.

The function returns a dict[prompt] -> dict[view_name] -> mask (bool ndarray).
Masks are saved alongside as PNGs so downstream tools can inspect them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter

# SAM3 import only succeeds in the `sam3` conda env where the package was
# installed editable from C:\Users\mgallai\sam3_work\sam3\. The caller is
# responsible for invoking us from that env.


SAM3_REPO_DEFAULT = Path(r"C:\Users\mgallai\sam3_work\sam3")


def _slugify(text: str, max_len: int = 40) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


def _load_sam3(checkpoint: Path, device: str):
    import inspect
    import torch
    if str(checkpoint.parent.parent) not in sys.path:
        sys.path.insert(0, str(checkpoint.parent.parent))
    from sam3.model_builder import build_sam3_image_model

    sig = inspect.signature(build_sam3_image_model)
    kwargs = {}
    if "device" in sig.parameters:          kwargs["device"] = device
    if "eval_mode" in sig.parameters:       kwargs["eval_mode"] = True
    if "checkpoint_path" in sig.parameters: kwargs["checkpoint_path"] = str(checkpoint)
    if "load_from_HF" in sig.parameters:    kwargs["load_from_HF"] = False
    model = build_sam3_image_model(**kwargs)
    model.eval()
    return model


def _composite_on_white(rgba: Image.Image) -> Image.Image:
    """Flatten RGBA -> RGB on white. Untouched RGB inputs pass through."""
    if rgba.mode != "RGBA":
        return rgba.convert("RGB")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg.convert("RGB")


def _normalize_masks(masks) -> np.ndarray:
    """SAM3 may return (N, H, W), (1, N, H, W), (H, W) etc. Normalise to bool (N, H, W)."""
    import torch
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    while masks.ndim > 3:
        if masks.shape[0] == 1:
            masks = masks[0]
        elif masks.shape[1] == 1:
            masks = masks[:, 0]
        else:
            break
    if masks.ndim == 2:
        masks = masks[np.newaxis]
    return masks > 0.5


def segment_view(processor, pil_img: Image.Image, prompt: str,
                 mode: str = "union", threshold: float = 0.5) -> np.ndarray:
    """One image, one prompt -> 2D bool mask (H, W).

    Guarantees the returned mask is exactly (pil_img.height, pil_img.width)
    so downstream voting can do a fast `mask[py, px]` lookup without a
    resize.  If SAM3 returns a different shape we resize NEAREST here.
    """
    import torch
    rgb = _composite_on_white(pil_img)
    expected_hw = (rgb.height, rgb.width)
    state = processor.set_image(rgb)
    with torch.inference_mode():
        out = processor.set_text_prompt(state=state, prompt=prompt)

    masks = out.get("masks")
    if masks is None:
        return np.zeros(expected_hw, dtype=bool)

    masks_bin = _normalize_masks(masks)
    if masks_bin.shape[0] == 0:
        return np.zeros(expected_hw, dtype=bool)

    if mode == "best":
        scores = out.get("scores")
        if scores is not None:
            scores = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
            final = masks_bin[int(np.argmax(scores))]
        else:
            final = masks_bin[0]
    else:
        # union: any region matching the prompt
        final = np.any(masks_bin, axis=0)

    # Defensive: SAM3 has been known to pad/crop internally on some inputs.
    # We always return a mask in the original image's shape (review bug H-6).
    if final.shape != expected_hw:
        pil_mask = Image.fromarray((final.astype(np.uint8) * 255))
        pil_mask = pil_mask.resize((rgb.width, rgb.height), Image.NEAREST)
        final = np.array(pil_mask) > 0
    return final


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    pil = Image.fromarray((mask * 255).astype(np.uint8))
    pil = pil.filter(ImageFilter.MinFilter(px * 2 + 1))   # erosion
    return np.array(pil) > 0


def segment_multi_prompt(
    image_paths: Sequence[Path],
    prompts: Sequence[str],
    masks_dir: Path,
    *,
    threshold: float = 0.5,
    mode: str = "union",
    erode_px: int = 2,
    device: str = "auto",
    checkpoint: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run SAM3 on every (image, prompt) pair. Returns nested dict
    `prompt -> { frame_stem: mask_bool }` and saves PNGs to masks_dir.

    Mask convention saved to PNG: 255 where the prompt's object is detected,
    0 elsewhere. (Note: this is INVERTED relative to nerfstudio's mask
    convention, where 0=ignore. Here we are extracting objects, so white =
    target object.)
    """
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = checkpoint or (SAM3_REPO_DEFAULT / "checkpoints" / "sam3.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {ckpt}")

    if verbose:
        print(f"[sam3] loading model from {ckpt} on {device} ...")
    from sam3.model.sam3_image_processor import Sam3Processor
    model = _load_sam3(ckpt, device)
    processor = Sam3Processor(model)

    masks_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, np.ndarray]] = {p: {} for p in prompts}

    total = len(image_paths) * len(prompts)
    done = 0
    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            continue
        try:
            pil = Image.open(img_path)
        except Exception as e:
            if verbose:
                print(f"[sam3] WARN: cannot open {img_path.name}: {e}")
            continue
        frame_stem = img_path.stem
        for prompt in prompts:
            slug = _slugify(prompt)
            mask = segment_view(processor, pil, prompt, mode=mode, threshold=threshold)
            if erode_px > 0:
                mask = _erode(mask, erode_px)
            results[prompt][frame_stem] = mask

            out_path = masks_dir / f"{frame_stem}__{slug}.png"
            Image.fromarray((mask.astype(np.uint8) * 255)).save(out_path)
            done += 1
            if verbose and done % 20 == 0:
                pct = mask.mean() * 100
                print(f"[sam3] {done}/{total} ({100*done/total:.0f}%)  "
                      f"last: {frame_stem} '{prompt}' cov={pct:.1f}%")

    if verbose:
        print(f"[sam3] done. {done} masks written to {masks_dir}")

    return results
