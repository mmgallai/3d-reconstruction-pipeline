"""Chair-removal inpainter using FLUX.1 Fill dev.

Comparison twin to `_inpaint_chair_lama.py`. Same CLI shape + mask handling,
so the outputs are drop-in compatible with the downstream splat fine-tune.

VRAM strategy for a 16 GB card:
  1. Load pipeline in bfloat16.
  2. Enable model-CPU-offload (pipe.enable_model_cpu_offload) — moves each
     module (T5, CLIP, transformer, VAE) to GPU only when it's actively used.
     Peak VRAM stays under ~10 GB; wall clock is ~2-3x slower than fp16-only.
  3. Cap inpaint resolution to 2048 on the long side (FLUX Fill's native range;
     going higher requires tiling and adds cost). Images are resized down for
     inference then upscaled to source size with bicubic — matches the LaMa
     workflow's "native" behavior close enough for A/B splat comparison.

Determinism: fixed --seed used for EVERY image so the same prompt + mask
region should get roughly the same fill across views (FLUX is still
stochastic to a degree, so per-view divergence is expected — that's precisely
what we're testing against LaMa's true determinism in the downstream splat
fine-tune).

HF auth: FLUX.1-Fill-dev is a gated model. Requires either
  `huggingface-cli login`  OR  the HF_TOKEN env var  to be set before running.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_PROMPT = (
    "seamless polished pine hardwood floor with visible wood grain, "
    "grey linen curtain in background, warm indoor tungsten lighting, "
    "uncluttered living room floor, photorealistic 35mm photo, "
    "no furniture, no chair, no shadows"
)


def _load_union_mask(masks_root: Path, stem: str, target_hw: tuple[int, int],
                     dilate_px: int) -> tuple[np.ndarray, bool]:
    """Union blue_armchair + teddy_bear SAM3 masks, dilate, resize to target."""
    from scipy.ndimage import binary_dilation
    h, w = target_hw
    combined = np.zeros((h, w), dtype=bool)
    had_any = False
    for slug in ("blue_armchair", "teddy_bear"):
        p = masks_root / f"{stem}__{slug}.png"
        if not p.exists():
            continue
        had_any = True
        m = np.array(Image.open(p).convert("L")) > 127
        if m.shape != (h, w):
            m = np.array(
                Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (w, h), Image.NEAREST
                )
            ) > 127
        combined |= m
    if dilate_px > 0 and combined.any():
        combined = binary_dilation(combined, iterations=int(dilate_px))
    return combined.astype(np.uint8) * 255, had_any


def _resize_for_flux(pil: Image.Image, long_edge: int) -> tuple[Image.Image, tuple[int, int]]:
    """Downscale image so its long edge <= long_edge and both dims are multiples of 16.
    Returns (resized, original_size)."""
    orig_w, orig_h = pil.size
    scale = min(1.0, long_edge / max(orig_w, orig_h))
    new_w = max(16, int(round(orig_w * scale / 16)) * 16)
    new_h = max(16, int(round(orig_h * scale / 16)) * 16)
    return pil.resize((new_w, new_h), Image.LANCZOS), (orig_w, orig_h)


def inpaint_one(pipe, img_path: Path, masks_root: Path,
                out_img_dir: Path, out_mask_dir: Path,
                dilate_px: int, long_edge: int,
                prompt: str, guidance: float, steps: int,
                seed: int) -> tuple[bool, bool, float]:
    import torch
    t0 = time.perf_counter()
    img_rgb = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img_rgb.size

    mask_np, had_mask = _load_union_mask(masks_root, img_path.stem, (orig_h, orig_w), dilate_px)
    Image.fromarray(mask_np).save(out_mask_dir / f"{img_path.stem}.png")

    out_path = out_img_dir / f"{img_path.stem}.jpg"
    if mask_np.sum() == 0:
        img_rgb.save(out_path, quality=95)
        return False, had_mask, time.perf_counter() - t0

    # Resize to FLUX-compatible size
    img_small, _ = _resize_for_flux(img_rgb, long_edge)
    mask_small_pil = Image.fromarray(mask_np).resize(img_small.size, Image.NEAREST)

    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            image=img_small,
            mask_image=mask_small_pil,
            height=img_small.size[1],
            width=img_small.size[0],
            guidance_scale=guidance,
            num_inference_steps=steps,
            max_sequence_length=256,
            generator=gen,
        ).images[0]

    # Upscale back to source resolution
    if out.size != (orig_w, orig_h):
        out = out.resize((orig_w, orig_h), Image.BICUBIC)
    out.save(out_path, quality=95)
    return True, had_mask, time.perf_counter() - t0


def _dir_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--masks-dir", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out-images-dir", required=True, type=Path)
    ap.add_argument("--out-masks-dir", required=True, type=Path)
    ap.add_argument("--dilate-px", type=int, default=40)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--guidance-scale", type=float, default=30.0,
                    help="FLUX Fill's recommended guidance is high (30-50 for removal).")
    ap.add_argument("--num-inference-steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--long-edge", type=int, default=2048,
                    help="Max inpaint resolution (long edge). Higher = better quality + more VRAM.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-Fill-dev")
    args = ap.parse_args()

    args.out_images_dir.mkdir(parents=True, exist_ok=True)
    args.out_masks_dir.mkdir(parents=True, exist_ok=True)

    if not (os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").exists()):
        raise SystemExit(
            "FLUX.1-Fill-dev is a gated model. Set HF_TOKEN env var OR run "
            "`huggingface-cli login` first. See "
            "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev"
        )

    print(f"[flux] loading pipeline {args.model_id} (nf4 transformer + bf16 others, full GPU) ...")
    import torch
    from diffusers import FluxFillPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig as BnbCfg, T5EncoderModel

    # nf4 quantize the 12B transformer down to ~6.5 GB VRAM
    bnb4 = BnbCfg(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.bfloat16)

    print("  loading transformer in nf4 ...")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer",
        quantization_config=bnb4, torch_dtype=torch.bfloat16,
    )
    print("  loading T5 text encoder in nf4 ...")
    from transformers import BitsAndBytesConfig as TBnbCfg
    tbnb4 = TBnbCfg(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)
    t5 = T5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder_2",
        quantization_config=tbnb4, torch_dtype=torch.bfloat16,
    )
    print("  loading full pipeline ...")
    pipe = FluxFillPipeline.from_pretrained(
        args.model_id, transformer=transformer, text_encoder_2=t5,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    print("[flux] pipeline ready (fits in ~10 GB VRAM).")

    all_imgs = sorted(
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit:
        all_imgs = all_imgs[: args.limit]

    print(f"[flux] {len(all_imgs)} images | dilate={args.dilate_px}px | long_edge={args.long_edge} "
          f"| steps={args.num_inference_steps} | guidance={args.guidance_scale} | seed={args.seed}")
    print(f"[flux] prompt: {args.prompt}")

    times = []
    filled = passthrough = no_mask = 0
    for i, p in enumerate(all_imgs, 1):
        did_fill, had_m, dt = inpaint_one(
            pipe, p, args.masks_dir, args.out_images_dir, args.out_masks_dir,
            args.dilate_px, args.long_edge, args.prompt, args.guidance_scale,
            args.num_inference_steps, args.seed,
        )
        times.append(dt)
        if did_fill:
            filled += 1
        elif not had_m:
            no_mask += 1
        else:
            passthrough += 1
        if i % 5 == 0:
            mean = sum(times[-5:]) / len(times[-5:])
            eta_s = mean * (len(all_imgs) - i)
            print(f"  [{i:4d}/{len(all_imgs)}] last5_mean={mean:.1f}s  eta={eta_s/60:.1f}min")

    total = sum(times)
    mean_all = total / max(1, len(times))
    print()
    print(f"[flux] done")
    print(f"  processed : {len(all_imgs)}  (filled={filled}, passthrough={passthrough}, no-mask={no_mask})")
    print(f"  mean/img  : {mean_all:.1f}s")
    print(f"  wall      : {total/60:.1f}min")
    print(f"  out imgs  : {_dir_size_mb(args.out_images_dir):.1f} MB")
    print(f"  out masks : {_dir_size_mb(args.out_masks_dir):.1f} MB")


if __name__ == "__main__":
    main()
