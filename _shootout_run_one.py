"""Shootout wrapper: run ONE inpaint/edit model on the chair test image.

Dispatches on --model-family, loads the correct diffusers/iopaint pipeline,
runs a single inference on (image, [mask], prompt), saves the result, and
tears down GPU memory before exit so a shootout harness can invoke this
script sequentially per model without OOM.

Design notes
------------
- Each family lives in its own `run_<family>()` function; main() only
  dispatches + does teardown, so adding a new model = add one function
  + one entry in FAMILY_TABLE.
- Heavy imports (torch, diffusers, transformers, iopaint) are done
  INSIDE each run_* function so `--help` works with only argparse.
- NF4 quantization pattern for the DiT-family models (FLUX / Qwen /
  FireRed) mirrors `_inpaint_chair_flux.py`: quantize the transformer
  (and T5 for FLUX) via bitsandbytes, load remaining subfolders in bf16.
- Teardown: `del pipe`, drop known cross-references, `gc.collect()`,
  `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`. Called even
  if inference raised, via try/finally.

Model-family table
------------------
family         pipeline class                          needs_mask  guidance (typical)
------         --------------                          ----------  -----------------
lama           iopaint.ModelManager("lama")            YES         n/a (deterministic)
flux_fill      FluxFillPipeline                        YES         30-50 (default 30)
flux_kontext   FluxKontextPipeline (mask=None)         no          2.5   (BFL default)
               FluxKontextInpaintPipeline (if mask)    YES         2.5
qwen_edit      QwenImageEditPlusPipeline               no          true_cfg=4.0, guidance=1.0
fired_edit     QwenImageEditPlusPipeline (same class)  no          true_cfg=4.0
z_image_edit   ZImageInpaintPipeline (Z-Image-Edit     YES         0.0  (turbo distilled)
               unreleased; use inpaint on Z-Image-Turbo)

CLI
---
    python _shootout_run_one.py \
        --model-family flux_kontext \
        --model-id     black-forest-labs/FLUX.1-Kontext-dev \
        --image        chair.png \
        --mask         chair_mask.png       # optional for kontext/qwen/fired
        --prompt       "Remove the chair..." \
        --out          out_kontext.png \
        --seed         42 --steps 28 --guidance 2.5 --long-edge 1024 \
        [--use-nf4]

Exit code 0 on success. On any failure the exception propagates (non-zero
exit) so the shootout harness records it as a run failure.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Optional


# Family -> pipeline class + needs-mask flag + typical guidance range,
# duplicated here (as data) so `--help` output and log lines can reference it.
FAMILY_TABLE = {
    "lama":         ("iopaint.ModelManager('lama')",         True,  "n/a"),
    "flux_fill":    ("FluxFillPipeline",                     True,  "30-50"),
    "flux_kontext": ("FluxKontextPipeline (mask optional)",  False, "2.5"),
    "qwen_edit":    ("QwenImageEditPlusPipeline",            False, "true_cfg=4.0"),
    "fired_edit":   ("QwenImageEditPlusPipeline",            False, "true_cfg=4.0"),
    "z_image_edit": ("ZImageInpaintPipeline (Turbo inpaint)", True, "0.0 (turbo)"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_rgb(path: Path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _load_mask(path: Optional[Path], size):
    """Load a binary mask, resize to `size` (w, h). Returns L-mode PIL or None."""
    if path is None:
        return None
    from PIL import Image
    m = Image.open(path).convert("L")
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    return m


def _resize_long_edge(pil, long_edge: int, multiple: int = 16):
    """Downscale so max(w, h) <= long_edge; snap both dims to `multiple`.
    Returns (resized_pil, (orig_w, orig_h))."""
    from PIL import Image
    ow, oh = pil.size
    scale = min(1.0, long_edge / max(ow, oh))
    nw = max(multiple, int(round(ow * scale / multiple)) * multiple)
    nh = max(multiple, int(round(oh * scale / multiple)) * multiple)
    return pil.resize((nw, nh), Image.LANCZOS), (ow, oh)


def _free_cuda():
    """Best-effort GPU memory teardown between runs."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Family runners
# Each returns nothing; each writes `out_path`; each is responsible for its
# own `del pipe` on the way out (but main() also calls _free_cuda()).
# ---------------------------------------------------------------------------
def run_lama(args) -> None:
    """IOPaint LaMa — CPU/GPU classical inpainter. Needs mask."""
    import numpy as np
    import torch
    from PIL import Image
    from iopaint.model_manager import ModelManager
    from iopaint.schema import HDStrategy, InpaintRequest, LDMSampler

    if args.mask is None:
        raise SystemExit("[lama] --mask is required for LaMa.")

    img_rgb = np.array(_load_rgb(args.image))
    h, w = img_rgb.shape[:2]
    mask_pil = _load_mask(args.mask, (w, h))
    mask = (np.array(mask_pil) > 127).astype(np.uint8) * 255

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lama] device={device}")
    model = ModelManager(name="lama", device=device)
    cfg = InpaintRequest(
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_trigger_size=2048,
        hd_strategy_crop_margin=196,
        hd_strategy_resize_limit=2048,
        ldm_steps=args.steps,
        ldm_sampler=LDMSampler.plms,  # harmless for LaMa
    )
    # IOPaint expects BGR uint8 input; returns RGB uint8.
    img_bgr = img_rgb[:, :, ::-1].copy()
    t0 = time.perf_counter()
    result_rgb = model(img_bgr, mask, cfg)
    print(f"[lama] inference: {time.perf_counter() - t0:.2f}s")
    Image.fromarray(result_rgb).save(args.out)

    del model


def _bnb_nf4_cfg():
    """bitsandbytes NF4 config with bf16 compute (matches _inpaint_chair_flux.py)."""
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def run_flux_fill(args) -> None:
    """FLUX.1 Fill dev — masked inpainter. High guidance (30-50)."""
    import torch
    from diffusers import FluxFillPipeline, FluxTransformer2DModel
    from transformers import T5EncoderModel

    if args.mask is None:
        raise SystemExit("[flux_fill] --mask is required.")

    model_id = args.model_id or "black-forest-labs/FLUX.1-Fill-dev"
    print(f"[flux_fill] loading {model_id} (nf4={args.use_nf4}) ...")

    if args.use_nf4:
        bnb = _bnb_nf4_cfg()
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer",
            quantization_config=bnb, torch_dtype=torch.bfloat16,
        )
        t5 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2",
            quantization_config=bnb, torch_dtype=torch.bfloat16,
        )
        pipe = FluxFillPipeline.from_pretrained(
            model_id, transformer=transformer, text_encoder_2=t5,
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = FluxFillPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    img_rgb = _load_rgb(args.image)
    img_small, _orig = _resize_long_edge(img_rgb, args.long_edge, multiple=16)
    mask_small = _load_mask(args.mask, img_small.size)

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = pipe(
            prompt=args.prompt,
            image=img_small,
            mask_image=mask_small,
            height=img_small.size[1], width=img_small.size[0],
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            max_sequence_length=256,
            generator=gen,
        ).images[0]
    print(f"[flux_fill] inference: {time.perf_counter() - t0:.2f}s")
    if out.size != img_rgb.size:
        from PIL import Image
        out = out.resize(img_rgb.size, Image.BICUBIC)
    out.save(args.out)

    del pipe


def run_flux_kontext(args) -> None:
    """FLUX.1 Kontext dev — instruction-based editor. No mask required."""
    import torch
    from diffusers import FluxTransformer2DModel
    from transformers import T5EncoderModel

    model_id = args.model_id or "black-forest-labs/FLUX.1-Kontext-dev"
    use_inpaint = args.mask is not None
    if use_inpaint:
        from diffusers import FluxKontextInpaintPipeline as Pipe
    else:
        from diffusers import FluxKontextPipeline as Pipe

    print(f"[flux_kontext] loading {model_id} "
          f"(pipeline={Pipe.__name__}, nf4={args.use_nf4}) ...")

    if args.use_nf4:
        bnb = _bnb_nf4_cfg()
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer",
            quantization_config=bnb, torch_dtype=torch.bfloat16,
        )
        t5 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2",
            quantization_config=bnb, torch_dtype=torch.bfloat16,
        )
        pipe = Pipe.from_pretrained(
            model_id, transformer=transformer, text_encoder_2=t5,
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = Pipe.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    img_rgb = _load_rgb(args.image)
    img_small, _orig = _resize_long_edge(img_rgb, args.long_edge, multiple=16)

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    kwargs = dict(
        image=img_small,
        prompt=args.prompt,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=gen,
    )
    if use_inpaint:
        mask_small = _load_mask(args.mask, img_small.size)
        kwargs["mask_image"] = mask_small
        kwargs["strength"] = 1.0

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = pipe(**kwargs).images[0]
    print(f"[flux_kontext] inference: {time.perf_counter() - t0:.2f}s")
    if out.size != img_rgb.size:
        from PIL import Image
        out = out.resize(img_rgb.size, Image.BICUBIC)
    out.save(args.out)

    del pipe


def _run_qwen_family(args, default_model_id: str, tag: str) -> None:
    """Shared impl for Qwen-Image-Edit-2511 and FireRed-Image-Edit-1.1.

    Both ship `QwenImageEditPlusPipeline` = Qwen2.5-VL-7B text encoder +
    20B QwenImageTransformer2DModel + Flow-Match Euler scheduler.

    16 GB VRAM budget:
      - nf4 transformer  (20B bf16 = 40 GB  ->  ~10 GB nf4)
      - nf4 text encoder (7B  bf16 = 14 GB  ->  ~4 GB nf4)
      - VAE + processors in bf16 (small)
      - enable_model_cpu_offload() moves inactive modules to CPU each step
        (peak stays ~12 GB; wall clock ~2x slower vs. all-on-GPU)."""
    import torch
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
    from transformers import Qwen2_5_VLForConditionalGeneration

    model_id = args.model_id or default_model_id
    print(f"[{tag}] loading {model_id} (nf4={args.use_nf4}) ...")

    if args.use_nf4:
        bnb = _bnb_nf4_cfg()
        print(f"[{tag}]   loading transformer (nf4) ...")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer",
            quantization_config=bnb, torch_dtype=torch.bfloat16,
        )
        print(f"[{tag}]   loading text encoder Qwen2.5-VL (nf4) ...")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, subfolder="text_encoder",
            quantization_config=bnb, torch_dtype=torch.bfloat16,
        )
        print(f"[{tag}]   assembling pipeline ...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id, transformer=transformer, text_encoder=text_encoder,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
    else:
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")

    img_rgb = _load_rgb(args.image)
    img_small, _orig = _resize_long_edge(img_rgb, args.long_edge, multiple=16)

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = pipe(
            image=[img_small],           # list, even for a single input
            prompt=args.prompt,
            negative_prompt=" ",
            true_cfg_scale=4.0,          # Qwen/FireRed recommended
            guidance_scale=args.guidance,  # base cfg (1.0 is fine)
            num_inference_steps=args.steps,
            generator=gen,
        ).images[0]
    print(f"[{tag}] inference: {time.perf_counter() - t0:.2f}s")
    if out.size != img_rgb.size:
        from PIL import Image
        out = out.resize(img_rgb.size, Image.BICUBIC)
    out.save(args.out)

    del pipe


def run_qwen_edit(args) -> None:
    _run_qwen_family(args, "Qwen/Qwen-Image-Edit-2511", tag="qwen_edit")


def run_fired_edit(args) -> None:
    _run_qwen_family(args, "FireRedTeam/FireRed-Image-Edit-1.1", tag="fired_edit")


def run_z_image_edit(args) -> None:
    """Z-Image-Edit is unreleased (as of 2026-08-06). Fall back to
    ZImageInpaintPipeline on Z-Image-Turbo, which requires a mask.
    Turbo is 8-NFE distilled; guidance_scale=0.0 recommended."""
    import torch
    from PIL import Image
    from diffusers import ZImageInpaintPipeline

    if args.mask is None:
        raise SystemExit(
            "[z_image_edit] Z-Image-Edit is unreleased; falling back to "
            "ZImageInpaintPipeline on Z-Image-Turbo — --mask is required."
        )
    model_id = args.model_id or "Tongyi-MAI/Z-Image-Turbo"
    print(f"[z_image_edit] loading {model_id} (inpaint fallback) ...")
    pipe = ZImageInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    img_rgb = _load_rgb(args.image)
    img_small, _orig = _resize_long_edge(img_rgb, args.long_edge, multiple=16)
    mask_small = _load_mask(args.mask, img_small.size)

    # Turbo is distilled: 8 steps, guidance 0.0. Honor CLI overrides but log.
    steps = args.steps if args.steps != 28 else 8
    guidance = args.guidance if args.guidance != 3.5 else 0.0
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = pipe(
            prompt=args.prompt,
            image=img_small,
            mask_image=mask_small,
            strength=1.0,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        ).images[0]
    print(f"[z_image_edit] inference: {time.perf_counter() - t0:.2f}s "
          f"(steps={steps}, guidance={guidance})")
    if out.size != img_rgb.size:
        out = out.resize(img_rgb.size, Image.BICUBIC)
    out.save(args.out)

    del pipe


# ---------------------------------------------------------------------------
DISPATCH = {
    "lama":         run_lama,
    "flux_fill":    run_flux_fill,
    "flux_kontext": run_flux_kontext,
    "qwen_edit":    run_qwen_edit,
    "fired_edit":   run_fired_edit,
    "z_image_edit": run_z_image_edit,
}


def _table_epilog() -> str:
    """Pretty-print FAMILY_TABLE for `--help` output."""
    rows = ["family        pipeline                                   mask  guidance"]
    rows.append("-" * len(rows[0]))
    for fam, (cls, needs, g) in FAMILY_TABLE.items():
        rows.append(f"{fam:<13} {cls:<42} {'yes ' if needs else 'no  '} {g}")
    return "Model-family table:\n" + "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run ONE inpaint/edit model on a single image + prompt "
                    "and save the output. Cleans GPU memory before exit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_table_epilog(),
    )
    ap.add_argument("--model-family", required=True, choices=list(DISPATCH),
                    help="Which model family to run.")
    ap.add_argument("--model-id", default=None,
                    help="HF repo path (diffusion models); ignored for lama. "
                         "If omitted, family default is used.")
    ap.add_argument("--image", required=True, type=Path,
                    help="Input RGB image.")
    ap.add_argument("--mask",  default=None, type=Path,
                    help="Binary mask (255=fill/edit). Required for lama, "
                         "flux_fill, z_image_edit. Optional for kontext/qwen/fired.")
    ap.add_argument("--prompt", required=True,
                    help="Instruction (kontext/qwen/fired) or fill description "
                         "(flux_fill/lama/z_image).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Where to write the output image.")
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--steps",    type=int,   default=28,
                    help="Sampling steps (Turbo variants auto-lower to 8).")
    ap.add_argument("--guidance", type=float, default=3.5,
                    help="Classifier-free guidance scale. Recommended: "
                         "flux_fill 30-50, flux_kontext 2.5, qwen/fired 1.0, "
                         "z_image_turbo 0.0.")
    ap.add_argument("--long-edge", type=int,  default=1024,
                    help="Downscale so max(w,h) <= long_edge (multiple of 16).")
    ap.add_argument("--use-nf4",  action="store_true",
                    help="Quantize transformer (+T5 for FLUX) to nf4 via "
                         "bitsandbytes to fit in ~10-12 GB VRAM.")
    args = ap.parse_args()

    args.image = args.image.resolve()
    if args.mask is not None:
        args.mask = args.mask.resolve()
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fn = DISPATCH[args.model_family]
    print(f"[shootout] family={args.model_family} model_id={args.model_id or '(default)'}")
    print(f"[shootout] image={args.image}")
    print(f"[shootout] mask ={args.mask}")
    print(f"[shootout] out  ={args.out}")
    print(f"[shootout] prompt: {args.prompt}")

    t0 = time.perf_counter()
    try:
        fn(args)
    finally:
        # Always try to reclaim GPU memory, even on failure, so the harness
        # can move on to the next family without needing to fork.
        _free_cuda()
    print(f"[shootout] wall={time.perf_counter() - t0:.1f}s  -> {args.out}")


if __name__ == "__main__":
    main()
