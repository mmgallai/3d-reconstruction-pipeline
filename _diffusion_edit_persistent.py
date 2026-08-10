"""Persistent-pipeline batch editor. Loads the diffusion pipeline ONCE
and processes many images in the same process.  Much faster than
spawning a fresh Python + reloading the pipeline per image.

Usage:
  python _diffusion_edit_persistent.py --scene data4 --model kontext
  python _diffusion_edit_persistent.py --scene room  --model qwen
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Force unbuffered stdout so progress lines land in the log promptly.
sys.stdout.reconfigure(line_buffering=True)

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")

INSTRUCT_PROMPT = {
    "data4": "Remove the yellow tape measure, the small green artificial plant, and the tan cardboard box from the desk. Show only the empty desk surface where they were.",
    "room":  "Remove the blue armchair and the cow-print teddy bear pillow from the room. Show the empty floor and curtains behind them.",
    "data3": "Remove the white water bottle, the blue Arducam box, and the red plastic lobster toy from the desk. Show only the empty desk surface where they were.",
}

LONG_EDGE = 1024
STEPS     = 28
SEED      = 42


def resize_long_edge(pil, long_edge=LONG_EDGE, multiple=16):
    from PIL import Image
    ow, oh = pil.size
    scale = min(1.0, long_edge / max(ow, oh))
    nw = max(multiple, int(round(ow * scale / multiple)) * multiple)
    nh = max(multiple, int(round(oh * scale / multiple)) * multiple)
    return pil.resize((nw, nh), Image.LANCZOS), (ow, oh)


def load_kontext():
    import torch
    from diffusers import FluxKontextPipeline, FluxTransformer2DModel
    from transformers import T5EncoderModel, BitsAndBytesConfig
    print("[load] FLUX.1-Kontext-dev nf4 ...")
    model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer",
        quantization_config=bnb, torch_dtype=torch.bfloat16)
    t5 = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder_2",
        quantization_config=bnb, torch_dtype=torch.bfloat16)
    pipe = FluxKontextPipeline.from_pretrained(
        model_id, transformer=transformer, text_encoder_2=t5,
        torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    print("[load] pipeline ready")

    def infer(img_pil, prompt):
        img_small, orig = resize_long_edge(img_pil, LONG_EDGE, 16)
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        with torch.inference_mode():
            out = pipe(
                image=img_small,
                prompt=prompt,
                guidance_scale=4.0,
                num_inference_steps=STEPS,
                generator=gen,
            ).images[0]
        if out.size != img_pil.size:
            from PIL import Image
            out = out.resize(img_pil.size, Image.BICUBIC)
        return out
    return infer


def load_qwen():
    import torch
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
    from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    print("[load] Qwen-Image-Edit-2511 nf4 (both) + cpu_offload ...")
    model_id = "Qwen/Qwen-Image-Edit-2511"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer",
        quantization_config=bnb, torch_dtype=torch.bfloat16)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder",
        quantization_config=bnb, torch_dtype=torch.bfloat16)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_id, transformer=transformer, text_encoder=text_encoder,
        torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    print("[load] pipeline ready (cpu_offload)")

    def infer(img_pil, prompt):
        img_small, orig = resize_long_edge(img_pil, LONG_EDGE, 16)
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        with torch.inference_mode():
            out = pipe(
                image=[img_small],
                prompt=prompt,
                negative_prompt=" ",
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                num_inference_steps=STEPS,
                generator=gen,
            ).images[0]
        if out.size != img_pil.size:
            from PIL import Image
            out = out.resize(img_pil.size, Image.BICUBIC)
        return out
    return infer


LOADERS = {"kontext": load_kontext, "qwen": load_qwen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=["data4", "room", "data3"])
    ap.add_argument("--model", required=True, choices=["kontext", "qwen"])
    args = ap.parse_args()

    from PIL import Image

    prep_dir = PROJECT / "output/diffusion_prep" / args.scene
    vis      = json.loads((prep_dir / "visibility.json").read_text())
    prompt   = INSTRUCT_PROMPT[args.scene]
    src_dir  = Path(vis["images_dir"])
    out_dir  = prep_dir / f"edited_{args.model}" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    visible   = vis["visible"]
    invisible = vis["invisible"]

    print(f"[edit:{args.scene}:{args.model}] visible={len(visible)}  invisible={len(invisible)}")
    print(f"[edit:{args.scene}:{args.model}] prompt: {prompt}")

    # Copy invisible views through unchanged
    n_copied_new = 0
    for name in invisible:
        dst = out_dir / name
        if not dst.exists():
            shutil.copy2(src_dir / name, dst)
            n_copied_new += 1
    print(f"[edit:{args.scene}:{args.model}] copied {n_copied_new} unchanged views ({len(invisible)} total)")

    # Load pipeline ONCE
    infer = LOADERS[args.model]()

    times = []
    for i, name in enumerate(visible, 1):
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[edit:{args.scene}:{args.model}] [{i:4d}/{len(visible)}] {name} exists — skip", flush=True)
            continue

        t0 = time.perf_counter()
        try:
            img = Image.open(src_dir / name).convert("RGB")
            out = infer(img, prompt)
            out.save(dst, quality=95)
        except Exception as e:
            print(f"[edit:{args.scene}:{args.model}] [{i:4d}/{len(visible)}] FAIL {name}: {e}", flush=True)
            continue
        dt = time.perf_counter() - t0
        times.append(dt)
        mean = sum(times[-5:]) / len(times[-5:])
        eta_min = mean * (len(visible) - i) / 60
        print(f"[edit:{args.scene}:{args.model}] [{i:4d}/{len(visible)}] {name}  {dt:.1f}s  mean5={mean:.1f}s  eta={eta_min:.1f}min",
              flush=True)

    total_min = sum(times) / 60
    (prep_dir / f"edited_{args.model}" / "summary.json").write_text(json.dumps({
        "scene": args.scene, "model": args.model, "prompt": prompt,
        "n_edited": len(times),
        "n_copied": len(invisible),
        "total_min": total_min,
        "mean_per_edit_s": sum(times)/max(1, len(times)),
    }, indent=2))
    print(f"[edit:{args.scene}:{args.model}] DONE. edited={len(times)}  wall={total_min:.1f}min")


if __name__ == "__main__":
    main()
