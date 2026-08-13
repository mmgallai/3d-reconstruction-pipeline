"""Batch-edit visible views for a (dataset, model) combination.

Reads visibility.json (produced by _diffusion_visibility.py) and runs the
selected editing model on ONLY the visible views. Non-visible views are
copied through unchanged so splatfacto sees a consistent full image set.

Outputs go to:
  output/diffusion_prep/<scene>/edited_<model>/images/<img_name>   (visible: edited, invisible: copy)
  output/diffusion_prep/<scene>/edited_<model>/summary.json

Usage:
  python _diffusion_batch_edit.py --scene data4 --model kontext
  python _diffusion_batch_edit.py --scene room --model qwen
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
PY      = "C:/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"
RUNNER  = PROJECT / "_shootout_run_one.py"

INSTRUCT_PROMPT = {
    "data4": "Remove the yellow tape measure, the small green artificial plant, and the tan cardboard box from the desk. Show only the empty desk surface where they were.",
    "room":  "Remove the blue armchair and the cow-print teddy bear pillow from the room. Show the empty floor and curtains behind them.",
}

MODEL_ARGS = {
    "kontext": [
        "--model-family", "flux_kontext",
        "--model-id", "black-forest-labs/FLUX.1-Kontext-dev",
        "--steps", "28", "--guidance", "4.0", "--long-edge", "1024",
        "--seed", "42", "--use-nf4",
    ],
    "qwen": [
        "--model-family", "qwen_edit",
        "--model-id", "Qwen/Qwen-Image-Edit-2511",
        "--steps", "28", "--guidance", "1.0", "--long-edge", "1024",
        "--seed", "42", "--use-nf4",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=["data4", "room"])
    ap.add_argument("--model", required=True, choices=["kontext", "qwen"])
    ap.add_argument("--start", type=int, default=0, help="Skip first N visible views (resume)")
    ap.add_argument("--limit", type=int, default=None, help="Only process N views (debug)")
    args = ap.parse_args()

    prep_dir = PROJECT / "output/diffusion_prep" / args.scene
    vis      = json.loads((prep_dir / "visibility.json").read_text())
    prompt   = INSTRUCT_PROMPT[args.scene]
    src_dir  = Path(vis["images_dir"])
    out_dir  = prep_dir / f"edited_{args.model}" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    visible   = vis["visible"]
    invisible = vis["invisible"]

    print(f"[edit:{args.scene}:{args.model}] visible={len(visible)}  invisible={len(invisible)}  "
          f"prompt='{prompt[:80]}...'")

    # First: copy invisible views through unchanged
    n_copied = 0
    for name in invisible:
        dst = out_dir / name
        if not dst.exists():
            shutil.copy2(src_dir / name, dst)
            n_copied += 1
    print(f"[edit:{args.scene}:{args.model}] copied {n_copied} unchanged views through "
          f"({len(invisible) - n_copied} already present)")

    # Then: edit visible views
    to_edit = visible[args.start:]
    if args.limit:
        to_edit = to_edit[:args.limit]

    times = []
    for i, name in enumerate(to_edit, 1):
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[edit:{args.scene}:{args.model}] [{i:4d}/{len(to_edit)}] {name} already exists — skip")
            continue

        t0 = time.perf_counter()
        cmd = [PY, str(RUNNER),
               "--image", str(src_dir / name),
               "--prompt", prompt,
               "--out", str(dst)] + MODEL_ARGS[args.model]
        res = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        times.append(dt)

        if res.returncode != 0:
            print(f"[edit:{args.scene}:{args.model}] [{i:4d}/{len(to_edit)}] FAIL {name} "
                  f"({dt:.1f}s)")
            print(res.stderr[-500:])
            continue

        mean = sum(times[-5:]) / len(times[-5:])
        eta_min = mean * (len(to_edit) - i) / 60
        print(f"[edit:{args.scene}:{args.model}] [{i:4d}/{len(to_edit)}] {name}  "
              f"{dt:.1f}s  mean5={mean:.1f}s  eta={eta_min:.1f}min")

    total_min = sum(times) / 60
    (prep_dir / f"edited_{args.model}" / "summary.json").write_text(json.dumps({
        "scene": args.scene, "model": args.model, "prompt": prompt,
        "n_edited": len(to_edit), "n_copied": len(invisible),
        "total_min": total_min,
        "mean_per_edit_s": sum(times)/max(1, len(times)),
    }, indent=2))
    print(f"[edit:{args.scene}:{args.model}] DONE. edited={len(to_edit)}  total={total_min:.1f}min")


if __name__ == "__main__":
    main()
