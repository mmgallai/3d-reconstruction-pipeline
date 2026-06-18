"""Clean-GS Option 2: refine our v5_shcolor per-object splats further by
running Clean-GS WITHIN the existing tight crop.

The original Clean-GS v1 run (output/clean_gs_v32_data3/) fed the FULL
scene splat (226k Gaussians) and got halos around each object because
Clean-GS's 2D mask projection can't distinguish "in mask AND on object"
from "in mask AND behind/beside object at a different depth".

By starting from the v5_shcolor per-object splat (which is already
spatially cropped to the object's 3D footprint via v9a_fp_v2), the
depth-ambiguity that caused halos is gone — Clean-GS can only further
prune within an already-tight crop.

Reuses the cameras.json + masked_images dirs from the v1 run.
"""
import argparse
import subprocess
import sys
from pathlib import Path

CLEAN_GS_SCRIPT = Path(r"C:\Users\mgallai\Downloads\clean-gs\clean-gs.py")
SAM3_ENV_PY = Path(r"C:\Users\mgallai\AppData\Local\miniconda3\envs\sam3\python.exe")
V5_DIR = Path("output/segmented_v32_data3_v9a_fp_v2_splat_v5_shcolor")
V1_CGS_DIR = Path("output/clean_gs_v32_data3")
OUT_DIR = Path("output/clean_gs_v2_v32_data3")

PROMPTS = ["white_water_bottle", "blue_box", "red_lobster_figurine"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outlier-mode", default="neighbor",
                   choices=["none", "spatial", "neighbor", "multiview", "combined"])
    p.add_argument("--color-threshold", type=float, default=0.40)
    args = p.parse_args()

    cameras_json = V1_CGS_DIR / "cameras.json"
    if not cameras_json.exists():
        sys.exit(f"missing {cameras_json} — run v1 first")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for prompt in PROMPTS:
        in_splat = V5_DIR / f"{prompt}_splat.ply"
        masks_dir = V1_CGS_DIR / f"masks_{prompt}"      # reuse from v1 run
        out_splat = OUT_DIR / f"clean_gs_v2_{prompt}.ply"

        if not in_splat.exists():
            print(f"[{prompt}] missing input splat {in_splat} — skipping")
            continue
        if not masks_dir.exists() or not any(masks_dir.iterdir()):
            print(f"[{prompt}] missing masks {masks_dir} — skipping")
            continue

        sz_in = in_splat.stat().st_size / 1_048_576
        print(f"\n=== {prompt} ===")
        print(f"  input:  {in_splat.name}  ({sz_in:.1f} MB)")
        print(f"  masks:  {masks_dir.name}  ({len(list(masks_dir.iterdir()))} files)")

        cmd = [
            str(SAM3_ENV_PY),
            str(CLEAN_GS_SCRIPT),
            "--ply", str(in_splat),
            "--cameras", str(cameras_json),
            "--masked_images", str(masks_dir),
            "--output", str(out_splat),
            "--mode", args.outlier_mode,
            "--color_threshold", str(args.color_threshold),
        ]
        result = subprocess.run(cmd, capture_output=False)
        if out_splat.exists():
            sz_out = out_splat.stat().st_size / 1_048_576
            print(f"  -> {out_splat.name}  ({sz_out:.1f} MB, "
                  f"{100*sz_out/sz_in:.0f}% of input size)")


if __name__ == "__main__":
    main()
