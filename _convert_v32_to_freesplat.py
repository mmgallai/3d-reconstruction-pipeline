"""Convert our V32 data3 into FreeSplat's ScanNet-style layout.

Output layout (per workflow investigation):
  datasets/scannet/test/v32_data3/
    color/0.jpg, 1.jpg, ... 139.jpg
    depth/0.png, 1.png, ... 139.png   (uint16 mm)
    intrinsic/intrinsic_color.txt     (4x4 with K embedded)
    intrinsic/intrinsic_depth.txt     (same as color since Femto shares intrinsics)
  datasets/scannet/test_idx.txt        ("v32_data3\n")
  assets/evaluation_index_scannet_8views.json  (sparse-view evaluation index)

extrinsics.npy at scene root: (140, 4, 4) float32 c2w OpenCV in METRES.

Chain for poses (workflow advice + our conventions):
  transforms.json holds c2w in OpenGL convention with COLMAP-unit translations.
  c2w_opencv = c2w_gl @ diag(1, -1, -1, 1)   (flip Y and Z columns)
  c2w_opencv[:3, 3] /= scale_factor_da3_to_colmap   (-> metres)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import cv2

_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=Path, default=Path("."),
                   help="V32 project root containing nerfstudio_data + colmap")
    p.add_argument("--dst-root", type=Path,
                   default=Path(r"C:\Users\mgallai\Downloads\FreeSplat\datasets"),
                   help="Where FreeSplat-format data should land. FreeSplat config "
                        "default looks under datasets/scannet/test/<scene>")
    p.add_argument("--scene-name", default="v32_data3")
    p.add_argument("--num-context", type=int, default=8,
                   help="How many views the evaluation index treats as context "
                        "(the rest become targets)")
    args = p.parse_args()

    src = args.src_root
    scene_dir = args.dst_root / "scannet" / "test" / args.scene_name
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    intr_dir = scene_dir / "intrinsic"
    for d in (color_dir, depth_dir, intr_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata
    bounds = json.loads((src / "colmap/dense/tof_bounds.json").read_text())
    scale_factor = float(bounds["scale_factor_da3_to_colmap"])
    colmap_to_metric = 1.0 / scale_factor
    fintr = json.loads((src / "nerfstudio_data/femto_intrinsics.json").read_text())
    ci = fintr["color_intrinsics"]
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]
    W, H = ci["width"], ci["height"]
    print(f"  intrinsics fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  {W}x{H}")
    print(f"  scale_factor_da3_to_colmap={scale_factor:.4f} -> colmap_to_metric={colmap_to_metric:.6f}")

    # 2. Load transforms.json
    meta = json.loads((src / "colmap/dense/transforms.json").read_text())
    frames = meta["frames"]
    print(f"  {len(frames)} frames in transforms.json")

    # 3. Per-frame copy + conversion
    extrinsics = []
    n_written = 0
    skipped_paths = []
    for idx, frame in enumerate(frames):
        fp = Path(frame["file_path"])
        stem = fp.stem                                  # IMG_femto_NNNN
        rgb_src = None
        for cand in (src / "colmap/dense" / fp, src / "nerfstudio_data" / fp,
                     src / "nerfstudio_data" / "images" / fp.name):
            if cand.exists():
                rgb_src = cand
                break
        depth_src = src / "nerfstudio_data" / "depths_femto" / f"{stem}.npy"
        if rgb_src is None or not depth_src.exists():
            skipped_paths.append(str(fp))
            continue
        # Copy RGB (rename to sequential)
        rgb_dst = color_dir / f"{idx}.jpg"
        if not rgb_dst.exists():
            shutil.copy2(rgb_src, rgb_dst)
        # Convert depth float32 m -> uint16 mm
        depth_dst = depth_dir / f"{idx}.png"
        if not depth_dst.exists():
            d = np.load(depth_src).astype(np.float32)
            d_mm = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(depth_dst), d_mm)
        # Compose c2w in OpenCV, metric metres
        c2w_gl = np.asarray(frame["transform_matrix"], dtype=np.float64)
        c2w_cv = c2w_gl @ _GL_TO_CV
        c2w_cv[:3, 3] *= colmap_to_metric
        extrinsics.append(c2w_cv.astype(np.float32))
        n_written += 1
        if (idx + 1) % 30 == 0:
            print(f"  {idx+1}/{len(frames)} frames done")

    extrinsics = np.stack(extrinsics, axis=0)            # (N, 4, 4)
    np.save(scene_dir / "extrinsics.npy", extrinsics)
    print(f"  wrote extrinsics.npy {extrinsics.shape}")

    # 4. Intrinsics file (4x4 with K embedded)
    K4 = np.eye(4, dtype=np.float64)
    K4[0, 0] = fx; K4[1, 1] = fy; K4[0, 2] = cx; K4[1, 2] = cy
    for txt_name in ("intrinsic_color.txt", "intrinsic_depth.txt"):
        with open(intr_dir / txt_name, "w") as f:
            for row in K4:
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
    print(f"  wrote intrinsic/intrinsic_color.txt + intrinsic_depth.txt")

    # 5. test_idx.txt
    test_idx_path = args.dst_root / "scannet" / "test_idx.txt"
    with open(test_idx_path, "w") as f:
        f.write(args.scene_name + "\n")
    print(f"  wrote {test_idx_path}")

    # 6. evaluation_index_scannet_<N>views.json — for the evaluation view sampler
    # The exact required schema: dict keyed by scene_name, each value has
    # context: list of frame indices, target: list of frame indices.
    eval_dir = Path(r"C:\Users\mgallai\Downloads\FreeSplat\assets")
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"evaluation_index_scannet_{args.num_context}views.json"
    n_total = n_written
    ctx_idx = np.linspace(0, n_total - 1, args.num_context, dtype=int).tolist()
    tgt_idx = [i for i in range(n_total) if i not in set(ctx_idx)]
    # IMPORTANT: dataset_scannet.py filters `merged_index = {k for k in
    # view_sampler.index if k[:-2] in test_idx}`. So the JSON key must
    # be `<scene_name>_XX` where the suffix length is 2 chars.
    key = f"{args.scene_name}_00"
    eval_index = {key: {"context": ctx_idx, "target": tgt_idx}}
    with open(eval_path, "w") as f:
        json.dump(eval_index, f, indent=2)
    print(f"  wrote {eval_path}  key={key} (ctx={len(ctx_idx)}, target={len(tgt_idx)})")

    if skipped_paths:
        print(f"\n  WARNING: skipped {len(skipped_paths)} frames (missing files): "
              f"{skipped_paths[:3]}")

    print(f"\n  =>  Total frames written: {n_written}")
    print(f"  =>  Scene dir: {scene_dir}")


if __name__ == "__main__":
    main()
