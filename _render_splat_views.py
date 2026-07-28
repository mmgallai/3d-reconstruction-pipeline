"""Render N views of an INRIA-format Gaussian-splat PLY using gsplat.

Runs inside the nerfstudio-blackwell Docker image (gsplat compiled for CUDA).

Handles the nerfstudio dataparser transform (rotation + scale) that splatfacto
applies to poses before training. The splat lives in dataparser-normalized
space; poses must be transformed the same way.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

import gsplat

SH_C0 = 0.28209479177387814


def load_ply_splat(path):
    ply = PlyData.read(path)
    v = ply["vertex"]
    N = len(v)
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
    opacity = np.array(v["opacity"], dtype=np.float32)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
    # INRIA/splatfacto quat order: w, x, y, z
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)

    rest_names = sorted([n for n in v.data.dtype.names if n.startswith("f_rest_")])
    if rest_names:
        rest = np.stack([v[k] for k in rest_names], axis=-1).astype(np.float32)
        n_rest = rest.shape[1] // 3
        # INRIA stores as (channels, coeffs): reshape (N, 3, n_rest) then transpose to (N, n_rest, 3)
        rest = rest.reshape(N, 3, n_rest).transpose(0, 2, 1)
        sh0 = dc.reshape(N, 1, 3)
        colors = np.concatenate([sh0, rest], axis=1)  # (N, K+1, 3)
    else:
        colors = dc.reshape(N, 1, 3)

    K = colors.shape[1]
    sh_degree = int(round(math.sqrt(K))) - 1

    print(f"  N={N}  sh_degree={sh_degree}  (K={K})", flush=True)
    return dict(
        means=torch.from_numpy(xyz).cuda(),
        colors=torch.from_numpy(colors).cuda(),
        opacities=torch.sigmoid(torch.from_numpy(opacity).cuda()),
        scales=torch.exp(torch.from_numpy(scales).cuda()),
        quats=torch.from_numpy(quats).cuda(),
        sh_degree=sh_degree,
        N=N,
    )


def load_poses(transforms_json, dataparser_json, indices):
    """Load N camera poses in the same normalized space as the trained splat.

    Nerfstudio applies:  c2w_new = dp_transform @ c2w_original
                         c2w_new[:3, 3] *= dp_scale
    Then converts GL → CV (flip Y and Z axes of rotation) at render time.
    """
    tj = json.loads(Path(transforms_json).read_text())
    dj = json.loads(Path(dataparser_json).read_text())
    dp_T = np.array(dj["transform"], dtype=np.float64)
    if dp_T.shape == (3, 4):
        dp_T = np.vstack([dp_T, [0.0, 0.0, 0.0, 1.0]])
    dp_scale = float(dj["scale"])

    W = int(tj.get("w", 1556))
    H = int(tj.get("h", 1038))
    fx = float(tj.get("fl_x"))
    fy = float(tj.get("fl_y"))
    cx = float(tj.get("cx"))
    cy = float(tj.get("cy"))

    frames = tj["frames"]
    viewmats, Ks, names = [], [], []
    for i in indices:
        if i < 0 or i >= len(frames):
            continue
        fr = frames[i]
        c2w_gl = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w_new = dp_T @ c2w_gl
        c2w_new[:3, 3] *= dp_scale
        # OpenGL → OpenCV: negate columns 1 and 2 of rotation
        c2w_cv = c2w_new.copy()
        c2w_cv[:3, 1] *= -1
        c2w_cv[:3, 2] *= -1
        w2c = np.linalg.inv(c2w_cv)
        viewmats.append(w2c.astype(np.float32))
        Ks.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32))
        names.append(Path(fr.get("file_path", f"frame_{i}")).stem)

    return (
        torch.from_numpy(np.stack(viewmats)).cuda(),
        torch.from_numpy(np.stack(Ks)).cuda(),
        W, H, names,
    )


def save_grid(rgb, alpha, out_path, names, cols=None):
    """rgb: (C, H, W, 3) float in [0, 1]. Alpha overlaid on white background."""
    imgs = rgb.clamp(0, 1)
    # Composite alpha over white
    a = alpha.clamp(0, 1)
    composited = imgs * a + (1.0 - a)
    arr = (composited.cpu().numpy() * 255).astype(np.uint8)
    C, H, W, _ = arr.shape
    if cols is None:
        cols = min(C, 3)
    rows = math.ceil(C / cols)
    grid = np.full((rows * H, cols * W, 3), 240, dtype=np.uint8)
    for i in range(C):
        r, c = i // cols, i % cols
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = arr[i]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
    print(f"  saved {out_path}  {grid.shape}  views={names}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--dataparser", required=True)
    ap.add_argument("--indices", default="0,60,120,180,240")
    ap.add_argument("--out", required=True)
    ap.add_argument("--downscale", type=int, default=2)
    args = ap.parse_args()

    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    print(f"[render] ply={args.ply}", flush=True)
    splat = load_ply_splat(args.ply)
    print(f"[render] poses={args.transforms}  dp={args.dataparser}  indices={indices}", flush=True)
    viewmats, Ks, W, H, names = load_poses(args.transforms, args.dataparser, indices)

    if args.downscale > 1:
        W //= args.downscale
        H //= args.downscale
        Ks[:, 0, 0] /= args.downscale
        Ks[:, 1, 1] /= args.downscale
        Ks[:, 0, 2] /= args.downscale
        Ks[:, 1, 2] /= args.downscale

    print(f"[render] rendering {viewmats.shape[0]} views at {W}x{H}", flush=True)
    with torch.no_grad():
        rgb, alpha, _ = gsplat.rasterization(
            means=splat["means"],
            quats=splat["quats"],
            scales=splat["scales"],
            opacities=splat["opacities"],
            colors=splat["colors"],
            viewmats=viewmats,
            Ks=Ks,
            width=W, height=H,
            sh_degree=splat["sh_degree"],
            render_mode="RGB",
            packed=False,
        )

    save_grid(rgb, alpha, args.out, names)


if __name__ == "__main__":
    main()
