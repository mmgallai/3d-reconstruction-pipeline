"""Inspect the FreeSplat-generated .ply and compare to our V32 splatfacto baseline."""
from pathlib import Path
import numpy as np
from plyfile import PlyData

FS = Path("output/freesplat_v32_data3/freesplat_v32_data3.ply")
V32 = Path("output/splat_v32_data3_noinit_pruned.ply")

for label, path in [("FreeSplat", FS), ("V32 splatfacto", V32)]:
    print(f"\n=== {label}: {path.name} ({path.stat().st_size/1_048_576:.1f} MB) ===")
    ply = PlyData.read(str(path))
    elem = ply["vertex"]
    print(f"  Gaussian count: {len(elem):,}")
    pos = np.stack([elem["x"], elem["y"], elem["z"]], axis=-1)
    print(f"  Position range:")
    for ax, n in zip(range(3), "xyz"):
        lo, hi = pos[:, ax].min(), pos[:, ax].max()
        p1, p99 = np.percentile(pos[:, ax], 1), np.percentile(pos[:, ax], 99)
        print(f"    {n}: [{lo:.2f}, {hi:.2f}]  p1-p99: [{p1:.2f}, {p99:.2f}]  span: {hi-lo:.2f}")

    # Opacity
    if "opacity" in elem.data.dtype.names:
        op_logit = np.asarray(elem["opacity"], dtype=np.float32)
        alpha = 1.0 / (1.0 + np.exp(-op_logit))
        print(f"  Opacity (alpha after sigmoid):")
        print(f"    median {np.median(alpha):.3f}  mean {alpha.mean():.3f}  "
              f"<0.1: {(alpha < 0.1).mean()*100:.1f}%  >0.9: {(alpha > 0.9).mean()*100:.1f}%")

    # Scale
    if all(f"scale_{i}" in elem.data.dtype.names for i in range(3)):
        s = np.stack([elem["scale_0"], elem["scale_1"], elem["scale_2"]], axis=-1)
        s_actual = np.exp(s)
        max_scale = s_actual.max(axis=-1)
        print(f"  Max-axis scale (raw, splat units):")
        print(f"    median {np.median(max_scale):.3f}  p90 {np.percentile(max_scale, 90):.3f}  "
              f"p99 {np.percentile(max_scale, 99):.3f}  max {max_scale.max():.3f}")

    # Mean SH (DC color)
    if "f_dc_0" in elem.data.dtype.names:
        SH_C0 = 0.28209479
        f_dc = np.stack([elem["f_dc_0"], elem["f_dc_1"], elem["f_dc_2"]], axis=-1).astype(np.float32)
        rgb = np.clip(f_dc * SH_C0 + 0.5, 0, 1) * 255
        print(f"  DC color (255-scale):")
        print(f"    mean R={rgb[:,0].mean():.0f} G={rgb[:,1].mean():.0f} B={rgb[:,2].mean():.0f}")
        print(f"    median R={np.median(rgb[:,0]):.0f} G={np.median(rgb[:,1]):.0f} B={np.median(rgb[:,2]):.0f}")
