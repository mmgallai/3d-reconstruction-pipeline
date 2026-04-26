"""
Analyze a Gaussian Splat .ply file and report quality indicators.
"""
import struct
import numpy as np
from pathlib import Path

PLY_PATH = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project/output/splat.ply")

# ── 1. Parse header ───────────────────────────────────────────────────────────
with open(PLY_PATH, "rb") as f:
    header_lines = []
    while True:
        line = f.readline().decode("ascii", errors="replace").strip()
        header_lines.append(line)
        if line == "end_header":
            break
    data_start = f.tell()

print("=== PLY HEADER ===")
for l in header_lines:
    print(l)

# Parse element / property info
n_gaussians = 0
properties = []
for line in header_lines:
    if line.startswith("element vertex"):
        n_gaussians = int(line.split()[-1])
    elif line.startswith("property float"):
        properties.append(line.split()[-1])

print(f"\n=== SUMMARY ===")
print(f"Gaussians: {n_gaussians:,}")
print(f"Properties per Gaussian: {len(properties)}")
print(f"Properties: {properties[:10]} ...")

# ── 2. Load data ──────────────────────────────────────────────────────────────
# Binary little-endian floats
n_props = len(properties)
dtype = np.dtype([("f" + str(i), np.float32) for i in range(n_props)])

with open(PLY_PATH, "rb") as f:
    f.seek(data_start)
    data = np.frombuffer(f.read(n_gaussians * n_props * 4), dtype=dtype)

# Map property names to column indices
prop_idx = {name: i for i, name in enumerate(properties)}
cols = np.stack([data[f"f{i}"] for i in range(n_props)], axis=1)

def col(name):
    return cols[:, prop_idx[name]]

# ── 3. Position analysis ──────────────────────────────────────────────────────
x, y, z = col("x"), col("y"), col("z")
print(f"\n=== POSITION ===")
print(f"X range: [{x.min():.2f}, {x.max():.2f}]  span={x.max()-x.min():.2f}")
print(f"Y range: [{y.min():.2f}, {y.max():.2f}]  span={y.max()-y.min():.2f}")
print(f"Z range: [{z.min():.2f}, {z.max():.2f}]  span={z.max()-z.min():.2f}")

# Outlier detection: how many are >3 std from center?
for ax, name in [(x,"x"),(y,"y"),(z,"z")]:
    mu, sd = ax.mean(), ax.std()
    outliers = np.sum(np.abs(ax - mu) > 3 * sd)
    print(f"  {name}: mean={mu:.2f}  std={sd:.2f}  outliers(>3σ)={outliers:,} ({100*outliers/n_gaussians:.1f}%)")

# ── 4. Opacity analysis ───────────────────────────────────────────────────────
if "opacity" in prop_idx:
    raw_opacity = col("opacity")
    # nerfstudio stores raw logit; convert with sigmoid
    opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
    print(f"\n=== OPACITY (sigmoid) ===")
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        pct = 100 * np.mean(opacity > thresh)
        print(f"  > {thresh:.1f}: {pct:.1f}%  ({int(n_gaussians * pct / 100):,} Gaussians)")
    print(f"  mean opacity: {opacity.mean():.3f}  median: {np.median(opacity):.3f}")

# ── 5. Scale analysis (log scale stored) ──────────────────────────────────────
scale_keys = [k for k in properties if k.startswith("scale_")]
if scale_keys:
    scales = np.stack([col(k) for k in scale_keys], axis=1)
    # nerfstudio stores log(scale)
    scales_exp = np.exp(scales)
    max_scale = scales_exp.max(axis=1)
    print(f"\n=== SCALE (exp of stored log-scale) ===")
    print(f"  Median max-scale: {np.median(max_scale):.4f}")
    print(f"  Mean max-scale:   {max_scale.mean():.4f}")
    print(f"  >1.0 (large/floater): {100*np.mean(max_scale>1.0):.1f}%")
    print(f"  >0.5: {100*np.mean(max_scale>0.5):.1f}%")
    print(f"  <0.01 (tiny/noise):   {100*np.mean(max_scale<0.01):.1f}%")

# ── 6. Coverage / density ─────────────────────────────────────────────────────
# Estimate scene volume and Gaussian density
vol = (x.max()-x.min()) * (y.max()-y.min()) * (z.max()-z.min())
print(f"\n=== SCENE ===")
print(f"  Bounding volume: {vol:.2f} cubic units")
print(f"  Density: {n_gaussians/vol:.0f} Gaussians/unit³")

# ── 7. Overall verdict ────────────────────────────────────────────────────────
print(f"\n=== QUALITY VERDICT ===")
issues = []
if "opacity" in prop_idx:
    low_op = np.mean(opacity < 0.1)
    if low_op > 0.3:
        issues.append(f"HIGH floater ratio: {low_op*100:.0f}% of Gaussians have opacity <0.1 (expect <20%)")
if scale_keys:
    large = np.mean(max_scale > 1.0)
    if large > 0.05:
        issues.append(f"LARGE Gaussians: {large*100:.1f}% have scale >1.0 (may cause blurry regions)")
    tiny = np.mean(max_scale < 0.001)
    if tiny > 0.1:
        issues.append(f"TINY Gaussians: {tiny*100:.1f}% have scale <0.001 (noise)")
for ax, name in [(x,"x"),(y,"y"),(z,"z")]:
    mu, sd = ax.mean(), ax.std()
    out_pct = np.mean(np.abs(ax - mu) > 5 * sd) * 100
    if out_pct > 1:
        issues.append(f"OUTLIERS in {name}: {out_pct:.1f}% beyond 5σ — likely sky/background floaters")

if not issues:
    print("  No major issues detected — splat looks statistically healthy.")
else:
    for iss in issues:
        print(f"  ⚠ {iss}")
