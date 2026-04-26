"""
Compare PLY files from Polycam, RealityScan, and our pipeline.
Works with both point clouds and Gaussian splat PLYs.
"""
import numpy as np
from pathlib import Path
import struct

FILES = {
    "Polycam":        r"C:\Users\mgallai\Projects\3d_automated\reconstruction_project\output\other apps\polycam.ply",
    "Ours_V4_pruned": r"C:\Users\mgallai\Projects\3d_automated\reconstruction_project\output\splat_v4_mvs_500k_pruned.ply",
    "Ours_V5_pruned": r"C:\Users\mgallai\Projects\3d_automated\reconstruction_project\output\splat_v5_da3_pruned.ply",
    "Ours_V6_full":   r"C:\Users\mgallai\Projects\3d_automated\reconstruction_project\output\splat_v6_da3.ply",
    "Ours_V6_pruned": r"C:\Users\mgallai\Projects\3d_automated\reconstruction_project\output\splat_v6_da3_pruned.ply",
}

PLY_TO_NP = {
    "float": np.float32, "float32": np.float32,
    "double": np.float64,"float64": np.float64,
    "uchar":  np.uint8,  "uint8":   np.uint8,
    "char":   np.int8,
    "short":  np.int16,  "ushort":  np.uint16,
    "int":    np.int32,  "uint":    np.uint32,
}

def read_ply_header(path):
    with open(path, "rb") as f:
        lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            lines.append(line)
            if line == "end_header":
                break
        data_start = f.tell()
    return lines, data_start

def parse_header(lines):
    n = 0
    props = []
    fmt = "binary_little_endian"
    for line in lines:
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("format"):
            fmt = line.split()[1]
        elif line.startswith("property"):
            parts = line.split()
            ptype, pname = parts[1], parts[2]
            props.append((pname, PLY_TO_NP.get(ptype, np.float32)))
    return n, props, fmt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

def analyze(name, path):
    path = Path(path)
    if not path.exists():
        print(f"\n{'='*55}")
        print(f"  {name}: FILE NOT FOUND")
        return

    size_mb = path.stat().st_size / 1_048_576
    header_lines, data_start = read_ply_header(path)
    n, props, fmt = parse_header(header_lines)
    prop_names = [p[0] for p in props]
    dtype = np.dtype(props)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  File size    : {size_mb:.0f} MB")
    print(f"  Format       : {fmt}")
    print(f"  Vertex count : {n:,}")
    print(f"  Properties   : {len(props)}  → {prop_names[:8]}" + (" ..." if len(prop_names)>8 else ""))

    # Load data (sample if huge)
    with open(path, "rb") as f:
        f.seek(data_start)
        raw = f.read(n * dtype.itemsize)

    try:
        data = np.frombuffer(raw, dtype=dtype)
    except Exception as e:
        print(f"  [WARN] Could not parse binary data: {e}")
        return

    # Position stats
    for ax in ("x","y","z"):
        if ax in prop_names:
            v = data[ax].astype(np.float32)
            print(f"  {ax} range : [{v.min():.2f}, {v.max():.2f}]  span={v.max()-v.min():.2f}  std={v.std():.2f}")

    # Bounding volume
    if all(a in prop_names for a in ("x","y","z")):
        sx = float(data["x"].max()-data["x"].min())
        sy = float(data["y"].max()-data["y"].min())
        sz = float(data["z"].max()-data["z"].min())
        print(f"  Bounding vol : {sx*sy*sz:.1f} cubic units  ({sx:.1f} x {sy:.1f} x {sz:.1f})")

    # Color / normals
    has_color   = any(c in prop_names for c in ("red","r","diffuse_red"))
    has_normals = "nx" in prop_names
    has_opacity = "opacity" in prop_names
    print(f"  Has color    : {has_color}")
    print(f"  Has normals  : {has_normals}")
    print(f"  Is Gaussian splat: {has_opacity}")

    # Gaussian-specific stats
    if has_opacity:
        raw_op = data["opacity"].astype(np.float32)
        op = sigmoid(raw_op)
        print(f"\n  --- Gaussian Opacity ---")
        for t in (0.1, 0.3, 0.5, 0.9):
            pct = 100*np.mean(op > t)
            print(f"  > {t}: {pct:.1f}%  ({int(n*pct/100):,} Gaussians)")
        print(f"  mean={op.mean():.3f}  median={np.median(op):.3f}")
        ghost_pct = 100*np.mean(op < 0.1)
        print(f"  Ghost ratio (opacity<0.1): {ghost_pct:.1f}%")

    # Color stats (point cloud)
    if has_color:
        for ch in ("red","green","blue"):
            if ch in prop_names:
                v = data[ch]
                print(f"  {ch}: mean={v.mean():.0f}  min={v.min()}  max={v.max()}")

    # Point density estimate
    if all(a in prop_names for a in ("x","y","z")):
        vol = sx * sy * sz
        if vol > 0:
            density = n / vol
            print(f"  Point density: {density:.1f} pts/unit³")

# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  COMPARATIVE ANALYSIS — 3D Reconstruction Outputs")
print("="*55)

for name, path in FILES.items():
    analyze(name, path)

print(f"\n{'='*55}")
print("  SUMMARY COMPARISON")
print(f"{'='*55}")
