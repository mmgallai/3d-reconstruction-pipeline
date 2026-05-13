"""
V23 (Femto ToF) deliverable inspection.

Loads V23 splat + mesh, compares against V6 (canonical splat) and V17 (canonical
mesh). Scenes differ (V6/V17 = iPhone capture, V23 = Femto desk) so absolute
sizes aren't comparable — focus is on distribution shapes and ratios.
"""
from pathlib import Path
import numpy as np
import trimesh

ROOT = Path(r"C:\Users\mgallai\Projects\3d_automated\reconstruction_project")


# ── Splat parsing (binary PLY with float32 properties) ────────────────────────

def load_splat(path):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        data_start = f.tell()
    n = 0
    props = []
    for ln in header_lines:
        if ln.startswith("element vertex"):
            n = int(ln.split()[-1])
        elif ln.startswith("property float"):
            props.append(ln.split()[-1])
    dtype = np.dtype([(f"f{i}", np.float32) for i in range(len(props))])
    with open(path, "rb") as f:
        f.seek(data_start)
        arr = np.frombuffer(f.read(n * len(props) * 4), dtype=dtype)
    cols = np.stack([arr[f"f{i}"] for i in range(len(props))], axis=1)
    return n, {p: cols[:, i] for i, p in enumerate(props)}


def splat_stats(name, path):
    print(f"\n{'─'*78}\n{name}  ({path.name}, {path.stat().st_size/1_048_576:.1f} MB)")
    n, p = load_splat(path)
    x, y, z = p["x"], p["y"], p["z"]
    bb = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])

    # Opacity (sigmoid of logit)
    op = 1.0 / (1.0 + np.exp(-p["opacity"]))
    # Scale (exp of log-scale)
    scale_keys = sorted([k for k in p if k.startswith("scale_")])
    scales = np.stack([p[k] for k in scale_keys], axis=1)
    max_scale = np.exp(scales).max(axis=1)

    out5sigma = 0
    for ax in (x, y, z):
        mu, sd = ax.mean(), ax.std()
        out5sigma += int(np.sum(np.abs(ax - mu) > 5 * sd))

    print(f"  Gaussians         : {n:>14,}")
    print(f"  Properties        : {len(p):>14}")
    print(f"  BB diag (units)   : {np.linalg.norm(bb):>14.3f}")
    print(f"  Opacity mean      : {op.mean():>14.3f}   median={np.median(op):.3f}")
    print(f"  Opacity <0.10 (%) : {100*np.mean(op<0.10):>14.2f}  (floaters)")
    print(f"  Opacity >0.50 (%) : {100*np.mean(op>0.50):>14.2f}  (solid)")
    print(f"  Max-scale median  : {np.median(max_scale):>14.5f}")
    print(f"  Max-scale >1.0 (%): {100*np.mean(max_scale>1.0):>14.2f}  (large/blur)")
    print(f"  Max-scale <0.001  : {100*np.mean(max_scale<0.001):>14.2f}  (tiny/noise)")
    print(f"  Outliers >5σ      : {out5sigma:>14,}  ({100*out5sigma/(3*n):.2f}% avg per-axis)")

    return dict(
        name=name, gaussians=n, bb_diag=float(np.linalg.norm(bb)),
        opacity_mean=float(op.mean()),
        floater_pct=float(100 * np.mean(op < 0.10)),
        solid_pct=float(100 * np.mean(op > 0.50)),
        max_scale_median=float(np.median(max_scale)),
        large_pct=float(100 * np.mean(max_scale > 1.0)),
        outliers_pct=float(100 * out5sigma / (3 * n)),
        size_mb=path.stat().st_size / 1_048_576,
    )


# ── Mesh inspection via trimesh ───────────────────────────────────────────────

def mesh_stats(name, path):
    print(f"\n{'─'*78}\n{name}  ({path.name}, {path.stat().st_size/1_048_576:.1f} MB)")
    m = trimesh.load(str(path), force="mesh", process=False)
    v, f = m.vertices, m.faces
    bb = v.max(0) - v.min(0)
    edges_ct = np.unique(m.edges_sorted, axis=0, return_counts=True)[1]
    boundary = int((edges_ct == 1).sum())
    nm       = int((edges_ct > 2).sum())
    cc = trimesh.graph.connected_components(m.face_adjacency, min_len=1)
    largest = max(len(c) for c in cc)

    has_uv = hasattr(m.visual, "uv") and m.visual.uv is not None
    try:
        has_vc = (m.visual is not None and hasattr(m.visual, "vertex_colors")
                  and m.visual.vertex_colors is not None
                  and len(m.visual.vertex_colors) > 0)
    except Exception:
        has_vc = False

    print(f"  Vertices              : {len(v):>14,}")
    print(f"  Faces                 : {len(f):>14,}")
    print(f"  BB diagonal           : {np.linalg.norm(bb):>14.3f}")
    print(f"  Surface area          : {m.area:>14.2f}")
    print(f"  Watertight            : {str(m.is_watertight):>14}")
    print(f"  Boundary (open) edges : {boundary:>14,}")
    print(f"  Non-manifold edges    : {nm:>14,}")
    print(f"  Connected components  : {len(cc):>14,}  (largest={largest:,}={largest/len(f)*100:.2f}%)")
    print(f"  UV mapped / Vert col  : {str(has_uv):>14} / {has_vc}")
    return dict(
        name=name, verts=len(v), faces=len(f),
        bb_diag=float(np.linalg.norm(bb)),
        area=float(m.area),
        watertight=bool(m.is_watertight),
        boundary=boundary, nm=nm,
        components=len(cc),
        largest_pct=float(largest / len(f) * 100),
        size_mb=path.stat().st_size / 1_048_576,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*78}\n SPLAT COMPARISON: V6 (DA3-init) vs V23 (Femto-ToF-init)\n{'='*78}")
    splat_targets = [
        ("V23 pruned",  ROOT / "output" / "splat_v23_noinit_pruned.ply"),
        ("V24 pruned",  ROOT / "output" / "splat_v24_noinit_pruned.ply"),
        ("V25 pruned",  ROOT / "output" / "splat_v25_noinit_pruned.ply"),
        ("V23 raw",     ROOT / "output" / "splat_v23_noinit.ply"),
        ("V24 raw",     ROOT / "output" / "splat_v24_noinit.ply"),
        ("V25 raw",     ROOT / "output" / "splat_v25_noinit.ply"),
    ]
    s_results = []
    for label, p in splat_targets:
        if not p.exists():
            print(f"\n  {label}: NOT FOUND ({p})")
            continue
        s_results.append(splat_stats(label, p))

    # Splat summary
    print(f"\n\n{'SPLAT SUMMARY':^78}\n{'='*78}")
    print(f"  {'Run':<12} {'Gauss':>12} {'Size MB':>9} {'Op mean':>9} "
          f"{'Float %':>8} {'Solid %':>8} {'Big %':>7} {'5σ %':>6}")
    for r in s_results:
        print(f"  {r['name']:<12} {r['gaussians']:>12,} {r['size_mb']:>9.1f} "
              f"{r['opacity_mean']:>9.3f} {r['floater_pct']:>8.2f} "
              f"{r['solid_pct']:>8.2f} {r['large_pct']:>7.2f} {r['outliers_pct']:>6.2f}")

    print(f"\n\n{'='*78}\n MESH COMPARISON: V17 vs V23 (high/mid/low LOD)\n{'='*78}")
    mesh_targets = [
        ("V23 HIGH",  ROOT / "output" / "mesh_v23"      / "mesh_v23_openmvs.ply"),
        ("V25 HIGH",  ROOT / "output" / "mesh_v25"      / "mesh_v25_openmvs.ply"),
        ("V26 TSDF",  ROOT / "output" / "mesh_v26_tsdf" / "mesh_v26_tsdf.ply"),
        ("V23 MID",   ROOT / "output" / "mesh_v23"      / "mesh_v23_openmvs_mid.ply"),
        ("V25 MID",   ROOT / "output" / "mesh_v25"      / "mesh_v25_openmvs_mid.ply"),
        ("V23 LOW",   ROOT / "output" / "mesh_v23"      / "mesh_v23_openmvs_low.ply"),
        ("V25 LOW",   ROOT / "output" / "mesh_v25"      / "mesh_v25_openmvs_low.ply"),
    ]
    m_results = []
    for label, p in mesh_targets:
        if not p.exists():
            print(f"\n  {label}: NOT FOUND ({p})")
            continue
        m_results.append(mesh_stats(label, p))

    # Mesh summary
    print(f"\n\n{'MESH SUMMARY':^78}\n{'='*78}")
    print(f"  {'Mesh':<11} {'Verts':>10} {'Faces':>10} {'Size MB':>8} "
          f"{'Holes':>8} {'NM':>5} {'Comps':>6} {'Largest%':>9} {'Watertight':>11}")
    for r in m_results:
        print(f"  {r['name']:<11} {r['verts']:>10,} {r['faces']:>10,} "
              f"{r['size_mb']:>8.1f} {r['boundary']:>8,} {r['nm']:>5,} "
              f"{r['components']:>6,} {r['largest_pct']:>8.2f}% {str(r['watertight']):>11}")


if __name__ == "__main__":
    main()
