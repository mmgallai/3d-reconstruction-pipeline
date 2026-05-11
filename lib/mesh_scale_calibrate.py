"""
Apply the DA3↔COLMAP scale factor to a PLY mesh so its vertex coordinates
end up in metric (meters) instead of arbitrary COLMAP units.

The scale factor is read from <project_root>/colmap/dense/da3_bounds.json
(field `scale_factor_da3_to_colmap`). Vertices are multiplied by 1/scale.

UV coordinates and texture references are preserved as-is (scaling XYZ
does not affect UV [0,1] mapping).

Usage:
    python lib/mesh_scale_calibrate.py <input.ply> <output.ply> [bounds.json]
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_ply_header(data: bytes) -> tuple[int, list[str]]:
    """Return (header_byte_length, list_of_header_lines)."""
    end_marker = b"end_header\n"
    idx = data.find(end_marker)
    if idx < 0:
        raise ValueError("PLY: end_header not found")
    header_len = idx + len(end_marker)
    header = data[:header_len].decode("ascii", errors="replace")
    lines = header.splitlines()
    return header_len, lines


def parse_format(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("format "):
            return line.split()[1]
    return "ascii"


def main(in_path: Path, out_path: Path, bounds_path: Path) -> int:
    bounds = json.loads(bounds_path.read_text())
    scale = bounds["scale_factor_da3_to_colmap"]
    inv = 1.0 / scale
    print(f"  scale_factor_da3_to_colmap = {scale:.6f}")
    print(f"  applying inverse = {inv:.6f}")

    data = bytearray(in_path.read_bytes())
    header_len, lines = read_ply_header(bytes(data))
    fmt = parse_format(lines)
    if fmt != "binary_little_endian":
        print(f"  ERROR: unsupported format '{fmt}' (need binary_little_endian)", flush=True)
        return 2

    # Find the vertex element and its property layout.
    n_verts = None
    vertex_props: list[tuple[str, str]] = []  # (type, name)
    in_vertex = False
    for line in lines:
        if line.startswith("element vertex "):
            n_verts = int(line.split()[2])
            in_vertex = True
        elif line.startswith("element ") and in_vertex:
            in_vertex = False
        elif line.startswith("property ") and in_vertex:
            parts = line.split()
            if len(parts) == 3:
                vertex_props.append((parts[1], parts[2]))
            else:
                # property list -- not supported here (vertex props shouldn't be lists)
                print(f"  ERROR: unexpected list property in vertex element: {line}", flush=True)
                return 3

    if n_verts is None:
        print("  ERROR: no vertex element found", flush=True)
        return 4

    # Compute size of one vertex record.
    type_size = {
        "char": 1, "uchar": 1, "int8": 1, "uint8": 1,
        "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
        "int": 4, "uint": 4, "int32": 4, "uint32": 4, "float": 4, "float32": 4,
        "double": 8, "float64": 8,
    }
    fmt_chars = {
        "char": "b", "int8": "b",
        "uchar": "B", "uint8": "B",
        "short": "h", "int16": "h",
        "ushort": "H", "uint16": "H",
        "int": "i", "int32": "i",
        "uint": "I", "uint32": "I",
        "float": "f", "float32": "f",
        "double": "d", "float64": "d",
    }
    record_size = sum(type_size[t] for t, _ in vertex_props)
    print(f"  vertex record: {record_size} bytes, {n_verts:,} vertices, {len(vertex_props)} props")

    # Identify x,y,z slot indices and offsets within the record.
    offsets = {}
    off = 0
    for t, name in vertex_props:
        if name in ("x", "y", "z"):
            offsets[name] = (off, t)
        off += type_size[t]
    if not all(k in offsets for k in ("x", "y", "z")):
        print(f"  ERROR: x/y/z properties not all found in: {vertex_props}", flush=True)
        return 5

    # Detect float vs double for x,y,z (PLY allows both)
    valid_float_types = ("float", "float32", "double", "float64")
    for k in ("x", "y", "z"):
        _, t = offsets[k]
        if t not in valid_float_types:
            print(f"  ERROR: vertex.{k} is type {t!r} (expected float/double)", flush=True)
            return 6
    coord_type = offsets["x"][1]
    is_double = coord_type in ("double", "float64")
    np_dtype = np.float64 if is_double else np.float32
    coord_size = 8 if is_double else 4
    print(f"  vertex coord type: {coord_type} ({coord_size} bytes)")

    # Read pre-scaling bbox for the report.
    raw = np.frombuffer(bytes(data[header_len:header_len + n_verts * record_size]), dtype=np.uint8)
    raw = raw.reshape(n_verts, record_size)
    xs = np.frombuffer(raw[:, offsets["x"][0]:offsets["x"][0] + coord_size].tobytes(), dtype=np_dtype)
    ys = np.frombuffer(raw[:, offsets["y"][0]:offsets["y"][0] + coord_size].tobytes(), dtype=np_dtype)
    zs = np.frombuffer(raw[:, offsets["z"][0]:offsets["z"][0] + coord_size].tobytes(), dtype=np_dtype)
    bb_before = (xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min())
    diag_before = (bb_before[0]**2 + bb_before[1]**2 + bb_before[2]**2) ** 0.5
    print(f"  bbox BEFORE: ({bb_before[0]:.3f}, {bb_before[1]:.3f}, {bb_before[2]:.3f})  diag={diag_before:.3f} (COLMAP units)")

    # Scale x,y,z. Preserve original dtype so the byte layout stays identical.
    xs_new = (xs * inv).astype(np_dtype, copy=False)
    ys_new = (ys * inv).astype(np_dtype, copy=False)
    zs_new = (zs * inv).astype(np_dtype, copy=False)

    vertex_section_len = n_verts * record_size
    vertex_bytes = bytearray(data[header_len:header_len + vertex_section_len])
    vbuf = np.frombuffer(vertex_bytes, dtype=np.uint8).reshape(n_verts, record_size)
    vbuf = vbuf.copy()  # writable
    vbuf[:, offsets["x"][0]:offsets["x"][0] + coord_size] = np.frombuffer(xs_new.tobytes(), dtype=np.uint8).reshape(n_verts, coord_size)
    vbuf[:, offsets["y"][0]:offsets["y"][0] + coord_size] = np.frombuffer(ys_new.tobytes(), dtype=np.uint8).reshape(n_verts, coord_size)
    vbuf[:, offsets["z"][0]:offsets["z"][0] + coord_size] = np.frombuffer(zs_new.tobytes(), dtype=np.uint8).reshape(n_verts, coord_size)

    out = bytearray(data)
    out[header_len:header_len + vertex_section_len] = vbuf.tobytes()

    bb_after = (xs_new.max() - xs_new.min(), ys_new.max() - ys_new.min(), zs_new.max() - zs_new.min())
    diag_after = (bb_after[0]**2 + bb_after[1]**2 + bb_after[2]**2) ** 0.5
    print(f"  bbox AFTER : ({bb_after[0]:.3f}, {bb_after[1]:.3f}, {bb_after[2]:.3f})  diag={diag_after:.3f} (meters)")

    out_path.write_bytes(bytes(out))
    print(f"  saved: {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: mesh_scale_calibrate.py <input.ply> <output.ply> [bounds.json]", file=sys.stderr)
        sys.exit(1)
    in_p = Path(sys.argv[1])
    out_p = Path(sys.argv[2])
    bounds_p = Path(sys.argv[3]) if len(sys.argv) > 3 else (
        in_p.parent.parent.parent / "colmap" / "dense" / "da3_bounds.json"
    )
    if not bounds_p.exists():
        print(f"  ERROR: bounds file not found: {bounds_p}", file=sys.stderr)
        sys.exit(7)
    sys.exit(main(in_p, out_p, bounds_p))
