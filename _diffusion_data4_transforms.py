"""Convert INRIA gaussian-splatting cameras.json + femto_intrinsics.json
into nerfstudio transforms.json format so splatfacto can retrain on data4.

INRIA cameras.json entry format (per scene/dataset_readers.py):
  {image_name, width, height, R (world-to-camera, OpenCV convention),
   T (world-to-camera translation, OpenCV), K (3x3 intrinsics)}

nerfstudio transforms.json expects camera-to-world in OpenGL convention:
  c2w_gl = c2w_ocv with columns 1 and 2 negated (X, -Y, -Z)
"""
import json
import numpy as np
from pathlib import Path

PROJECT   = Path("C:/Users/mgallai/Projects/3d_automated/reconstruction_project")
CAMS_JSON = PROJECT / "output/clean_gs_data4/cameras.json"
INTR_JSON = PROJECT / "assets/data4/nerfstudio_data/femto_intrinsics.json"
OUT_TX    = PROJECT / "output/diffusion_prep/data4/transforms.json"

cams = json.loads(CAMS_JSON.read_text())
intr = json.loads(INTR_JSON.read_text())["color_intrinsics"]
print(f"loaded {len(cams)} camera entries; W={intr['width']} H={intr['height']}")

frames = []
for c in cams:
    R = np.array(c["R"], dtype=np.float64)  # (3,3) world-to-camera OpenCV
    T = np.array(c["T"], dtype=np.float64)  # (3,)  world-to-camera OpenCV
    w2c_ocv = np.eye(4)
    w2c_ocv[:3, :3] = R
    w2c_ocv[:3, 3]  = T
    c2w_ocv = np.linalg.inv(w2c_ocv)
    # OpenCV -> OpenGL: negate Y and Z columns of c2w rotation
    c2w_gl = c2w_ocv.copy()
    c2w_gl[:3, 1] *= -1
    c2w_gl[:3, 2] *= -1
    frames.append({
        "file_path": f"images/{c['image_name']}",
        "transform_matrix": c2w_gl.tolist(),
    })

transforms = {
    "camera_model": "OPENCV",
    "fl_x": intr["fx"],
    "fl_y": intr["fy"],
    "cx":   intr["cx"],
    "cy":   intr["cy"],
    "w":    intr["width"],
    "h":    intr["height"],
    "k1":   0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
    "frames": frames,
}
OUT_TX.parent.mkdir(parents=True, exist_ok=True)
OUT_TX.write_text(json.dumps(transforms, indent=2))
print(f"wrote {OUT_TX}  n_frames={len(frames)}")
