"""View source: load real RGB photos + COLMAP poses from a V32-pipeline run.

The original automated_trimming used synthetic pyrender views (6 views, plastic
shading). This module replaces that with the real input photos that already
exist in `nerfstudio_data/images/` + per-frame poses from `colmap/dense/
transforms.json`. Two material advantages:

  * SAM3 was trained on photos -> performs noticeably better than on renders
  * 100+ real views >> 6 synthetic views -> almost nothing stays occluded
  * No render-time, no lighting setup, no synthetic textures

The class returns a uniform `ViewInfo` object that the rest of the
segmenter consumes; you can swap in a synthetic-render source later without
touching downstream code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np


# nerfstudio transforms.json uses OpenGL camera convention (Y-up, -Z forward).
# COLMAP / OpenCV uses (Y-down, +Z forward). To bring a transforms.json c2w
# back to OpenCV convention, flip the Y and Z columns of the rotation:
#   c2w_cv = c2w_gl @ diag(1, -1, -1, 1)
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class ViewInfo:
    """One real-world view: an image path + its camera in OpenCV convention.

    All coordinates are in METRES (matching the scale-calibrated mesh).
    """
    name: str                # frame stem, e.g. "IMG_femto_0042"
    image_path: Path
    width: int
    height: int
    K: np.ndarray            # 3x3 intrinsics (px)
    w2c: np.ndarray          # 4x4 world-to-camera (OpenCV), translation in metres
    c2w: np.ndarray          # 4x4 camera-to-world (OpenCV), translation in metres

    @property
    def camera_position(self) -> np.ndarray:
        """Camera origin in world-space metres, shape (3,)."""
        return self.c2w[:3, 3]

    @property
    def camera_forward(self) -> np.ndarray:
        """Unit vector along camera +Z (OpenCV: into the scene)."""
        f = self.c2w[:3, 2]
        return f / (np.linalg.norm(f) + 1e-12)


class V32ViewSource:
    """Loads photo views from the V32-pipeline standard layout.

    Expects:
        <project_root>/nerfstudio_data/images/IMG_femto_NNNN.jpg
        <project_root>/colmap/dense/transforms.json  (nerfstudio format,
                                                     translations in COLMAP units)
        <project_root>/colmap/dense/tof_bounds.json  (provides scale factor)

    The transforms.json is in OpenGL/Y-up convention and translations are in
    COLMAP arbitrary units. We convert both at load time so callers receive
    OpenCV-convention metric poses ready to project against the
    scale-calibrated mesh.
    """

    def __init__(
        self,
        project_root: Path | str,
        transforms_json: Optional[Path | str] = None,
        bounds_json: Optional[Path | str] = None,
        images_root: Optional[Path | str] = None,
        scale_override: Optional[float] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.transforms_json = Path(transforms_json) if transforms_json \
            else self.project_root / "colmap" / "dense" / "transforms.json"
        self.bounds_json = Path(bounds_json) if bounds_json \
            else self.project_root / "colmap" / "dense" / "tof_bounds.json"
        self.images_root = Path(images_root) if images_root \
            else self.project_root / "colmap" / "dense"

        if not self.transforms_json.exists():
            raise FileNotFoundError(f"transforms.json not found: {self.transforms_json}")

        meta = json.loads(self.transforms_json.read_text())
        self._meta = meta

        # ---- Scale factor: divide COLMAP-unit translations to get metres ----
        if scale_override is not None:
            self.colmap_to_metric = 1.0 / float(scale_override)
        else:
            if not self.bounds_json.exists():
                # tof_bounds may be absent for legacy / non-Femto pipelines.
                # In that case assume poses are already metric.
                self.colmap_to_metric = 1.0
            else:
                b = json.loads(self.bounds_json.read_text())
                self.colmap_to_metric = 1.0 / float(b["scale_factor_da3_to_colmap"])

        # ---- Pre-parse all frames ----
        self._views: List[ViewInfo] = []
        self._skipped_paths: List[str] = []   # diagnostic — frames we couldn't load
        # Global intrinsics (transforms.json may put them at the top level)
        gl_w  = int(meta.get("w", 1920))
        gl_h  = int(meta.get("h", 1080))
        gl_fx = float(meta.get("fl_x", 1125.0))
        gl_fy = float(meta.get("fl_y", 1125.0))
        gl_cx = float(meta.get("cx", gl_w / 2))
        gl_cy = float(meta.get("cy", gl_h / 2))

        for frame in meta["frames"]:
            fp = Path(frame["file_path"])
            # The dataparser will look up <data_dir> / <file_path>. We want the
            # actual jpg on disk: <images_root> / <file_path>.
            img_path = (self.images_root / fp).resolve()
            if not img_path.exists():
                # Try alternate roots (some pipelines write file_path as just
                # `images/IMG_femto_NNNN.jpg`).
                alt = (self.project_root / "nerfstudio_data" / fp).resolve()
                if alt.exists():
                    img_path = alt
                else:
                    self._skipped_paths.append(str(fp))
                    continue

            # Per-frame overrides on intrinsics if present
            fx = float(frame.get("fl_x", gl_fx))
            fy = float(frame.get("fl_y", gl_fy))
            cx = float(frame.get("cx", gl_cx))
            cy = float(frame.get("cy", gl_cy))
            w  = int(frame.get("w", gl_w))
            h  = int(frame.get("h", gl_h))

            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float64)

            # c2w in OpenGL convention, translation in COLMAP units
            c2w_gl = np.array(frame["transform_matrix"], dtype=np.float64)
            # Convert OpenGL -> OpenCV
            c2w_cv = c2w_gl @ _GL_TO_CV
            # Rescale translation to metres
            c2w_cv[:3, 3] *= self.colmap_to_metric
            w2c_cv = np.linalg.inv(c2w_cv)

            name = img_path.stem
            self._views.append(ViewInfo(
                name=name, image_path=img_path,
                width=w, height=h,
                K=K, w2c=w2c_cv, c2w=c2w_cv,
            ))

        # Loudly report any missed frames so users don't silently lose data
        # (review bug H-7).
        n_frames = len(meta.get("frames", []))
        n_loaded = len(self._views)
        if self._skipped_paths:
            head = self._skipped_paths[:3]
            print(f"[V32ViewSource] WARNING: loaded {n_loaded}/{n_frames} frames; "
                  f"skipped {len(self._skipped_paths)} (image not found). "
                  f"First missing: {head}")

    def __len__(self) -> int:
        return len(self._views)

    def __iter__(self) -> Iterator[ViewInfo]:
        return iter(self._views)

    def all_views(self) -> List[ViewInfo]:
        return list(self._views)

    def sample(self, n: int, seed: int = 0) -> List[ViewInfo]:
        """Return n evenly-spaced views (deterministic). Useful for testing."""
        if n >= len(self._views):
            return list(self._views)
        idx = np.linspace(0, len(self._views) - 1, n, dtype=int)
        return [self._views[i] for i in idx]
