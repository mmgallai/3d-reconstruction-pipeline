"""
Capture an RGB + depth dataset from the Orbbec Femto Mega for the splat / mesh
pipeline.

Outputs (matches the existing pipeline's expected layout):
  nerfstudio_data/images/IMG_femto_NNNN.jpg             RGB, JPEG
  nerfstudio_data/depths_femto/IMG_femto_NNNN.npy       float32 depth in meters,
                                                          D2C-aligned to RGB pixels
  nerfstudio_data/depths_femto/IMG_femto_NNNN_conf.npy  uint8 mask: 1=valid, 0=invalid
  nerfstudio_data/depths_femto/IMG_femto_NNNN_ir.png    uint16 IR/intensity image
                                                          (optional; saved if IR stream available)
  nerfstudio_data/femto_intrinsics.json                 written once at session end

Live preview shows:
  - Current RGB frame, scaled to fit window
  - Coloured border: GREEN = good frame, RED = motion blur / out-of-range
  - HUD: frames captured, in-range %, blur metric, FPS, AND:
      * center-pixel distance (so you stay in the 0.5–3m sweet spot)
      * range hint colour: green ok, yellow too-far, red too-close
  - Optional depth overlay (D toggle), now colour-coded by range zone
  - Optional sparse-flow motion vectors (M toggle), used by motion-auto

Hotkeys / buttons:
  SPACE         save current frame (manual saves red frames too, with a warning)
  A             toggle auto-capture by time (starts ON; 1 frame every AUTO_INTERVAL_SEC sec)
  M             toggle motion-triggered auto-capture (1 frame per ~AUTO_MOTION_CM cm)
  D             toggle depth preview overlay (range-zone colour-coded)
  C             toggle coverage minimap (yaw x pitch grid, where you've pointed)
  S             toggle audio cues (shutter on save / warning on out-of-range)
  ESC / Ctrl+C  finish session, write intrinsics, exit

Usage:
  conda activate da3
  python capture_femto.py
  python capture_femto.py --output-dir custom_path --auto-interval 0.3 --max-frames 200
  python capture_femto.py --motion-cm 4.0       # motion-auto: one frame every 4 cm
  python capture_femto.py --no-ir               # skip IR stream

Camera setup notes:
  - For our 0.25–5.46m ToF range, position yourself so most of the room
    is within reach. For larger rooms, do multiple "stations".
  - Direct sun on a wall returns garbage depth — close blinds or shoot diffuse.
  - Sweep slowly. Loop back to your start point to help COLMAP SfM close the loop.
  - Use motion-auto (M) for the most consistent overlap — captures by *distance
    moved* (~5 cm default) rather than wall-clock time, so a slow start and
    fast end both produce evenly-spaced frames.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python missing. Install: pip install opencv-python")
    sys.exit(1)

import pyorbbecsdk as ob


# Capture settings — tuned for indoor room sweep
COLOR_W, COLOR_H = 1920, 1080      # RGB stream resolution (request from camera)
COLOR_FPS         = 30
DEPTH_W, DEPTH_H = 640, 576        # NFOV depth — 30 fps + 75°×65° matches RGB FOV better than WFOV
DEPTH_FPS         = 30
IR_W, IR_H        = 1024, 1024     # Femto Mega IR sensor (matches depth FOV)
IR_FPS            = 30
JPEG_QUALITY      = 92             # match heic_converter default
SAVE_W, SAVE_H    = 1512, 2016     # match existing pipeline's images_2 dimensions (portrait)
DEFAULT_AUTO_INTERVAL_SEC = 0.5    # 2 fps when time-auto-capture is on
DEFAULT_AUTO_MOTION_CM    = 5.0    # ~5 cm of camera motion per frame in motion-auto
BLUR_THRESH       = 18.0           # Laplacian variance below this = motion blur
DEPTH_MIN_M       = 0.30           # mask depths closer than this (Femto spec: 0.25)
DEPTH_MAX_M       = 5.30           # mask depths beyond this (Femto spec: 5.46)
DEPTH_CONF_TH     = 0              # Confidence threshold (sensor-dep; 0 = take everything valid)
# Sweet-spot range for capture quality — outside this, results degrade fast
SWEET_NEAR_M      = 0.50           # below this depth, ToF starts to noise up
SWEET_FAR_M       = 3.00           # beyond this, IR return is weak / depth gets sparse


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="nerfstudio_data",
                   help="Base directory for the capture session "
                        "(default: nerfstudio_data/, matches pipeline expectations)")
    p.add_argument("--auto-interval", type=float, default=DEFAULT_AUTO_INTERVAL_SEC,
                   help="Seconds between time-based auto-captures (default 0.5)")
    p.add_argument("--motion-cm", type=float, default=DEFAULT_AUTO_MOTION_CM,
                   help="Camera motion (cm) between motion-triggered auto-captures (default 5)")
    p.add_argument("--max-frames", type=int, default=400,
                   help="Stop auto-capture after this many frames (default 400)")
    p.add_argument("--no-preview", action="store_true",
                   help="Headless mode (no OpenCV window — for remote sessions)")
    p.add_argument("--manual-start", action="store_true",
                   help="Start with auto-capture OFF; press A or M to begin capturing")
    p.add_argument("--no-ir", action="store_true",
                   help="Skip the IR stream (saves a bit of bandwidth + storage)")
    p.add_argument("--no-audio", action="store_true",
                   help="Disable audio cues (beeps for save / out-of-range warnings)")
    return p.parse_args()


def _blur_metric(gray: np.ndarray) -> float:
    """Variance of Laplacian — higher = sharper. <50 = motion-blurred."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _color_frame_to_bgr(frame) -> np.ndarray:
    """Convert an Orbbec color frame to a BGR numpy image regardless of source format."""
    fmt = frame.get_format()
    w = frame.get_width()
    h = frame.get_height()
    data = np.asarray(frame.get_data())
    if fmt == ob.OBFormat.MJPG:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == ob.OBFormat.NV12:
        yuv = data.reshape(h * 3 // 2, w)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    elif fmt == ob.OBFormat.RGB:
        bgr = cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    elif fmt == ob.OBFormat.BGR:
        bgr = data.reshape(h, w, 3)
    elif fmt == ob.OBFormat.YUYV:
        bgr = cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    else:
        raise RuntimeError(f"unhandled color format {fmt}")
    return bgr


def _depth_frame_to_meters(frame) -> np.ndarray:
    """Orbbec depth is uint16 in millimeters by default."""
    w = frame.get_width()
    h = frame.get_height()
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(h, w)
    try:
        scale = float(frame.get_depth_scale())  # mm per raw unit (usually 1.0)
    except AttributeError:
        scale = 1.0
    return raw.astype(np.float32) * (scale / 1000.0)


def _ir_frame_to_u16(frame) -> Optional[np.ndarray]:
    """Orbbec IR frame -> uint16 (H, W). Returns None if shape is unexpected."""
    if frame is None:
        return None
    w = frame.get_width()
    h = frame.get_height()
    try:
        raw = np.frombuffer(frame.get_data(), dtype=np.uint16)
        if raw.size == h * w:
            return raw.reshape(h, w)
        raw = np.frombuffer(frame.get_data(), dtype=np.uint8)
        if raw.size == h * w:
            return raw.reshape(h, w).astype(np.uint16) * 257  # lift to uint16
    except Exception:
        return None
    return None


def _intrinsics_to_dict(intr) -> dict:
    return {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.cx),
        "cy": float(intr.cy),
        "width": int(intr.width),
        "height": int(intr.height),
    }


def _extrinsics_to_dict(extr) -> dict:
    """Orbbec OBExtrinsic: `.rot` is a 3x3 numpy ndarray, `.transform` is a
    3-element translation vector. After D2C HW alignment both are identity/zero."""
    rot = np.asarray(extr.rot).tolist() if hasattr(extr, "rot") else None
    # field is `transform`, not `trans`
    trans = np.asarray(extr.transform).tolist() if hasattr(extr, "transform") else None
    return {"rotation_row_major_3x3": rot, "translation_xyz_m": trans}


def _imwrite_or_raise(path: Path, image: np.ndarray, params=None) -> None:
    if params is None:
        ok = cv2.imwrite(str(path), image)
    else:
        ok = cv2.imwrite(str(path), image, params)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")


def setup_pipeline(enable_ir: bool = True) -> tuple[ob.Pipeline, ob.AlignFilter, bool]:
    """Return (pipeline, align_filter, ir_enabled). ir_enabled may be False
    if the camera lacks an IR sensor or the requested profile is unavailable."""
    pipeline = ob.Pipeline()
    config = ob.Config()

    # ── Color stream ─────────────────────────────────────────────────────────
    color_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    try:
        color_profile = color_profiles.get_video_stream_profile(
            COLOR_W, COLOR_H, ob.OBFormat.MJPG, COLOR_FPS
        )
    except Exception:
        color_profile = color_profiles.get_default_video_stream_profile()
        print(f"  WARN: requested {COLOR_W}x{COLOR_H} MJPG@{COLOR_FPS} unavailable, "
              f"using default {color_profile.get_width()}x{color_profile.get_height()} "
              f"{color_profile.get_format()}@{color_profile.get_fps()}")
    config.enable_stream(color_profile)

    # ── Depth stream ─────────────────────────────────────────────────────────
    depth_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
    try:
        depth_profile = depth_profiles.get_video_stream_profile(
            DEPTH_W, DEPTH_H, ob.OBFormat.Y16, DEPTH_FPS
        )
    except Exception:
        depth_profile = depth_profiles.get_default_video_stream_profile()
        print(f"  WARN: requested {DEPTH_W}x{DEPTH_H} Y16@{DEPTH_FPS} unavailable, "
              f"using default {depth_profile.get_width()}x{depth_profile.get_height()} "
              f"{depth_profile.get_format()}@{depth_profile.get_fps()}")
    config.enable_stream(depth_profile)

    # ── IR stream (optional) ─────────────────────────────────────────────────
    ir_enabled = False
    if enable_ir:
        try:
            ir_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.IR_SENSOR)
            try:
                ir_profile = ir_profiles.get_video_stream_profile(
                    IR_W, IR_H, ob.OBFormat.Y16, IR_FPS
                )
            except Exception:
                ir_profile = ir_profiles.get_default_video_stream_profile()
                print(f"  WARN: requested IR {IR_W}x{IR_H} Y16@{IR_FPS} unavailable, "
                      f"using default {ir_profile.get_width()}x{ir_profile.get_height()} "
                      f"{ir_profile.get_format()}@{ir_profile.get_fps()}")
            config.enable_stream(ir_profile)
            ir_enabled = True
        except Exception as e:
            print(f"  INFO: IR stream unavailable ({e}); continuing without IR")

    # D2C alignment: depth aligned to color image space
    config.set_align_mode(ob.OBAlignMode.HW_MODE)

    pipeline.start(config)
    align_filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
    return pipeline, align_filter, ir_enabled


def main() -> int:
    args = _parse_args()

    out_root = Path(args.output_dir).resolve()
    images_dir = out_root / "images"
    depths_dir = out_root / "depths_femto"
    images_dir.mkdir(parents=True, exist_ok=True)
    depths_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Femto Mega capture ===")
    print(f"  RGB → {images_dir}")
    print(f"  Depth → {depths_dir}")

    pipeline, align_filter, ir_enabled = setup_pipeline(enable_ir=not args.no_ir)
    if ir_enabled:
        print(f"  IR stream enabled (saved as *_ir.png)")
    print(f"  Pipeline started. Time-auto is ON. Click buttons or use SPACE/A/M/D/C/S/ESC. Ctrl+C finishes safely.")

    # Audio cues — winsound on Windows, fall back to terminal bell.
    audio_on = not args.no_audio
    try:
        import winsound  # type: ignore
        def _beep(freq=1200, dur_ms=80):
            if audio_on:
                try: winsound.Beep(freq, dur_ms)
                except Exception: pass
        def _shutter_sound():
            """Two-tone 'shutter click' for save events. Loud + distinctive,
            fires regardless of SPACE / time-auto / motion-auto."""
            if not audio_on:
                return
            try:
                winsound.Beep(1800, 50)
                winsound.Beep(1100, 60)
            except Exception:
                pass
    except ImportError:
        def _beep(freq=1200, dur_ms=80):
            if audio_on:
                sys.stdout.write("\a"); sys.stdout.flush()
        def _shutter_sound():
            if audio_on:
                sys.stdout.write("\a\a"); sys.stdout.flush()

    saved_count = 0
    red_manual_count = 0
    last_auto_t = 0.0
    auto_mode   = not args.manual_start    # time-based
    motion_mode = False    # motion-based (toggle M)
    show_depth  = False
    last_intrinsics = None
    fps_t0, fps_n = time.time(), 0
    requested_quit = False
    requested_save = False
    last_warn_t = 0.0

    # Motion-trigger state: ORB feature tracking + center-depth conversion
    # so "px displacement" converts to "approx metres displacement".
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=4)
    prev_gray_small = None
    prev_kp = None
    prev_desc = None
    motion_cm_accum = 0.0     # accumulated camera motion since last save (cm)
    motion_target_cm = float(args.motion_cm)

    # Coverage minimap state: integrate pixel-displacement -> yaw/pitch (radians)
    # using fx/fy and small-angle approx. Drifts over time, but for a few-minute
    # sweep it's a useful "where have I pointed the camera" overview.
    show_coverage = False                   # toggled by C
    COV_YAW_BINS, COV_PITCH_BINS = 24, 8    # 15 deg yaw, ~22 deg pitch
    cov_yaw_rad   = 0.0                     # cumulative orientation, 0 at start
    cov_pitch_rad = 0.0
    cov_hits: dict[tuple[int, int], int] = {}    # bin -> capture count

    # Mouse-clickable button rectangles. Recomputed each frame so they scale
    # with the preview, but stored at module-level for the callback closure.
    BUTTON_HEIGHT = 60
    button_rects: dict[str, tuple] = {}  # label -> (x1, y1, x2, y2)

    def on_mouse(event, x, y, flags, param):
        nonlocal saved_count, auto_mode, motion_mode, show_depth, show_coverage
        nonlocal audio_on, requested_quit, requested_save
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for label, (x1, y1, x2, y2) in button_rects.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if label == "SAVE":
                    requested_save = True
                elif label == "TIME":
                    auto_mode = not auto_mode
                    if auto_mode and motion_mode:
                        motion_mode = False   # exclusive
                elif label == "MOTION":
                    motion_mode = not motion_mode
                    if motion_mode and auto_mode:
                        auto_mode = False     # exclusive
                elif label == "DEPTH":
                    show_depth = not show_depth
                elif label == "COVERAGE":
                    show_coverage = not show_coverage
                elif label == "SOUND":
                    audio_on = not audio_on
                elif label == "FINISH":
                    requested_quit = True
                return

    window_name = "Femto capture (click buttons or SPACE/A/M/D/C/S/ESC)"
    if not args.no_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            aligned = align_filter.process(frames)
            if aligned is None:
                continue
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if cf is None or df is None:
                continue
            # IR frame is not in the aligned set (alignment is depth->color only);
            # pull it from the raw frame-set.
            ir_u16 = None
            if ir_enabled:
                try:
                    ir_frame = frames.get_ir_frame()
                    ir_u16 = _ir_frame_to_u16(ir_frame)
                except Exception:
                    ir_u16 = None

            bgr = _color_frame_to_bgr(cf)
            depth_m = _depth_frame_to_meters(df)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            blur = _blur_metric(gray)

            # Validity mask
            valid = (depth_m > DEPTH_MIN_M) & (depth_m < DEPTH_MAX_M)
            in_range_pct = valid.mean() * 100

            # Center-pixel distance (for sweet-spot HUD + motion-cm conversion).
            ch, cw = depth_m.shape[0] // 2, depth_m.shape[1] // 2
            patch = depth_m[max(0, ch-5):ch+5, max(0, cw-5):cw+5]
            patch_valid = patch[(patch > DEPTH_MIN_M) & (patch < DEPTH_MAX_M)]
            center_depth = float(np.median(patch_valid)) if patch_valid.size >= 10 else 0.0
            # Range zone for HUD colour
            if center_depth == 0.0:
                range_zone = "void"
                range_color = (120, 120, 120)
            elif center_depth < SWEET_NEAR_M:
                range_zone = "close"
                range_color = (0, 0, 220)
            elif center_depth > SWEET_FAR_M:
                range_zone = "far"
                range_color = (0, 200, 220)
            else:
                range_zone = "ok"
                range_color = (0, 200, 0)

            good_frame = (blur >= BLUR_THRESH) and (in_range_pct >= 6.0)
            border = (0, 220, 0) if good_frame else (0, 0, 220)

            # Motion estimate: track ORB features on a downsampled gray + median
            # px displacement. Convert px -> metres via fx + centre depth, and
            # also (separately, by axis) into yaw/pitch radians for the
            # coverage minimap.
            gray_small = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4))
            kp = orb.detect(gray_small, None)
            kp, desc = orb.compute(gray_small, kp)
            motion_cm_this_frame = 0.0
            if prev_desc is not None and desc is not None and len(prev_kp) >= 10 and len(kp) >= 10:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(prev_desc, desc)
                if len(matches) >= 10:
                    dxs, dys = [], []
                    for m in matches:
                        pp = prev_kp[m.queryIdx].pt
                        qp = kp[m.trainIdx].pt
                        dxs.append(qp[0] - pp[0])
                        dys.append(qp[1] - pp[1])
                    dx_med = float(np.median(dxs)) * 4.0   # un-do 4x downsample
                    dy_med = float(np.median(dys)) * 4.0
                    px_med = (dx_med ** 2 + dy_med ** 2) ** 0.5
                    # Convert pixel motion to metres: use color intrinsics fx + center depth
                    fx = 1100.0
                    fy = 1100.0
                    if last_intrinsics is not None:
                        fx = float(last_intrinsics["color_intrinsics"]["fx"])
                        fy = float(last_intrinsics["color_intrinsics"]["fy"])
                    if center_depth > 0:
                        motion_m = (px_med / fx) * center_depth
                        motion_cm_this_frame = motion_m * 100.0
                        motion_cm_accum += motion_cm_this_frame
                    # Coverage: features shift LEFT (-dx) when camera pans RIGHT.
                    # Small-angle approx: dyaw ≈ -dx_px / fx, dpitch ≈ +dy_px / fy.
                    # (dy positive when camera tilts DOWN; here we use +dy so
                    # tilting down -> pitch increases positive == "below horizon")
                    cov_yaw_rad   += -dx_med / fx
                    cov_pitch_rad +=  dy_med / fy
            prev_kp, prev_desc = kp, desc

            # Capture if asked
            now = time.time()
            capture_reason: Optional[str] = None
            if auto_mode and (now - last_auto_t) >= args.auto_interval and good_frame:
                capture_reason = "time-auto"
                last_auto_t = now
            if motion_mode and motion_cm_accum >= motion_target_cm and good_frame:
                capture_reason = "motion-auto"
                motion_cm_accum = 0.0

            # Out-of-range audio warning (rate-limit to 2 sec)
            if range_zone in ("close", "far") and (now - last_warn_t) > 2.0:
                _beep(freq=600 if range_zone == "close" else 900, dur_ms=70)
                last_warn_t = now

            # Cache intrinsics from any frame (they don't change run-to-run).
            # In pyorbbecsdk, intrinsics/extrinsics live on the StreamProfile,
            # not on the Frame.
            if last_intrinsics is None:
                color_profile = cf.get_stream_profile()
                depth_profile = df.get_stream_profile()
                color_intr = color_profile.get_intrinsic()
                depth_intr = depth_profile.get_intrinsic()
                try:
                    extr = color_profile.get_extrinsic_to(depth_profile)
                    extr_dict = _extrinsics_to_dict(extr)
                except Exception as _e:
                    extr_dict = {"error": str(_e)}
                # depth_scale may live on DepthFrame; fall back to 1.0
                try:
                    depth_scale = float(df.get_depth_scale())
                except AttributeError:
                    depth_scale = 1.0
                last_intrinsics = {
                    "color_intrinsics": _intrinsics_to_dict(color_intr),
                    "depth_intrinsics": _intrinsics_to_dict(depth_intr),
                    "color_to_depth_extrinsic": extr_dict,
                    "color_format_requested": str(cf.get_format()),
                    "depth_format_requested": str(df.get_format()),
                    "depth_scale_mm_per_unit": depth_scale,
                    "depth_units": "meters (float32)",
                    "captured_with": "pyorbbecsdk",
                    "session_started_iso": datetime.utcnow().isoformat() + "Z",
                }

            # Build preview
            if not args.no_preview:
                preview = bgr.copy()
                if show_depth:
                    # Range-zone-coded overlay: red <SWEET_NEAR, green sweet-spot,
                    # yellow >SWEET_FAR, gray invalid. Clearer than rainbow.
                    d_vis = np.zeros((depth_m.shape[0], depth_m.shape[1], 3), dtype=np.uint8)
                    mask_close = (depth_m > DEPTH_MIN_M) & (depth_m < SWEET_NEAR_M)
                    mask_ok    = (depth_m >= SWEET_NEAR_M) & (depth_m <= SWEET_FAR_M)
                    mask_far   = (depth_m > SWEET_FAR_M) & (depth_m < DEPTH_MAX_M)
                    d_vis[mask_close] = (0, 0, 220)    # BGR: red
                    d_vis[mask_ok]    = (0, 220, 0)    # green
                    d_vis[mask_far]   = (0, 200, 220)  # yellow
                    d_vis_res = cv2.resize(d_vis, (preview.shape[1], preview.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                    preview = cv2.addWeighted(preview, 0.55, d_vis_res, 0.45, 0)
                fps_n += 1
                fps = fps_n / (now - fps_t0) if now > fps_t0 else 0

                # Top status line — frame quality numbers
                top_text = (f"Saved: {saved_count}  |  in-range: {in_range_pct:5.1f}%  "
                            f"|  blur: {blur:.0f}  |  fps: {fps:.1f}")
                cv2.rectangle(preview, (0, 0), (preview.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(preview, top_text, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Second HUD line — center depth + motion accumulator
                center_text = (f"center: {center_depth:.2f}m [{range_zone:5s}]  "
                               f"|  motion since save: {motion_cm_accum:4.1f}/{motion_target_cm:.1f} cm")
                cv2.rectangle(preview, (0, 40), (preview.shape[1], 76), (0, 0, 0), -1)
                cv2.putText(preview, center_text, (10, 66),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, range_color, 2, cv2.LINE_AA)

                # Center crosshair (helps point the camera in motion-auto mode)
                Hh, Wh = preview.shape[:2]
                cv2.drawMarker(preview, (Wh // 2, Hh // 2), range_color,
                               markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

                # Coverage minimap (C toggle): yaw x pitch grid in top-right,
                # showing which directions have been captured. Like Polycam's
                # coverage overlay, but estimated from ORB-feature integration
                # (no IMU), so it drifts on long sweeps - still useful as a
                # "where have I pointed the camera" sanity check.
                if show_coverage:
                    pad_r = 12
                    map_w, map_h = 280, 100      # minimap pixel size
                    map_x0 = preview.shape[1] - map_w - pad_r
                    map_y0 = 90
                    cell_w = map_w // COV_YAW_BINS
                    cell_h = map_h // COV_PITCH_BINS
                    # Background panel
                    cv2.rectangle(preview, (map_x0 - 4, map_y0 - 22),
                                  (map_x0 + map_w + 4, map_y0 + map_h + 4),
                                  (20, 20, 20), -1)
                    pct = 100.0 * len(cov_hits) / (COV_YAW_BINS * COV_PITCH_BINS)
                    cv2.putText(preview, f"coverage: {len(cov_hits)} cells ({pct:.0f}%)",
                                (map_x0, map_y0 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                    # Paint cells
                    max_count = max(cov_hits.values()) if cov_hits else 1
                    for y in range(COV_PITCH_BINS):
                        for x in range(COV_YAW_BINS):
                            cx1 = map_x0 + x * cell_w
                            cy1 = map_y0 + y * cell_h
                            cx2 = cx1 + cell_w - 1
                            cy2 = cy1 + cell_h - 1
                            n = cov_hits.get((x, y), 0)
                            if n == 0:
                                cell_color = (40, 40, 100)        # dark red = uncovered
                            else:
                                # green intensity scales with frame count, capped
                                intensity = min(255, 100 + int(155 * n / max_count))
                                cell_color = (0, intensity, 0)
                            cv2.rectangle(preview, (cx1, cy1), (cx2, cy2), cell_color, -1)
                    # Grid lines
                    for x in range(COV_YAW_BINS + 1):
                        gx = map_x0 + x * cell_w
                        cv2.line(preview, (gx, map_y0), (gx, map_y0 + map_h), (80, 80, 80), 1)
                    for y in range(COV_PITCH_BINS + 1):
                        gy = map_y0 + y * cell_h
                        cv2.line(preview, (map_x0, gy), (map_x0 + map_w, gy), (80, 80, 80), 1)
                    # Mark current orientation with a yellow dot
                    import math as _m
                    yaw_norm = ((cov_yaw_rad / (2 * _m.pi)) + 0.5) % 1.0
                    cur_x_bin  = int(yaw_norm * COV_YAW_BINS) % COV_YAW_BINS
                    pitch_norm = max(0.0, min(0.9999, (cov_pitch_rad / _m.pi) + 0.5))
                    cur_y_bin  = int(pitch_norm * COV_PITCH_BINS)
                    dot_x = map_x0 + cur_x_bin * cell_w + cell_w // 2
                    dot_y = map_y0 + cur_y_bin * cell_h + cell_h // 2
                    cv2.circle(preview, (dot_x, dot_y), 4, (0, 255, 255), -1)
                    cv2.circle(preview, (dot_x, dot_y), 5, (0, 0, 0), 1)

                # Bottom button bar
                H, W = preview.shape[:2]
                bar_h = BUTTON_HEIGHT
                cv2.rectangle(preview, (0, H - bar_h), (W, H), (30, 30, 30), -1)
                btn_defs = [
                    ("SAVE",     "[SPACE] Save",
                     (0, 180, 0) if good_frame else (0, 140, 220)),
                    ("TIME",     f"[A] Time-auto: {'ON' if auto_mode else 'OFF'}",
                     (0, 180, 0) if auto_mode else (80, 80, 200)),
                    ("MOTION",   f"[M] Motion-auto: {'ON' if motion_mode else 'OFF'}",
                     (0, 180, 0) if motion_mode else (80, 80, 200)),
                    ("DEPTH",    f"[D] Depth: {'ON' if show_depth else 'OFF'}",
                     (0, 180, 0) if show_depth else (80, 80, 200)),
                    ("COVERAGE", f"[C] Coverage: {'ON' if show_coverage else 'OFF'}",
                     (0, 180, 0) if show_coverage else (80, 80, 200)),
                    ("SOUND",    f"[S] Sound: {'ON' if audio_on else 'OFF'}",
                     (0, 180, 0) if audio_on else (80, 80, 200)),
                    ("FINISH",   "[ESC] Finish",
                     (40, 40, 200)),
                ]
                slot_w = W // len(btn_defs)
                pad = 8
                button_rects.clear()
                for i, (label, text, color) in enumerate(btn_defs):
                    x1 = i * slot_w + pad
                    x2 = (i + 1) * slot_w - pad
                    y1 = H - bar_h + pad
                    y2 = H - pad
                    button_rects[label] = (x1, y1, x2, y2)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), color, -1)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    # Center text in button
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    tx = x1 + (x2 - x1 - tw) // 2
                    ty = y1 + (y2 - y1 + th) // 2
                    cv2.putText(preview, text, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # Outer status border (still indicates frame quality)
                cv2.rectangle(preview, (0, 76), (W - 1, H - bar_h - 1), border, 6)
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or requested_quit:  # ESC or Finish button
                    break
                elif key == 32 or requested_save:  # SPACE or Save button
                    capture_reason = "manual"
                    requested_save = False
                elif key in (ord("a"), ord("A")):
                    auto_mode = not auto_mode
                    if auto_mode and motion_mode: motion_mode = False
                    print(f"  time-auto-capture: {'ON' if auto_mode else 'OFF'}")
                elif key in (ord("m"), ord("M")):
                    motion_mode = not motion_mode
                    if motion_mode and auto_mode: auto_mode = False
                    print(f"  motion-auto-capture: {'ON' if motion_mode else 'OFF'} "
                          f"(every {motion_target_cm:.1f} cm)")
                elif key in (ord("d"), ord("D")):
                    show_depth = not show_depth
                elif key in (ord("c"), ord("C")):
                    show_coverage = not show_coverage
                    print(f"  coverage minimap: {'ON' if show_coverage else 'OFF'}")
                elif key in (ord("s"), ord("S")):
                    audio_on = not audio_on
                    print(f"  audio cues: {'ON' if audio_on else 'OFF'}")

            if capture_reason is not None:
                manual_override = capture_reason == "manual"
                if not good_frame and not manual_override:
                    print(f"  skipped {capture_reason} capture: red frame "
                          f"(in-range {in_range_pct:.1f}%, blur {blur:.0f}, "
                          f"center {center_depth:.2f}m)")
                    continue

                idx = saved_count + 1
                stem = f"IMG_femto_{idx:04d}"
                _imwrite_or_raise(images_dir / f"{stem}.jpg", bgr,
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                np.save(str(depths_dir / f"{stem}.npy"), depth_m.astype(np.float32))
                np.save(str(depths_dir / f"{stem}_conf.npy"), valid.astype(np.uint8))
                if ir_u16 is not None:
                    # IR is at its native resolution (e.g. 1024x1024) — saved as
                    # 16-bit PNG to preserve full dynamic range.
                    _imwrite_or_raise(depths_dir / f"{stem}_ir.png", ir_u16)
                saved_count += 1
                if manual_override and not good_frame:
                    red_manual_count += 1
                _shutter_sound()              # 2-tone shutter click on every save
                motion_cm_accum = 0.0
                # Record coverage bin for this frame's orientation
                import math as _m
                yaw_norm = ((cov_yaw_rad / (2 * _m.pi)) + 0.5) % 1.0
                yaw_bin  = int(yaw_norm * COV_YAW_BINS) % COV_YAW_BINS
                pitch_norm = (cov_pitch_rad / _m.pi) + 0.5
                pitch_norm = max(0.0, min(0.9999, pitch_norm))
                pitch_bin = int(pitch_norm * COV_PITCH_BINS)
                cov_hits[(yaw_bin, pitch_bin)] = cov_hits.get((yaw_bin, pitch_bin), 0) + 1

                if manual_override:
                    quality = "green" if good_frame else "red"
                    print(f"  saved {stem} manually ({quality} frame, "
                          f"in-range {in_range_pct:.1f}%, blur {blur:.0f}, "
                          f"center {center_depth:.2f}m)")
                elif saved_count % 10 == 0 or args.no_preview:
                    print(f"  captured {saved_count} frames  "
                          f"(in-range {in_range_pct:.1f}%, blur {blur:.0f}, "
                          f"center {center_depth:.2f}m, "
                          f"coverage {len(cov_hits)}/{COV_YAW_BINS*COV_PITCH_BINS} cells)")
                if saved_count >= args.max_frames:
                    print(f"  reached max_frames={args.max_frames}, stopping.")
                    break

    except KeyboardInterrupt:
        print("\n  Ctrl+C received; finishing session.")
    finally:
        pipeline.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()

    # Write intrinsics
    if last_intrinsics is not None:
        intr_path = out_root / "femto_intrinsics.json"
        intr_path.write_text(json.dumps(last_intrinsics, indent=2))
        print(f"  saved: {intr_path}")

    print(f"\n=== Session complete ===")
    print(f"  Captured {saved_count} frame pair(s)")
    if red_manual_count:
        print(f"  Manual red-frame saves: {red_manual_count}")
    print(f"  RGB images:  {images_dir}")
    print(f"  Depth maps:  {depths_dir}")
    print(f"  Intrinsics:  {out_root / 'femto_intrinsics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
