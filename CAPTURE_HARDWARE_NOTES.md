# Capture Hardware Notes — Drone / LiDAR / Gimbal for the V32 Pipeline

Reference document consolidating a discussion (2026-07-08 → 2026-07-12) about
automating scene capture for the reconstruction pipeline. Covers three
candidate devices, their fit with our pipeline, and how they combine.

---

## 1. What the pipeline needs from any capture device

The V32 pipeline (`reconstruct_realityscan.py --use-femto-depth` → COLMAP SfM
→ splatfacto → OpenMVS) ingests:

**Required:**
- RGB frames (JPG / PNG), ideally ≥ 4 MP
- Camera intrinsics (focal length, principal point, distortion) — from EXIF
  or a JSON prior similar to `femto_intrinsics.json`
- ~100–150 frames per scene with ~60–80 % forward overlap

**Optional (accelerates + improves quality):**
- Aligned per-pixel depth (Femto ToF style) — used to seed splat init
- Per-frame IMU / GNSS / VIO pose — skips or improves COLMAP
- Gimbal stabilisation — reduces motion blur (kills SfM if bad)

**Dealbreakers:**
- Ultra-narrow FoV (< 30°) — matching fails
- No timestamps / no cross-sensor sync
- Encrypted / proprietary output
- Rolling shutter combined with fast platform motion

---

## 2. ModalAI Starling 2 Max — the drone

**Verdict: FIT.** Turnkey aerial capture platform with a Linux/ROS 2 SDK.

### Actual camera loadout (verified from ModalAI docs)

- **C28 (standard):** 4 cameras total
  - 2× Sony IMX412, 12 MP color, **forward-facing** — capture / streaming
  - 2× onsemi AR0144, 1 MP mono global-shutter fisheye, **forward-facing** —
    VIO / SLAM (not for reconstruction)
- **C29 (enhanced):** same 4 cameras + 1× PMD ToF depth sensor (also
  forward-facing). ToF model / range unpublished.

**Important correction:** all cameras face FORWARD, not down.
Starling 2 Max is a forward-VIO design. There is no built-in nadir camera —
if the drone hovers above an object, none of the built-in cameras can see it.

### Specs summary
- 566 g takeoff weight, 500 g payload capacity
- ~55 min flight (Amprius Li-ion) / ~40 min (standard Li-Ion)
- 120° H × 93° V FoV on IMX412 — fisheye-model friendly
- Rolling shutter (needs slow flight for clean SfM)
- Weak GNSS on current unit (revision in progress)
- SDK: VOXL MPA, ROS 2 Foxy, PX4/MAVLink, snapshot API, Docker
- Frames are hardware-timestamped in MPA pipes
- Price: $3,199 (C28) – $5,799 (C29)

### Pipeline compatibility
- **RGB:** ✅ 12 MP exceeds 4 MP minimum
- **Intrinsics:** available via VOXL SDK per unit; convert to `femto_intrinsics.json` format (~30 lines Python, one-time per drone)
- **Wide fisheye lens:** switch COLMAP to `OPENCV_FISHEYE` model
- **Depth:** C29 has ToF; C28 falls back to pure photogrammetry (still works)
- **Automation:** native fit — scriptable capture missions

### Integration effort
~1 week to write the MPA-snapshot → `frames/*.jpg` + `poses.json` bridge
and validate motion-blur tolerance.

---

## 3. Livox MID-360 — the LiDAR

**Verdict: NOT FIT as a standalone capture device. USEFUL as a pose /
depth co-sensor.**

Pure LiDAR scanner with **no RGB camera**. Our pipeline is
photogrammetry-first, so MID-360 can't feed it standalone.

### Specs summary
- 905 nm non-repetitive scanning LiDAR
- Range: 0.1 m (min) – 40 m at 10 % reflectivity – 70 m at 80 %
- Precision: ~2 cm at 10 m, ~3 cm close range
- 360° H × 59° V FoV
- 200,000 pts/s (first return), 10 Hz sweeps
- Built-in 200 Hz IMU (ICM40609)
- Ethernet + PTPv2 or GPS PPS sync
- SDK: Livox-SDK2, `livox_ros_driver2` (ROS 1 Noetic / ROS 2 Foxy/Humble)
- FAST-LIO2 supported
- 265 g, IP67, 6.5 W avg
- Price: ~$480–$1,350

### How MID-360 could be used in our pipeline

1. **As splat init** — the accumulated LiDAR point cloud can seed
   `tof_init.ply` directly (no per-pixel back-projection). Cleaner than
   Femto's back-projection step. Requires LiDAR ↔ camera extrinsic
   calibration + PTP sync + sweep fusion (via FAST-LIO2).
2. **As pose source** — FAST-LIO2 gives accurate camera poses via
   LiDAR-inertial odometry, skipping COLMAP or seeding it.

### Comparison with Femto Mega ToF

| Aspect | Femto Mega ToF | Livox MID-360 |
|---|---|---|
| Type | RGB-D camera | LiDAR scanner (no RGB) |
| Depth density | Dense per-pixel | Sparse point cloud |
| RGB alignment | Native | Requires extrinsic calibration |
| Range | 0.25 – 5.5 m | 0.1 – 70 m |
| Accuracy | ~1 cm at 1 m | ~2 cm at 10 m |
| Sweet spot | Tabletop / close-up | Room to outdoor |
| Setup | Plug-and-play | 2+ weeks integration |
| Sunlight | Poor | OK |

### Which is better for what?
- **Tabletop (0.3 – 1.5 m):** Femto wins — dense, aligned, close-range
  optimised. MID-360 achieves ~90–95 % of Femto's quality with proper
  calibration + accumulation, but has beam-divergence issues under 30 cm.
- **Room-scale (2 – 5 m):** Equal or MID-360 better.
- **> 5 m / outdoor:** MID-360 wins decisively; Femto physically can't
  reach.

**Rule of thumb:** MID-360 is right for anything the drone would fly to;
Femto is right for anything you hold in your hand.

### On LiDAR quality vs speed

LiDAR doesn't just add speed:
- **Mesh quality:** no impact (OpenMVS is pure photogrammetry).
- **Splat quality:** real gain in low-texture regions (blank walls, uniform
  surfaces) — where pure photometric loss struggles and produces the
  monitor-blob / needle artifacts we've fought.
- **SfM poses:** more robust in textureless / fast-motion / repeating-pattern
  scenes.

### Projecting LiDAR points to camera pixels (technical caveat)

The math is standard (extrinsic + intrinsic transform), but per-frame
coverage is very sparse — with a 120° camera FoV, only ~30 k of MID-360's
200 k pts/s land in the frame, vs 12.3 M pixels. That's 0.24 % of pixels
with a real LiDAR sample per frame. For dense per-pixel depth you'd need
either accumulation over time or a depth-completion network (GuideNet,
NLSPN). For our pipeline's use case (splat init), sparse is fine — no
per-pixel projection needed.

---

## 4. SIYI A8 mini — the gimbal camera

**Verdict: PARTIAL FIT (usable with caveats).**

### Specs summary
- 8 MP Sony 1/1.7" CMOS starlight sensor
- Stills: 3840 × 2160 JPG (8.3 MP)
- Video: 4K @ 25 fps H.265 MP4
- Fixed lens: f = 21 mm equiv, f/2.8
- **81° horizontal FoV** (93° diagonal) — well-behaved
- Rolling shutter (assumed)
- 3-axis gimbal stabilisation
- SDK: UART/UDP protocol, RTSP stream, MAVLink integration; mature
  Python `siyi-sdk` (PyPI) + `mzahana/siyi_sdk` + `julianoes` MAVSDK
  camera manager
- 95 g, ~$254

### Pipeline compatibility

| Aspect | Result |
|---|---|
| 8 MP resolution | ✅ meets ≥ 4 MP minimum |
| 81° H FoV | ✅ standard OPENCV pinhole model works, no fisheye needed |
| Fixed intrinsics | ✅ hardcode fx ≈ 2240 px (3840 × 21 / 36); solve `k1, k2, p1, p2` via checkerboard |
| Depth | ❌ none |
| Per-frame pose | ❌ none inherent — must pull from flight-controller MAVLink |
| Automation | ✅ SDK exposes photo-trigger over UDP |
| Rolling shutter | ⚠️ mitigated by stop-and-shoot capture, angular rate < 10°/s |

### Why a gimbal matters (this was the key insight)

Without a gimbal on the Starling, the drone's cameras only point forward.
If it hovers above an object, none of the built-in cameras can see it.
A gimbal decouples camera angle from drone attitude:

1. **Angle diversity feeds SfM** — 3D reconstruction needs the same
   feature from many angles. Gimbal + orbital flight = automatic angular
   coverage.
2. **Motion-blur killer** — active vibration compensation.
3. **Nadir + oblique combos** — best-practice photogrammetry mixes
   straight-down and 30–60° tilted shots. Only possible with a gimbal.
4. **Look up / sideways** — bridge undersides, facades, sculptures.

**Given Starling's forward-only cameras, the A8 mini is not optional —
it's the primary way to point a camera at anything not directly in front
of the drone.**

### Integration effort
Small (~1–2 days). Script SDK photo triggers, pull JPGs off SD card
(or RTSP frame-grab fallback), align to flight-controller MAVLink
timestamps for pose priors. Start COLMAP with `SIMPLE_RADIAL`, upgrade to
`OPENCV` if needed.

---

## 5. Combining all three as a capture rig

The three devices are **complementary, not overlapping**:

- **Starling 2 Max** — autopilot host + default RGB (12 MP forward) +
  IMU + GNSS. Provides the flight platform and forward-VIO cameras.
- **Livox MID-360** — payload for 360° metric LiDAR + FAST-LIO2 poses
  (fixes Starling's rolling-shutter drift, replaces ToF-init at scale).
- **SIYI A8 mini** — gimbal-stabilised detail camera for close-in passes
  and for anything not directly forward (nadir, oblique, upward,
  orbital).

Together they form a coherent aerial capture stack: one autopilot, one
metric-pose LiDAR, one detail camera. Total added payload is well within
the Starling's 500 g budget (MID-360 265 g + A8 mini 95 g = 360 g).

**Integration cost for full rig:** ~3 weeks of hardware, sync, and
calibration work before the pipeline sees clean inputs.

---

## 6. Key decisions / findings from this discussion

1. **Depth is optional.** The pipeline runs pure photogrammetry cleanly
   without any depth sensor. Depth is a quality / speed lever, not a
   requirement.
2. **All Starling cameras face forward** — corrected mid-discussion; this
   makes the gimbal essential rather than nice-to-have.
3. **MID-360 is not a Femto replacement at tabletop scale**, but is the
   correct choice at room-and-larger scale where Femto physically can't
   reach.
4. **Rolling shutter is the recurring risk** across Starling, A8 mini,
   and most consumer sensors. Mitigation: fly slow, stop-and-shoot,
   angular rate < 10°/s.
5. **PI's "per-fragment opacity mask" idea for needles** (discussed in
   passing): would require rasterizer-level changes to gsplat, not a
   post-processing edit. Cheaper alternative: query distance at the
   Gaussian's *far tip* instead of its centre — same signal, catches
   needles, no rasterizer surgery. Not yet implemented.
6. **A per-drone calibration step is a one-time cost** (~30 lines of
   Python) — extract VOXL SDK intrinsics, format as pipeline JSON, save
   per unit. Trivial once done.

---

## References
- ModalAI Starling 2 Max: <https://www.modalai.com/products/starling-2-max>
- Starling 2 Max datasheet: <https://docs.modalai.com/starling-2-max-datasheet/>
- VOXL camera server: <https://docs.modalai.com/voxl-camera-server/>
- Livox MID-360: <https://www.livoxtech.com/mid-360>
- Livox SDK 2: <https://github.com/Livox-SDK/livox_ros_driver2>
- Livox time sync: <https://livox-wiki-en.readthedocs.io/en/latest/tutorials/new_product/common/time_sync.html>
- SIYI A8 mini: <https://www.siyi.biz/en/product/tri-axis-single-camera-gimbal/a8-mini/spec/>
- `mzahana/siyi_sdk` (Python): <https://github.com/mzahana/siyi_sdk>
- `siyi-sdk` PyPI: <https://pypi.org/project/siyi-sdk/>
- `julianoes/siyi-a8-mini-camera-manager` (MAVSDK): <https://github.com/julianoes/siyi-a8-mini-camera-manager>
