# Femto Mega capture protocol

Quality of every downstream stage (COLMAP SfM, ToF init, splat training,
OpenMVS mesh) is bottlenecked by capture quality. Following this checklist
typically improves final results more than any pipeline change.

## Trajectory

- **Move slowly** — 5 cm/sec or less. Motion blur ruins COLMAP feature
  matching and smears ToF time-of-flight phase measurements.
- **Orbit, do not pan** — keep the subject roughly centered, walk around it
  in a circle. Constant distance keeps ToF in its sweet spot (0.5–3 m is
  ideal; 0.25–5.46 m is the hard limit) and keeps COLMAP feature density
  uniform.
- **Multi-height pass** — do at least two orbits at different heights (e.g.
  table-top and shoulder), then a top-down sweep. Single-height captures
  produce mesh holes on top/bottom surfaces.
- **70–80% frame-to-frame overlap** — at 30 FPS and 5 cm/sec that means
  each pixel of subject is seen in ~5+ consecutive frames. COLMAP needs
  this to triangulate.
- **No skipping** — every step of the orbit should have at least one frame.
  Skipped angles become mesh holes.

## Lighting

- **Diffuse** — overhead softbox / cloudy day / shaded outdoor area.
- **No moving shadows** — the camera shadow on the subject moves with you;
  pre-light from multiple angles so the subject is not lit from behind your
  back.
- **No direct sunlight on subject** — IR from sunlight corrupts the Femto's
  850 nm ToF measurements. Indoor or shade only for ToF accuracy.
- **No mirrors / windows in frame** — both COLMAP and ToF fail on them.
  Cover or shoot around them.

## Subject

- **Texture matters for COLMAP** — featureless walls (white drywall,
  uniform table) cause COLMAP to fail to register frames. If the subject
  is a textureless object, surround it with textured backdrop (newspaper,
  marker scribbles, projector pattern).
- **Avoid glass / transparent / specular** — both COLMAP and ToF return
  garbage on them. Skin (sweat) and chrome (reflections) are also hard.
  Cover with matte spray or accept holes.

## Scale guidance

Femto is metric. To verify scale after reconstruction:
1. Place a known-size object in the scene (a 30 cm ruler, a known box).
2. Measure it in the final mesh. Should match within ~2 cm at 1 m distance.
3. If off by a constant factor: `tof_bounds.json scale_factor_da3_to_colmap`
   is wrong — re-solve.

## How many frames?

- **Minimum: 60** — fewer and COLMAP rarely registers everything.
- **Recommended: 120–200** — what V23 used (129). Good coverage with
  manageable training time.
- **Diminishing returns: >300** — slows training without adding info, since
  voxel-subsampling caps init density anyway.

## Pre-capture quick check

Before committing to a long capture, do a 30-frame test sweep and run:
```
python capture_femto.py                 # capture the sweep
python reconstruct_realityscan.py --use-femto-depth --quick
```
Quick mode runs 1000 iters at 4× downscale (~3 min). If the result is
recognizable, your protocol is fine. If COLMAP fails to register most
frames, fix lighting/motion/overlap before the real capture.

## Post-capture inspection

After running the pipeline, look for:
- **`colmap/dense/sparse/0/`** — should contain ALL captured frames. If a
  fraction registered, your trajectory had non-overlapping segments.
- **`colmap/dense/tof_bounds.json` `scale_info.std`** — should be <2 m for
  a tight scale-solve. >5 m means depth and COLMAP are disagreeing badly
  (probably reflections or out-of-range surfaces poisoning the solve).
- **`output/splat_v23_noinit_pruned.ply`** opacity median — should be >0.9.
  Lower means training found a lot of ambiguous geometry → recapture with
  better lighting / coverage.
