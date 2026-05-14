# Future work — not implemented in V23–V29

## 0. V27 TSDF + UV atlas hybrid (blocked on OpenMVS)

Wanted to combine V26's TSDF geometry (better coverage + connectivity) with
V23's OpenMVS-textured UV atlas. Three attempts all crashed OpenMVS's
TextureMesh with `std::out_of_range _Map_base::at` (SIGSEGV) during
atlas-packing — regardless of mesh size (tried 1.74M and decimated to 500K
faces), vertex-color stripping, or resolution-level (0 = 8K, 1 = 4K atlas).

OpenMVS's atlas-packer appears fundamentally incompatible with the topology
that Open3D's marching cubes produces from TSDF volumes (likely T-junctions,
near-degenerate triangles, or PLY property ordering differences). The same
TextureMesh code happily textures V23's OpenMVS-native mesh.

Working alternatives:
- **Custom texturing** in Python: project each face to its best-view camera,
  sample RGB, bake to a texture atlas via xatlas. ~200-300 lines of code.
- **Blender / MeshLab** automatic UV unwrap + texture-transfer from V23 mesh
  onto V27 geometry. Manual but reliable.
- **Use a different TSDF generator** that produces OpenMVS-friendly topology
  (e.g. vdbfusion, which is already a dn-splatter dependency).

Effort: 1-2 days for the custom Python texturer; few hours for the
Blender route per mesh. The TSDF script (V26) and the scaling/orchestration
in `reconstruct_v27_tsdf_atlas.py` are ready when the texturing path is.



Three improvements that need work beyond what this session covered.

## 1. Real depth-supervised splatfacto-big (the real Paradigm B)

V25 tried this via `ags-mesh` (dn-splatter) but had to disable densification
because gsplat 1.5+ doesn't expose the legacy tensors (`xys`, `conics`,
`max_2Dsize`) dn-splatter uses. Result: 50× fewer Gaussians, unrenderable.

**What needs to change:**
- gsplat 1.5+ replaced its mutable strategy API with a clean `Strategy`
  abstract base. dn-splatter still calls `MyStrategy.step_post_backward(...)`
  with the old positional args that expected `xys`, `conics`, etc.
- Port dn-splatter's `DefaultStrategy.step_post_backward` to read from
  gsplat 1.5's new `info` dict (which exposes `means2d`, `radii`, etc.
  per-batch via the strategy hook).
- File: `dn-splatter/dn_splatter/dn_model.py` (the densification code is
  guarded behind the patches we applied in V22).

**Effort estimate:** 2–4 days. Requires understanding both old gsplat
1.0 internals and new gsplat 1.5 strategy hooks. Once done, V25's depth
+ normal supervision becomes a real lever instead of a regression.

**Alternative:** wait for upstream dn-splatter to support gsplat 1.5
(maintainer is intermittent on this).

**Alternative #2:** roll a depth loss into vanilla `splatfacto-big`
directly. nerfstudio's `SplatfactoModel.get_loss_dict` is where the L1+SSIM
RGB loss lives; add `depth_loss = mse(rendered_depth, gt_depth)` weighted
by `depth_lambda`. About 50–100 lines + per-view depth rendering. Avoids
dn-splatter entirely.

## 2. Femto IR channel as a supervision input

The Femto Mega exposes a separate IR/intensity image (the raw return
amplitude from the 850 nm illumination). Useful for:
- Surfaces where RGB has no texture (white walls, monochrome objects)
  but IR has subtle variation.
- Surfaces where RGB and depth disagree (glass partially reflective —
  RGB sees through, depth sees the surface, IR shows the surface
  texture).

**What needs to change:**
- `capture_femto.py` already gets the IR frame from pyorbbecsdk (look
  for `OBSensorType.IR_SENSOR`), it just doesn't save it. Add saving as
  `IMG_femto_NNNN_ir.png` (uint16 grayscale).
- For supervision, add a fourth-channel input to splatfacto's loss or
  use IR as an auxiliary regularizer (push Gaussian opacities to match
  IR intensity in textureless regions).
- A simpler immediate win: use IR for **edge detection** at glass / chrome
  surfaces where COLMAP fails — feed those edge points as priors to
  the COLMAP feature matcher (`--ImageReader.mask_path`).

**Effort estimate:** 1 day for capture-side, 2–3 days for supervision.

**Requires:** a new capture session with IR saving enabled (current 129
frames in `nerfstudio_data/images/` have no IR).

## 3. Multi-pass capture registration (loop closure for large scenes)

Single-session captures are limited by what you can sweep in one
continuous orbit. For a whole room or building you need multiple sessions
with overlapping views, registered into a unified coordinate frame.

**What needs to change:**
- `register_multipass.py` (not yet written): given N independent capture
  sessions each with their own `nerfstudio_data/depths_femto/` + COLMAP
  poses, find overlapping views and use `refine_poses_icp.py`'s ICP +
  pose-graph optimization across sessions.
- Approach: pick anchor frames in each session that show a common region.
  Run colored ICP between anchor pairs across sessions. Build a global
  pose graph and optimize. Output: merged transforms.json with all frames.
- Then feed merged transforms.json into the existing splat / mesh pipeline.

**Effort estimate:** 2–3 days. The math is identical to `refine_poses_icp`
(V29) just at a higher level — instead of frame-to-frame edges, session-
to-session anchor pairs.

**Why not done in V29:** only have one capture session. Would need at
least two overlapping captures of the same scene to test against.

**Quick workaround for now:** treat multiple sessions as one big folder.
Just dump all RGB + depth frames into `nerfstudio_data/images/` +
`depths_femto/`. COLMAP will re-solve poses across all of them. Loses
the per-session structure but works if scene overlap is sufficient.
