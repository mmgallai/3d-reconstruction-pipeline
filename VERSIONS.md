# Versions

Comprehensive history of versions in this repo. Each version is a branch +
same-name tag. Output artifacts live under `output/`.

| Version | Branch + tag | Type | Status | Key change |
|---|---|---|---|---|
| V6  | `v6`  | splat | canonical (legacy, iPhone capture) | DA3 monocular depth → splat init |
| V17 | `v17` | mesh  | canonical (legacy, iPhone capture) | OpenMVS Refine + texture + LOD + AO |
| V22 | `v22` | splat | regression (kept as reference) | AGS-Mesh experiment, broke densification on gsplat 1.5 |
| V23 | `v23` | both  | canonical for **Femto data** | Pure ToF init (Paradigm A) |
| V24 | `v24` | both  | **best of Femto branch — current default** | Hybrid ToF + DA3 fill init + scale-calibration fix |
| V25 | `v25` | splat | experiment, **regression** | ags-mesh + ToF supervision (same V22 densification bug) |
| V26 | `v26` | mesh  | experiment, no atlas | TSDF volumetric fusion (KinectFusion-style, Paradigm C) |
| V29 | `v29` | tooling | docs + framework | ICP pose-refinement framework + scoping docs |
| V30 | `v30` | splat | experiment, **visually worse than V24** | Real depth-supervised splatfacto-big via splat_tof plugin |
| V31 | `v31` | mesh  | experiment, **visually worse than V24** | TSDF + custom xatlas UV baker |

## Legacy (iPhone data, before Femto camera)

### V6 — `splat_v6_da3_pruned.ply`
Canonical Gaussian splat for the iPhone capture set. Used Depth Anything 3
(DA3) monocular depth network to back-project a 500K-point colored cloud,
then trained splatfacto-big initialized from that cloud.

- 2,949,830 pruned Gaussians, 698 MB
- BB diagonal: 9.91 (COLMAP units; ~5 m metric after rescale)
- Opacity median 0.864, % solid (>0.5) = 64.0%

### V17 — `output/mesh_v17/`
Canonical textured mesh for the iPhone capture. Tier-1 + Tier-2 mesh
improvements: OpenMVS DensifyPointCloud → ReconstructMesh → RefineMesh →
TextureMesh, plus LOD decimation (mid 500K / low 100K) and ambient-occlusion
bake. Mesh in metric meters after `mesh_scale_calibrate.py`.

- 4,608,811 verts / 7,733,519 faces / 354 MB (HIGH)
- 8K UV-atlas texture (32 MB PNG)
- 250K / 50K vert LODs at 9 MB / 1.8 MB

### V22 — `splat_v22_agsmesh*.ply`
**Experimental regression, kept for reference.** First attempt at depth +
normal supervision via DN-Splatter's AGS-Mesh variant. Required 8 source
patches to port DN-Splatter from gsplat 1.0 → 1.5 (needed for Blackwell
sm_120 GPU support). Patch #8 disabled densification entirely because
gsplat 1.5 doesn't expose the legacy `xys`/`conics`/`max_2Dsize` tensors
that DN-Splatter's strategy reads. Result: ~260K Gaussians (20× fewer than
V14-V17), 10.7 MB pruned splat. Visually surface-aligned where covered,
but coverage too sparse to render usefully.

## Femto Mega capture branch (current pipeline)

The user upgraded to an Orbbec Femto Mega ToF camera. All V23+ runs use
the same 129-frame indoor desk capture (`nerfstudio_data/depths_femto/`).
The three integration paradigms were:
- **A:** ToF replaces DA3 monocular depth as the splat init source
- **B:** ToF supplies per-iteration depth supervision during splat training
- **C:** ToF + COLMAP poses → KinectFusion-style TSDF fusion, no
  photogrammetry

### V23 — Paradigm A (canonical Femto)
Pure Femto ToF init: `femto_to_init.py` back-projects per-frame metric
depths through COLMAP poses (with sparse-point/depth ratio for COLMAP→
metric scale solve), produces a colored 270K-point cloud → splatfacto-big
trains from it. End-to-end run in ~30 min.

- 654,232 pruned Gaussians, 154.7 MB
- BB diagonal 13.89 (~4 m metric); opacity median 0.975, solid 72.8%
- Mesh: 840,413 verts / 1,338,563 faces, 56 MB, 8K UV atlas
- LODs: 500K / 100K triangle variants with AO

Key files: `femto_to_init.py`, `capture_femto.py`, `lib/openmvs_pipeline.py`.

### V24 — Hybrid init (current default)
**`v24` is the recommended branch / tag for new work.** Same as V23 but
the init cloud can blend Femto ToF with DA3-fill in regions ToF can't
reach (<0.25 m, >5.46 m). On the desk capture the hybrid was visually
indistinguishable from V23 (everything fit in ToF range), but the
capability matters for larger scenes.

V24 also fixes a scale-calibration bug in `lib/openmvs_pipeline.py:312`:
the mesh-rescale step now prefers `tof_bounds.json` (Femto's measured
COLMAP→metric scale, 23.98) over `da3_bounds.json` (DA3's estimated scale,
22.32 — ~7% off in tested scenes). V23 mesh was retroactively rescaled
with this fix.

- 642,745 pruned Gaussians, 152.0 MB (~V23)
- BB diagonal 15.40 (~11% wider than V23 — extended scene coverage)
- α-median 0.91 ± 7%, β-median 0.02 m for the per-frame ToF↔DA3 fit
- Tested against V23: visually identical on the desk; +11% coverage on
  walls/ceiling beyond Femto's 5.46 m cap

Key files: `femto_to_init.py` (--use-da3-fallback flag), all V23 files.

### V25 — Paradigm B attempt #1 (regression)
Tried depth supervision via `ags-mesh` (DN-Splatter family) on top of V22's
patched codebase. Same densification limit as V22 means only ~13K
Gaussians survived opacity culling — even worse than V22 because ToF
supervision is stronger than DA3 supervision and culls harder.

Pipeline change: `lib/nerfstudio_pipeline.py:export_splat` now installs
DN-Splatter in the export container too (V25 export originally failed
because the config.yml references `dn_splatter.dn_pipeline`).

- 13,269 pruned Gaussians, 3.1 MB
- Surface-aligned but too sparse to render. **Use V30 instead** for the
  real Paradigm B implementation.

### V26 — Paradigm C (TSDF fusion)
KinectFusion-style: skip COLMAP MVS + splatting entirely. Integrate 129
Femto RGB-D frames into an Open3D ScalableTSDFVolume using the COLMAP
poses (scaled to metres), then marching-cubes a colored mesh in ~30 sec.

- 944,635 verts / 1,744,213 faces / 67.6 MB
- 2.7× larger surface area than V23 OpenMVS mesh (TSDF accepts every
  depth measurement; OpenMVS drops surfaces without multi-view stereo)
- 60% fewer boundary edges; largest connected component 54% of mesh
  vs 3.5% for V23 (TSDF much better topology connectivity)
- Per-vertex colors only — no UV atlas. V31 was the attempt to add one.

Key files: `tsdf_fusion_femto.py`.

### V29 — Tooling: ICP pose refinement + scoping
- `refine_poses_icp.py`: Open3D coloured-ICP between adjacent Femto
  point clouds + pose-graph optimization. On the dense desk capture,
  global optimization found **0 mm pose changes** — COLMAP is already as
  accurate as ICP can refine. Framework is ready for sparse-overlap or
  multi-pass captures.
- `CAPTURE_GUIDE.md`: capture-protocol checklist (trajectory, lighting,
  subject, scale verification, frame-count guidance).
- `FUTURE_WORK.md`: scoped IR-channel supervision, multi-pass framework,
  gsplat 1.5 densification fix, V27 TSDF-atlas blocker.

### V30 — Paradigm B attempt #2 (visually worse than V24)
The "real" depth-supervised splatfacto-big. New `splat_tof/` plugin
subclasses `SplatfactoModel` + `FullImageDatamanager` to add an L1 depth
loss against Femto metric depths, while keeping gsplat 1.5's standard
densification active (bypasses DN-Splatter entirely).

- 4,042,719 raw Gaussians → 1,069,474 pruned (V23: 1.28 M → 654 K)
- Scene BB shrank 37%, per-Gaussian scale shrank 47% — depth loss pulled
  Gaussians off the periphery onto the measured surface, finer detail
- Median opacity dropped from 0.975 → 0.631 — many partially-transparent
  thin shells. **Looked worse than V23/V24 visually.** Probable cause:
  `depth_lambda=0.2` is too aggressive on this capture.

Key files: `splat_tof/` (plugin package), `lib/nerfstudio_pipeline.py`
recognizes `--train-method splatfacto-tof`.

### V31 — TSDF + custom UV atlas (visually worse than V24)
Unblocks V27 (which OpenMVS TextureMesh SIGSEGVed on TSDF marching-cubes
topology) with a Python pipeline: xatlas UV unwrap + per-face best-view
projection + flat-per-face atlas bake.

- 500K faces (decimated from 1.74M), 4K PNG atlas
- 100% face coverage, 70.3% texel coverage, ~30 s runtime
- **Visually faceted** because each face gets one solid colour from its
  best-view centroid pixel. Per-pixel barycentric sampling would fix
  this (~50 extra lines).

Key files: `bake_tsdf_atlas.py`.

## Pattern across V23-V31

Six algorithmic experiments on this Femto desk capture, zero visual wins
over V23/V24. The bottleneck is the data (handheld 129-frame loose sweep),
not the pipeline. Next real improvement is a better recapture per
`CAPTURE_GUIDE.md`, then re-running V24 + optionally V30 on the new data.

## Quick reference: where to start

- **Render a Femto reconstruction now:** branch `v24`, run
  `python reconstruct_realityscan.py --skip-mvs --use-femto-depth`.
  Outputs land in `output/`.
- **Capture new data:** branch `v24`, run `python capture_femto.py`,
  follow `CAPTURE_GUIDE.md`.
- **Compare splats across versions:** `python inspect_v23.py` (uses da3
  conda env for trimesh).
- **Try TSDF-only mesh:** `python tsdf_fusion_femto.py` (output in
  `output/mesh_v26_tsdf/`).
- **Reuse V29 ICP framework on a future multi-pass capture:**
  `python refine_poses_icp.py` after building per-pass clouds.
