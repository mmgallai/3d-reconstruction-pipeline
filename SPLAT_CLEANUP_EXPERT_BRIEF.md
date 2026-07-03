# V32 splat cleanup — what we tried, what failed, and an expert question

**Date:** 2026-06-23

We have iterated six post-process pruning variants on a 226,665-Gaussian V32 splat of an indoor desk scene, paired with its OpenMVS textured mesh, and we have hit a wall. Every global rule we have written either leaves the leafy perimeter fringe in place or removes legitimate flat-on-surface Gaussians and visibly degrades the desk and back wall. Manual brushing in SuperSplat removes the same fringe with zero collateral damage, which tells us the separation exists spatially but not in any (scale, opacity, mesh-distance, anisotropy) feature space we have tried. This document records the experiments, the user's diagnostic observations, and a short list of targeted questions for a 3DGS expert.

> **How to use this document**
>
> If you want the actionable plan, the active work is in `PROGRESS_LOG.md` and `COMPREHENSIVE_PLAN.md`. This document is for a single targeted question to a 3DGS expert about the post-training scene-cleanup problem.

## Table of contents

- [1. Problem framing](#1-problem-framing)
- [2. Methods we have tried (v1 to v6)](#2-methods-we-have-tried-v1-to-v6)
- [3. The user's observations and proposals](#3-the-users-observations-and-proposals)
- [4. The remaining problem + questions for the expert](#4-the-remaining-problem--questions-for-the-expert)
- [Summary in one paragraph for the expert's eye](#summary-in-one-paragraph-for-the-experts-eye)
- [Appendix](#appendix)

---

# 1. Problem framing

**We need a post-processing-only floater cleanup for a 226,665-Gaussian V32 splat of an indoor desk scene, paired with its OpenMVS mesh, without degrading the desk, back wall, or grabbable objects, and without re-running splatfacto every iteration.**

## What we're building

An open-source pipeline that turns a single consumer-grade Femto Mega ToF scan into a Meta Quest 3 VR digital twin. The pipeline runs Femto Mega ToF capture → COLMAP → splatfacto (V32 config) and produces two paired deliverables for every scene:

- A **3D Gaussian splat** for visual fidelity in the headset.
- An **OpenMVS textured mesh** for physics, colliders, and grab interactions.

These are not alternatives. Splat + mesh travel together: the splat is what the user sees, the mesh is what the user touches. Per-object outputs are produced by `scene_segmenter v9a_fp_v2`, which crops each grabbable object as its own splat+mesh pair so Unity can attach a rigid body and grab handle per object.

## The artifact

V32 looks good on the desk surface, the objects, and the front of the back wall, but it has three visible defects:

1. **Leafy/feathery fringe around the desk perimeter** — thin elongated Gaussians sticking outward from desk edges.
2. **A halo of similar fringe above and behind the back wall** — Gaussians floating in empty space above the wall plane.
3. **View-dependent specular blobs on reflective surfaces (monitors)** — these move with the camera and read as ghosting in stereo.

## Why it matters

In a Quest 3 headset, floaters break the illusion in a way they don't on a desktop monitor. Stereo disparity makes off-surface Gaussians read as semi-transparent "stuff" hovering in front of real surfaces, and head movement makes view-dependent blobs swim. The desk and back wall are the scene's largest continuous surfaces, so any global cleanup rule that nibbles those is immediately visible. The objects are small and have to survive verbatim — they are the interactive elements.

## Constraint: post-process only

V32 training takes ~30 min on an RTX 5070 Ti. Iterating cleanup ideas at retrain cadence is prohibitive. Every method in this brief operates on the trained `.ply` (and optionally the mesh) without touching splatfacto.

## Inputs

- **`splat_v32_data3_noinit_pruned.ply`** — 226,665 Gaussians, 53.6 MB, in nerfstudio splat-space coords (1 metric metre = 1.5227 splat units; dp_scale=0.17306, scale_factor_da3_to_colmap=8.7986).
- **OpenMVS scene mesh** — 805,424 verts / 1,296,127 faces, metric world coords.
- **`dataparser_transforms.json`** — the splat-space ↔ metric world transform from the splatfacto run.
- **Per-object segmenter outputs** — 3 objects (bottle, box, lobster) from `scene_segmenter v9a_fp_v2`, each as its own splat+mesh+collider bundle.

The scene's dominant supporting plane (the desk) is **tilted ~34.5° from world +Y** (RANSAC normal +0.10, +0.82, +0.56), so any "flatten with world axes" assumption is wrong out of the gate.

---

## 2. Methods we have tried (v1 to v6)

All six variants share the same prefilter: a per-axis scale-percentile cap (p95 over the max scale) plus an opacity floor of >= 0.30. They differ in the spatial/shape mask applied on top. Inputs are V32 `splat_v32_data3_noinit_pruned.ply` (226,665 Gaussians) and the OpenMVS textured mesh (805k verts, 1.3M faces). All distances below are reported in metric world units after inverse-transforming Gaussian centers from splat-space (1 m = 1.5227 splat units).

### v1 - `aggressive_prune.py`: scale + opacity + ToF AABB
Combines the shared scale/opacity prefilter with an axis-aligned bounding box from `tof_bounds.json` (20 percent slack). The hope was that the ToF capture volume would naturally bound the scene and drop everything outside.

- **V32: 177,038 kept (78.1 percent).**
- **Succeeded at:** trimming a thin top-Z slab (accidentally, because Z happened to be roughly aligned).
- **Failed at:** the bounds were in COLMAP units and the splat is in splat-space, so the AABB was effectively a no-op on X/Y. The leafy fringe was untouched. This was a coord-space bug, not a method bug.

### v2 - `aggressive_prune_v2.py`: scale + opacity + per-object union AABB
Same prefilter; AABB now derived from the union of per-object splat AABBs from `scene_segmenter v9a_fp_v2`, plus 30 cm margin. Coordinate spaces corrected.

- **V32: 99,733 kept (44.0 percent).**
- **Succeeded at:** the units are right, and the fringe above the volume is gone.
- **Failed at:** over-cropped. The per-object union spans the desk and graspable objects but does **not** reach the back wall or the desk's outer perimeter, so structural scene geometry was deleted along with the floaters.

### v3 - `aggressive_prune_v3.py`: scale + opacity + scene-mesh AABB
Same prefilter; AABB now derived from the V32 OpenMVS scene mesh vertices in splat-space, with 10 cm margin and 1-99 percentile clip to suppress mesh outliers.

- **V32: 153,752 kept (67.9 percent).**
- **Succeeded at:** stopped cropping the desk perimeter that v2 cut.
- **Failed at:** the OpenMVS mesh itself does not fully cover the back wall, so the wall-top fringe (and the wall behind it) was still clipped. AABB-based cropping is the wrong tool for a non-axis-aligned scene anyway: the actual desk plane is tilted 34.5 degrees from +Y (RANSAC normal `(+0.10, +0.82, +0.56)`).

### v4 - `aggressive_prune_v4.py`: scale + opacity + Open3D density outlier removal
Same prefilter; then Open3D `remove_statistical_outlier` / `remove_radius_outlier` on Gaussian centers as a point cloud.

| Params | Kept | % | Verdict |
|---|---|---|---|
| stat std_ratio=2.0 | 167,530 | 73.9 | density only removed 3.6 percent extra; no visible change |
| stat std_ratio=1.0 | 154,541 | 68.2 | density removed 11 percent; best of v4 but fringe still present |
| radius nb=10 r=0.07 | 172,898 | 76.3 | no-op (0.5 percent) |
| radius nb=20 r=0.10 | 173,307 | 76.5 | no-op (0.3 percent) |

- **Succeeded at:** confirmed that density outlier removal is the wrong instrument here.
- **Failed at:** the splat is dense everywhere, including in the fringe, because the leafy needles are themselves clustered. Radius-based variants found no isolated points; statistical at std_ratio=1.0 helped but still left visible fringe and started clipping legitimate Gaussians.

### v5 - `aggressive_prune_v5.py`: scale + opacity + mesh-distance shell
Same prefilter; then build an Open3D `RaycastingScene` BVH over the OpenMVS mesh, inverse-transform each Gaussian center to metric world, and keep those with unsigned distance to nearest face <= `--max-dist-m`. Median surviving surface distance is 2.24 cm with p95 = 33.5 cm (bimodal).

| max-dist-m | Kept | % | Verdict |
|---|---|---|---|
| 0.01 | 56,806 | 25.1 | too tight; below the 2.24 cm median surface distance |
| 0.025 | 91,540 | 40.4 | borderline; carves real surfaces |
| 0.05 | 117,826 | 52.0 | mid; still kills surface coverage |
| 0.10 | 139,206 | 61.4 | step drops about 20 percent; still leafy |
| 0.15 | 148,407 | 65.5 | leafy survives |
| 0.25 | 155,696 | 68.7 | user verdict: still leafy |
| 0.50 | 172,426 | 76.1 | effective no-op past p95 = 33.5 cm |

- **Succeeded at:** removed isolated far floaters and any Gaussian whose center genuinely sits in empty space.
- **Failed at:** the leafy needles' **centers** sit close to the mesh (they pass any reasonable shell threshold) but their elongated bodies extend outward, which is what we see as "leaves". A center-distance shell cannot reject them at any threshold without also destroying surface coverage.

### v6 - `aggressive_prune_v6.py`: scale + opacity + four shape modes
Adds `eff-extent`, `aniso`, `combo`, and `off` modes on top of v5's `mesh-dist`. The key one is **eff-extent**: keep if `center_dist + alpha * max_axis <= max_dist_m`, i.e. compare the Gaussian's farthest projected point (not its center) to the shell. `aniso` thresholds `max_axis / min_axis` directly. `combo` ANDs all of them.

| Mode / params | Kept | % | Verdict |
|---|---|---|---|
| eff-extent d=0.05 a=1.0 | 103,431 | 45.6 | too aggressive; cuts surfaces |
| eff-extent d=0.10 a=1.0 | 135,772 | 59.9 | best cleanup, but desk gets holes |
| eff-extent d=0.10 a=0.5 | 137,783 | 60.8 | barely moves vs plain mesh-dist |
| eff-extent d=0.15 a=1.0 | 147,273 | 65.0 | desk holes persist |
| eff-extent d=0.25 a=1.0 | 155,251 | 68.5 | leafy starts coming back |
| eff-extent d=0.30 a=1.0 | 159,525 | 70.4 | bigger shell, desk still degraded vs raw |
| aniso<=50 only | 135,538 | 59.8 | kills needles by shape but lets near-isotropic ghosts through |
| combo d=0.05 a=1.0 aniso<=50 | 74,571 | 32.9 | over-prune |

- **Succeeded at:** eff-extent is the correct geometric statement of the problem (it knows the Gaussian is a body, not a point). Of the surviving Gaussians at v5 d=25cm, 12.6 percent are simultaneously >2 cm long and >5x anisotropic, and eff-extent is the only mode that targets exactly this population.
- **Failed at:** the global p95-scale + opacity prefilter still catches legitimate flat-on-surface Gaussians whose scale fingerprint happens to match a floater's. Tightening the shell to kill the last fringe always opens holes in the desk; loosening it lets the fringe back in.

### Where this leaves us
After six variants - AABB-based (v1, v2, v3), density-based (v4), center-distance shell (v5), and shape-aware effective-extent (v6) - we have not found a global rule that removes the leafy fringe without visibly degrading the surfaces immediately under it. The user's SuperSplat brush comparison sharpens this: manual erasing of the same fringe leaves the desk visibly intact, because the brush is purely spatial and per-Gaussian, whereas every script we have written mixes a global shape/opacity prior with a geometric shell and inevitably catches surface-resident Gaussians whose statistics overlap the floaters'. The fringe and the surface are separable visually but not in the (scale, opacity, distance, anisotropy) feature space we have been filtering on.

---

## 3. The user's observations and proposals

Throughout the cleanup iterations the user drove the strategy with four concrete proposals. Each was implemented, swept, and produced a non-trivial empirical result.

### 3.1 "Use the mesh as a shape mask to crop the splat"

The user noticed that the OpenMVS mesh (805,424 verts / 1,296,127 faces) already encodes the true scene support and proposed using it as a shape-aware crop instead of axis-aligned bounds. This became `aggressive_prune_v3.py` (mesh-AABB in splat-space, +10 cm margin, 1-99 percentile clip) which kept 153,752 / 226,665 Gaussians (68%) but still cut the back wall because the OpenMVS mesh itself was incomplete there. The idea matured into `v5`, where an Open3D `RaycastingScene` BVH computed each Gaussian center's unsigned distance to the nearest mesh face after inverse-transforming back to metric world units. Sweeping `--max-dist-m` from 1 cm (56,806 / 25.1%) through 50 cm (172,426 / 76.1%) showed the filter was *correctly* gating by surface proximity, but the leafy fringe survived at every threshold. Diagnosis: the needles' centers sit on the surface (passing the filter) while their elongated bodies extend outward. That observation directly motivated v6's `eff-extent` mode, `center_dist + alpha*max_axis`.

### 3.2 "The scene is tilted, not aligned with world axes"

The user pointed out that AABB-based filters were doomed because the desk is not world-aligned. We ran RANSAC on mesh vertices near each object footprint and recovered a plane normal of (+0.10, +0.82, +0.56), i.e. **34.5° tilt from +Y**. This explains why every AABB variant (v1, v2, v3) either leaked or over-cropped, and it is now used to place object-patch coordinates for the per-object segmenter.

### 3.3 "Make the mask bigger / more tolerable"

After v5 at 25 cm still left the fringe, the user pushed thresholds upward. The v6 sweep went `d=10cm a=1.0` (135,772 / 59.9%, winner for fringe but holes in the desk) up to `d=30cm a=1.0` (159,525 / 70.4%, larger mask but desk quality still worse than raw). This bracketed the no-win zone: smaller masks gut the desk, larger masks let the fringe back in.

### 3.4 "SuperSplat brush doesn't degrade the scene, why do your scripts?"

The user manually erased the fringe in SuperSplat and observed zero collateral on the desk, while every auto-prune degraded it. The diagnosis: our filters are **global** (scale > p95, opacity < 0.30, mesh-dist > N, max-axis > X) and any global rule catches some legitimate flat-on-surface Gaussians that share a fingerprint with floaters. SuperSplat's brush is purely **spatial-local**, so it has no collateral. This is the central constraint on any future auto-pruner.

### Key empirical findings from the V32 splat

| Quantity | Value |
|---|---|
| Median Gaussian-center distance to nearest mesh face (after scale+opacity prune) | **2.24 cm** |
| p95 center-to-mesh distance | **33.5 cm** (bimodal: dense surface + sparse fringe) |
| Median anisotropy (max_axis / min_axis) | **12.7x** |
| p90 anisotropy | **245x** |
| p99 anisotropy | **11,123x** |
| "Leafy needles" (>2 cm long AND >5x anisotropic, at v5 d=25cm) | **12.6%** of survivors |
| Metric-to-splat scale (1 m -> splat units) | **1.5227** (dp_scale 0.17306 x DA3-to-COLMAP 8.7986) |

---

## 4. The remaining problem + questions for the expert

### a) The core trade-off we cannot escape with post-process pruning

After six iterations of post-process filters (v1 through v6), we have converged on an uncomfortable conclusion: **any rule-based, global filter we write has measurable collateral damage on the surfaces we are trying to preserve, while still leaving residual leafy artifacts at the perimeter.** The user's SuperSplat brush comparison is the most damning evidence. When the user manually brushes the fringe Gaussians away in SuperSplat, the desk and back wall retain their original apparent fidelity. When *any* of our scripts run — including the gentlest configurations (v4 statistical std_ratio=2.0 keeping 73.9%, or v6 eff-extent d=25cm a=1.0 keeping 68.5%) — the surfaces visibly degrade. The brush has zero collateral damage; our scripts always have some.

The mechanism is now clear. Our filters use **global statistics**: scale > p95, opacity < 0.30, mesh-distance > N cm, max-axis > X cm, anisotropy > Y. The leafy fringe Gaussians and a non-trivial subset of legitimate, flat-on-surface scene Gaussians share overlapping fingerprints in those feature spaces. For example, the v5 mesh-distance analysis showed needle centers sitting close to the mesh (passing any reasonable distance threshold up to 25 cm) while their **bodies** extend outward — which is why v6 eff-extent at d=10cm a=1.0 kept only 59.9% but began punching holes in the desk. Conversely, the 12.6% of survivors that are simultaneously >2 cm long and >5× anisotropic include both the floaters we want gone and the legitimate elongated Gaussians along the desk's tilted edge.

A second compounding factor is splatfacto's **alpha-blended rendering**. Even when we cut a Gaussian that is statistically a clear outlier, every remaining surface pixel was originally rendered as the integration of many overlapping Gaussians. Removing any of those contributors — even ones whose centers are 10–25 cm off the mesh — perceptually softens or punches micro-holes into the surface, because the local color/opacity budget was tuned during training to include them. The brush, by contrast, only removes Gaussians that contribute to pixels the user has identified as fringe; it does not touch the support set of any preserved surface pixel.

### b) Specific questions for an expert

1. **View-dependent floater detection.** Is there a published post-process method that uses **view-dependent signals** (e.g. per-Gaussian utilization across the training cameras, spherical-harmonic energy concentrated in narrow view cones, or render-vs-mask agreement) to label floaters? Our specular-blob artifacts near the monitors pass every geometric filter we have tried because their centers are near the monitor mesh — but they only "exist" from certain viewing angles.

2. **Surface-aware mesh-distance metric.** Could a smarter mesh-distance metric — e.g. **signed distance projected along the Gaussian's max-axis direction in surface-normal coordinates** (parallel-to-surface distance vs perpendicular distance) — distinguish a flat surface Gaussian (long axis tangent to the mesh) from a needle whose long axis is normal-to-surface and pokes outward? Our scalar `center_dist + alpha*max_axis` (v6 eff-extent) does not separate these two cases.

3. **SpotLessSplats viability.** Is **SpotLessSplats** considered SOTA for this failure mode, and what is the realistic implementation cost on top of nerfstudio's `splatfacto-big`? Is there a more lightweight training-time fix we should consider first (e.g. an opacity-regularizer change, anisotropy clamp, or a depth-loss term that we have not tried since v25 dn-splatter)?

4. **Best practical answer for our specific case.** Given the constraint of *no retrain* and a V32 scene-level leafy fringe, is the honest answer that **hand-brushing in SuperSplat is the best available tool today**, and our energy is better spent on tooling around that workflow (batch ROI, undo, etc.) than on more pruning heuristics?

5. **Train-time transient detection.** Is there a published technique that flags **transient or view-dependent Gaussians at train-time** — e.g. utilization maps over training views, as in SpotLessSplats — that would let us export a per-Gaussian "trustworthiness" channel into the .ply, then prune on that signal instead of on shape/position?

6. **Hybrid auto + interactive workflow.** Could a **conservative auto-prune of the perimeter** (something like v4 stat_std_ratio=2.0, which only removes the obvious outliers and keeps 73.9%) followed by **SuperSplat brush refinement** be the right pipeline — and is there a better interactive tool than SuperSplat for the refinement step (real-time preview of multiple thresholds, region-locked brushing, mesh-overlay guidance)?

### c) What we tried but did NOT explore

Methods we are aware of but have not tested:

- **SpotLessSplats** — train-time transient/floater suppression via utilization maps. Highest-priority unknown.
- **ToF-IR pre-clean** (Track 0 of our comprehensive plan) — clean the Femto Mega ToF depth/IR streams *before* they enter COLMAP/splatfacto initialization, attacking ghost initialization at the source rather than post-processing.
- **3DGUT optics-aware rasterizer** — alternative rasterizer that may handle reflective-surface artifacts differently from splatfacto's default.
- **Recapture with monitors off** — capture-time mitigation; eliminates the view-dependent specular blobs by removing the offending surfaces from the scene during acquisition.
- **NVIDIA NuRec** — listed as a candidate for the digital-twin pipeline but not benchmarked against our V32 baseline.

---

## Summary in one paragraph for the expert's eye

We have a 226,665-Gaussian V32 splat of a tilted (34.5° from +Y) indoor desk scene with three defects — a leafy perimeter fringe on the desk, a floater halo above the back wall, and view-dependent specular blobs on monitors — and a paired OpenMVS mesh (805k verts / 1.3M faces) we can use as a geometric prior. Across six post-process variants (AABB-based v1–v3, density-based v4, mesh-distance shell v5, shape-aware effective-extent v6) we cannot find a global rule that removes the fringe without measurably degrading the desk and back wall, because the floater population and a non-trivial subset of legitimate flat-on-surface Gaussians overlap in every (scale, opacity, mesh-distance, anisotropy) feature space we have tried — and splatfacto's alpha-blended rendering means that cutting *any* contributor to a preserved surface pixel softens it. SuperSplat manual brushing of the same fringe leaves the surface visually intact, so the separation is spatial-local, not statistical. We are seeking guidance on view-dependent floater signals (SpotLessSplats and similar), surface-frame mesh distances (parallel vs perpendicular to local normal), and whether a hybrid conservative-auto + interactive-brush workflow is the realistic ceiling without retraining.

---

## Appendix

### File paths

- `c:\Users\mgallai\Projects\3d_automated\blender_version\splat_v32_data3_noinit_pruned.ply` — V32 trained splat, 53.6 MB
- `c:\Users\mgallai\Projects\3d_automated\blender_version\dataparser_transforms.json` — splat-space ↔ metric world transform
- `c:\Users\mgallai\Projects\3d_automated\blender_version\aggressive_prune.py` — v1 (ToF AABB)
- `c:\Users\mgallai\Projects\3d_automated\blender_version\aggressive_prune_v2.py` — v2 (per-object union AABB)
- `c:\Users\mgallai\Projects\3d_automated\blender_version\aggressive_prune_v3.py` — v3 (scene-mesh AABB)
- `c:\Users\mgallai\Projects\3d_automated\blender_version\aggressive_prune_v4.py` — v4 (Open3D density outlier removal)
- `c:\Users\mgallai\Projects\3d_automated\blender_version\aggressive_prune_v5.py` — v5 (mesh-distance shell)
- `c:\Users\mgallai\Projects\3d_automated\blender_version\aggressive_prune_v6.py` — v6 (eff-extent, aniso, combo)
- `c:\Users\mgallai\Projects\3d_automated\blender_version\tof_bounds.json` — ToF capture AABB (COLMAP units)
- `c:\Users\mgallai\Projects\3d_automated\blender_version\PROGRESS_LOG.md` — active work log
- `c:\Users\mgallai\Projects\3d_automated\blender_version\COMPREHENSIVE_PLAN.md` — overall plan including Track 0 ToF-IR pre-clean
- `c:\Users\mgallai\Projects\3d_automated\blender_version\SCENE_SEGMENTER_NOTES.md` — `scene_segmenter v9a_fp_v2` design + version graveyard

### Exact V32 numbers

- Gaussians in input `splat_v32_data3_noinit_pruned.ply`: **226,665**
- File size: **53.6 MB**
- OpenMVS scene mesh: **805,424 verts / 1,296,127 faces**
- Splat-space scale: **1 metre = 1.5227 splat units** (dp_scale = 0.17306, scale_factor_da3_to_colmap = 8.7986)
- Desk plane RANSAC normal: **(+0.10, +0.82, +0.56)** → **34.5° tilt from world +Y**
- Median Gaussian-center distance to nearest mesh face (after scale+opacity prefilter): **2.24 cm**
- p95 center-to-mesh distance: **33.5 cm** (bimodal)
- Median anisotropy (max_axis / min_axis): **12.7×**
- p90 anisotropy: **245×**
- p99 anisotropy: **11,123×**
- "Leafy needle" population (>2 cm long AND >5× anisotropic, measured at v5 d=25 cm): **12.6 %** of survivors
- Per-object segmenter outputs from `scene_segmenter v9a_fp_v2`: **3 objects** (bottle, box, lobster), each with paired splat + mesh + collider JSON
- Variant keep counts on V32 (out of 226,665):
  - v1 (ToF AABB): **177,038 (78.1 %)**
  - v2 (per-object union AABB +30 cm): **99,733 (44.0 %)**
  - v3 (scene-mesh AABB +10 cm, 1–99 % clip): **153,752 (67.9 %)**
  - v4 stat std_ratio=2.0: **167,530 (73.9 %)**
  - v4 stat std_ratio=1.0: **154,541 (68.2 %)**
  - v4 radius nb=10 r=0.07: **172,898 (76.3 %)**
  - v4 radius nb=20 r=0.10: **173,307 (76.5 %)**
  - v5 mesh-dist 1 cm: **56,806 (25.1 %)**
  - v5 mesh-dist 2.5 cm: **91,540 (40.4 %)**
  - v5 mesh-dist 5 cm: **117,826 (52.0 %)**
  - v5 mesh-dist 10 cm: **139,206 (61.4 %)**
  - v5 mesh-dist 15 cm: **148,407 (65.5 %)**
  - v5 mesh-dist 25 cm: **155,696 (68.7 %)**
  - v5 mesh-dist 50 cm: **172,426 (76.1 %)**
  - v6 eff-extent d=5 cm a=1.0: **103,431 (45.6 %)**
  - v6 eff-extent d=10 cm a=1.0: **135,772 (59.9 %)**
  - v6 eff-extent d=10 cm a=0.5: **137,783 (60.8 %)**
  - v6 eff-extent d=15 cm a=1.0: **147,273 (65.0 %)**
  - v6 eff-extent d=25 cm a=1.0: **155,251 (68.5 %)**
  - v6 eff-extent d=30 cm a=1.0: **159,525 (70.4 %)**
  - v6 aniso ≤ 50 only: **135,538 (59.8 %)**
  - v6 combo d=5 cm a=1.0 aniso ≤ 50: **74,571 (32.9 %)**
