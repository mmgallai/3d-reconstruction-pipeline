# Scene Segmenter — automated 3D object extraction from V32 photogrammetry

**Last updated:** 2026-06-18
**Status:** Working. Production output in `output/segmented_v32_data3_v9a_fp_v2/` (mesh) and
`output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned/` (splat).

This document is the canonical record of what works, what didn't, and why,
so we don't re-derive everything next time. **Read this before changing
the pipeline.**

---

## Canonical production picks (user-confirmed)

After a long tuning round on both sides, these are the versions the user
locked in. Use these unless you have a specific reason to deviate.

### Mesh extraction — **v9a_fp_v2**

- Directory: [`output/segmented_v32_data3_v9a_fp_v2/`](output/segmented_v32_data3_v9a_fp_v2/)
- Per-object outputs: `<prompt>_extracted.{obj,mtl,png,ply,glb}` + `_collider.json`
- Recipe: rasterized SAM3 voting → seeds (score ≥ 0.7) → spatial-voxel
  CC filter on seeds → 2-D X-Z footprint with **0.5 cm dilation** +
  **0.5 cm Y margin** → spatial crop of original 1.3 M-face mesh.
- Production CLI: see "Production command" section below.

### Splat extraction — **v6_cleaned** (CURRENT PRODUCTION PICK, 2026-06-18)

Two-stage process. **v5_shcolor** produces the per-object splat; **v6_cleaned**
refines it via Clean-GS pruning. Always run both.

#### Stage A — v5_shcolor crop

- Output: `output/segmented_v32_data3_v9a_fp_v2_splat_v5_shcolor/<prompt>_splat.ply`
- Recipe: footprint crop (driven by the v9a_fp_v2 extracted mesh's vertices)
  + SH-viewport color filter `--exclude-color "124,113,71" --color-tolerance 18`,
  **no scale filter on the body** (user verdict: body quality is priority).
- CLI:
  ```
  python _spatial_crop_splat.py \
    --splat output/splat_v32_data3_noinit_pruned.ply \
    --dataparser nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json \
    --segmented-dir output/segmented_v32_data3_v9a_fp_v2 \
    --prompts "white water bottle,blue box,red lobster figurine" \
    --output-dir output/segmented_v32_data3_v9a_fp_v2_splat_v5_shcolor \
    --crop-style footprint --xz-dilate-cm 0.5 --y-margin-cm 0.5 \
    --exclude-color "124,113,71" --color-tolerance 18 --sh-view-dir "0,0,1"
  ```

#### Stage B — v6_cleaned (Clean-GS pruning on the v5_shcolor crops)

- Output: `output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned/<prompt>_splat.ply`
  (same nerfstudio PLY schema as v5; drop-in replacement)
- Recipe: feed v5_shcolor per-object splat + 20 evenly-subsampled masked photos
  + dataset `cameras.json` into Clean-GS's whitelist → colour-validation →
  k-NN outlier-prune chain. Because the input is already 3D-tight, the
  2D-mask depth-ambiguity that caused halos in the v1 "full-scene-in"
  experiment is gone — this only prunes WITHIN the existing envelope.
- Measured impact (V32 data3): bounding box unchanged, **12-19 % Gaussian
  reduction per object**, slight rise in per-Gaussian opacity confidence
  (lobster +2.5 pp), 12-19 % smaller files. Real-floater removal, not
  surface erosion.
- Cost: ~5-15 s per object on CPU. No GPU. SAM3 masks (5 min upstream)
  are reused from `pipeline.py`'s cache — no additional masking pass.
- CLI:
  ```
  python _clean_splat.py
  ```
  All defaults are wired for V32 data3. Important flags if porting to
  another scene: `--splat-dir`, `--cameras`, `--sam3-cache`, `--photo-dir`,
  `--output-dir`, `--prompts`, `--max-views` (default 20; Clean-GS
  over-prunes catastrophically with all 140).

> **Note on why v6 not v5:** v5_shcolor was the production pick until
> 2026-06-18, when v6_cleaned was added and visually confirmed by user
> as the better deliverable (cleaner fringe around bottle's base / lobster's
> claws / box edges). v5 remains a useful fallback if Clean-GS over-prunes
> on a future scene — just skip Stage B.

### ⚠ Open thread — possible upgrade: v12 (two-tier base-only scale)

- Directory: [`output/segmented_v32_data3_v9a_fp_v2_splat_v12_twotier/`](output/segmented_v32_data3_v9a_fp_v2_splat_v12_twotier/)
- Same as v5, plus a **base-only** scale filter (`--max-scale-cm-base 0.6
  --scale-filter-band-cm 4.0`, no body filter). On paper this *strictly
  dominates* v5 — same body quality (no body Gaussians touched) +
  ~10% extra desk-halo cleanup at the base.
- **User has not visually compared v5 vs v12 yet.** Default remains v5
  until they confirm. If v12 looks at least as good on the objects in
  SuperSplat side-by-side, swap to v12 as the production pick.

---

## Goal

For a given object on a tabletop scene (e.g. `"white water bottle"`),
extract the subset of the V32 photogrammetry mesh's ~1.3 M triangles that
belong to it. Output Unity-ready GLB + textured OBJ + collider hints for
VR digital-twin grab interaction (Meta Quest 3). Fully automated, text
prompt only, no manual Blender selection.

## Inputs

- **Mesh**: `output/mesh_v32_data3/mesh_v32_data3_openmvs.ply`
  (~805K verts / 1.3 M faces, scaled to metric metres via ToF bounds)
- **Atlas**: `output/mesh_v32_data3/scene_textured0.png` (4096×4096)
- **Photos**: `nerfstudio_data/images/IMG_femto_*.jpg` (~140 views)
- **Poses**: `colmap/dense/transforms.json` (nerfstudio format,
  OpenGL convention, COLMAP-unit translations)
- **Scale**: `colmap/dense/tof_bounds.json` (provides
  `scale_factor_da3_to_colmap`)

## Final pipeline (what works)

The winning path is **rasterized SAM3 voting → spatial cropping** by 2D
footprint, NOT face-by-face mask voting. See [`_spatial_crop.py`](_spatial_crop.py):

1. **SAM3 multi-prompt masking** — `scene_segmenter/sam3_segment.py` runs
   SAM3 once per (view, prompt). One model load, 140 views × N prompts.
   Masks cached as PNGs (one per (view, prompt)).
2. **Mesh ↔ camera setup** — `scene_segmenter/views.py` loads
   `transforms.json`, converts OpenGL c2w → OpenCV w2c, rescales
   translations to metres. Mesh is already metric.
3. **Open3D BVH raycasting per view** —
   `scene_segmenter/extract_faces.py::_build_raycasting_scene` builds the
   BVH once (~1 s), then `_rasterize_view_face_ids` casts rays through
   every pixel for each view (~45 ms / view on CPU). Returns per-pixel
   FaceID + depth. **Replaces** the buggy sparse-sample Z-buffer
   approach from v1–v6.
4. **3-way consensus voting** —
   `accumulate_face_scores_rasterized`. For every visible pixel:
   - Erode SAM3 mask by 3 px → POSITIVE region (confident in-object)
   - Dilate by 3 px → ~ NEGATIVE (confident out-of-object)
   - Band in between → uncertain, no vote
   - Per face: `score = pos_pixels / (pos_pixels + neg_pixels)`
   This is much sharper than the original
   `score = pos_pixels / total_visible_pixels` because the SAM3
   boundary fuzz doesn't drag faces toward 0.5.
5. **Seeds** — faces with `score ≥ 0.7` (3-way evidence threshold).
   These are the "definitely the object" faces.
6. **Largest spatial seed cluster** —
   `_largest_seed_component` (in `_spatial_crop.py`). Voxelize seed
   centroids at 1 cm and keep the largest 26-connected component.
   Drops SAM3 false-positive isolated seeds elsewhere in the scene.
   **NOT mesh-edge adjacency** — OpenMVS meshes have non-manifold edges
   that fragment that graph into uselessly small pieces. Voxel
   adjacency in metric world space is robust.
7. **2D X-Z footprint** — project the cleaned seeds onto the horizontal
   plane at 5 mm resolution → binary occupancy mask of "where the
   object is from above". Dilate by ~5 mm (0.5 cm) as a safety halo.
8. **Spatial crop** — for every face in the original mesh, keep iff:
   - its centroid's `(x, z)` lands in the True region of the footprint
     mask, AND
   - its centroid's `y` is in `[seeds_y_min - 0.5 cm, seeds_y_max + 0.5 cm]`
9. **Export** — OBJ + MTL + PNG + VCG-textured PLY + GLB + collider
   JSON per object. See [`scene_segmenter/exporter.py`](scene_segmenter/exporter.py).

## Production command

```bash
# 1. Run the full segmenter once to generate cached SAM3 masks
# (5 min for 140 views × N prompts on RTX 5070 Ti)
conda run -n sam3 python -m scene_segmenter.pipeline \
  --mesh output/mesh_v32_data3/mesh_v32_data3_openmvs.ply \
  --atlas output/mesh_v32_data3/scene_textured0.png \
  --project-root . \
  --prompts "white water bottle,blue box,red lobster figurine" \
  --output-dir output/segmented_v32_data3_v9a_fp_v2

# 2. Then re-run the spatial crop (~100 s, mostly raycasting)
conda run -n sam3 python _spatial_crop.py --mode a \
  --output-dir output/segmented_v32_data3_v9a_fp_v2 \
  --crop-style footprint \
  --xz-dilate-cm 0.5 \
  --y-margin-cm 0.5 \
  --seed-threshold 0.7 \
  --no-remainder

# 3. Splat crop (v5_shcolor) -- see "Splat extraction" section above
conda run -n sam3 python _spatial_crop_splat.py [args]

# 4. Clean-GS pruning (v6_cleaned, the final production splat)
conda run -n sam3 python _clean_splat.py
```

Cached SAM3 masks live in `<output_dir>/masks/`. Re-runs that change
spatial-crop knobs but reuse the same prompts skip SAM3 entirely.

## Output structure (per prompt)

```
output/segmented_v32_data3_v9a_fp_v2/<prompt_slug>/
├── <prompt>_extracted.obj       ← MeshLab + Unity (best general format)
├── <prompt>_extracted.mtl       ← references the PNG below
├── <prompt>_extracted_mat.png   ← embedded atlas slice
├── <prompt>_extracted.ply       ← VCG textured PLY (MeshLab loads textured)
├── <prompt>_extracted.glb       ← Unity / Blender single-file
└── <prompt>_extracted_collider.json
                                  AABB, OBB, convex-hull volume, surface
                                  area, recommended Unity collider type
                                  (Box / Capsule / Convex), aspect ratios
```

`scene_textured0.png` is also copied alongside for the textured PLY's
`comment TextureFile` directive.

## Configuration cheat sheet

| Knob | File | Default | What it does |
|---|---|---|---|
| `--mask-margin-px` | pipeline.py | 3 | Erode/dilate width for 3-way evidence. Higher = ignore more boundary fuzz. |
| `--seed-threshold` | _spatial_crop.py | 0.7 | Min score to be a seed. Higher = stricter. |
| `--xz-dilate-cm` | _spatial_crop.py | 0.5 | Safety halo around the seed outline. Lower = tighter crop. |
| `--y-margin-cm` | _spatial_crop.py | 0.5 | Vertical extension above/below seed Y extent. Lower = thinner desk slice. |

The defaults are tuned for the data3 tabletop. For bigger/smaller objects
adjust proportionally.

## Versions tried — what failed and why

This is a graveyard so we don't repeat the experiments.

| Ver | Approach | Result | Why it failed |
|---|---|---|---|
| v1 | 3-vertex face sampling + sparse Z-buffer + weighted mean | Swiss-cheese holes everywhere | Background ghost faces bled through the sparse Z-buffer; per-face score was noisy at 0/⅓/⅔/1 |
| v2 | Same but more permissive thresholds | Still holes | Threshold tuning doesn't fix the underlying voting noise |
| v3 | + Laplacian face-score smoothing + `trimesh.fill_holes()` | Marginally better | Smoothing helps interior holes; geometric hole-fill only handles small loops |
| v4 | + aggressive `--min-component-fraction 0.20` | Object barely there | CC filter killed legitimate surface |
| v5 | + dilation 2 rings + smoothing 3 | Still holes per user | Dilation grows boundary but can't recover faces voting rejected |
| v6 | 7-sample face sampling (verts + edge midpoints + centroid) | Marginal | Faces still hit ~1 pixel at typical mesh density; sub-pixel samples can't disambiguate |
| **v7** | **Open3D ray-cast FaceID maps + consensus voting (pixels_in_mask / total_visible)** | Sharp bimodal scores. Fewer faces than v5 — confirmed v1-v5 were including ghost faces! | Voting is now correct; new bottleneck is mesh topology fragmentation |
| v8 | v7 + 3-way evidence (positive / negative / uncertain) | Slight improvement over v7 | Helps marginal-score faces; doesn't fix topology fragmentation |
| v9a (AABB) | Use v7-style score ≥ 0.7 seeds → AABB → crop ALL faces in box | **No more holes — solid objects** ✅ | Desk slab is rectangular, includes ~2 cm of desk past object footprint, especially bad for lobster (claws spread the AABB) |
| v9a_fp | AABB → **2D X-Z footprint mask** + 1 cm dilation, 1 cm Y margin | Tight to object outline, lobster slab now follows body+claws | Some boundary halo remaining |
| **v9a_fp_v2** | v9a_fp + **spatial voxel CC filter on seeds** + 0.5 cm dilation + 0.5 cm Y margin | **Production result.** Footprint is 34-50% of AABB area. ✅ | Final tuning sweet spot |

## Key insights (lessons learned)

1. **Sparse Z-buffers don't work for face visibility.** If you only write
   sample-point depths into the Z-buffer, most pixels never get written,
   and background ghost faces falsely pass the occlusion test. Always
   use a real triangle rasterizer (we use Open3D's BVH raycaster).
2. **Per-face voting always has noise.** Even with rasterized voting,
   the per-triangle score is bimodal but the threshold cutoff still
   fragments connected surfaces. **Switch to spatial cropping** (AABB
   or 2D footprint) once you have rough seeds — no swiss cheese ever.
3. **Mesh-edge adjacency CC filtering is dangerous on OpenMVS output.**
   Non-manifold edges and T-junctions fragment what should be a single
   connected component into hundreds of pieces. Use **spatial voxel
   adjacency** instead.
4. **The PLY format trimesh writes is NOT MeshLab-textured by default.**
   trimesh writes per-vertex `property double s/t`; MeshLab requires
   `comment TextureFile <name>` in the header AND per-face
   `property list uchar float texcoord` (per-wedge UVs, float not
   double). We write our own VCG-compatible PLY in
   [`exporter.py::_write_textured_ply_vcg`](scene_segmenter/exporter.py).
5. **MeshLab's GLB importer drops textures** (known long-standing bug).
   Don't point users at the GLB for MeshLab — give them the OBJ
   (OBJ+MTL+PNG path is the most reliable textured format MeshLab
   handles).
6. **trimesh dedupes identical PNGs by content hash** when exporting OBJ.
   So our snapshot-based diff to identify the just-written texture file
   fails — instead, follow the OBJ's `mtllib` line to find the MTL,
   then the MTL's `map_Kd` line to find the PNG.
7. **TextureVisuals can silently downgrade to ColorVisuals on
   `mesh.submesh(append=True)`.** Always rebind via
   `_ensure_texture_visuals` after slicing or the export comes out
   untextured.

## Coordinate / convention conventions

- **World**: metric metres, OpenCV convention (Y down, +Z forward), but
  rendering tools (MeshLab, Blender) treat the data with Y up — works
  because the V32 capture has the table approximately on Y = 0 and
  objects above.
- **transforms.json**: OpenGL c2w with COLMAP-unit translations.
  Convert via `c2w_cv = c2w_gl @ diag(1,-1,-1,1)`, then divide
  translation by `scale_factor_da3_to_colmap`.
- **UV**: GLB / trimesh use top-left origin (V=0 at top). VCG / OpenGL
  / MeshLab use bottom-left. Our VCG PLY writer flips V → `1 - v`.

## File map

```
scene_segmenter/                # the importable pipeline package
├── pipeline.py                 # CLI orchestrator
├── views.py                    # V32ViewSource: photos + COLMAP poses
├── sam3_segment.py             # SAM3 multi-prompt; caches PNG masks
├── extract_faces.py            # voting math (rasterized + 3-way evidence)
└── exporter.py                 # OBJ/MTL/PNG/PLY/GLB + collider JSON

_spatial_crop.py                # spatial-crop (footprint mode) standalone
SCENE_SEGMENTER_NOTES.md        # THIS FILE
```

## What we deliberately did NOT do

- **Graph cut / MRF post-processing** — both experts recommended it.
  We skipped it because spatial cropping (v9a + v9a_fp) sidesteps the
  fragmentation problem more simply. Reconsider if we hit a scene where
  spatial crops can't be tight enough (e.g., overlapping objects).
- **PyTorch3D / nvdiffrast** — would give GPU rasterization but
  install is painful on Windows. Open3D BVH on CPU is already fast
  enough (~45 ms / view, 6 s total for all 140 views).
- **3D Gaussian Splat segmentation (SAGA, Gaussian Grouping)** — wrong
  output format. We need Unity-ready textured *mesh* faces, not
  segmented Gaussians.
- **UV-atlas-space voting** — interesting alternative (vote in 2D atlas
  space, then select faces by UV coverage), but spatial cropping reached
  acceptable quality first. Documented for future revisit.
- **TSDF mesh as input** — V26 showed TSDF has cleaner topology + 60%
  fewer holes than OpenMVS, but loses the UV atlas. Potential
  improvement: bake atlas onto TSDF geometry, then run segmenter on
  the hybrid.

## Things to try if this isn't good enough later

1. **Lower `--xz-dilate-cm 0`** for surgical-precision crops (may nibble
   object boundary faces).
2. **Multi-prompt mask fusion** — OR together SAM3 masks for related
   prompts ("water bottle, bottle cap, plastic bottle") to fill SAM3
   inconsistency gaps. Code is already prompt-list based; just add
   prompt synonyms and combine masks before voting.
3. **Graph-cut MRF** — `pip install pymaxflow`, formulate face labeling
   as min-cut with data term = score, smoothness term = dihedral angle.
   Both experts recommended this.
4. **Switch input to TSDF** — re-run V26 TSDF, bake the atlas onto it,
   point the segmenter at the hybrid mesh. Expect dramatically less
   topology noise.
5. **Replace AABB Y range with normal-aware desk separation** — desk
   faces have ~vertical-up normals; use that to apply zero-dilation
   footprint to desk faces, large-dilation footprint to object body
   faces.

---

## Related-papers shortlist (from `3d_papers_enhancement.xlsx`, 2026-06-15)

This is a brief inventory of methods the PI flagged for evaluation, with a
one-line description of what each fixes and our current verdict on
whether it belongs in our pipeline.

### 1. NanoGS — 3DGS post-hoc simplification

- **Paper:** [arXiv 2603.16103](https://arxiv.org/abs/2603.16103) · **Code:** [saliteta/NanoGS](https://github.com/saliteta/NanoGS) (pip-install, CC BY-NC 4.0)
- **Method:** Training-free Gaussian merging via k-NN graph + mass-preserving moment matching; merges nearby Gaussian pairs into single primitives while preserving the standard .ply format. Reduces splat count 10×-1000× without retraining or images.
- **Verdict:** ⚙ **Already wired in.** Tested on V32 splat at ratios 0.5 / 0.25 / 0.1 (results in `output/nanogs_v32_data3/`). Best use: compress the full-scene splat for Quest 3 backdrop, or compress per-object splats before bundling for Unity. Recommend keeping this as a documented optional post-step.

### 2. Point Cloud Noise (Wolff et al., Disney) — MVS point-cloud cleanup

- **Paper:** [Disney Research](https://la.disneyresearch.com/wp-content/uploads/Point-Cloud-Noise-and-Outlier-Removal-for-Image-Based-3D-Reconstruction-Paper.pdf) · **Code:** none public; method is patented (US10074160B2)
- **Method:** Filters per-view depth-map point clouds by geometric (signed-distance to other views' surfaces) + photometric (cross-view color std-dev) consistency. Goes between depth-map computation and meshing.
- **Verdict:** ❌ **Not for us.** Implemented clean-room from the paper (`_wolff_filter.py`) and tested on Femto ToF depths — over-filters (kept 0.2% with paper defaults, 1.3% even when loosened) because ToF data is already much cleaner than the noisy MVS clouds the paper targets. Insertion point would also require splicing into OpenMVS or replacing it; payoff doesn't justify the cost. Not the bottleneck on our pipeline.

### 3. Multi-Scale Geometric Consistency (ACMM / ACMMP)

- **Paper:** [arXiv 1904.08103](https://arxiv.org/abs/1904.08103) (CVPR 2019) · **Code:** [GhiXu/ACMM](https://github.com/GhiXu/ACMM), [GhiXu/ACMMP](https://github.com/GhiXu/ACMMP) (TPAMI 2022 follow-up)
- **Method:** A multi-view stereo *depth estimator* (replaces the depth-map step in OpenMVS / COLMAP). Uses adaptive checkerboard sampling + multi-scale geometric consistency to improve depth quality, especially in low-texture regions. Requires CUDA + OpenCV; takes COLMAP poses as input.
- **Verdict:** 🟡 **Possibly useful, but invasive.** Same role as OpenMVS's `ComputeDepthMap` step. Would replace that stage entirely; the rest of OpenMVS (densify + mesh + texture atlas) could still run on top. Likely produces cleaner depths than vanilla OpenMVS for tabletop scenes with low-texture surfaces (like the white bottle's body, the blue box's flat sides). Worth a side experiment IF we revisit upstream mesh quality, but our current pain point isn't depth quality — it's segmentation, which is downstream.

### 4. Clean-GS — semantic-mask floater removal for 3DGS

- **Paper:** [arXiv 2601.00913](https://arxiv.org/abs/2601.00913) · **Code:** [smlab-niser/clean-gs](https://github.com/smlab-niser/clean-gs) (CC BY-NC-SA 4.0)
- **Method:** Removes floaters / background clutter from an existing 3DGS model using **sparse semantic masks (as few as 3 masks from 1% of views)** as guidance. Achieves 60-80% compression while preserving object quality.
- **Verdict:** ✅ **Highly relevant.** This is essentially what we built in `_spatial_crop_splat.py` but more principled — uses semantic masks (we already have 420 SAM3 masks cached!) to guide a per-Gaussian importance score. Could **replace our current SH-color filter + scale filter heuristic**. The author's compression numbers (125 MB → 47 MB) suggest it's substantially more aggressive than our current per-object cropping. Strong candidate for a v14 splat pipeline. **Recommend testing.**

### 5. FreeSplat — feed-forward 3DGS generation

- **Paper:** [arXiv 2405.17958](https://arxiv.org/abs/2405.17958) (NeurIPS 2024) · **Code:** [wangys16/FreeSplat](https://github.com/wangys16/FreeSplat) (no LICENSE file)
- **Method:** Feed-forward network — given 2-30 RGB images + poses, outputs a 3DGS scene in a single forward pass (seconds, no per-scene training). Trained on ScanNet (indoor rooms).
- **Verdict:** ❌ **Not for us.** Replaces splatfacto (the splat *generator*), not anything we have a problem with. Domain mismatch (trained on 3-10 m rooms, we have 1 m tabletops). Splatfacto-trained outputs beat feed-forward methods on quality given time + compute, which we have. Doesn't touch the segmentation step that's our actual bottleneck.

### 6. FreeSplat++ — extension of FreeSplat for long sequences

- **Paper:** [arXiv 2503.22986](https://arxiv.org/abs/2503.22986) (2025) · Same code repo as FreeSplat
- **Method:** Extends FreeSplat to "extremely long" multi-view sequences with a cross-view aggregation framework, pixel-wise triplet fusion, and a weighted floater-removal strategy. Still feed-forward, still indoor / room-scale.
- **Verdict:** ❌ **Same as FreeSplat — not for us.** Slightly more robust to many views but same domain mismatch with our tabletop scans, and still solves the wrong problem (generation, not segmentation).

### 7. StableGS — floater-free 3DGS training framework

- **Paper:** [arXiv 2503.18458](https://arxiv.org/abs/2503.18458) (2025) · **Code:** [MooreThreads/StableGS](https://github.com/MooreThreads/StableGS)
- **Method:** Modifies the 3DGS optimization loop with a Dual Opacity architecture (separate geometric + appearance paths) + DUSt3R depth prior to prevent floaters from forming during training. Bakes anti-floater into the training itself rather than removing them post-hoc.
- **Verdict:** 🟡 **Maybe relevant for V34+.** Would replace our nerfstudio splatfacto step with a different training recipe. If the V32 splat has training-time floaters (e.g. the cream halo around objects) baked in, StableGS could produce a cleaner splat *before* our segmenter ever sees it. Tradeoff: would require re-training the splat (~30 min) with a less mature training pipeline than splatfacto. Worth testing if Clean-GS (post-hoc) isn't enough.

### Ranking by likely usefulness for us

| Rank | Method | Why |
|---|---|---|
| 1 | **Clean-GS** | Targets *our exact problem* (floater removal from existing 3DGS) using inputs we already have (SAM3 masks). Drop-in for the splat pipeline post-crop. |
| 2 | **NanoGS** | Already wired in; useful for Quest 3 deployment compression. |
| 3 | **StableGS** | Could prevent floaters from forming in the first place; needs re-training the splat. |
| 4 | **ACMM / ACMMP** | Better depth → potentially cleaner mesh. Invasive to OpenMVS pipeline. Not blocking. |
| 5 | **Wolff** | Already implemented & tested; doesn't fit our data well. Park it. |
| 6-7 | **FreeSplat / FreeSplat++** | Solve a problem we don't have (slow training, sparse views). Skip. |
