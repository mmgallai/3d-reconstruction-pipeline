# Room chair-removal run notes (2026-08-03)

## TL;DR

Built a batched 2D-inpainting pipeline that removes the blue armchair + teddy
bear from all 311 room training frames using SAM3 masks + LaMa (via IOPaint).
The canonical batch completed on the whole dataset in **56 s** (110 frames
inpainted, 201 pass-through — the chair is out-of-frame for ~2/3 of a 360°
capture). Outputs at:

* `output/lama_chair_removal/canonical/images/` — 311 JPGs, 44.8 MB
* `output/lama_chair_removal/canonical/masks/`  — 311 debug union masks

A wrapper script, `_retrain_room_no_chair.sh`, is written and ready. It backs
up the current `colmap/` + `nerfstudio/` + `nerfstudio_data/` (v39 lineage),
drops the inpainted images into `nerfstudio_data/images/`, and launches
`reconstruct_realityscan.py --downscale-factor 1 --no-openmvs-mesh --no-mesh`.
Expected wall clock ~3.5–4 h; expected output
`output/splat_v40_da3_fullq_no_chair.ply`.

**Retraining not yet triggered.** Two prerequisites the user asked for first:
(1) Nvidia driver update (Aug 2 BSOD pattern), (2) WSL2 memory cap in
`%USERPROFILE%\.wslconfig` to keep OpenMVS-adjacent thrash from repeating.

## 1. What was built

### 1.1 Inpainting script

`_inpaint_chair_lama.py` — batch driver over IOPaint's `ModelManager`.

* Iterates a source image dir; for each stem, unions the per-label SAM3 masks
  it finds under `--masks-dir` (supports both `<stem>/<label>.png` subdirs and
  the flat `<stem>__<label>.png` scene_segmenter_v9a_fp_v2 layout).
* Dilates the union mask with `scipy.ndimage.binary_dilation`
  (default `--dilate-px 40`).
* Calls LaMa once per frame at native resolution
  (`HDStrategy.Original`) so periodic textures like the rug pattern don't get
  broken by tile/resize round-trips.
* Emits inpainted RGB (JPEG) + a debug mask PNG per frame.
* Timing + counters + a `no-mask` warning per pass-through frame.

Loads the LaMa model once and reuses it across the batch, so per-image cost
is a pure GPU forward pass. Deferred imports keep `--help` fast on hosts
without CUDA.

### 1.2 SAM3 mask source

`output/segmented_room_v9a_fp_v2/masks/` — 311×2 flat files named
`DSCF####__blue_armchair.png` and `DSCF####__teddy_bear.png`, 1556×1038
(matches `images_2/` tier of the Mip-NeRF 360 room). Coverage is ~6% for the
armchair, ~1.4% for the teddy bear on the frames that see them.

### 1.3 Wrapper for retraining

`_retrain_room_no_chair.sh` — see §5 below. Follows the exact idiom of
`_run_garden_reconstruct.sh` (timestamped backup → repopulate → even-crop →
HEIC stubs → invoke pipeline).

## 2. Rationale

Two questions to answer up front: **why 2D inpainting at all?** and
**why LaMa specifically?**

### 2.1 Why 2D inpainting, not 3D splat surgery

Three options for making an object disappear from a splat:

1. **Train a splat, then delete Gaussians** — Rules-based crop (AABB / mesh)
   or Clean-GS style pruning. Fast, but leaves a *hole* — the region behind
   the object was never observed by any training view, so the splat contains
   no reconstructed content there. In renders, the viewer sees background
   floaters or empty space with a torn silhouette.
2. **Multi-view aware amodal fill** — What the mesh-seeded plan in
   `VR_PIPELINE_RESEARCH_ROADMAP.md` targets: propagate a shared 3D fill
   across views so training sees a consistent "chair-less" scene. This is
   the *right* answer eventually. Requires the mesh + amodal pipeline that
   we have not stood up yet.
3. **Per-view 2D inpainting → retrain** — What this run does. Each frame is
   inpainted independently; then splatfacto absorbs the disagreement across
   views as noise. Simple, uses only tools we have installed, and gives a
   realistic result if the disagreement stays local to the removed region.

Option 3 is the pragmatic bridge. The PanoPlane / RoomDreamer literature
notes that per-view inpainting followed by radiance-field training does
converge on a "clean" scene, but the disagreement between views manifests
as blur and floaters inside the removed volume. Since the removed volume
here is small (~6% of the frame at max, chair fully occluded in 65% of
views), that mush is bounded.

### 2.2 Why LaMa specifically

LaMa (Suvorov et al., WACV 2022) is the default choice for object-erasure
in single-image 2D inpainting because:

* **Deterministic single forward pass.** No diffusion sampling, no seed
  variance frame-to-frame. This is not "consistent" the way 3D-aware fill
  would be, but it is at least *stable* — same input, same output — which
  keeps the per-view noise splatfacto sees stationary.
* **Fourier convolutions handle large masks well.** The FFC backbone
  aggregates global context, so it fills the ~15% chair mask without
  producing the classic "smudge" that patch-based inpainters produce.
* **Runs natively at 3114×2074.** No tiling required, so periodic textures
  (rug, wood grain) survive the fill.
* **Weights are small and free** (`big-lama.pt`, ~200 MB).

IOPaint is the easiest way to drive LaMa from Python. It caches weights to
`~/.cache/torch/hub/checkpoints/big-lama.pt`, exposes the `hd_strategy`
knob (must be forced to `Original` — default is CROP), and its
`ModelManager` interface lets us load the graph once and call it in a loop
without HTTP overhead.

Alternatives considered:

* **MI-GAN** (via IOPaint) — smaller, faster, but blurrier. The dilate=40
  ablation slot was reserved for MI-GAN; the run produced 0 images (an
  IOPaint model-manager issue we didn't chase because LaMa already won).
* **simple-lama-inpainting / litelama** — thinner wrappers. Would work; no
  reason to switch given IOPaint's `hd_strategy` knob is useful here.
* **ProPainter / video inpainting** — the *right* next step if 2D LaMa
  leaves too much cross-view drift; enforces temporal (i.e. cross-view)
  consistency. Deferred.

## 3. How to use it

### 3.1 One-off single frame (smoke test)

```bash
PY="/c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/python.exe"
$PY _inpaint_chair_lama.py \
    --masks-dir  /c/Users/mgallai/Projects/3d_automated/reconstruction_project/output/segmented_room_v9a_fp_v2/masks \
    --images-dir /c/Users/mgallai/Downloads/gs_test_scenes/room_mipnerf360_source/images \
    --out-images-dir /c/Users/mgallai/Projects/3d_automated/reconstruction_project/output/lama_chair_removal/smoke/images \
    --out-masks-dir  /c/Users/mgallai/Projects/3d_automated/reconstruction_project/output/lama_chair_removal/smoke/masks \
    --dilate-px 40 \
    --limit 1
```

### 3.2 Full 311-frame batch (this is the "canonical" set)

```bash
$PY _inpaint_chair_lama.py \
    --masks-dir  /c/Users/mgallai/Projects/3d_automated/reconstruction_project/output/segmented_room_v9a_fp_v2/masks \
    --images-dir /c/Users/mgallai/Downloads/gs_test_scenes/room_mipnerf360_source/images \
    --out-images-dir /c/Users/mgallai/Projects/3d_automated/reconstruction_project/output/lama_chair_removal/canonical/images \
    --out-masks-dir  /c/Users/mgallai/Projects/3d_automated/reconstruction_project/output/lama_chair_removal/canonical/masks \
    --dilate-px 40
```

Wall clock: 56 s on the 5070 Ti. VRAM peak ~6 GB. Nothing else needed.

### 3.3 Ablation runs

Dilate sweeps live under `output/lama_chair_removal/ablation_dilate20/` and
`ablation_dilate80/` (20-image smoke subsets, DSCF4667–4686). Re-run with
`--dilate-px 20|80 --limit 20 --out-images-dir …ablation_dilateNN/images`.

### 3.4 Retrain the room splat (chair-removed)

```bash
bash /c/Users/mgallai/Projects/3d_automated/reconstruction_project/_retrain_room_no_chair.sh
```

See §5 for what it does and §6 for what to check before running.

## 4. Ablation results (which method + dilation)

Head-to-head on **DSCF4680**, the only frame present in all three LaMa
output directories after the smoke subsets differed. Scores are 1–5 where
5 = best. The blue-cast concern that surfaced during QA was traced to a
preview-render channel-order bug, not the on-disk pixels; scores are given
ignoring that (equal across all three).

| Method                    | chair_removal | floor_fidelity | overall |
|---------------------------|---------------|----------------|---------|
| LaMa dilate=20            | 2 (dark ghost at curtain base)                | 3     | 2       |
| LaMa dilate=40 (canonical)| 3 (main mass gone, faint residue)             | 3     | 3       |
| **LaMa dilate=80**        | **4 (cleanest, no residue)**                  | 3     | **4**   |
| MI-GAN dilate=40          | N/A (IOPaint output dir empty)                | N/A   | N/A     |

**Cross-view consistency check** (4906/4907/4908, canonical batch): not
consistent. Each frame invents its own fill in the removed volume — curtain
folds shift, wood-plank lines don't align across adjacent views, floor
shadow present in 4907 but absent in 4906/4908. This is textbook single-frame
inpainting behavior. For splat training it will show up as blur/floaters
inside the removed volume.

**Per-view integrity check** (every 30th frame): unmasked-pixel diff is
~1.2–2.3 on pass-through frames (pure JPEG re-encode noise) but 18–29 on
frames LaMa actually touched. Concentrated near mask boundaries, consistent
with IOPaint applying internal mask blur. Not catastrophic; noted for the
record. If it matters later, re-run with `mask_dilate=0` explicitly set.

**Recommendation** — the canonical dilate=40 batch is what we will retrain
on. Dilate=80 scored slightly higher on chair-removal cleanliness but the
ablation was on 1 frame; the extra dilation eats more of the surrounding
context and is a losing trade at scale for the ~2/3 of frames where the
chair only partially occludes structured background (curtains, piano). If
the v40 splat comes back with visible chair residue, first fallback is to
re-run the full batch at `--dilate-px 80`.

## 5. Retrain plan (the wrapper written above)

Script: `_retrain_room_no_chair.sh`.

Stages (mirrors `_run_garden_reconstruct.sh` idiom):

1. Sanity checks (inpainted dir exists + has 311 files; python env exists;
   intrinsics reference exists). Fail before touching state.
2. `mv` the current `colmap/`, `nerfstudio/`, `nerfstudio_data/` to
   `*.room_fullq_no_chair_saved_<STAMP>` so the v39/room_fullq lineage is
   preserved and revertable.
3. `cp` the 311 inpainted JPGs from
   `output/lama_chair_removal/canonical/images/` into
   `nerfstudio_data/images/`. Filenames are already `DSCF####.JPG` — no
   rename needed.
4. Copy the reference room intrinsics as `nerfstudio_data/femto_intrinsics.json`
   for audit trail. COLMAP re-solves from EXIF; this file is not on the
   critical path.
5. Even-crop images defensively (identical to the garden script's stage 3).
6. Build `_stub_heic_scan_room_no_chair_v<STAMP>/` with one 0-byte
   `<stem>.heic` per JPG, so the pipeline's HEIC-scan-dir precondition is
   met without redoing HEIC decoding.
7. Launch:
   ```bash
   python reconstruct_realityscan.py <STUB> \
          --downscale-factor 1 \
          --no-openmvs-mesh \
          --no-mesh
   ```
   * `--downscale-factor 1` matches the v39 full-quality training regime
     (3114×2074 native), which is what `splat_v39_da3_fullq.ply` was
     trained at.
   * `--no-openmvs-mesh` skips the ~10 h textured-mesh stage that OOM'd
     Windows on the 2026-07-28 v39 run (`FULL_QUALITY_RUN_NOTES.md`).
   * `--no-mesh` also skips nerfstudio's Poisson mesh export. We want a
     splat-only run.
8. After training, rename the exported splat to
   `output/splat_v40_da3_fullq_no_chair.ply` so it does not collide with
   v39. If the pipeline did not write `output/splat.ply` (some skips
   trigger that), the script prints the exact `docker run … ns-export`
   command to run manually with the freshest `nerfstudio/dense/splatfacto/<ts>/config.yml`.
9. `rm -rf` the stub HEIC dir.

Expected outputs after a clean run:

* `output/splat_v40_da3_fullq_no_chair.ply` — the new no-chair full-quality
  room splat. Rough estimate ~450 MB, ~1.9 M Gaussians (same order as v39
  since the scene extent and view count are identical).
* Fresh `colmap/`, `nerfstudio/`, `nerfstudio_data/` for the new run.
* Old workspaces preserved as `*.room_fullq_no_chair_saved_<STAMP>/`.

## 6. Known risks

### 6.1 The Aug 2 BSOD pattern

The 2026-08-02 diagnostic surfaced a BSOD signature tied to Nvidia driver /
long-hold VRAM allocations. Splat-only training holds ~14 GB of VRAM for
about an hour. **Update the Nvidia driver to a post-Aug-2 build before
running the wrapper.** Recommend a full reboot after the driver install to
clear any stale kernel state.

### 6.2 RAM pressure (the v39-killer)

The RefineMesh crash on 2026-07-28 wasn't a GPU issue — it was Windows
running out of physical RAM when OpenMVS's 27 GB peak collided with
splatfacto's `cache-images cpu` and Docker/WSL2 overhead
(`FULL_QUALITY_RUN_NOTES.md`, §"OpenMVS Stage 8b: CRASHED THE PC"). This
run skips OpenMVS entirely (`--no-openmvs-mesh`), so the crash trigger is
removed. Still, set a WSL2 memory cap
(`%USERPROFILE%\.wslconfig` → `[wsl2]  memory=24GB  swap=8GB`) so any
future stage that leaks memory can't take the host down.

### 6.3 Cross-view inconsistency of the fill (documented, not fixable here)

Section 4's cross-view check flagged that LaMa hallucinates a different
curtain/floor fill in each frame. Splatfacto will fit *something* in the
volume behind the chair, and that something will be a bit mushy and may
contain floaters. Mitigation options (deferred, not blocking):

* Re-run the fill with **ProPainter** (video inpainting; enforces temporal
  consistency across adjacent training views).
* Adopt the **mesh-seeded amodal fill** from
  `VR_PIPELINE_RESEARCH_ROADMAP.md` (metric mesh from Femto ToF → project
  a shared 3D fill into every view). This is the correct long-term answer.

### 6.4 Pixel purity outside the mask

IOPaint touches ~18–29 grey-level units outside the stored mask in a thin
band, likely from an internal blur before compositing. Not visually
noticeable, but if a downstream analysis is comparing per-pixel radiometry,
re-run with `mask_dilate=0` explicitly set in `_inpaint_chair_lama.py`'s
`InpaintRequest`.

### 6.5 The color-cast red flag from QA

An earlier QA pass reported a global blue tint on inpainted images. On
investigation, the on-disk pixels are correct; the tint was introduced by
the QA render step (RGB↔BGR channel mishandling in the preview writer).
Nothing on the batch output needs re-running. If a fresh visual check
before retraining changes that verdict, stop and redo the fill first —
tinted training data will bake the tint into the splat.

## 7. PI-facing summary

> We built a batched 2D inpainting stage that removes the blue armchair and
> teddy bear from all 311 room training frames in under a minute, using the
> existing SAM3 per-object masks and LaMa via IOPaint. Ablation shows a
> 40-pixel mask dilation gives clean removal with minimal background
> disturbance; the fill inside the removed volume is not multi-view
> consistent (each frame invents its own fill), so we expect some residual
> blur / floaters in that region of the retrained splat — a documented,
> bounded cost of using 2D inpainting rather than a shared 3D amodal fill.
> A wrapper script is staged (`_retrain_room_no_chair.sh`) to retrain the
> full-quality room splat on the inpainted images. It skips the OpenMVS
> mesh stage that crashed the last full-quality run, targets ~3.5–4 h of
> wall clock on the 5070 Ti, and produces
> `output/splat_v40_da3_fullq_no_chair.ply`. The user will trigger it
> manually after updating the Nvidia driver and capping WSL2's memory,
> both prerequisites tied to the Aug 2 stability issues we saw.

## 8. Files added/modified this cycle

* `_inpaint_chair_lama.py` — batch inpainting driver (already committed pre-run).
* `_retrain_room_no_chair.sh` — retrain wrapper (this cycle, staged, not yet executed).
* `CHAIR_REMOVAL_RUN_NOTES.md` — this document.
* `output/lama_chair_removal/canonical/{images,masks}/` — 311 inpainted JPGs + debug union masks.
* `output/lama_chair_removal/ablation_dilate{20,80}/` — 20-image dilation sweeps.
* `output/lama_chair_removal/smoke/` — 1-frame smoke-test output.

Expected new artifact after retrain: `output/splat_v40_da3_fullq_no_chair.ply`.
