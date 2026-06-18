# Run guide — Femto Mega → 3D reconstruction pipeline

End-to-end setup + run instructions. Follow top to bottom on a fresh machine,
or skip to **§ Run the pipeline** if you already have the repo + a captured
dataset.

## Repository

- **GitHub:** https://github.com/mmgallai/3d-reconstruction-pipeline
- **Recommended branch / tag:** **`v32`** — the latest, has vendor-intrinsic
  COLMAP priors + all V24 capture upgrades (coverage minimap, motion-auto,
  IR save, shutter sound, range zones)
- See [VERSIONS.md](VERSIONS.md) for the full version history (v6 → v32) and
  what changed in each
- See [CAPTURE_GUIDE.md](CAPTURE_GUIDE.md) for capture-protocol details and
  in-app capture aids
- See [FUTURE_WORK.md](FUTURE_WORK.md) for ideas not yet implemented

---

## Hardware requirements

| Component | Minimum | Recommended (full capacity) |
|---|---|---|
| GPU | 8 GB VRAM | **24 GB+** (RTX 4090, 5090, A6000, A100) |
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB+ |
| Disk | 50 GB free | 200 GB free (intermediate + outputs) |
| OS | Win 10/11, Linux | Win 11 with WSL2 GPU passthrough |
| Camera | Orbbec Femto Mega over USB-3 | — |

The pipeline runs all CUDA stages (COLMAP, OpenMVS, nerfstudio splatfacto,
TSDF fusion) inside Docker. The host machine needs Docker with GPU access.

---

## One-time setup

### 1. Clone the repo

```powershell
git clone https://github.com/mmgallai/3d-reconstruction-pipeline.git
cd 3d-reconstruction-pipeline
git checkout v32
```

### 2. Install Docker Desktop + GPU passthrough

- Download Docker Desktop: https://www.docker.com/products/docker-desktop/
- Enable WSL2 backend in settings → Resources → WSL Integration
- Install the NVIDIA Container Toolkit so containers can see the GPU:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Verify with: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`

### 3. Install Miniconda + create the `da3` env

The `da3` conda env runs all host-side Python (capture script, DA3 init,
inspection, scale calibration). The training/mesh stages happen in Docker.

```powershell
# Miniconda: https://docs.conda.io/projects/miniconda/en/latest/
conda create -n da3 python=3.12 -y
conda activate da3
pip install opencv-python numpy trimesh open3d xatlas pillow torch
```

If you want to capture from the Femto on this machine (not just process data
copied over), also install pyorbbecsdk:

```powershell
# Download the wheel that matches your Python version + OS from:
#   https://github.com/orbbec/pyorbbecsdk/releases
# Example: pyorbbecsdk2-2.0.18-cp312-cp312-win_amd64.whl
pip install path\to\pyorbbecsdk2-*.whl
```

### 4. First pipeline run builds the Docker image (~10 min)

The first time you run `reconstruct_realityscan.py` it builds the
`nerfstudio-blackwell` image from the local Dockerfile. Subsequent runs reuse
the cached image. Other images (`colmap/colmap`, `yeicor/openmvs-ubuntu-cuda`)
are pulled from Docker Hub on demand.

---

## Input files — what the pipeline needs

The pipeline expects the captured dataset to live under `nerfstudio_data/`
inside the repo root. Two ways to fill that folder.

### Option A — Capture fresh on this machine

Plug in the Femto Mega over USB-3, then:

```powershell
conda activate da3
python capture_femto.py
```

A live preview window opens. Recommended workflow:
1. Stand at orbit start, point at subject — verify GREEN crosshair (depth in
   0.5–3 m sweet spot)
2. Press **C** to show the coverage minimap (top-right corner)
3. Press **M** to enable motion-auto (1 frame per 5 cm of camera motion)
4. Walk a slow orbit. High beep on each save. Mid/low beep means you've
   drifted out of the sweet spot — adjust distance
5. Aim for **120–200 frames** at minimum, ideally with 2–3 orbits at
   different heights
6. Press **ESC** to finish — this writes `femto_intrinsics.json`

Full hotkey reference: top of `capture_femto.py` docstring.

### Option B — Copy data from another machine

Three locations to copy. Layout must match exactly:

| Path (relative to repo root) | What | How many files | Approx size |
|---|---|---|---|
| `nerfstudio_data/images/` | RGB JPEGs named `IMG_femto_NNNN.jpg` (1920×1080, q92) | 120-300 | 200-300 MB |
| `nerfstudio_data/depths_femto/` | `IMG_femto_NNNN.npy` (float32 depth, metres, 1920×1080) + `IMG_femto_NNNN_conf.npy` (uint8 valid mask) + optional `IMG_femto_NNNN_ir.png` (uint16 IR at 1024×1024) | 2× JPEG count (+IR if present) | 2–4 GB |
| `nerfstudio_data/femto_intrinsics.json` | Vendor-calibrated intrinsics (fx, fy, cx, cy + color↔depth extrinsics) — written by `capture_femto.py` at end of session | 1 | < 2 KB |

**Total dataset size: roughly 3 GB for a 263-frame capture.**

### Recommended transfer methods (fastest → slowest)

The 3 GB dataset is small enough that almost any method works in under
30 minutes. Pick whichever is most convenient.

**Step 1 (always): compress to a single archive on the source machine**

Putting ~800 small files into one archive transfers about 10× faster than
copying file-by-file over any network or USB protocol. From the repo root
on the source machine:

```powershell
# Windows PowerShell
Compress-Archive -Path nerfstudio_data -DestinationPath nerfstudio_data.zip
# Result: roughly 2.5 GB single file (JPGs are already compressed, .npy
# files compress moderately well)
```

Or with 7-Zip for slightly better compression + much faster on huge file
counts:
```powershell
7z a -mx=5 nerfstudio_data.7z nerfstudio_data
```

**Step 2: ship the archive**

| Method | Setup needed | Transfer time (3 GB) | When to use |
|---|---|---|---|
| **USB-3 external SSD / stick** | Format NTFS or exFAT | 30–60 sec | Both machines are physically nearby |
| **Direct LAN SMB share** | Right-click folder → Properties → Share | 1–2 min on gigabit LAN | Both on the same Wi-Fi / Ethernet |
| **`scp` / `rsync` over LAN** | SSH server enabled on dest | 1–2 min | Linux dest or Win11 with OpenSSH |
| **OneDrive / Google Drive** | None (assuming logged-in) | 5–15 min upload + same to download | No physical access; cross-network |
| **WeTransfer / Dropbox Transfer** | None | 10–30 min total | Same as above, no cloud account on dest |

**Step 3: extract on the destination**

After copying the archive into the new repo's root:
```powershell
Expand-Archive -Path nerfstudio_data.zip -DestinationPath .
# or: 7z x nerfstudio_data.7z
```

**Sanity-check after extract:**

```powershell
(Get-ChildItem nerfstudio_data\images -Filter *.jpg).Count
# Should equal:
(Get-ChildItem nerfstudio_data\depths_femto -Filter *.npy | Where-Object { $_.Name -notlike "*_conf*" }).Count
# Should also have:
Test-Path nerfstudio_data\femto_intrinsics.json   # → True
```

The current 263-frame dataset on **this** machine lives at:
```
C:\Users\mgallai\Projects\3d_automated\reconstruction_project\nerfstudio_data\
```

---

## Run the pipeline

### Standard run (V23/V24 default — 2× downscale, 30K iters)

```powershell
conda activate da3
python reconstruct_realityscan.py --skip-mvs --use-femto-depth
```

About 30 min on a single RTX 4080 with a 130-frame dataset.

### Full-capacity run (recommended for 24 GB+ GPU)

```powershell
conda activate da3
python reconstruct_realityscan.py `
    --skip-mvs `
    --use-femto-depth `
    --downscale-factor 1 `
    --iters 50000
```

About 50–75 min on a 24-32 GB GPU with a 263-frame dataset (~50 min splat
training at full res + ~10 min OpenMVS + ~5 min packaging).

If your subject is room-scale or strongly textured, also add
`--mesh-quality high` (or `best`); skip it for desk-scale scenes (`fast`
default gives identical visual output, in our test).

### Mesh-only re-run (after splat training already finished)

If the splat is done and you only want to upgrade the mesh quality (e.g.
re-run today's V32 run at higher mesh quality), skip training:

```powershell
python reconstruct_realityscan.py `
    --skip-mvs `
    --use-femto-depth `
    --skip-training `
    --mesh-quality best
```

`--mesh-quality high` or `best` forces RefineMesh to re-run even if a
previous (lower-quality) `scene_mesh_refine.ply` is cached.

### Flag reference

| Flag | Effect | Notes |
|---|---|---|
| `--use-femto-depth` | Use Femto ToF depth maps as Gaussian splat init source (V23 path / Paradigm A) | Required for any Femto-based run. Without it, the pipeline tries DA3 monocular depth |
| `--skip-mvs` | Skip OpenMVS DensifyPointCloud → use sparse COLMAP points (or hybrid init) instead | Faster; recommended for V23 path |
| `--downscale-factor N` | Train splat at 1/N image resolution. **N=1 full res, N=2 half, N=4 quarter** | Higher N = less VRAM, faster, lower quality. Default 2 |
| `--iters N` | Train for N iterations. Default 30000. **50000+ on strong GPUs** | More iters = sharper but diminishing returns past 30-50K |
| `--mesh-quality {fast,high,best}` | OpenMVS RefineMesh resolution. fast=res-2 (default, ~5 min), high=res-1 (~20 min), **best=res-0 full-image-resolution (~80 min)** | Affects mesh only, splat unchanged. high / best force RefineMesh to re-run even if a cached output exists |
| `--train-method NAME` | `splatfacto-big` (default), `splatfacto`, `splatfacto-tof` (depth-supervised, V30), `ags-mesh` (V25 regression) | V30 visually worse than V24 on tested scenes |
| `--quick` | 1000 iters @ 4× downscale + skip MVS | 3-min sanity-check, low-quality output |
| `--skip-training` | Skip splat training, jump to export from existing checkpoint | Useful for re-running mesh/export stages |
| `--resume` | Continue training from last checkpoint | Useful after a crash |
| `--no-splat` | Skip splat export (mesh only) | |
| `--no-mesh` | Skip Poisson mesh (splat only) | |

### Recommended config per GPU class

| GPU VRAM | `--downscale-factor` | `--iters` | `--mesh-quality` | Total time (263 frames) |
|---|---|---|---|---|
| 8 GB (3070, 4060) | 4 | 20000 | fast | ~25 min |
| 12 GB (3060, 4070) | 2 | 30000 | fast | ~30 min |
| 16 GB (4080, A4000) | 2 | 30000 | fast | ~35 min |
| **24 GB (4090, 3090)** | **1** | **50000** | **fast** (desk) / **high** (room) | ~50–75 min |
| **32 GB+ (5090, A6000, A100)** | **1** | **50000** | **fast** (desk) / **best** (large/textured scene) | ~50 min – 3 hours |

**Why `fast` even on a strong GPU:** on the 263-frame desk capture,
`fast` and `best` mesh outputs were visually identical despite `best`
producing +5% verts/faces. The longer RefineMesh helps on **larger or
more textured scenes**, not on tight desk-scale captures. Default to
`fast` and only escalate if the mesh visibly lacks detail in regions
where you expect surface texture.

**`--mesh-quality` is the single biggest mesh-quality lever** — it controls
OpenMVS's `RefineMesh --resolution-level`. The splat is unaffected by this
flag, the mesh is affected.

| Setting | Maps to | RefineMesh time | Mesh sharpness |
|---|---|---|---|
| `fast` (default) | res-level 2 (quarter-res images) | ~5 min | Decent |
| `high` | res-level 1 (half-res images) | ~20 min | Noticeably sharper (on larger scenes) |
| `best` | res-level 0 (full image resolution) | ~80 min on big meshes / ~18 min on desk-scale | Photographic fidelity — when it helps |

**Test result on the 263-frame desk capture:** `fast` and `best` produced
**visually identical** outputs (the +5% vert/face count in the `best`
stats didn't translate to perceptible detail). For desk-sized scenes
`fast` is enough. Try `high` or `best` only on:
  - Larger scenes (whole room, building exterior)
  - Strongly textured subjects (carved wood, fabric)
  - When you've already maxed `--iters` and want one more knob

## Cleaning floaters after training

The pipeline's default prune (opacity > 0.15 + connected-component
cleanup) removes most haze, but reflective surfaces (glossy monitors,
windows, mirrors) can produce stubborn blobs that pass the default
filter. The `aggressive_prune.py` script (added in v32) applies three
extra filters without retraining:

```powershell
python aggressive_prune.py output/splat_vN_noinit_pruned.ply --use-aabb
# Outputs: splat_vN_noinit_pruned_clean.ply
# Filters:
#   1. Drop top 5% largest-scale Gaussians (default --scale-pct 95)
#   2. Stricter opacity (default --opacity-min 0.30)
#   3. Clip to scene AABB from tof_bounds.json (--use-aabb)
```

**Caveat from V32 testing:** scale-based filtering catches some floaters
but does NOT remove view-dependent monitor reflections that the trainer
spawned to satisfy the photometric loss across moving reflections. Those
blobs are inside the scene volume, at normal-ish scale, and at high
opacity — visually indistinguishable from real geometry by post-process
filters alone. The real fixes for reflection blobs are:
  - Recapture with the monitors off / draped (best ROI)
  - SAM-based screen masking + retrain (heavier infra work)

---

## Per-object extraction (Unity / Quest 3 deliverables)

Once you have the scene-level splat + textured mesh from the main pipeline,
the **scene segmenter** crops Unity-ready per-object splats + meshes by
text prompt. Fully automated, no manual Blender work. Used for the VR
digital-twin grab-interaction targets.

### Current canonical chain (V32 data3)

```
SAM3 masks (cached)                   <- pipeline.py (~5 min, GPU)
   |
   v
v9a_fp_v2 mesh per object             <- _spatial_crop.py     (~100 s, CPU)
   |
   v
v5_shcolor splat per object           <- _spatial_crop_splat.py (~30 s, CPU)
   |
   v
v6_cleaned splat per object  (FINAL)  <- _clean_splat.py      (~5-15 s/object, CPU)
```

Each stage feeds the next; outputs land in
`output/segmented_<scene>_v9a_fp_v2*/`. Deep design notes + version
graveyard in [SCENE_SEGMENTER_NOTES.md](SCENE_SEGMENTER_NOTES.md).

### Production commands (V32 data3 defaults)

```powershell
# 1. SAM3 voting + voxel-CC seeds -> v9a_fp_v2 per-object mesh
conda run -n sam3 python -m scene_segmenter.pipeline `
  --mesh output/mesh_v32_data3/mesh_v32_data3_openmvs.ply `
  --atlas output/mesh_v32_data3/scene_textured0.png `
  --project-root . `
  --prompts "white water bottle,blue box,red lobster figurine" `
  --output-dir output/segmented_v32_data3_v9a_fp_v2

conda run -n sam3 python _spatial_crop.py --mode a `
  --output-dir output/segmented_v32_data3_v9a_fp_v2 `
  --crop-style footprint --xz-dilate-cm 0.5 --y-margin-cm 0.5 `
  --seed-threshold 0.7 --no-remainder

# 2. Footprint crop on the scene splat -> v5_shcolor per-object splat
conda run -n sam3 python _spatial_crop_splat.py `
  --splat output/splat_v32_data3_noinit_pruned.ply `
  --dataparser nerfstudio/dense/splatfacto/<run>/dataparser_transforms.json `
  --segmented-dir output/segmented_v32_data3_v9a_fp_v2 `
  --prompts "white water bottle,blue box,red lobster figurine" `
  --output-dir output/segmented_v32_data3_v9a_fp_v2_splat_v5_shcolor `
  --crop-style footprint --xz-dilate-cm 0.5 --y-margin-cm 0.5 `
  --exclude-color "124,113,71" --color-tolerance 18 --sh-view-dir "0,0,1"

# 3. Clean-GS pruning on the v5_shcolor crops -> v6_cleaned (FINAL)
conda run -n sam3 python _clean_splat.py
```

`_clean_splat.py` has every V32 data3 default baked in — just `python _clean_splat.py`
reproduces the production output. For another scene, override `--splat-dir`,
`--cameras`, `--sam3-cache`, `--photo-dir`, `--output-dir`, `--prompts`.

### Per-object outputs

```
output/segmented_<scene>_v9a_fp_v2/<prompt_slug>/
├── <prompt>_extracted.obj/.mtl/.png    <- Unity (OBJ+MTL+PNG; most reliable)
├── <prompt>_extracted.ply              <- VCG textured PLY (MeshLab)
├── <prompt>_extracted.glb              <- Single-file Unity / Blender
└── <prompt>_extracted_collider.json    <- AABB / OBB / convex hull + recommended Unity collider

output/segmented_<scene>_v9a_fp_v2_splat_v6_cleaned/
└── <prompt>_splat.ply                  <- Final per-object Gaussian splat (FINAL)
```

The per-object splat is in the same nerfstudio PLY schema as the scene
splat (positions / normals / SH coefficients / opacity / scales /
rotations) — drop-in loadable by SuperSplat, Aras-P Unity splat plugin,
Three.js gsplat viewers, and Babylon's `GaussianSplattingMesh`.

### Why v6_cleaned beats v5_shcolor

Measured on V32 data3: v6 prunes 12–19 % of Gaussians per object,
**bounding box unchanged**, slight rise in per-Gaussian opacity confidence
(lobster +2.5 pp), and 12–19 % smaller files. The prunes are real floaters
(the fuzzy fringe around the bottle base / lobster claws / box edges),
not surface erosion. v5 remains a useful fallback if Clean-GS over-prunes
on a new scene — just skip stage 3.

---

## Pipeline stages — what happens when you run it

1. **Stage 1 — HEIC→JPEG** (skipped when `--use-femto-depth`)
2. **Stage 2 — COLMAP SfM** — feature extraction, exhaustive matching,
   sparse reconstruction, image undistortion. **Vendor intrinsics from
   `femto_intrinsics.json` are passed in V32+.** ~5-15 min depending on
   frame count.
3. **Stage 4 — Femto ToF init** — back-projects depth maps through COLMAP
   poses → `colmap/dense/tof_init.ply` (500K colored points). ~30 sec.
4. **Stage 5 — Convert COLMAP model → transforms.json** for nerfstudio
5. **Stage 6 — Pre-generate `images_N/`** downscaled image set
6. **Stage 7 — Training** — splatfacto-big trains a Gaussian Splat from
   `tof_init.ply` init. **The longest stage** (~18 min at N=2/30K iters,
   ~40 min at N=1/50K iters)
7. **Stage 8 — 3DGS Poisson export** (expected to fail — 3DGS lacks normals)
8. **Stage 8b — OpenMVS textured mesh** — DensifyPointCloud → ReconstructMesh
   → RefineMesh → TextureMesh (8K UV atlas). ~8 min the first time, ~2 min
   on reruns thanks to caching.
9. **Stage 9 — Export splat as .ply**
10. **Stage 10 — Packaging + LOD decimation + ambient-occlusion bake**

---

## Outputs — where the deliverables land

After a successful run, look in `output/`:

| Path | What | Size |
|---|---|---|
| `output/splat_vN_noinit.ply` | Raw trained splat (all Gaussians) | 200-600 MB |
| `output/splat_vN_noinit_pruned.ply` | After opacity + connected-component prune (typical 50-70% retention) | 50-300 MB |
| `output/mesh_vN/mesh_vN_openmvs.ply` | Textured mesh (HIGH LOD, 600K-1M verts) | 50-100 MB |
| `output/mesh_vN/mesh_vN_openmvs_ao.ply` | HIGH mesh + ambient-occlusion bake | 25-50 MB |
| `output/mesh_vN/mesh_vN_openmvs_mid.ply` + `_mid_ao.ply` | Mid LOD (500K faces) | 9-12 MB |
| `output/mesh_vN/mesh_vN_openmvs_low.ply` + `_low_ao.ply` | Low LOD (100K faces) | 1.8-2.1 MB |
| `output/mesh_vN/scene_textured0.png` | 8K UV-atlas texture | 30-35 MB |

The version number `N` is auto-incremented from existing splats in `output/`.

### Verifying the result

```powershell
# Splat stats
conda run -n da3 python inspect_v23.py
```

What to look for in a **good** Femto run:
- Splat pruned Gaussians: **400K – 1.5M**
- Opacity median: **> 0.9** (low → ambiguous geometry / recapture)
- BB diagonal (after scale-calibrate): should match real scene size in metres
- Mesh boundary (open) edges: lower the better; expect 100-400K for a 1M-face mesh
- Mesh largest connected component: > 90% at MID / LOW LOD; lower = fragmented

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `docker: not found` or `error during connect` | Docker Desktop not running | Start Docker Desktop from Start menu, wait ~30 sec |
| `pip install pyorbbecsdk` fails | Not on PyPI | Download wheel matching your Python from https://github.com/orbbec/pyorbbecsdk/releases |
| COLMAP registers fewer than 80% of frames | Capture too fast / poor overlap / textureless | Re-shoot following [CAPTURE_GUIDE.md](CAPTURE_GUIDE.md). Coverage minimap helps. |
| Splat opacity median < 0.7 | Training found ambiguous geometry — usually a capture quality issue | Re-capture with diffuse lighting, slower orbit, more frames |
| `ns-export gaussian-splat` fails with `No module named 'splat_tof'` | Forgot the plugin install when using `--train-method splatfacto-tof` | The pipeline auto-installs it; verify `splat_tof/` exists in repo root |
| `std::out_of_range _Map_base::at` in OpenMVS | TextureMesh choking on a mesh with degenerate topology | Use `bake_tsdf_atlas.py` as a fallback texturing path |
| OOM during training | GPU too small for current downscale | Bump `--downscale-factor` from 1 → 2 or 2 → 4 |

For anything else, check `reconstruct_v*.log` and `reconstruct_v*.err.log`
in the project root — those are the captured stdout / stderr of the most
recent pipeline run.

---

## Quick reference — common commands

```powershell
# Capture
conda activate da3
python capture_femto.py

# Standard run
python reconstruct_realityscan.py --skip-mvs --use-femto-depth

# Full-capacity run on a strong GPU
python reconstruct_realityscan.py --skip-mvs --use-femto-depth --downscale-factor 1 --iters 50000

# Quick sanity-check (~3 min)
python reconstruct_realityscan.py --use-femto-depth --quick

# Resume an interrupted training
python reconstruct_realityscan.py --skip-mvs --use-femto-depth --resume

# Inspect splat + mesh stats
conda run -n da3 python inspect_v23.py

# TSDF-only mesh (no COLMAP MVS) — fast scanning-style output
conda run -n da3 python tsdf_fusion_femto.py

# Custom UV-atlas baker for TSDF meshes (unblocks the V27 attempt)
conda run -n da3 python bake_tsdf_atlas.py

# Per-object extraction (Unity / Quest 3) -- after the main pipeline finishes
conda run -n sam3 python -m scene_segmenter.pipeline --prompts "white water bottle,..."
conda run -n sam3 python _spatial_crop.py --mode a --crop-style footprint
conda run -n sam3 python _spatial_crop_splat.py [...args]
conda run -n sam3 python _clean_splat.py    # final v6_cleaned splats
```
