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

Copy via scp / SMB / external drive. After copying, sanity-check the counts:

```powershell
(Get-ChildItem nerfstudio_data\images -Filter *.jpg).Count
# Should match:
(Get-ChildItem nerfstudio_data\depths_femto -Filter *.npy | Where-Object { $_.Name -notlike "*_conf*" }).Count
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
    --iters 50000 `
    --mesh-quality best
```

About 2–3 hours on a 32 GB GPU with a 263-frame dataset (~1 hour splat
training + ~80 min RefineMesh at full image resolution + ~15 min for the
other OpenMVS stages and packaging).

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
| 16 GB (4080, A4000) | 2 | 30000 | high | ~50 min |
| **24 GB (4090, 3090)** | **1** | **50000** | **high** | ~75 min |
| **32 GB+ (5090, A6000, A100)** | **1** | **50000** | **best** | ~3 hours |

**`--mesh-quality` is the single biggest mesh-quality lever** — it controls
OpenMVS's `RefineMesh --resolution-level`. The splat is unaffected by this
flag, the mesh is dramatically affected.

| Setting | Maps to | RefineMesh time | Mesh sharpness |
|---|---|---|---|
| `fast` (default) | res-level 2 (quarter-res images) | ~5 min | Decent |
| `high` | res-level 1 (half-res images) | ~20 min | Noticeably sharper |
| `best` | res-level 0 (full image resolution) | ~80 min | Photographic fidelity |

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
```
