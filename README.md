# 3D Reconstruction Pipeline

Turns a folder of photos into a **Gaussian Splat** (`.ply`) and optionally a **textured mesh**.

```
photos/  →  [COLMAP SfM]  →  [SAM3 masking]  →  [Nerfstudio splatfacto]  →  splat.ply
```

---

## Setup on a new machine

Follow these steps to get the pipeline running on a fresh device.

### 1. Install host-level prerequisites
- **Windows 11**: WSL2 + Docker Desktop (with GPU support enabled)
- **Linux**: Docker + NVIDIA Container Toolkit
- **All platforms**: Miniconda (or Anaconda) + Git + an NVIDIA GPU with CUDA 12.x drivers

### 2. Clone this repo
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 3. Clone external dependencies into the project root
These are not vendored in the repo — clone them next to the source:
```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
git clone https://github.com/maturk/dn-splatter.git    # (optional, for V6 onward)
```

### 4. Create the conda environment
```bash
conda create -n da3 python=3.10 -y
conda activate da3
pip install -r requirements.txt
pip install -e ./Depth-Anything-3
```

> If `pip install torch` fails, install the CUDA-matched build manually:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> ```

### 5. Build the Nerfstudio Docker image (one-time, ~10 min)
```bash
docker build -t nerfstudio-blackwell .
```

### 6. Pull the COLMAP image
```bash
docker pull colmap/colmap:latest
```

### 7. Drop your iPhone scan into `images/`
Either RealityScan HEIC files or any folder of overlapping JPEGs.

### 8. Run the pipeline
```bash
# RealityScan HEIC scan (full quality)
python reconstruct_realityscan.py

# Quick test (1000 iters, fast)
python reconstruct_realityscan.py --quick

# Skip MVS (use existing fused.ply for init)
python reconstruct_realityscan.py --skip-mvs
```

### 9. Verify the install (optional)
```bash
python check_setup.py
```

> **Note:** Your input photos, COLMAP outputs, training checkpoints, and final splats are all gitignored. They live on your machine only.

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | [python.org](https://www.python.org) |
| Docker Desktop with GPU support | [docs.docker.com](https://docs.docker.com/desktop/gpu/) |
| NVIDIA GPU + drivers | CUDA 12.x recommended |
| SAM3 repo + checkpoint | See below |

**Python packages:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy pillow
```

---

## First-Time Setup

### 1. Pull COLMAP Docker image
```bash
docker pull colmap/colmap:latest
```

### 2. Build the Nerfstudio Docker image
Run this once from the `reconstruction_project/` folder:
```bash
docker build -t nerfstudio-blackwell .
```
> This takes ~10 minutes. It compiles gsplat for your GPU (Blackwell / sm_120).

### 3. (Optional) Pull OpenMVS for textured mesh export
```bash
docker pull yeicor/openmvs-ubuntu-cuda:v2.3.0
```

### 4. Install SAM3
```bash
# Clone SAM3
git clone https://github.com/facebookresearch/sam2 C:/Users/<you>/sam3_work/sam3

# Download checkpoint (sam3.pt) and place it at:
# C:/Users/<you>/sam3_work/sam3/checkpoints/sam3.pt
```

### 5. Configure paths
Open `config.py` and set `SAM3_REPO_PATH` to where you cloned SAM3.
Alternatively, set an environment variable:
```bash
set SAM3_REPO_PATH=C:\Users\<you>\sam3_work\sam3
```

### 6. Verify your setup
```bash
python check_setup.py
```
Fix any `[FAIL]` items before continuing.

---

## Running the Pipeline

### Basic run (no masking)
```bash
python reconstruct.py images/skull
```

### With SAM3 masking (removes background, keeps the object)
```bash
python reconstruct.py images/skull --prompt "skull"
```

### Also export a textured mesh
```bash
python reconstruct.py images/skull --prompt "skull" --mesh
```

### Quick test run (fast, low quality — good for verifying setup)
```bash
python reconstruct.py images/skull --quick
```

### Skip re-training (re-export only)
```bash
python reconstruct.py images/skull --skip-training
```

---

## Output

All outputs are placed in `output/` by default:

| File | Description |
|------|-------------|
| `output/splat.ply` | Gaussian Splat — open in [SuperSplat](https://playcanvas.com/supersplat/editor) or Unity |
| `output/scene_textured.ply` | Textured mesh (only if `--mesh` was used) |
| `output/scene_textured0.png` | Texture atlas for the mesh |

Intermediate files (COLMAP sparse model, dense images, masks, nerfstudio checkpoints) are kept in the project folder in case you need to re-run individual stages.

---

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 1 | Stage images to workspace |
| 2 | Unmasked SfM — estimates camera poses + undistorts images |
| 3 | SAM3 masking — segments target object per image *(only with `--prompt`)* |
| 4 | Masked SfM — refines sparse model using masks *(only with `--prompt`)* |
| 5 | Blank mask safety check — fills in any missing masks |
| 6 | Nerfstudio training — Gaussian Splat (splatfacto) |
| 7 | Gaussian Splat export → `splat.ply` |
| 8 | OpenMVS textured mesh *(only with `--mesh`)* |
| 9 | Package outputs to `output/` |

---

## Configuration

All settings live in `config.py`. Key options:

| Setting | Default | Description |
|---------|---------|-------------|
| `SAM3_REPO_PATH` | env var or hardcoded | Path to SAM3 repository |
| `NERF_MAX_ITERATIONS` | `30000` | Training iterations (use `--quick` for 1000) |
| `NERF_DOWNSCALE_FACTOR` | `4` | Image downscale (1=full, 2=half, 4=quarter) |
| `COLMAP_CAMERA_MODEL` | `PINHOLE` | Camera model (PINHOLE works well for phone cameras) |
| `COLMAP_SINGLE_CAMERA` | `1` | Treat all images as same camera (recommended for one device) |

---

## Folder Structure

```
reconstruction_project/
├── config.py               # All settings — edit this
├── reconstruct.py          # Main pipeline script
├── check_setup.py          # Environment validator
├── Dockerfile              # Builds nerfstudio-blackwell image
│
├── lib/                    # Pipeline modules (do not edit unless needed)
│   ├── colmap_pipeline.py
│   ├── nerfstudio_pipeline.py
│   ├── sam3_helper.py
│   └── docker_runner.py
│
├── images/                 # Put your input photo sets here
│   ├── skull/
│   └── dice/
│
├── output/                 # Final outputs (splat.ply, mesh)
│
└── experimental/           # Archived older pipeline versions (v1–v5)
```

---

## Tips

- **Photo count:** 60–120 photos per object gives best results. Use the ScanGuide iOS app for guided capture.
- **Lighting:** Diffuse, even lighting — avoid harsh shadows and reflections.
- **Overlap:** At least 60% overlap between consecutive shots.
- **Object size:** Keep the object centered and fill ~60% of the frame.
- **Background:** A simple, textured background helps COLMAP find feature matches.

---

## Troubleshooting

**COLMAP finds 0 cameras / no sparse model**
- Too few photos, or too little texture/overlap.
- Try shooting more angles.

**Nerfstudio training crashes (CUDA OOM)**
- Increase `NERF_DOWNSCALE_FACTOR` in `config.py` (try 8).
- Or use `--quick` mode.

**`nerfstudio-blackwell` image not found**
- Run `docker build -t nerfstudio-blackwell .` from this folder.

**GPU not accessible in Docker**
- Ensure Docker Desktop has GPU support enabled (Settings → Resources → GPU).
- Run `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` to test.

**SAM3 masking fails**
- Check `SAM3_REPO_PATH` and that `sam3.pt` checkpoint exists.
- Pipeline will continue without masking if SAM3 fails.
