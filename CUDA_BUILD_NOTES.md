# CUDA Build Chain Notes (sam3 env)

The dev environment is **set up and verified** for building PyTorch CUDA
extensions on this Windows machine. All 5 extensions needed by StableGS /
FreeSplat / FreeSplat++ now build and import cleanly.

## Environment summary

| Component | Version | Path |
|---|---|---|
| PyTorch | 2.8.0+cu129 | sam3 env |
| CUDA Toolkit | 12.9.41 | `$ENV\Library\bin\nvcc.exe` (installed via conda from `nvidia/label/cuda-12.9.0`) |
| MSVC | 14.44.35207 | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\` |
| Ninja | 1.13.0 | sam3 env pip |
| GPU compute capability | sm_120 (RTX 5070 Ti) | — |

## How to build a new CUDA extension

Use the helper batch file `C:\Users\mgallai\Downloads\_build_cuda_ext.bat`:

```bat
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=C:\Users\mgallai\AppData\Local\miniconda3\envs\sam3\Library"
set "CUDA_PATH=C:\Users\mgallai\AppData\Local\miniconda3\envs\sam3\Library"
set "DISTUTILS_USE_SDK=1"
set "PATH=C:\Users\mgallai\AppData\Local\miniconda3\envs\sam3\Library\bin;%PATH%"
set "LIB=C:\Users\mgallai\AppData\Local\miniconda3\envs\sam3\Library\lib;%LIB%"
set "TORCH_CUDA_ARCH_LIST=12.0"
cd /d "%~1"
C:\Users\mgallai\AppData\Local\miniconda3\envs\sam3\python.exe setup.py build_ext --inplace -j8
```

Invoke from PowerShell:
```powershell
cmd /c "C:\Users\mgallai\Downloads\_build_cuda_ext.bat <path-to-extension-dir>"
```

The five non-obvious env vars are all critical:
- `DISTUTILS_USE_SDK=1`: tells PyTorch's cpp_extension we already activated MSVC, don't try to re-activate it.
- `CUDA_HOME` / `CUDA_PATH`: PyTorch's `CUDAExtension` checks both.
- `LIB`: adds conda's CUDA lib dir to linker search path (conda puts libs in
  `Library\lib\`, but PyTorch by default only adds `Library\lib\x64\`).
- `TORCH_CUDA_ARCH_LIST=12.0`: target the 5070 Ti's compute capability directly.

## Per-extension notes

After build, the package directory is created INSIDE the extension's source
tree by the in-place install. If the source tree doesn't have a `<pkg>/` dir,
the in-place copy errors out — create it manually with an empty `__init__.py`:

```bash
mkdir -p <ext>/<pkg_name>
touch <ext>/<pkg_name>/__init__.py
```

For global installation, copy the resulting package dir into the env's
site-packages:

```bash
cp -r <ext>/<pkg_name> /c/Users/mgallai/AppData/Local/miniconda3/envs/sam3/Lib/site-packages/
```

### Patches applied

- **StableGS `diff-gaussian-rasterization_radegs/cuda_rasterizer/utils.cu:29`** —
  used `std::sqrt` / `std::pow` from a `__global__` kernel, which CUDA 12.x
  rejects (host-only functions). Replaced with `sqrtf` / `powf` floats.

## Currently installed CUDA extensions (importable from sam3 env)

| Extension | Site-packages path | Used by |
|---|---|---|
| `simple_knn` | `Lib\site-packages\simple_knn\_C.cp312-win_amd64.pyd` | StableGS |
| `diff_gaussian_rasterization_radegs` | same | StableGS (RaDeGS variant) |
| `diff_gaussian_rasterization_taming` | same | StableGS (Taming variant) |
| `fused_ssim_cuda` | `Lib\site-packages\fused_ssim_cuda.cp312-win_amd64.pyd` | StableGS |
| `diff_gaussian_rasterization` | `Lib\site-packages\diff_gaussian_rasterization\_C.cp312-win_amd64.pyd` | FreeSplat / FreeSplatPP |

Quick smoke test:
```python
import torch
from simple_knn import _C
from diff_gaussian_rasterization_radegs import _C
from diff_gaussian_rasterization_taming import _C
from diff_gaussian_rasterization import _C
import fused_ssim_cuda
print("All CUDA extensions importable.")
```

## What's still needed to actually RUN each method

Now that the CUDA build chain works, the remaining per-method steps are:

### StableGS
1. Prep V32 data into their expected COLMAP layout (`images/`, `images_2/`,
   `sparse/0/`).
2. Run `python mvs.py -s dataset/v32_data3 ...` to generate `pairs_uni.pth`.
3. Run `python train_pair.py -s dataset/v32_data3 -m output_stablegs/v32_data3
   --mode "final_count" --eval --budget <N> --sh_lower --dual --single_loss
   --lambda_depth_pair 0.05 --lambda_depth_normal 0.05` (~30 min on RTX 5070 Ti).
4. Output is a `point_cloud.ply` in `output_stablegs/v32_data3/`.
5. Then run our `_spatial_crop_splat.py` on it for object extraction.

**Est. wall-clock to run end-to-end: ~1-2 hours.**

### FreeSplat
1. Download pretrained checkpoint from
   <https://drive.google.com/drive/folders/1_KqJnSfNrNxSMguBwFtR1cxTPxdLG7Sc>
   (Google Drive — manual download needed, ~500 MB).
2. Convert V32 data to ScanNet-like format:
   - `color/<idx>.jpg` (sequential numbering)
   - `depth/<idx>.png` (16-bit, mm-scale)
   - `intrinsic/intrinsic_color.txt` and `intrinsic_depth.txt`
   - `extrinsics.npy` (4x4 c2w per frame in OpenCV convention)
3. Pick 2-30 input views (FreeSplat is designed for sparse views).
4. Run inference script with the v32 scene path.

**Est. wall-clock: ~2 hours (mostly conversion + checkpoint download).**
**Risk: domain mismatch** — checkpoint trained on ScanNet rooms, not tabletop.

### FreeSplat++
Same as FreeSplat but with a different checkpoint and config. Note: cloned to
`Downloads/FreeSplatPP/` but no actual code there yet (the page says "code
available", but the repo at clone-time was the original FreeSplat — the new
code may be in a feature branch).

### ACMM
Separate from the CUDA extension build chain (this is a standalone C++/CUDA
project, not a PyTorch extension):
1. Need to install OpenCV for Windows + CMake config.
2. `cd ACMM && cmake -G "Visual Studio 17 2022" -A x64 . && cmake --build . --config Release`
3. Convert COLMAP output via `colmap2mvsnet_acm.py`
4. Run `./ACMM <data_folder>`

**Est. wall-clock: ~2-3 hours.** Likely needs additional Windows-specific
CMake tweaks (README says only Ubuntu 14.04 tested).
