# Methods-from-PI's-spreadsheet — Running Progress Log

**Purpose of this file:** Persistent multi-session memory anchor. Read this
FIRST at the start of any new session. Update it at the end of every
substantive step so the next developer (or future me) knows the state
without re-deriving everything.

**Primary references this log points back to:**
- [`SCENE_SEGMENTER_NOTES.md`](SCENE_SEGMENTER_NOTES.md) — the main
  scene_segmenter project notes (canonical pipeline picks + related-papers
  shortlist)
- [`CUDA_BUILD_NOTES.md`](CUDA_BUILD_NOTES.md) — exact CUDA + MSVC + ext
  build recipe on this Windows machine (env vars, batch helper, patches)
- `C:\Users\mgallai\Downloads\3d_papers_run_status.xlsx` — color-coded
  per-method status table

---

## At-a-glance status (LAST UPDATED 2026-06-18)

| Method | Code | Build | Data prep | Run | Output |
|---|---|---|---|---|---|
| NanoGS | ✅ | n/a (pure Py) | ✅ | ✅ | `output/nanogs_v32_data3/` |
| Clean-GS v1 (full-scene input) | ✅ | n/a (pure Py) | ✅ | ✅ rejected (halos) | `output/clean_gs_v32_data3/` |
| **Clean-GS v2 (per-object input)** | ✅ | n/a (pure Py) | ✅ | ✅ **promoted to production (v6_cleaned)** | `output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned/` via [`_clean_splat.py`](_clean_splat.py) |
| Wolff (Disney) | clean-room (no public code) | n/a (pure Py) | ✅ | ✅ | `output/wolff_v32_data3*/` |
| StableGS | ✅ cloned | ✅ 4 CUDA ext built | n/a | ❌ blocked | Repo missing `gaussian_renderer.render_rade` + dual_alpha rasterizer wrappers |
| **FreeSplat** | ✅ cloned | ✅ 1 CUDA ext built | ✅ | ✅ | `output/freesplat_v32_data3/raw_renders/` (PSNR 21.65) |
| FreeSplat++ | ✅ cloned | n/a | n/a | ❌ skip | No code released — repo is README-only |
| ACMM | ✅ cloned | ❌ Windows cmake port not viable in session | n/a yet | ❌ | Use WSL2 for original Linux build |

**Session summary**: 4 of 7 methods produced outputs. StableGS's CUDA build chain works but the public repo is incomplete (missing critical Python wrappers — verified via grep that `render_rade` is called but never defined). FreeSplat++ has no public code. ACMM build on Windows estimated at 4-8 h of porting (recommend WSL2 instead).

---

## Environment recap

| Component | Value | Note |
|---|---|---|
| Project working dir | `c:\Users\mgallai\Projects\3d_automated\reconstruction_project` | All scene_segmenter pipeline code + outputs |
| Conda env for code | `sam3` | Has SAM3 + nerfstudio + Open3D + torch 2.8.0+cu129 + all 5 CUDA exts |
| Python in env | 3.12 | `cp312-win_amd64` binaries |
| GPU | NVIDIA GeForce RTX 5070 Ti | sm_120 |
| CUDA Toolkit | 12.9.41 in env's `Library\bin\nvcc.exe` | Installed via `conda install -c nvidia/label/cuda-12.9.0 cuda-toolkit` |
| MSVC | 14.44.35207 at `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\` | Already present before our work |
| Ninja | 1.13.0 (pip) | For PyTorch ext builds |
| OS | Windows 11 | |

### Repos cloned to `C:\Users\mgallai\Downloads\`

- `NanoGS\` — pip-installed in env; ran ✅
- `clean-gs\` — used via Python script; ran ✅
- `StableGS\` — 4 CUDA ext built ✅; training not yet attempted
- `FreeSplat\` — CUDA ext built ✅; checkpoint not yet downloaded
- `FreeSplatPP\` — cloned; shares CUDA ext with FreeSplat
- `ACMM\` — cloned; cmake build not yet attempted
- `diff-gaussian-rasterization-w-depth\` — built ✅ for FreeSplat

### Build helper

`C:\Users\mgallai\Downloads\_build_cuda_ext.bat` — invoke as
`cmd /c "<bat-path> <ext-dir>"` from PowerShell. Sets MSVC + CUDA_HOME +
CUDA_PATH + DISTUTILS_USE_SDK + LIB + TORCH_CUDA_ARCH_LIST=12.0.

### CUDA extensions currently installed in `sam3` env

(check by running `python _test_cuda_imports.py` if that script still
exists, or just `python -c "from simple_knn import _C; ..."`)

| Extension | Site-packages dir | Used by |
|---|---|---|
| `simple_knn` | `Lib\site-packages\simple_knn\` | StableGS |
| `diff_gaussian_rasterization_radegs` | `Lib\site-packages\diff_gaussian_rasterization_radegs\` | StableGS RaDeGS variant |
| `diff_gaussian_rasterization_taming` | `Lib\site-packages\diff_gaussian_rasterization_taming\` | StableGS Taming variant |
| `fused_ssim_cuda` (loose .pyd) | `Lib\site-packages\fused_ssim_cuda.cp312-win_amd64.pyd` | StableGS |
| `diff_gaussian_rasterization` | `Lib\site-packages\diff_gaussian_rasterization\` | FreeSplat / FreeSplatPP |

Patches applied:
- `StableGS\submodules\diff-gaussian-rasterization_radegs\cuda_rasterizer\utils.cu:29` —
  `std::sqrt`/`pow` calls inside a `__global__` kernel rejected by CUDA 12.x;
  replaced with `sqrtf`/`powf` (float versions).

### Inputs we have for V32 data3

- **RGB images**: `nerfstudio_data/images/IMG_femto_*.jpg` and `colmap/dense/images/` (140 each)
- **ToF depth maps**: `nerfstudio_data/depths_femto/IMG_femto_*.npy` (140 of them, float32 metric metres; plus `*_conf.npy` confidence)
- **Intrinsics (Femto Mega native)**: `nerfstudio_data/femto_intrinsics.json` — fx=1125.36, fy=1125.07, cx=966.03, cy=519.94, W=1920, H=1080
- **COLMAP poses**: `colmap/dense/transforms.json` (nerfstudio-style — OpenGL convention, COLMAP-unit translations)
- **Scale conversion**: `colmap/dense/tof_bounds.json` — `scale_factor_da3_to_colmap = 8.798561645475129`. To convert COLMAP-unit translations to metric metres, divide by this number.
- **nerfstudio dataparser** (for splat space): `nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json` — applied during splatfacto training. Documented in `_spatial_crop_splat.py`.
- **V32 splat**: `output/splat_v32_data3_noinit_pruned.ply` (226,665 Gaussians; pruned-but-not-extracted scene splat)
- **V32 mesh**: `output/mesh_v32_data3/mesh_v32_data3_openmvs.ply` + `scene_textured0.png` atlas
- **Existing SAM3 masks** (cached): `output/segmented_v32_data3/masks/IMG_femto_NNNN__<prompt>.png` for prompts `white_water_bottle, blue_box, red_lobster_figurine`

---

## Per-method action lists

For each unfinished method below: what needs to happen, in order, with file
paths and command sketches. Tick `[x]` as steps complete.

### FreeSplat (NeurIPS 2024 — feed-forward 3DGS for indoor scenes)

Repo: `C:\Users\mgallai\Downloads\FreeSplat\`
Code path uses: `pytorch_lightning` + `hydra-core` + cached config in `config\`

- [ ] **Install Python deps** (besides CUDA ext): `pip install -r FreeSplat\requirements.txt`. Pinned `kornia==0.6.7` may need adjustment for Py3.12. Skip `wandb` if it complains.
- [ ] **Download pretrained checkpoint** from <https://drive.google.com/drive/folders/1_KqJnSfNrNxSMguBwFtR1cxTPxdLG7Sc> — manual click-through (no `gdown` URL).
- [ ] **Identify minimal inference entry point** — search `FreeSplat\src` for `main` / `inference` / `eval` script. Lightning convention so probably `main.py` with hydra config.
- [ ] **Convert V32 data to expected format** — see `FreeSplat\src\datasets` (ScanNet-style: `color/`, `depth/`, `intrinsic/`, `extrinsics.npy`). Pick 2-30 view subset (paper sweet spot).
- [ ] **Run inference** — output a `.ply` splat predicted from sparse views.
- [ ] **Compare to V32 splatfacto output** quality-wise (PSNR vs ground truth view OR side-by-side visual).
- [ ] **Save outputs** to `output/freesplat_v32_data3/`.

Known risks: domain mismatch (trained on 3-10 m rooms; we have 1 m tabletop).
Even if the inference runs, output quality may be visibly worse than splatfacto.

### FreeSplat++ (2025 extension)

Repo: `C:\Users\mgallai\Downloads\FreeSplatPP\`

- [ ] If FreeSplat ran successfully: only difference is a new checkpoint + slightly different config.
- [ ] **Download FreeSplat++ checkpoint** (separate Drive folder; check repo readme).
- [ ] **Re-run inference** with the same V32 data subset.
- [ ] **Save outputs** to `output/freesplatpp_v32_data3/`.

### StableGS (2025 — floater-free 3DGS training)

Repo: `C:\Users\mgallai\Downloads\StableGS\`
Training script: `train_pair.py`. Pair-gen script: `mvs.py`.

- [ ] **Install Python deps** — repo has no `requirements.txt`; cross-reference imports in `train_pair.py`. Likely needs: tqdm, lpips, plyfile, tensorboard, scene-utils (already present).
- [ ] **Prep V32 data into expected COLMAP layout** — `dataset/v32_data3/{images,images_2,sparse/0/}`. They use `images_2` as the downsample-by-2 path; for us images_2 can just be 960×540 versions OR same as images.
- [ ] **Run mvs.py** to generate `pairs_uni.pth`: `python mvs.py -s dataset/v32_data3 ...` (exact args TBD).
- [ ] **Train**: `python train_pair.py -s dataset/v32_data3 -m output_stablegs/v32_data3 -i images_2 --mode "final_count" --eval --budget 226665 --sh_lower --dual --single_loss --lambda_depth_pair 0.05 --lambda_depth_normal 0.05` — match our V32 splat budget so we're comparing apples to apples. ~30 min on RTX 5070 Ti.
- [ ] **Verify the trained .ply** opens in SuperSplat and renders.
- [ ] **Save outputs** to `output/stablegs_v32_data3/`.
- [ ] **(Optional)** Run `_spatial_crop_splat.py` on the StableGS-trained splat to extract per-object splats and compare to V32-splatfacto results.

### ACMM (CVPR 2019 — multi-scale geometric consistency MVS)

Repo: `C:\Users\mgallai\Downloads\ACMM\`

- [ ] **Standalone C++ project — different toolchain.** Not a PyTorch extension. Needs CMake + nvcc + OpenCV linked at build time. README mentions Ubuntu 14.04 only.
- [ ] **Install OpenCV for MSVC** — easiest is `conda install -c conda-forge opencv` then point cmake at it; or download prebuilt binaries from <https://opencv.org/releases/>.
- [ ] **CMake configure**: `cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=<path>`. Expect Windows-specific patches (Linux-style includes, etc.).
- [ ] **Build**: `cmake --build . --config Release -j8`.
- [ ] **Convert COLMAP output via `colmap2mvsnet_acm.py`** (from ACMM repo) to ACMM's expected layout.
- [ ] **Run**: `./ACMM.exe <data_folder>` produces depth maps + fused point cloud.
- [ ] **Compare** point cloud + Poisson-meshed result to our V32 OpenMVS mesh (similar style to the Wolff test).
- [ ] **Save outputs** to `output/acmm_v32_data3/`.

Risk: Windows-port-specific compile errors are likely. Budget 2-3 h, treat
as exploration-only if it stalls.

---

## Decisions / conventions used across runs

- **Always run from project root** `reconstruction_project/` — scripts use relative paths.
- **All Python invocations use the `sam3` conda env.** Prefer `python.exe` from `C:\Users\mgallai\AppData\Local\miniconda3\envs\sam3\` directly when conda's wrapper misbehaves with multiline `-c`.
- **PyTorch CUDA exts must be built with the env active** + the `_build_cuda_ext.bat` helper. See CUDA_BUILD_NOTES.md.
- **`numpy<2` pin in env** — repeatedly. Reinstall any package that upgrades numpy beyond 1.26 (trimesh and our older numpy code break on 2.x).
- **Per-method outputs go to** `output/<method_name>_v32_data3/` so they're discoverable.

---

## Session change log

### 2026-06-17 (this session, before this log was created)

- Installed CUDA Toolkit 12.9.41 into `sam3` env via conda (no admin needed)
- Built + installed 5 PyTorch CUDA extensions:
  - simple_knn (StableGS)
  - diff_gaussian_rasterization_radegs (StableGS)
  - diff_gaussian_rasterization_taming (StableGS)
  - fused_ssim_cuda (StableGS)
  - diff_gaussian_rasterization (-w-depth) (FreeSplat / FreeSplatPP)
- Patched `StableGS .../utils.cu:29` (host-only `std::sqrt` in `__global__` kernel → `sqrtf`)
- Verified all 5 extensions import in `sam3` env
- Wrote `CUDA_BUILD_NOTES.md`
- Updated `3d_papers_run_status.xlsx`

### 2026-06-17 (this session, after log was created)

**Workflow investigation completed.** 5 agents profiled all 4 methods in parallel. Key results:

**Recommended order: StableGS → FreeSplat → ACMM (skip FreeSplat++).**

Rationale:
- **StableGS**: per-scene optimization, no domain mismatch (works on tabletop). All 4 CUDA exts already built (huge sunk cost recovered). Risks: needs DUSt3R checkpoint (~2.5 GB) for `mvs.py`, AND the cloned StableGS repo may be MISSING the upstream `arguments.py` from the original 3DGS repo (workflow agent flagged this — need to copy from gaussian-splatting upstream).
- **FreeSplat**: ScanNet-trained pretrained, real domain mismatch risk for our 1 m tabletop (trained on 0.5-15 m rooms). CUDA ext already built; remaining work is checkpoint download (Google Drive, manual) + data conversion (uint16 mm depth PNGs, OpenGL→OpenCV w2c, scale-factor divide). Inference itself is fast (30-60 s).
- **ACMM**: Classical MVS, no neural domain mismatch BUT separate C++/CMake build chain. Workflow agent identified concrete Linux→Windows patches needed: POSIX `mkdir`/`sys/stat.h`, GCC-only compiler flags, ancient CMake 2.8 + outdated CUDA compute caps. Could be 1-2 days of C++ surgery on Windows. Deprioritize.
- **FreeSplat++**: workflow confirmed repo is README-only — no code or checkpoint released. SKIP entirely.

**Common V32-conversion helpers worth writing once** (workflow recommendation):
- `colmap_sparse_to_dataset_dir(src, dst)` — shared by StableGS + ACMM
- `nerfstudio_c2w_opengl_to_opencv_w2c(transforms_json, scale_factor=8.798)` — shared by FreeSplat + any port
- `tof_npy_to_uint16_png_mm(npy_path, png_path)` — FreeSplat depth conversion
- `femto_intrinsics_to_K_txt(femto_json, out_path, normalize=False)`
- `make_freesplat_eval_index_json(scene_name, num_frames=140, num_context=8)`
- `validate_colmap_pinhole(cameras_bin_path)` — fail-fast guard for both StableGS + ACMM

Workflow output dump: `C:\Users\mgallai\AppData\Local\Temp\claude\<session>\tasks\wrmwse8ib.output` — preserves exact entry-point commands, data formats, and pitfalls for each method.

**StableGS attempted, BLOCKED by upstream-repo incompleteness.**
- Cloned upstream INRIA `gaussian-splatting` to `Downloads/gaussian-splatting-upstream/` and grafted:
  - `arguments/` (ModelParams, PipelineParams, OptimizationParams) → into `StableGS/arguments/` ✅
  - `gaussian_renderer/` → into `StableGS/gaussian_renderer/` ✅ (but only `render`, not `render_rade`)
  - `fused_ssim/` Python wrapper → into `sam3 env site-packages` ✅
- **Missing & unreproducible without paper-reimplementation**:
  - `gaussian_renderer.render_rade(view, gaussians, pipe, bg, dual_alpha=...)` — StableGS's custom Dual-Opacity entry point, never defined in the public repo (`grep` confirms it's CALLED in `train_pair.py`/`render*.py` but never DEFINED anywhere)
  - Python wrappers around `diff_gaussian_rasterization_radegs._C` and `_taming._C` (the raw C symbols are exposed but no Python wrapper sets up `GaussianRasterizationSettings(dual_alpha=…)` etc.)
- Conclusion: the **public StableGS repo cannot be run end-to-end as-is**. Two options if revisiting:
  1. Email the MooreThreads StableGS authors to request the missing files.
  2. Reimplement the Dual Opacity logic from the arXiv paper — multi-day project, low payoff vs alternative methods.
- Built artifacts that ARE working: all 4 CUDA extensions (simple_knn, radegs, taming, fused_ssim_cuda) compile cleanly. So if/when the missing Python files arrive, we can run training in ~30 min.

**FreeSplat — RAN SUCCESSFULLY ✅**
- Downloaded the 3 checkpoints (2views.ckpt, 3views.ckpt, fvt.ckpt, ~560 MB each) via `gdown --folder` into `FreeSplat/checkpoints/freesplat_models/`
- Wrote V32→ScanNet converter at [`_convert_v32_to_freesplat.py`](_convert_v32_to_freesplat.py) — applies OpenGL→OpenCV pose flip + `/scale_factor_da3_to_colmap` for metric metres + uint16 mm depth PNGs + 4x4 K txt
- Data lives at `C:\Users\mgallai\Downloads\FreeSplat\datasets\scannet\test\v32_data3_00\` (note `_00` suffix — ScanNet convention requires `<scene>_<NN>` dir naming so that `key[:-2]` matches `test_idx.txt` entry `v32_data3_` with trailing underscore)
- Eval index JSON at `FreeSplat/assets/evaluation_index_scannet_8views.json` with key `v32_data3_00`, 8 context views + 132 targets
- **Six Windows / Py3.12 / torch-2.8 patches** required to make the FreeSplat code run:
  1. `lightning_fabric/utilities/cloud_io.py`: force `weights_only=False` for trusted local checkpoints (PyTorch 2.6+ default changed)
  2. `src/dataset/dataset_scannet.py` line 92: replaced `str(path).split('/')[-1]` with `Path(path).name` (Windows backslashes)
  3. `src/model/decoder/cuda_splatting.py` line 112+213: removed `debug=False` kwarg (not in older `diff_gaussian_rasterization` wrapper)
  4. `src/model/decoder/cuda_splatting.py` line 120: changed `image, radii, depth, _ = rasterizer(...)` to 3-tuple unpack
  5. `src/model/decoder/cuda_splatting.py` line 130: guarded depth `.unsqueeze(0)` behind `if depth.dim()==2`
  6. `src/model/model_wrapper.py` line 400: squeezed `depth_gt` before matplotlib call
  7. Aliased `site-packages/diff_gaussian_rasterization` → `site-packages/diff_gaussian_rasterization_depth` (FreeSplat code imports under the second name)
- Also need extra Python deps not flagged in requirements: `pytorch_lightning hydra-core jaxtyping beartype kornia==0.6.7 timm dacite lpips e3nn tabulate svg.py mmcv-lite moviepy<2 wandb gdown`. `numpy<2` must be reinstalled after every `pip install` that bumps it.
- **Run command** (env: WANDB_MODE=disabled to bypass login):
  ```
  WANDB_MODE=disabled python -m src.main +experiment=scannet/fvt +output_dir=output_v32_data3 \
    mode=test dataset/view_sampler=evaluation \
    checkpointing.load=checkpoints/freesplat_models/fvt.ckpt \
    dataset.view_sampler.num_context_views=8 'dataset.roots=[datasets/scannet]'
  ```
- **8-view Metrics**: PSNR 21.65, SSIM 0.808, depth_abs_diff 0.145 m, depth_rel_diff 17.4%, delta_25 91.8%, delta_10 81.0%, 756,333 Gaussians generated. Outputs (132 rendered + GT pairs + depth) at `output/freesplat_v32_data3/raw_renders/scannet/v32_data3_00/` (75 MB). PLY at `output/freesplat_v32_data3/freesplat_v32_data3.ply` (49 MB).
- **30-view follow-up (2026-06-18 11:59)**: Re-ran with fvt's max context views to test if dense input would fix the smear. Used `_regen_freesplat_eval_index.py --num-context 30`.
  - **Metrics**: PSNR 22.25 (+0.6 dB), SSIM 0.813 (+0.005, ~flat), depth_abs_diff **0.078 m (−47 %)**, depth_rel_diff **9.0 % (−48 %)**, delta_25 89.7% (regressed -2.1pp), delta_10 81.7% (+0.7pp), 1,842,416 Gaussians (+2.4×). Outputs: `output/freesplat_v32_data3/freesplat_v32_data3_30views.ply` (125 MB) + `raw_renders_30views/`.
  - **Workflow verdict**: depth quality nearly halved (good), but per-Gaussian quality REGRESSED: high-confidence Gaussians (alpha>0.9) dropped 55% → 23%; low-opacity filler (alpha<0.2) grew 33.5% → 41.1% (~757k of 1.84M are transparent fluff); extreme position outliers (>±15 m) went 0 → 581 — adding views introduced floaters instead of converging.
  - **vs V32 splatfacto baseline**: not competitive. FreeSplat 30-view needs 8× more Gaussians (1.84M vs 227k) and 2.3× more disk (125 MB vs 53.6 MB) but is <half as confident per-Gaussian (23% vs 55% alpha>0.9). The ScanNet-vs-tabletop domain mismatch is architectural — view count cannot fix it.
  - **Final FreeSplat verdict**: drop from the visual deliverable. V32 splatfacto stays canonical.

**Clean-GS v2 — RAN SUCCESSFULLY ✅ (improvement on Clean-GS v1)**
- **Motivation**: v1 (feeding the full 226k-Gaussian scene splat into Clean-GS with 2D mask projection) produced visible halos around each object — diagnosed as depth-ambiguity (Gaussians behind/beside the object also project into the 2D mask, so they survived filtering).
- **Fix**: feed v9a_fp_v2's already-3D-cropped per-object splat (`output/segmented_v32_data3_v9a_fp_v2_splat_v5_shcolor/<prompt>_splat.ply`) into Clean-GS instead. Since the input is already spatially tight, depth-ambiguity is gone — Clean-GS only does color-validation + k-NN outlier pruning within the crop.
- **Script**: [`_run_clean_gs_v2.py`](_run_clean_gs_v2.py) — reuses `cameras.json` + `masks_<prompt>/` from v1's directory. Args: `--mode neighbor --color_threshold 0.40`.
- **Results** (comparison via workflow `wqqj1p11n`):

  | Object | v5_shcolor in | v1 (full-scene in) | **v2 (per-object in)** |
  |---|---|---|---|
  | Bottle | 5,210 / 1.23 MB / 2.2×4.6×5.9 cm | 16,086 / 3.81 MB / **25×16×11 cm halo** | **4,580 / 1.08 MB / 2×4×6 cm** |
  | Box | 5,202 / 1.23 MB / 2.6×2.7×3.0 cm | 5,503 / 1.30 MB / 6×6×3 cm mild halo | **4,213 / 1.00 MB / 2×3×3 cm** |
  | Lobster | 2,463 / 0.58 MB / 3.1×2.8×2.5 cm | 2,424 / 0.57 MB / 5×4×3 cm (alpha-tail halo) | **2,052 / 0.49 MB / 3×3×2 cm** |

- **Why v2 is a real win (not noise)**:
  1. Span is unchanged across all three objects → no legitimate geometry pruned.
  2. Lobster's `opacity_high_pct` went UP (46.3 → 48.8 %, +2.5 pp) while count went DOWN (-17 %) — textbook fingerprint of removing genuine floaters, not nibbling surface.
  3. 12–19 % smaller files → meaningfully cheaper Quest 3 streaming.
- **New canonical chain**: `v9a_fp_v2 spatial-crop → v5_shcolor SH-color filter → clean_gs_v2 refinement`. Replaces v5_shcolor as the production pick for per-object splats.
- **Caveat**: gains are modest, not dramatic. Do a Quest 3 visual A/B (especially on the bottle, where v5 already looked clean) before fully retiring v5_shcolor as a fallback.
- **Workflow output** for full inspection details: `tasks/wqqj1p11n.output`.

**Clean-GS v2 integrated into scene_segmenter pipeline as `v6_cleaned` (2026-06-18)**
- New canonical splat stage at [`_clean_splat.py`](_clean_splat.py). End-to-end
  CLI mirrors `_spatial_crop_splat.py` style; defaults wired for V32 data3
  so a no-arg invocation reproduces production output.
- Pipeline now: `pipeline.py` (SAM3 + raycasting + voting → seeds) →
  `_spatial_crop.py` (footprint + spatial crop → v9a_fp_v2 mesh) →
  `_spatial_crop_splat.py` (v5_shcolor splat per object) →
  **`_clean_splat.py` (v6_cleaned: Clean-GS prune within the v5 crop)**.
- Two non-obvious gotchas resolved while wiring this up:
  1. Clean-GS's `--masked_images` flag expects **RGB photos with non-object
     pixels blacked out**, NOT raw binary masks. Feeding it the canonical
     SAM3 cache (`segmented_v32_data3/masks/IMG_*__<prompt>.png`, which is
     0/255 binary) caused 98 % over-pruning because the colour-validation
     step compared Gaussians' SH-decoded colour to mostly-black mask images.
     `_clean_splat.py` now multiplies the binary mask with the source photo
     from `nerfstudio_data/images/` at staging time.
  2. Clean-GS's `print("✓ Saved …")` crashes on Windows cp1252 stdout
     (`UnicodeEncodeError`) AFTER the PLY is already written, producing
     rc=1. `_clean_splat.py` sets `PYTHONIOENCODING=utf-8` in the subprocess
     env to avoid the false-negative.
  3. Clean-GS's colour-validation step gets monotonically stricter with
     more views (any one disagreement drops a Gaussian). Feeding all 140
     cached views over-prunes; ~20 evenly-subsampled views reproduce
     the sweet-spot 12-19 % pruning. `_clean_splat.py` defaults to
     `--max-views 20`.
- Outputs verified equivalent to the standalone `_run_clean_gs_v2.py` run
  (bottle 1.09 MB vs 1.08, box 1.00 MB vs 1.00, lobster 2,042 vs 2,052
  Gaussians; ≤0.5 % delta from view-sampling jitter).
- SCENE_SEGMENTER_NOTES.md updated to mark `v6_cleaned` as production pick
  and add the Stage B section + CLI.

**Track 1 (COMPREHENSIVE_PLAN) — Truly-unseen core measurement DONE (2026-06-18)**
- Script: [`_unseen_core_map.py`](_unseen_core_map.py). Casts rays from every captured
  view through a 5 mm desk-plane grid under each object's XZ footprint; counts how
  many views had unobstructed line of sight. Uses scene_segmenter's V32ViewSource +
  Open3D BVH raycaster over the V32 OpenMVS mesh.
- **Critical knob discovered:** per-object extracted-mesh `y_min` is NOT a reliable
  desk plane for objects whose contact base was occluded during capture (box's
  extracted-mesh bottom is 12 cm above the actual desk because OpenMVS couldn't
  reconstruct the under-box geometry). Use `--desk-y 0.002` (V32 desk surface,
  taken from the bottle's well-reconstructed base) for all objects on the V32 desk.
- **Result (140 views, 5 mm grid):**

  | Object | Truly unseen | < 5 views | >= 20 views | median views/pt |
  |---|---|---|---|---|
  | white_water_bottle | 0.1 % (1/1204) | 0.2 % | 99.6 % | 120 |
  | blue_box | 0.0 % (0/960) | 0.0 % | 100.0 % | 126 |
  | red_lobster_figurine | 10.8 % (118/1088) | 14.2 % | 79.0 % | 64 |

- **Plan-level decision:** every object < 30 % unseen → **NN-copy from observed
  footprint is sufficient for V32 data3**. Skip Track 2a M2 (LaMa) and Track 2b
  (GSFix3D off-the-shelf). The full diffusion path is no longer load-bearing.
- Lobster's 10.8 % unseen is one contiguous triangular shadow under the body;
  surrounding ring is well-observed → NN extrapolation will give uniform desk
  colour, perfectly adequate for VR.
- Outputs at `output/unseen_core_v32_data3/`: per-object heatmap PNG + grid NPZ
  (positions + per-point view count) for downstream patch-Gaussian seeding.

**Track 2a (COMPREHENSIVE_PLAN, modified post-Track-1) — desk-patch Gaussian seeder DONE (2026-06-18)**
- Script: [`_seed_desk_patch.py`](_seed_desk_patch.py). Unified M1+M2+M3 (no LaMa
  diffusion needed at V32's <30% unseen regime).
- Per grid point: visibility-pass + best-view scoring (cos(off-normal) / distance)
  → bilinear RGB sample from that view's source photo; truly-unseen points get
  SciPy `cKDTree` NN-copy from the nearest observed point.
- Bug caught early: initial best_score=-1 made every point claim a "best view"
  even at score 0 (visible=False), bypassing NN-copy. Fixed by gating
  `better = visible & (score > best_score)`. Now NN-copy fires for the 118
  lobster unseen points as expected.
- Output: per-object splat-space PLY at `output/desk_patch_v32_data3/desk_patch_<slug>.ply`.
  Schema matches existing v6_cleaned (SH degree 3 zero-padded, same property order).
- Transform chain verified: metric `(x, y_desk, z)` → COLMAP-units → splat space
  via dataparser_transforms.json (R, t, dp_scale=0.17306; bottle base maps to
  splat Y≈0.83, splat Z≈0.04 — matches v6 bottle splat's high-Z end exactly).
- Desk normal in splat space: `[0.051, 0.064, -0.997]` (≈ -Z); patch quaternion
  `(0.73, -0.683, 0, -0.035)` aligns Gaussian local +Y with that normal.
- Patch sizes: bottle 1,204 Gaussians (293 KB), box 960 (234 KB), lobster
  1,088 (265 KB) — additive to the V32 scene splat (227k) is < 2 % overhead.
- **Track 2a M4 (localized splatfacto retraining) NOT YET tested** — first
  step is a visual check: render a held-out view through `V32 scene splat MINUS
  object + patch` to see whether the seeded patch already looks plausible
  without retraining. If yes, M4 can be skipped entirely.
- **Track 2a deliverable scenes** generated via [`_merge_patched_scene.py`](_merge_patched_scene.py):
  takes V32 scene splat (`splat_v32_data3_noinit_pruned.ply`, 227k Gaussians),
  removes Gaussians inside each object's v9a_fp_v2 footprint, splices in the
  desk-patch. Outputs at `output/scene_patched_v32_data3/`:

  | File | Gaussians | Size | What |
  |---|---|---|---|
  | `scene_patched_white_water_bottle.ply` | 222,438 | 52.6 MB | scene minus bottle + desk patch |
  | `scene_patched_blue_box.ply` | 222,255 | 52.6 MB | scene minus box + desk patch |
  | `scene_patched_red_lobster_figurine.ply` | 225,247 | 53.3 MB | scene minus lobster + desk patch |
  | `scene_patched_ALL.ply` | 216,610 | 51.2 MB | scene minus all 3 + 3 desk patches |

- **Open these in SuperSplat to visually verify**: the desk surface under each
  removed object should now be present (no visible hole). If the patches look
  flat/cardboard-like (esp. lobster's 10.8 % NN-copied region), run M4 to
  polish. If they already blend cleanly, M4 is unnecessary.

**Phase: trim + inpaint -- mesh-side mirror DONE (2026-06-18)**
- Orchestrator: [`_phase_trim_inpaint.py`](_phase_trim_inpaint.py). Produces
  apples-to-apples splat-and-mesh deliverables under one root.
- **Mesh trim**: `_trim_mesh_by_footprint()` drops scene-mesh faces whose
  centroids fall inside the same XZ footprint + Y range the splat side uses
  (`_test_inside_footprint` reused from `_spatial_crop_splat.py`). Per-object
  removed face counts: bottle 48,014 / box 27,877 / lobster 17,140 (1.3-3.7 %
  of the 1.296 M-face V32 scene mesh).
- **Mesh patch**: `_build_mesh_patch()` triangulates the Track-1 desk-plane
  grid (5 mm spacing, ~1 k vertices/object) and assigns per-vertex RGB from
  the same `_best_view_per_point` + `_sample_color_for_points` + `_nn_fill_missing`
  chain the splat patch uses. 1.8-2.3 k tris per patch.
- **Output layout** at `output/phase_trim_inpaint_v32_data3/`:

  ```
  splat/
    scene_full.ply                                # V32 scene splat untouched
    per_object/<slug>/
      object_only.ply                             # v6_cleaned bottle/box/lobster splat
      desk_patch_only.ply                         # NN-copy patch Gaussians alone
      scene_minus_object_patched.ply              # scene splat with object gone + patch
  mesh/
    scene_full.ply                                # V32 OpenMVS mesh untouched
    per_object/<slug>/
      object_only.ply / .obj / .mtl / .png / .glb # v9a_fp_v2 extracted, fully textured
      desk_patch_only.ply                         # planar mesh patch w/ per-vertex RGB
      scene_minus_object_patched.ply              # scene mesh with object faces gone + patch
  ```

- 36 files total. Texture sidecars (`scene_textured0.png` + per-object
  `_mat.png` + OBJ/MTL/GLB) copied so each mesh per-object dir is
  self-contained for MeshLab / Unity.
- The mesh patch uses per-vertex colors (PLY format). If the user wants a
  proper UV atlas instead (better Unity blending), upgrade to OBJ + xatlas
  + baked atlas — left as TODO once the per-vertex test passes visual A/B.

**Mesh-side colour fix (2026-06-18)**
- User reported `mesh/scene_full.ply` showed an "scene_textured0.png not
  loaded" warning in MeshLab and rendered uncolored. Root cause: the source
  V32 PLY is VCG-textured (`comment TextureFile scene_textured0.png` +
  per-face uchar/float texcoord) and references the atlas as a sidecar. The
  raw `shutil.copy2` brought the PLY but not the PNG.
- **Fix part 1**: `scene_textured0.png` copied next to `mesh/scene_full.ply`
  (and into each `mesh/per_object/<slug>/` dir as a safety net). Full scene
  now opens textured in MeshLab.
- **Fix part 2** (orchestrator): `scene_mesh.visual.to_color()` bakes the
  V32 atlas into per-vertex RGBA BEFORE the trim step. The resulting
  `scene_minus_object_patched.ply` files are now self-contained per-vertex-
  coloured PLYs (no sidecar required, ~775-794k verts each, mean RGB
  [132, 122, 75] = desk brown). Renders correctly in any PLY viewer.
- Trade-off acknowledged: per-vertex bake loses sub-vertex atlas detail
  (e.g. fine wood grain between vertices). Acceptable for the visual A/B
  test. For Unity production, will need to switch back to UV+atlas via a
  VCG-textured writer + xatlas-baked extra-atlas for the patch faces.

**`aggressive_prune_v2.py` — peripheral-artifact cleanup, coord-bug fix (2026-06-22)**
- User reported a "leafy fringe" surrounding the V32 desk in scene_full
  renders. Diagnosis: v1's `aggressive_prune.py --use-aabb` reads bounds
  from `colmap/dense/tof_bounds.json` (COLMAP units, ~9× larger than
  splat-space) but compares them DIRECTLY against the PLY's positions
  (splat-space). Net: only z accidentally cropped, x/y are no-ops. The
  20 % slack made it even more permissive. Verified by replaying the
  filter on V32: kept 78 % of Gaussians as an arbitrary half-space cut,
  not a desk box. Hence the fringe survives.
- Fix: new script [`aggressive_prune_v2.py`](aggressive_prune_v2.py).
  Same scale + opacity filters; the AABB step is rewritten entirely in
  splat-space. Three sources of an AABB, in precedence:
    1. Explicit `--aabb-x-min/.../z-max` six numbers
    2. `--aabb-json` (any 6-key JSON in splat-space)
    3. `--auto-aabb` (default ON): union of per-object splats in
       `output/segmented_<scene>_v9a_fp_v2_splat_v6_cleaned/` +
       a 30 cm metric margin (`--margin-m 0.30`)
  Scale + opacity defaults UNCHANGED (95th percentile, 0.30) so the
  desk and object interior detail are not touched.
- V32 trial run:
  - Input: `output/splat_v32_data3_noinit_pruned.ply` (226,665 G, 53.6 MB)
  - Output: `output/splat_v32_data3_noinit_pruned_v2_desk.ply` (99,733 G, 23.6 MB)
  - Auto-AABB (splat units, after 30 cm margin):
    x[-1.16, +1.00] y[+0.23, +1.52] z[-0.86, +0.49]
  - Filter breakdown: scale -11,334 (5 %); opacity -41,482;
    AABB v2 -74,116 (the v1 AABB missed all of these on x/y due to the
    coord-bug).
- Doc: RUN_GUIDE.md "Cleaning floaters after training" section now
  documents v1 (with bug noted) and v2 (recommended) side-by-side.
- Pending: visual verification by user. Expect the leafy peripheral
  fringe in the V32 scene render to be gone in `*_v2_desk.ply`, with
  no visible loss on the desk / bottle / box / lobster.

**`aggressive_prune_v3.py` — mesh-derived AABB, replaces v2 (2026-06-23)**
- User visual A/B showed v2 was too aggressive: clipped the back wall
  and the desk's perimeter because the AABB came from only the three
  object splats (their union is roughly 80 × 35 × 45 cm of splat-space,
  much smaller than the full reconstructed scene). 30 cm margin wasn't
  enough to cover the back wall.
- Fix: new script [`aggressive_prune_v3.py`](aggressive_prune_v3.py).
  Derives the AABB from the V32 OpenMVS scene mesh's vertices, forward-
  transformed to splat-space via the same `dataparser_transforms.json`
  the splatfacto training used. Why this source is correct:
  the scene mesh is the "ground truth" of what was actually triangulated;
  the leafy fringe floaters do NOT exist in the mesh because they have
  no real 3D geometry. So the mesh's spatial extent IS exactly the
  envelope we want to keep, by construction.
- Algorithm:
    1. Read scene mesh vertices in metric.
    2. Apply metric -> splat forward transform.
    3. 1st-99th percentile AABB per axis (drops stray OpenMVS outliers).
    4. Expand by 10 cm metric margin (`--margin-m 0.10`).
    5. Same scale + opacity prune as v1/v2.
- V32 trial run:
  - Input: `output/splat_v32_data3_noinit_pruned.ply` (226,665 G, 53.6 MB)
  - Output: `output/splat_v32_data3__trial_v3_mesh_aabb.ply` (153,752 G, 36.4 MB)
  - Final AABB (splat units): x[-1.39, +1.13] y[+0.45, +1.68] z[-0.70, +0.81]
  - Compared to v2 box (x[-1.16,+1.00] y[+0.23,+1.52] z[-0.86,+0.49]):
    v3 box is bigger on x, y high, and z high (= more back-wall +
    desk-perimeter coverage); slightly tighter on z low (= mesh ends
    before the floater tail starts).
  - Filter-3 drops: 20,097 (8.9 %) — exactly the floaters outside the
    reconstructed envelope. v2 dropped 74,116; v1 dropped ~49 k
    arbitrarily.
- v2 output renamed to `output/splat_v32_data3__trial_v2_object_aabb.ply`
  for side-by-side comparison.
- RUN_GUIDE.md documents v1 / v2 / v3 side-by-side with the trade-offs
  and when to use each.

### 2026-07-03 — PI soft-fade (v10 opacity, v11 SDF) integrated as canonical splat cleanup

**Trigger.** PI proposed two ideas for the leafy-fringe / hard-edge cut problem
we hit with v1-v9 (all hard binary keep/drop masks):
1. **Opacity modulation** — instead of dropping outside-AABB Gaussians, apply a
   smoothstep opacity falloff so ellipsoids fade to zero rather than pop.
2. **SDF-based suppression** — compute (signed) distance to the mesh and
   dynamically shrink each Gaussian's scale or opacity as the distance grows.

Neither idea had been tried in v1-v9; all previous variants were binary
threshold cuts.

**v10 (opacity fade at mesh-AABB) — new script.**
[`aggressive_prune_v10.py`](aggressive_prune_v10.py). Same mesh-AABB
(splat-space, expanded by margin) as v3, but instead of hard-cropping,
computes `dist_outside` per Gaussian center and multiplies the sigmoid
opacity by `smoothstep(inner_edge, outer_edge, dist)`. No Gaussians are
dropped by position; only `--drop-if-below` removes ones whose new
sigmoid falls under a floor.

  | Trial | Falloff | % kept | n_full | n_partial | n_zeroed | Path |
  |---|---|---:|---:|---:|---:|---|
  | v10_margin10cm_f5cm  | 5 cm  | 91.90 % | 202,648 | 6,188  | 18,317 | `output/splat_v32_data3__v10_margin10cm_f5cm.ply` |
  | v10_margin10cm_f10cm | 10 cm | 93.85 % | 203,262 | 10,126 | 13,765 | `output/splat_v32_data3__v10_margin10cm_f10cm.ply` |
  | v10_margin10cm_f20cm | 20 cm | 96.73 % | 204,354 | 15,561 | 7,238  | `output/splat_v32_data3__v10_margin10cm_f20cm.ply` |

Falloff-width analysis: 5 cm behaves almost identically to v3's hard crop
(partial band captures only 2.7 % of Gaussians vs 8.1 % zeroed). Median
`dist_outside` = 10.6 cm, p90 = 32 cm, max 78 cm. Only 20 cm falloff
places the transition zone over the meaningful p50 → p90 range.

**v11 (mesh-distance smoothstep, scale/opacity/both) — new script.**
[`aggressive_prune_v11.py`](aggressive_prune_v11.py). Uses Open3D
`RaycastingScene.compute_distance` for unsigned distance to the nearest
mesh face (true SDF isn't defined for our open OpenMVS mesh; unsigned
distance is the well-defined approximation). Smoothstep multiplier
between `--inner-m` and `--outer-m`, applied to either scale
(`log(scale) += log(mult)` across scale_0/1/2), opacity (same math as
v10), or both. Distance distribution on V32: median 2.45 cm, p90
26 cm — most Gaussians sit close to the mesh, so a per-mesh-face fade is
much finer-grained than v10's AABB.

  | Trial | Mode | Inner→Outer | % kept | n_full | n_partial | n_zeroed | Path |
  |---|---|---|---:|---:|---:|---:|---|
  | v11_scale_i2cm_o20cm    | scale   | 2 → 20 cm | 85.43 % | 125,811 | 72,302 | 29,040 | `output/splat_v32_data3__v11_scale_i2cm_o20cm.ply` |
  | v11_scale_i5cm_o30cm    | scale   | 5 → 30 cm | 89.21 % | 161,151 | 46,193 | 19,809 | `output/splat_v32_data3__v11_scale_i5cm_o30cm.ply` |
  | v11_opacity_i2cm_o20cm  | opacity | 2 → 20 cm | 86.99 % | 125,811 | 72,302 | 29,040 | `output/splat_v32_data3__v11_opacity_i2cm_o20cm.ply` |
  | **v11_both_i2cm_o20cm** | both    | 2 → 20 cm | 85.41 % | 125,811 | 72,302 | 29,040 | `output/splat_v32_data3__v11_both_i2cm_o20cm.ply` **← chosen** |
  | v11_both_i5cm_o30cm     | both    | 5 → 30 cm | 89.20 % | 161,151 | 46,193 | 19,809 | `output/splat_v32_data3__v11_both_i5cm_o30cm.ply` |

**Verdict (user visual inspection).** `v11_both_i2cm_o20cm` wins:
mesh-distance driver is finer-grained than v10's AABB shell, and doing
both scale-shrink AND opacity-fade attenuates far-from-mesh floaters
harder than either alone.

**Integration.** [`_pipeline_full.py`](_pipeline_full.py) stage 2 now
subprocess-calls `aggressive_prune_v11.py` with `--mode both --inner-m
0.02 --outer-m 0.20`. Old `--margin-m` CLI arg dropped, replaced with
`--cleanup-mode` / `--cleanup-inner-m` / `--cleanup-outer-m`.
`_write_readme` + summary.json now record the new params.

**Smoke test** at `output/pipeline_v32_v11_smoketest/`: full pipeline
runs end-to-end, 227,153 → 194,016 Gaussians (85.41 %) matching the
standalone v11 trial exactly. Scene-without-objects downstream stage
unaffected (drops 12,732 Gaussians via object footprints as before).

---

### 2026-07-06 — scene_without_objects patches fixed (color, placement, ghosts)

**Trigger.** User visually flagged three failure modes in the mesh patches
that cover the removed-object holes:
1. **Lobster patch** correctly placed but tinted RED (object colour bleed).
2. **Bottle patch** correct but a dark **mesh + splat "stub" survives** above
   the patch (partial object leftover).
3. **Blue box patch** placed on wrong plane AND tinted BLUE.

**Root causes found + landed.**

- **Fix 1: SAM3 mask guard bypass.** [`_pipeline_full.py`](_pipeline_full.py)
  `_resample_patch_rgb` was calling `_best_view_per_point(...)` without
  passing `sam3_masks`. Guard defaulted to `None`, so RGB sampling picked
  from views where the object was still visible → object colour baked in.
  Switched to `_topk_views_per_point` + `_sample_color_topk` (top-5 views +
  per-channel median RGB, robust to mask anti-aliasing edge bleed) and
  plumbed `sam3_cache = project_root/output/segmented_<scene>/masks` +
  `prompt_slug` through the call site. Also propagates `desk_normal_metric`
  from the NPZ when present (improves top-K scoring).

- **Fix 2: blue-box patch placed on wrong plane.**
  [`_unseen_core_map.py`](_unseen_core_map.py) `fit_local_desk_plane` had a
  35° tilt guard on the RANSAC-fitted normal to reject wall lock-ins.
  V32's desk tilts 34.7° for bottle/lobster halos (accepted) but the box
  halo fit at exactly **35.0°** → rejected → fell back to a flat
  axis-aligned `y=0.118 m` plane while the real desk near the box is at
  Y ≈ 0.219 m. Widened guard default to **45°** (walls tilt >70°, so still
  safe) and exposed `--auto-desk-max-tilt-deg` CLI arg.

- **Fix 3: bottle stub survived above the patch.** Same file, stage 4 was
  using the extracted-mesh Y_max as the crop's `y_max` (0.234 m for a
  bottle that's really 32 cm tall — SAM3 misses the translucent top).
  Now extends `y_max = y_min + 1.0 m` for BOTH the mesh crop AND the splat
  crop (originally only mesh, but adversarial review found 3,491 splat
  ghosts above the bottle from Y=0.239 up to 0.462 m — Fix 3 was
  asymmetric and left the splat side broken). Unified into a single
  `footprints` list; XZ is still the tight silhouette + 0.5 cm.

- **Fix 4 (from adversarial review): Open3D RANSAC nondeterminism.**
  `pcd.segment_plane(...)` doesn't accept a seed and Open3D's global RNG
  wasn't set. So Fix 2's success (blue box passing at exactly 35.0°) was
  one RNG realization; a re-run could produce 46° and silently regress to
  flat-plane fallback. Added `o3d.utility.random.seed(0)` at the top of
  `_unseen_core_map.py::main` (also configurable via `--auto-desk-seed`).
  Confirmed determinism across two consecutive runs.

**Numerical proof (output/pipeline_v32_v11_patches_v2/).**

| Slug | Old patch mean RGB | New patch mean RGB | Splat ghosts above patch |
|---|---|---|---:|
| water bottle | (170, 158, 112) — bleached by white bottle | **(155, 143, 98)** — tan | 0 (was 3,491) |
| blue box | *0 grid-points matched to mesh* — wrong plane | **(142, 133, 88)** — tan, B lowest ✓ | 0 (was 1,511) |
| red lobster | (167, 137, 99) — reddish pull | **(159, 138, 92)** — tan | 0 (was 499) |

100 % of NPZ grid points now match mesh patch vertices for every object
(blue box was 0/960 before Fix 2). SAM3 guard rejected 5,659 - 8,557
(point, view) pairs per prompt during resampling. Pipeline pass produced
`output/pipeline_v32_v11_patches_v2/` deliverable.

**Adversarial review verdict (4 parallel agents, workflow `wvvyq0ryz`).**
Reviewers found three "blocking" issues; two were real (splat-side
asymmetry + RANSAC nondeterminism, both addressed above) and one was
empirically false on V32 (reviewer assumed desk sat at world Y ≈ 0 but
the desk is tilted 35°, so the local desk elevation at each object's XZ
centroid matches its `Y_min`, not world Y = 0). Non-blocking follow-ups
noted for later:
- No assertion that SAM3 mask resolution matches source photo resolution
  (would silently mis-sample on mixed-res inputs)
- `_nn_fill_missing` crashes rather than gracefully filling when every
  point has NaN colour (unrealistic on V32 but worth a guard)
- Dead code: `_best_view_per_point` + `_sample_color_for_points` no
  longer called by `_pipeline_full.py` after Fix 1

---

### 2026-07-06 (later) — Stage 4 crop rewritten: 3D distance-to-object-mesh (was XZ silhouette)

**Trigger.** After the 2026-07-06 patch-color / blue-plane / bottle-stub
fixes, user visually flagged that stubs, red blobs, and a "blue vertical
stripe" were still visible in the v2 output. Root-cause analysis showed
the XZ silhouette + Y range crop (with a 0.5 cm and later 3 cm dilation)
was missing ~800 anisotropic Gaussians per object sitting 3-5 cm outside
the tight silhouette but whose ellipsoids extended into the object.

**Fix.** Replaced the XZ silhouette crop with a **3D distance-to-mesh
predicate**. For each object:

  pred = _build_object_distance_predicate(
      extracted_ply, dist_threshold_m=0.05, y_ceiling_offset_m=1.0)

Uses Open3D `RaycastingScene.compute_distance` (unsigned distance to
nearest triangle) with a Y-ceiling cap (Y_min + 1.0 m) to prevent the
query from picking up distant ceiling/wall faces above the desk.

Both mesh and splat sides use the same predicate:
  - **Splat**: drop Gaussians whose CENTER is within 5 cm of any object.
  - **Mesh**: drop any face where ANY vertex is within 5 cm of any object
    (was: centroid-only; the any-vertex test fixes dangling boundary faces).

**Numerical verification (output/pipeline_v32_v11_patches_v4/):**

| Metric | v2 (silhouette) | v3 (3 cm dilation) | v4 (3D distance) |
|---|---:|---:|---:|
| Splat gaussians removed | 12,732 | 16,492 | 17,070 |
| Mesh faces removed | 93,484 | 117,156 | 156,283 |
| Non-patch splat within 5 cm | 3,205 | 1,352 | **0** ✓ |
| Non-patch mesh verts within 5 cm | 33,854 | 13,918 | **0** ✓ |
| Anisotropic (>5x) within 5 cm | 2,479 | 1,033 | ~0 |
| Patch mean RGB (bottle) | (155,143,98) | — | (155,143,98) |
| Patch mean RGB (blue box) | (142,133,88) | — | (142,133,88) |
| Patch mean RGB (lobster) | (159,138,92) | — | (159,138,92) |

**Regression sniff (SHA-verified):** `scene_full.ply`, `scene_cleaned.ply`,
and all 6 per-object PLYs are byte-identical between v2 and v4 (stages
1-3 and 5 unchanged; only stage 4 differs).

**Adversarial review (4-agent workflow `wizpox1tf`).** Three of four
reviewers ok. R1 initially flagged 1,074 non-patch survivors within 5 cm
but their patch-identification undercounted the actual 3,319 desk patches
by ~1,000 — the survivors R1 counted were misclassified patch Gaussians.
Verified via NPZ-grid-proximity patch matching that the true non-patch
count within 5 cm is 0 across mesh and splat.

**Follow-ups (non-blocking):**
- Removed dead imports of `_build_footprint_from_extracted_mesh` and
  `_test_inside_footprint` (no longer used after the 3D-distance rewrite).
- SAM3 mask resolution assertion + `_nn_fill_missing` all-NaN guard from
  the previous review still open.
- View-dependent monitor reflection artifacts ("blue stripe") persist
  even after v4 — this is a train-time issue (SpotLessSplats / SAM3
  masking + retrain) rather than a post-process fix.

---

**ACMM — BLOCKED on Windows build chain, multiple-hour port required.**
- CMakeLists targets ancient `cmake_minimum_required(2.8)` — modern CMake removed compatibility (need `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` workaround, or modernize).
- `find_package(CUDA)` was removed in CMake 4.x (replaced by `FindCUDAToolkit`). Even with the `cmake_minimum` workaround, configure fails on `Specify CUDA_TOOLKIT_ROOT_DIR`.
- `OpenCV REQUIRED` — conda's `opencv-python` ships only Python bindings, not C++ headers/libs. Would need `vcpkg install opencv` or a manual OpenCV-SDK install (~1-2 GB, 30-60 min).
- Per workflow investigation, additional patches needed once cmake configures:
  - `mkdir()` POSIX-only at `ACMM.cpp:831, main.cpp:81,314` → use `<filesystem>::create_directories()`
  - `sys/stat.h` / `sys/types.h` in `main.h` → wrap with `#ifdef _WIN32` or remove
  - GCC-only flags in CMakeLists (-pthread, -ffast-math, -march=native) → guard with CMAKE_COMPILER_IS_GNUCXX (already done in CMakeLists, but `-ffast-math` etc still leak)
  - CUDA compute caps `compute_30,sm_30,compute_52,sm_52` are deprecated; need `compute_70+` and ideally `sm_120` for our RTX 5070 Ti
  - `np.asscalar()` deprecated in NumPy 1.16+ — replace with `float(item)` at `colmap2mvsnet_acm.py:374`
- Honest estimate: 4-8 hours of focused C++ porting work on Windows to make it build, then ~15-40 min runtime. **Recommendation**: do this in WSL2 (Ubuntu) where the original Linux build works as-is, rather than porting to Windows MSVC. Out of scope for current session.
