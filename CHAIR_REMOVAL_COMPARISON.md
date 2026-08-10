# Chair-removal from full-quality room splat — LaMa vs FLUX comparison

**Date:** 2026-08-03
**Baseline splat:** `output/splat_v39_da3_fullq.ply` (native-quality room reconstruction, 1.95M gaussians, blue armchair + teddy bear present)
**Goal:** remove the armchair from the splat by (1) inpainting each source view to erase the chair, (2) retraining the splat on the edited images.

## Result: v40 (LaMa-trained) is the canonical chair-removed room splat

- **File:** `output/splat_v40_da3_no_chair_lama.ply` (448 MB, 1.89M gaussians)
- **QA:** `output/july/room_no_chair_v40/qa/v40_lama_5views.png` (5 chair-visible views) vs `v39_baseline_5views.png` (same views with chair)
- **Verdict:** chair + teddy bear cleanly removed across all views; some soft blur in the vacated region (expected — no ground-truth geometry existed behind the chair), but no ghosting, floaters, or craters.

## FLUX Fill dev: attempted, categorically unsuitable

Three separate configurations tested; all failed the same way. **FLUX Fill dev is trained for object INSERTION/REPLACEMENT, not object removal.** With a chair-shaped mask, the model consistently invents a *new* chair in place of the old one regardless of prompt or guidance:

| Config | Prompt | Guidance | Dilate | Result |
|---|---|---|---|---|
| Smoke 1 | "wood floor, curtain, no chair, no furniture..." | 30 | 80 px | Beefy grey armchair with cow-print pillow |
| Smoke 2 | "wooden floor, grey curtain" | 3.5 | 150 px | Mid-century grey armchair with grey pillow |
| Smoke 3 | `""` (empty) | 1.5 | 250 px | Dark green wing-back recliner with tan pillow |

Sample outputs saved at `output/flux_chair_removal/native/smoke*/images/`.

**Why this happens:** the mask shape gives FLUX Fill strong geometric prior "put something chair-shaped here." Since it was trained on object-insertion data, it satisfies that shape by generating plausible furniture, ignoring negative-prompt hints and low-guidance settings. This is fundamental training bias, not a solvable tuning issue.

**Independent-view inconsistency** would compound the problem: FLUX with fixed seed still generates a *different* chair per view (each mask has a slightly different shape), so retraining a splat on those 110 images would produce a splat with a "ghost furniture" region rather than an empty floor.

## Pipeline built + benchmarks

Both wrappers live at project root:
- `_inpaint_chair_lama.py` — IOPaint's LaMa (Apache 2.0), deterministic feedforward CNN, ~0.15 s/img on native 3114×2074, ~1.5 GB VRAM
- `_inpaint_chair_flux.py` — HuggingFace `FluxFillPipeline` with nf4 quantized transformer + T5 (fits in ~10 GB VRAM), ~25 s/img at 1024 long-edge

## Recommendation for the PI

- **For object removal in this pipeline: use LaMa.** It's the correct tool for the job — deterministic (multi-view consistent), fast, no license restrictions, and specifically designed for filling based on surrounding pixel context.
- **FLUX Fill dev is the wrong tool for removal.** Use FLUX for OBJECT REPLACEMENT tasks (e.g., swap a chair for a different chair, or add a new object) — that's what it was trained for. It excels at those tasks; it just doesn't do removal.
- If we ever need higher-quality removal than LaMa provides (e.g., if fine surface texture is critical), the right SOTA options in mid-2026 are **PowerPaint v2** or **OmniEraser** — both trained on paired frames specifically for removal, both fit on 16 GB VRAM. Not tested here because LaMa v40 output is already sufficient for our splat pipeline.

## Files produced (all in `reconstruction_project/`)

- `output/splat_v40_da3_no_chair_lama.ply` — final chair-removed splat
- `output/splat_v39_da3_fullq.ply` — baseline with chair
- `output/lama_chair_removal/native_2074/images/` — 311 LaMa-inpainted images used for v40 training
- `output/flux_chair_removal/native/smoke*/images/` — FLUX smoke outputs (documentary evidence of the failure mode)
- `output/july/room_no_chair_v40/qa/` — 5-view render comparison v40 vs v39
- `nerfstudio/dense/splatfacto/2026-08-03_113945/` — v40 training checkpoint
- `_inpaint_chair_lama.py`, `_inpaint_chair_flux.py` — both wrappers, kept for future reference

## Also worth documenting

Two bugs found + fixed along the way:
1. **Path bug:** initial fine-tune wrapper populated `nerfstudio_data/images/` but splatfacto reads from `colmap/dense/images/`. Wasted ~2 hr before catching. Fix: images go into `colmap/dense/images/` for retraining.
2. **Learning-rate bug:** `--load-dir` style resume uses the checkpoint's already-decayed LR schedule; at step 30 000 LR is 1.6e-6 (final value), which produces sub-fp32 updates. Fine-tuning off a completed checkpoint doesn't actually change weights. Fix: retrain from scratch on inpainted images (~50 min).
