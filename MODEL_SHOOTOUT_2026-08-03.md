# Chair edit model shootout — LaMa vs FLUX Fill vs FLUX Kontext vs Qwen-Image-Edit-2511 vs FireRed-Image-Edit-1.1
**Date:** 2026-08-06 (updated 2026-08-07)
**Test view:** colmap/dense/images.orig_with_chair/DSCF4907.jpg
**Test mask:** output/segmented_room_v9a_fp_v2/masks/DSCF4907__blue_armchair.png (blue_armchair SAM3 mask)
**Prompts:** REMOVE + REPLACE-with-plant (see full prompts in test script _shootout_v2.sh)
**Comparison grid:** output/model_shootout/all_grid_v2.png
**Reference baseline:** v40 LaMa-trained splat (CHAIR_REMOVAL_COMPARISON.md)

## Executive summary (PI-facing)

LaMa remains the only splat-viable model for chair REMOVAL in this pipeline — no other candidate in the shootout displaced it. Qwen-Image-Edit-2511 and FireRed-Image-Edit-1.1 are semantically stronger than LaMa in principle (they understand "remove the blue armchair and the teddy-bear pillow" as a scene-level instruction rather than a paint-out mask), but both operate at a 1024 long-edge and posterize the entire 3114x2074 frame on the round-trip, which would propagate as a globally blurred scene into splatfacto training. FLUX Kontext dev is the story of two failed configurations: with a mask (FluxKontextInpaintPipeline + strength=1.0 + nf4) it is a no-op passthrough; without a mask (instruct-only FluxKontextPipeline at guidance 4.0) it edits but re-imagines unmasked regions per view, causing catastrophic multiview drift across 311 frames. FLUX Kontext no-mask does replace FLUX Fill's "insert a new chair" failure mode for one-off REPLACE renders, but is still not splat-viable at scale. The v40 verdict stands: LaMa for REMOVE; REPLACE is deferred until a native-resolution, mask-anchored, deterministic editor exists.

## Results table

| Model | REMOVE score | REPLACE score | Splat-viability | Wall clock / image (long_edge=1024) |
| --- | --- | --- | --- | --- |
| LaMa (v40 batch) | 9 | n/a | 9 | ~0.15 s (native 3114x2074, no downscale) |
| Qwen-Image-Edit-2511 | 6 | 6 | 3 (remove) / 2 (replace) | ~180 s |
| FireRed-Image-Edit-1.1 | 3 | 2 | 2 (remove) / 1 (replace) | ~180 s |
| FLUX Kontext dev (no-mask / instruct) | 2 | 4 | 3 (remove) / 4 (replace) | ~90 s |
| FLUX Kontext dev (mask / inpaint) | 1 | 1 | 3 (remove) / 2 (replace) | ~90 s |
| FLUX Fill dev | see CHAIR_REMOVAL_COMPARISON.md ("insert new chair" failure) | — | — | ~90 s |

## Per-model findings

### LaMa (v40 batch)
**Removal:** Chair and teddy-bear pillow are cleanly and completely deleted; the fill area shows only a mild soft-focus smear where the chair back used to be, with no ghost objects introduced. All other elements — ottoman, curtains, slippers, wood floor, right-side speaker — are preserved bit-perfect at native 3114x2074.
**Replacement:** Out of scope. LaMa is a paint-out CNN and cannot introduce novel content like a potted plant.
**Quality + splat-suitability:** The only splat-viable option in the shootout. Native-resolution processing means 300 of 311 views come out bit-identical to source, eliminating ghost furniture and multiview texture drift. Fully deterministic — same mask + same input yields the same output on every re-run. The full 311-view dataset runs in ~47 seconds. The only weakness is a faint low-frequency smudge on the floor where the chair sat, which splatfacto-big averages out across viewpoints.

### FLUX Fill dev
**Removal / Replacement:** See CHAIR_REMOVAL_COMPARISON.md — FLUX Fill's dominant failure mode on this mask was to *insert a new chair* into the masked region rather than remove or replace, which is why it was dropped from the v2 shootout in favor of FLUX Kontext for the augmentation slot.
**Quality + splat-suitability:** Not re-tested here; superseded for the REPLACE role by FLUX Kontext dev (no-mask).

### FLUX Kontext dev (mask-mode, FluxKontextInpaintPipeline)
**Removal:** Essentially a no-op passthrough — the blue armchair and teddy-bear pillow are fully present and indistinguishable from the original. The removal edit did not execute at all.
**Replacement:** Same story — chair and teddy pillow remain untouched, no potted plant introduced. Output is a near-pixel copy of the original, indicating the pipeline is not applying the edit inside the mask region under this configuration (strength=1.0 + nf4 quantization).
**Quality + splat-suitability:** Sharpness is trivially perfect because the unmasked pixels are untouched at native resolution, and the pipeline is fully deterministic per-view — but that is only useful if the model actually edits. Running this configuration across 311 views would cost ~7.75 hours of GPU time and reproduce the input dataset unchanged.

### FLUX Kontext dev (no-mask / instruct, FluxKontextPipeline, guidance 4.0)
**Removal:** Chair is NOT removed — the blue armchair and a re-hallucinated cow-print pillow both remain — AND the ottoman has been additionally erased and replaced by an oversized rug. Worst-case failure: target kept, non-target destroyed.
**Replacement:** Visually the most convincing REPLACE in the shootout — a plausible potted plant with sharp foliage appears roughly where the chair was — but the rug pattern, floorboards, ottoman edge, and teddy shape have all shifted vs source because the pipeline re-imagines the full frame without a mask anchor.
**Quality + splat-suitability:** For a one-off render, replace v2 would win the replace category outright. For 311-view splat training, the per-view drift is catastrophic: with 311 different viewpoints and no mask to anchor unmodified regions, the floor, rug, and speaker resolve differently in every view and splatfacto would train through a soup of drifting textures — the plant itself would resolve as a fuzzy green blob. Roughly 7.75 hours to produce something unusable at scale.

### Qwen-Image-Edit-2511
**Removal:** Both the chair and the teddy-bear pillow are semantically removed and the curtains/floor fill is plausible — the highest-integrity REMOVE semantics of any diffusion model in the shootout.
**Replacement:** The only model that actually produced a large potted plant with tall broad leaves in a black ceramic pot. Pot and foliage are photoreal and prompt-faithful. It fails the strict "replace" contract though: the plant is composited in front of the chair rather than swapping it, so the blue chair and teddy remain visible behind the new plant.
**Quality + splat-suitability:** The 1024 long-edge round-trip destroys texture across the whole 3114x2074 frame. Floorboards, curtains, and rug read as painterly blobs at native resolution; the splat trained on these would inherit that blur globally, not just where the chair was. ~15.5 hours of compute to produce a globally-degraded 311-view training set. Native-resolution inference on this scene would need >24 GB VRAM per view, which is out of budget on the current GPU. Non-viable at pipeline scale as-configured.

### FireRed-Image-Edit-1.1
**Removal:** Teddy-bear pillow is gone but the chair silhouette is only partially removed — a dark blocky invented object still sits where the chair back was. On top of the failed edit, the whole frame is coated in a coarse painterly texture that degrades every preserved element.
**Replacement:** A vague plant-shaped dark mass and pot silhouette appear near the correct location, but the whole frame is destroyed by heavy painterly/impasto artifacts. Chair and plant merge into a smear; no recognizable black ceramic pot or broad-leaf detail survives. Unusable as training data.
**Quality + splat-suitability:** Worst of both worlds — same 1024-upscale posterization as Qwen, but *also* fails to remove the chair. Strictly dominated by both LaMa (removal) and Qwen (replacement) on every axis for this pipeline.

## PI-facing verdict

LaMa remains the tool of record for REMOVE in this pipeline; the v40 LaMa-trained splat is still the reference and nothing in the shootout justifies retraining against a different remover. FLUX Kontext dev (no-mask / instruct) replaces FLUX Fill in the augmentation slot — Kontext at least produces a convincing potted plant in one-off renders rather than FLUX Fill's "insert a new chair" pathology, so use Kontext no-mask when a single publicity render of a *replaced* scene is needed. Qwen-Image-Edit-2511 and FireRed-Image-Edit-1.1 are novel and semantically the strongest at understanding REMOVE at the scene level, but their forced 1024 long-edge inference posterizes the entire 3114x2074 frame and makes them a poor choice for splat training without further work — native-resolution inference would need >24 GB VRAM per view, which is out of budget on this GPU. Bottom line: LaMa for REMOVE; Kontext no-mask for one-off REPLACE renders only; Qwen/FireRed shelved until a native-resolution path exists.

## Bugs + fixes documented

- **Kontext w/ mask (v1 test)** — FluxKontextInpaintPipeline + `strength=1.0` + nf4 quantization produced near-identical-to-input output (no-op passthrough). Fixed by removing the mask and switching to plain FluxKontextPipeline with an instruct-only prompt at `guidance=4.0`. Note: this fix trades the no-op failure for a per-view drift failure — usable for one-off renders, still not splat-viable at 311 views.
- **Qwen / FireRed initial OOM** — 20B DiT + Qwen2.5-VL text encoder both loaded bf16 = 28.5 GB, over budget on the 24 GB GPU. Fixed by nf4-quantizing BOTH the transformer AND the text encoder, plus enabling `enable_model_cpu_offload()`. This is what forces the 1024 long-edge cap; native 3114x2074 inference remains out of reach even after quantization.

## Files produced

- `output/model_shootout/{flux_kontext,qwen_edit,fired_edit,flux_fill}/*.jpg`
- `_shootout_v2.sh`, `_shootout_run_one.py` — reproducible harness
- `_make_shootout_grid.py` — grid composer
- `output/model_shootout/all_grid_v2.png` — grid
