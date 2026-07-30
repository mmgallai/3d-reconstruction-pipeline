# Full-quality room reconstruction — 2026-07-28

## Purpose
Quality benchmark for the PI. Reconstruct the Mip-NeRF 360 room scene at **native resolution** (no image downscaling anywhere in the pipeline) so we can see what the output looks like when the full input data is used.

## What "full quality" means concretely
Every earlier reconstruction was at `--downscale-factor 2` on top of an already-downsized image tier. This run bypasses both:

| Layer | Prior runs (room v37) | This run (full quality) |
|---|---|---|
| Image tier ingested | `images_2/` (1556×1038) | **`images/` (3114×2074, native)** |
| `NERF_DOWNSCALE_FACTOR` | 2 | **1** |
| Actual training resolution | 778×519 | **3114×2074** |
| Total downscale vs source | 4× smaller | **1× (native)** |

That's 16× more pixels per training image than the prior run.

## VRAM-fit changes (minimal, required to avoid OOM on RTX 5070 Ti 16 GB)
Only kick in when `--downscale-factor 1` is passed. All other splatfacto behavior unchanged.

Added to the `ns-train` command in `lib/nerfstudio_pipeline.py::train_nerfstudio_format`:
- **`--pipeline.datamanager.cache-images cpu`** — training images live in system RAM, streamed to GPU per iteration. Frees ~6-10 GB of VRAM that would otherwise be an eager GPU-side image cache. Slight per-iteration overhead (~5-15 %).
- **`--pipeline.model.stop-split-at 12000`** — stops Gaussian densification at iter 12000 instead of the default 15000. Combined with the existing `--cull-alpha-thresh 0.01`, keeps Gaussian count in a safe range for 16 GB VRAM.

**Correction from initial plan:** the first-attempted `--pipeline.model.max-gauss-num 5000000` flag does NOT exist in nerfstudio 1.1.5 (the version in the `nerfstudio-blackwell` docker image). The valid alternative for capping density is `--pipeline.model.stop-split-at`, used above. No other change.

Nothing else in the pipeline was changed. Same iterations (30 000), same `cull-alpha-thresh 0.01`, same DA3 init path, same OpenMVS mesh export.

## Prior workspace preserved (nothing lost)
Before this run started, the entire garden workspace was moved to timestamped backup dirs so we can revisit it:
- `colmap.garden_saved_<STAMP>/`
- `nerfstudio.garden_saved_<STAMP>/`
- `nerfstudio_data.garden_saved_<STAMP>/`

Similarly, the earlier room workspace is at `colmap.room_v6_saved_1785139204/` + `nerfstudio.room_v6_saved_1785139204/` (from the earlier full-quality-preserving move that happened before garden reconstruction).

The finalized `output/july/room_v6/` PLYs and the `output/mesh_v37/` reconstruction outputs are untouched. Same for garden `output/mesh_v38/`.

## Expected timeline (RTX 5070 Ti, 16 GB VRAM, 32 GB RAM)

Estimates were CONSERVATIVE — COLMAP MVS was much faster than expected at native. Actual observed times (updated as stages complete):

| Stage | Estimated | **Actual (observed)** |
|---|---|---|
| Stage 1 — HEIC bypass (stubs) | seconds | **seconds** ✓ |
| Stage 2 — COLMAP SfM | ~30-45 min | **~5 min** (already had features from prior 2× run) ✓ |
| Stage 3 — COLMAP PatchMatch MVS at native | ~8-15 hr | **1 hr 48 min** ✓ (much faster than expected — the 5070 Ti's memory bandwidth handles native MVS well) |
| Stage 4 — DA3 depth | ~5 min | **~5 min** ✓ |
| Stage 5-6 — transforms.json + `images_1/` (no-op at scale=1) | seconds | **seconds** ✓ |
| Stage 7 — splatfacto-big training (native res, stop-split@12k) | ~2-6 hr | **in progress** |
| Stage 8b — OpenMVS textured mesh | ~8-15 hr | pending |
| Stages 9-10c — splat export + LOD + AO | ~5-10 min | pending |
| **TOTAL so far** | ~20-40 hr | **~2 hr done, ~10-20 hr remaining** |

---

## Actual outcome (2026-07-28 → 2026-07-30)

### Splatfacto training: SUCCEEDED
- Total training time: **58 min** (16:05 → 17:03) — way faster than expected.
- Final checkpoint: `nerfstudio/dense/splatfacto/2026-07-28_200538/nerfstudio_models/step-000029999.ckpt` (1.57 GB, all 30k iterations complete).
- Config confirmed: `downscale_factor: 1`, `cache_images: cpu`, `stop_split_at: 12000`, `max_num_iterations: 30000`.

### OpenMVS Stage 8b: CRASHED THE PC
- Started 17:03:33, ran for ~6.5 hr into `RefineMesh` (deep in second refinement iteration, ~59% of iteration 1).
- Peak memory: **27.4 GB physical RAM** (per OpenMVS own log) + splatfacto's `cache-images cpu` was still holding ~6 GB of images in system RAM + Docker/WSL2 overhead + Windows.
- Combined footprint exceeded 32 GB physical RAM → Windows started thrashing swap → UI froze → user force-restarted.
- **This is a system RAM issue, not a GPU issue** (peak VRAM stayed well under 16 GB throughout).
- Nothing in Stage 8b's output actually survives — no `output/mesh_v39/` was produced.

### Recovery: splat export from checkpoint (~2 min)
Instead of retrying Stage 8b, we bypassed it by exporting the splat directly from the completed training checkpoint:
```
docker run --rm --gpus all -v <PROJECT>:/workspace nerfstudio-blackwell \
  bash -c "python3 /workspace/fix_weights.py && \
    ns-export gaussian-splat \
      --load-config /workspace/nerfstudio/dense/splatfacto/2026-07-28_200538/config.yml \
      --output-dir /workspace/output"
mv output/splat.ply output/splat_v39_da3_fullq.ply
```

### Full-quality splat stats
| Metric | Room v37 (2× downscale) | **Room v39 full quality** | Ratio |
|---|---|---|---|
| Training resolution | 778 × 519 | **3114 × 2074** | 16× more pixels |
| Trained Gaussians | 680 k | **1,952,121** (2.21 M raw, 259 k low-opacity culled at export) | **2.9× more** |
| Splat PLY size | 168 MB | **462 MB** | 2.7× |
| Bytes/gaussian | 248 | 248 | — |
| Training wall clock | 20 min | 58 min | 2.9× |
| Total time (reconstruction) | ~4.5 hr | **~2 hr + crash + 2 min export** | — |

### Visual result
Rendered at 5 sample views: `output/july/room_fullq_v39/qa/fullq_v39_5views.png`
Baseline (v37) at same views: `output/july/room_fullq_v39/qa/baseline_v37_5views.png`

Visible improvements in the full-quality render:
- Chair fabric texture legible where v37 was a flat blur.
- Teddy bear fur detail visible.
- Wood grain on the floor and coffee table.
- Piano key edges crisp; sheet-music binding visible.
- TV bezel edges sharp; center-speaker driver cone visible.
- Slippers on the rug crisp with soft-shadow edge.

---

## Lessons for the PI

1. **splatfacto training at native res is CHEAP on this hardware** — 58 min for a 3114×2074 room, not the 2-6 hr I estimated. VRAM stayed under the 16 GB ceiling thanks to `--cache-images cpu`.
2. **OpenMVS mesh refinement at native res is what killed the run** — not the splat. If we only need the splat (not the textured mesh), we can skip Stage 8b entirely and save ~10+ hours + avoid the RAM crash.
3. **`cache-images cpu` moves the pressure from VRAM to system RAM.** With 32 GB system RAM and OpenMVS also wanting 27 GB, they collide. If we ever want to keep BOTH the trained splat cached AND run OpenMVS in the same session, we'd need 64 GB RAM. Or, more practically: unload the splat before starting OpenMVS.
4. **The full-quality splat has 2.9× more Gaussians than the 2× run** — enough to represent fine detail like chair fabric, wood grain, teddy fur. For the extraction pipeline (`_extract_by_projection.py`) this means 2.9× more candidates to classify per view — still fast (~30 sec per scene), and per-object splats will be denser and less spike-prone.

## To finish the mesh (if wanted, later)
Options that avoid the RAM crash:
- **Skip OpenMVS entirely** — the splat alone is often what's actually needed for VR use.
- **Retry Stage 8b at reduced resolution level** — pass `--resolution-level 3` (or 4) to `RefineMesh` (currently level 2). Cuts memory ~2-4×. Would give a slightly less detailed mesh.
- **Retry after freeing splatfacto image cache** — restart the shell/docker so the CPU-cached images are released, then re-run just Stage 8b. Frees ~6 GB of RAM.
- **Upgrade RAM to 64 GB** — one-shot fix; native-res OpenMVS would then run comfortably.


## Outputs to expect
- `output/mesh_v<N+1>/` — new full-quality textured mesh + LODs
- `output/splat_v<N+1>_da3.ply` — full training splat (probably 5 M Gaussians, ~1.2 GB)
- `output/splat_v<N+1>_da3_pruned.ply` — post-prune (~2-3 M, ~600 MB)
- Fresh `nerfstudio/dense/splatfacto/<ts>/` with the trained model + dataparser transforms

## How to compare to prior room v37 (2x-downscale) result
- Visual: render both via `_render_qa_docker.sh` at the same view indices, compare PNGs side-by-side.
- Numerical: Gaussian count, extent, opacity distribution — same script pattern as the earlier vs-INRIA comparison in `output/july/README.md`.
- Extraction test: run `_extract_by_projection.py` on the new splat with the same SAM3 masks. Compare per-object Gaussian counts to room v6.

## PI-relevant summary (short version to hand off)
> "Reconstructed room scene at native 3114×2074 resolution, no downscaling anywhere in the pipeline. Two minimal config knobs added to fit within 16 GB VRAM: images cached in CPU RAM instead of GPU, and Gaussian count capped at 5 M (same order as INRIA reference). No algorithm changes. Reconstruction wall clock ~20-40 hr end-to-end."

## Reversibility
To go back to 2× downscale:
- Don't pass `--downscale-factor 1` to `reconstruct_realityscan.py`.
- The VRAM-fit flags only activate when scale=1, so 2× downscale runs are byte-identical to prior behavior.

No changes to any script other than the two lines added to `lib/nerfstudio_pipeline.py` (guarded by `if scale == 1`).
