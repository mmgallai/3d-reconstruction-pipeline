# Comprehensive plan — scene-vs-object hole problem (VR digital twin pipeline)

*Synthesis of three independent expert opinions (Gemini, Claude, GPT) + 6-agent verification sweep of every claim that mattered. Multi-route plan with explicit decision points.*

**Status:** draft for execution — 2026-06-18

---

## TL;DR

The three experts converged on the same headline answer: **use the OpenMVS / TSDF mesh as a depth scaffold under the cropped-out object, and synthesise only the missing RGB texture** (not depth, not geometry — those are already in the mesh). My verification sweep confirms this is also where the entire open-source field has converged in the last 12 months (InFusion, GScream, AuraFusion360, 3DGIC, GSFix3D all do depth-guided variants).

The project's edge that nobody else has: **we already own a metric ToF mesh**. Every published method's hardest sub-problem is hallucinating the depth under the object; we skip that step entirely.

The plan below has **five parallel tracks**, each with kill criteria. None of them block the others. We commit to evidence as we run them, not now.

| Track | What | Owner-time | Kill criteria | Verdict on day-0 |
|---|---|---|---|---|
| **0** | Orthogonal infrastructure (GLOMAP, SPZ, aras-p, EdgeAware-L1 depth, ToF-IR pre-clean) | 1 week | n/a — no-regret | **GO** |
| **1** | Measure: how big is the truly-unseen core after cross-view reprojection? | 2 days | n/a — informs everything | **GO** |
| **2a** | Build our own mesh-seeded fill (Gemini's 4 modules, LaMa for RGB) | 2–3 weeks | Truly-unseen core > 70 % desk patch AND LaMa fill visibly inconsistent | **GO** — primary path |
| **2b** | Test GSFix3D `--dual_input` as an off-the-shelf comparator | 1–2 weeks | Doesn't build on Windows + sm_120, OR can't get SD2 weights, OR &gt; 16 GB VRAM | **TEST** |
| **3** | Capture-side disocclusion sweep (SAM3 + COLMAP + transforms.json mask) | 1 week plumbing + needs reshoots | User can't reasonably perform the sweep on their objects | **GO** — parallel track |
| **4** | Unity hide-and-reveal + shadow plate fallback | 3 days | Tracks 2 or 3 ship a visually clean fill | **HOLD** — only if 2 & 3 fail |

**No-regret moves to start today:** Track 0 (infrastructure) + Track 1 (measurement). Both are cheap, both inform every other decision.

---

## 1 · What all three experts agreed on (verified)

| Claim | Gem | Cl | GPT | Verified? |
|---|---|---|---|---|
| Mesh / ToF gives us the depth scaffold; **only RGB needs synthesis** | ✅ | ✅ | ✅ | ✅ confirmed by the entire 2024-26 depth-guided 3DGS inpainting literature |
| **Naive 2D inpaint of all 140 views → retrain** is multi-view-inconsistent and produces smear | ✅ | ✅ | ✅ | ✅ confirmed — this is Gaussian Grouping's mechanism and SplatFill's within-mask PSNR of 15.67 shows it's hard for all methods |
| **LaMa beats SDXL / FLUX for repetitive non-semantic desk textures** | ✅ | ✅ | (~) | ✅ confirmed — Fourier-conv global receptive field is designed for periodic textures; LaMa is *deterministic* so 1-3 anchor views stay consistent by construction |
| **Hide-and-reveal alone (option C in our brief) is emergency UX only**, breaks on relocation | ✅ | ✅ | ✅ | ✅ confirmed — fails the user's own "put it back somewhere else" case |
| Capture-side disocclusion sweep (option D variant) is **highly underrated** | ✅ | ✅ | ✅ | ✅ confirmed — COLMAP `--ImageReader.mask_path` + SAM3 hand-masking is plumbing not research |
| Full-scene generative regeneration (TRELLIS, Marble) is **unsuitable for a faithful twin** | (-) | ✅ | (-) | ✅ confirmed — hallucinated content fails the project's "faithful" requirement |

## 2 · What survived verification vs what didn't

### Survives — methods we can actually use today

| Method | Status | License | Bottom line |
|---|---|---|---|
| **LaMa** via `simple-lama-inpainting` / `IOPaint` | Apache-2.0 weights + code; mature; plain PyTorch sm_120 OK; ~4 GB VRAM, &lt;1 s/call | Apache-2.0 | **USE** — the 2D RGB synthesiser, deterministic, purpose-built for repetitive textures |
| **GSFix3D** (Nov 2025 release) | Recent; **`--dual_input` flag verified real** — consumes paired mesh-render + 3DGS-render in latent space; SOTA on Replica (26.49 PSNR) / ScanNet++ (25.63) | Apache-2.0 code + RAIL++-M weights + 3DGS-license (research OK, commercial restricted) | **TEST** — the only off-the-shelf method that natively does what the experts prescribed |
| **GLOMAP** (now in COLMAP 4.0 as global mapper) | Standalone repo archived Mar 2026 → folded into COLMAP. Windows binaries available. Identical sparse-model format | BSD-3-Clause | **USE** — drop-in via `colmap mapper` + `ns-process-data --skip-colmap` |
| **Niantic SPZ v4** | MIT, May 2026 release, ~10× smaller than PLY, ZSTD compression, no point cap; read natively by aras-p UnityGaussianSplatting | MIT | **USE** — convert at packaging time |
| **aras-p UnityGaussianSplatting main** | Has ninjamode's VR fixes upstreamed (AMD FFX radix sort + Quest-compatible sort); SPZ import built-in | MIT | **USE** — runtime renderer for Quest 3 |
| **SAM3** + **COLMAP `--ImageReader.mask_path`** | SAM3 Apache-2.0, open-vocabulary text prompts; COLMAP BSD; well-documented mask pipeline | Apache + BSD | **USE** — disocclusion-sweep plumbing |
| **3DGIC cross-view reprojection** (the IDEA, not the repo) | Repo is honestly labelled "suboptimal version" — full code is corporate-held. The reprojection LOGIC is portable | research-only / custom | **TEST** — port only the reprojection logic as a standalone preprocessing step |
| **ToF-Splatting L1-quantile filter** (the IDEA, not the code) | No code released; multi-view filter is baked into a full SLAM pipeline — but the core algorithm is ~50-100 LOC | n/a (reimplement) | **TEST** — DA3 vs Femto ToF L1, drop top quantile; directly attacks the monitor blob |

### Discarded — claims that did not survive verification

| Claim | Verdict |
|---|---|
| InFusion is the canonical "depth as scaffold" we can adapt | ❌ Stale repo (Jul 2024), built on vanilla 3DGS not gsplat 1.5.3, depth-completion network is **non-swappable** — borrow the *idea* (unproject from depth), not the codebase |
| AuraFusion360 is the strongest off-the-shelf inpainter | ⚠️ Credible CVPR 2025 baseline (Apache-2.0, 17.90 PSNR / 0.424 LPIPS on 360-USID), but architecturally **360-tuned** — will over-hallucinate on a small near-planar desk patch (the failure mode the Claude reviewer himself flagged) |
| GScream's depth-prior + cross-attention is a useful drop-in | ⚠️ Useful ideas, but **NO LICENSE FILE** (legal risk) + Scaffold-GS dependency + sm_86 (RTX 3090) targeting + relies on Marigold+SD+LaMa external pipeline; would need a Scaffold-GS port first |
| SplatFill is the SOTA we should aim for | ❌ **NO public code**, A100-80GB training, only +0.01 PSNR over GScream, paper-only |
| FLUX.1 Fill is the modern diffusion inpainter to use | ❌ **Non-Commercial License v2.0** on weights — blocks open-source pipeline distribution; also 24 GB VRAM bf16 won't fit our 16 GB; also stochastic so risks inconsistent grain across anchor views |
| Mobile-GS sort-free is the room-scale Quest renderer | ❌ Quest 3 numbers **do not exist** — all reported FPS are Snapdragon 8 Gen 3 PHONE; mobile Vulkan code is closed-source by company policy; no Unity/Quest integration path |
| ninjamode VR fork is the production renderer | ⚠️ Frozen reference (last commit Oct 2024); README itself says **"VR support has been upstreamed to aras-p/UnityGaussianSplatting; this repository is likely a better base for your work"** — use aras-p main |
| FisherRF gives 70 FPS active-view selection for real-time capture guidance | ⚠️ Independent reports (Active View Selector, arXiv 2506.19844) measure **5–10 s/view in real deployments**; nerfstudio fork (FisherRF-ns) has 2 stars and looks abandoned. Right idea, wrong maturity |
| Mocking a static duplicate of the object onto an unmodified scene splat is acceptable for VR | ❌ This is the failure mode the user already reported. Hide-and-reveal alone is also inadequate for relocation |

### The single load-bearing claim that is NOT YET verified

GSFix3D's `--dual_input` flag was trained on Replica + ScanNet++ datasets. Whether it **generalises zero-shot to a Femto-Mega capture of an arbitrary desk** — on Windows + sm_120 Blackwell with 16 GB VRAM, after sourcing SD2 weights from a third-party mirror (HF removed them) — is unverified. The entire "off-the-shelf path" hinges on this. **Track 2b's first deliverable is to answer this question.**

---

## 3 · The plan — five parallel tracks

### Track 0 — Orthogonal infrastructure upgrades (no-regret, do now)

Independent of the hole problem; every one is a verified win.

| Task | Where it lands | Effort | Verified |
|---|---|---|---|
| Switch SfM to **GLOMAP** (global mapper in COLMAP 4.0) | `lib/colmap_pipeline.py` (new) — wrap `colmap mapper --Mapper.ba_global_function_tolerance ...` + invoke `ns-process-data --skip-colmap --colmap-model-path` | 1 day | ✅ |
| Add **SPZ v4 export** at packaging time | Add `lib/spz_export.py` that uses Niantic CLI or pip bindings; called from `reconstruct_realityscan.py`'s final stage | 0.5 day | ✅ |
| Switch Unity runtime to **aras-p UnityGaussianSplatting main** (drop ninjamode reference) | Unity project; verify SPZ import + AMD FFX radix sort on Quest 3 | 1 day | ✅ |
| Add **EdgeAwareLogL1 depth loss** to splatfacto-big on the **native gsplat 1.5.3 densifier** (NOT DN-Splatter's broken gsplat-1.0 densifier — that was V25's regression) | `splat_tof/splat_tof/model.py` (extend existing tof plugin) | 2-3 days | ✅ (loss is sound; densifier choice is what V25 got wrong) |
| Add **ToF-IR-ghost L1-quantile pre-clean** before back-projection init | New `_tof_ir_prefilter.py` — run DA3 depth per frame, compare to Femto ToF, drop top 25 %ile L1 residuals, init splatfacto from filtered cloud | 1 day | ✅ (algorithm verified; we already have DA3) |

**Kill criteria:** none — every one is independently useful.
**Acceptance:** all 5 plumbed and a fresh V32 run reproduces V32 quality or better with the same Gaussian count.

### Track 1 — Measure the truly-unseen core (do first, ~2 days)

Before we commit to a fill strategy, **quantify how big the unseen patch actually is**. The expert sweep flagged this as the single most under-investigated assumption: many frames may have seen the desk at some sweep angle, even with the object present.

Implementation:
1. For each object's SAM3 footprint, render the V32 OpenMVS mesh to per-view depth at every training view.
2. For each footprint pixel, check across all 140 views: is there a view where that 3D point is NOT inside any SAM3 object mask AND is in-frame AND not behind another occluder?
3. Output: per-object **truly-unseen mask** + size statistic (%) + a heatmap visualisation.

**Why this matters:** if the truly-unseen core is, say, 15 % of the bottle's footprint (just directly under the cap), we don't need diffusion at all — nearest-neighbour copy from the rest of the footprint that other frames DID see is enough. If it's 90 %, we need the full mesh-seeded fill pipeline.

**Implementation:** ~150 LOC, reuses our existing Open3D BVH raycaster + SAM3 cache + transforms.json.
**Deliverable:** `_unseen_core_map.py` + a per-object size table.
**Decision impact:** if the unseen core is < 30 % for all three V32 objects, simplify the Track 2 fill to NN-copy + minimal blending; skip diffusion entirely.

### Track 2 — Mesh-seeded fill (primary path)

Two branches; we'll run both and compare on the same V32 holes.

#### Track 2a — Our own implementation (Gemini's 4-module recipe)

Highest control, builds on what we already have, no new framework integration.

| Module | What it does | Reuses | New code |
|---|---|---|---|
| **M1 — Mesh depth raycast** | Pick 1-3 anchor views per object. For each, render the V32 OpenMVS mesh to a depth map masked to the SAM3 footprint. RANSAC plane-fit to the surrounding mesh patch to enforce strict planarity under the object base (Gemini's risk mitigation). | `_spatial_crop.py`'s BVH raycaster + Open3D | ~80 LOC |
| **M2 — Sparse 2D inpaint** | Pass each anchor view through **LaMa** (`SimpleLama()(image, mask)`). 1-3 inpainted RGB images of the patch. Optional: render the masked anchor through `prompt = "remove shadows, match ambient lighting"` to suppress baked AO under the object (Gemini's lighting-bake risk). | `simple-lama-inpainting` pip package | ~50 LOC |
| **M3 — Guided unprojection** | Use M1's mesh-depth to unproject M2's RGB pixels into 3D. Spawn new Gaussians ON the mesh surface; initialize their scales to flat-on-plane (using mesh normals), opacity from a small positive logit, SH-DC color from the inpainted pixel. | `_spatial_crop_splat.py`'s position-transform chain | ~120 LOC |
| **M4 — Localized optimization** | Merge new Gaussians into the V32 scene splat. Freeze the original Gaussians. Run splatfacto for ~500 iters optimizing ONLY the new ones, with TV smoothness loss + photometric loss masked to the patch + monitor mask to keep the blob from leaking gradients (Claude's reflective-corruption mitigation). | nerfstudio splatfacto-big + freeze hook | ~200 LOC |
| **M5 — Same loop for the mesh** | Seed faces on the mesh's interpolated surface in the patch region. Bake LaMa-inpainted texture into the existing 8K UV atlas. Update `mesh_v32_data3_openmvs.ply`. | OpenMVS atlas writer + xatlas | ~100 LOC |

**Total new code:** ~600 LOC. Few days to write, one-week shakeout.
**Deliverable:** patched V32 scene splat + patched scene mesh per object; both load cleanly in SuperSplat / Unity.
**Acceptance metrics (Track 3 eval — see §4 below):**
- No visible hole in the patched splat under a held-out trajectory (LPIPS &lt; 0.15 across the boundary)
- No baked duplicate (Remove360 semantic-residual probe — confirm SAM3 prompt for the object returns no Gaussians in the patched scene)
- Patch &lt; 5k Gaussians per object (Quest 3 budget)
- 72 FPS standalone Quest 3 maintained

**Kill criteria:**
- LaMa output is visibly inconsistent across anchor views even though they share fixed geometry → escalate to SDXL-Inpaint with locked seed + identical prompt, then Track 3 capture-side
- M4 freeze-and-fit fails to blend the patch into surrounding desk → drop the localized optimization, use M3's frozen seeded Gaussians as-is (cheaper, just blurrier)
- Truly-unseen core (Track 1) is so large that NN-copy + LaMa cannot synthesize plausible texture → fall to Track 3 (capture)

#### Track 2b — GSFix3D off-the-shelf comparator

The only published method that natively does mesh-render + 3DGS-render → diffusion inpaint. **It either works and validates our whole approach, or fails fast and tells us what's hard.**

Day-1 sanity check (1 day):
1. Clone `https://github.com/GSFix3D/GSFix3D` — verify last commit is recent.
2. Check VRAM ceiling claim: their testbed was 24 GB RTX 4500 Ada Ubuntu; our box is 16 GB RTX 5070 Ti Windows.
3. Source SD2 weights from a third-party mirror (HF removed them) — note the workaround in CUDA_BUILD_NOTES.md.
4. Try to install on WSL2 first (their env was Ubuntu); Windows-native is a stretch.
5. Convert a single V32 hole (the bottle) to their Replica-format input.
6. Run with `--dual_input` consuming our OpenMVS mesh render + V32 splat render.

Pass: it produces output without OOM, runs in &lt; 1 h. Fail: OOM, missing weights, format incompatibility.

If pass → continue with the full V32 dataset, compare side-by-side with Track 2a. If GSFix3D wins on patched-region PSNR/LPIPS by a clear margin, switch primary path. If 2a wins or they tie, keep 2a (zero new dependency).

**Effort:** 1-2 weeks max. Kill at 1 day if blocking issues surface.

### Track 3 — Capture-side disocclusion sweep (parallel, highest faithfulness)

If achievable, this is the gold-standard answer: real observed pixels, not synthesized.

Plumbing (1 week):
1. **ScanGuide UI step**: after the standard 140-frame sweep, prompt the user to "lift each grabbable object briefly and replace it." Record an extra 10-20 frames of empty desk during the lift.
2. **SAM3 hand mask**: for each lift-phase frame, run SAM3 with prompts `["arm", "hand", "fingers"]`. Cache as `IMG_femto_NNNN__hand.png`.
3. **COLMAP integration**: pass the hand masks via `--ImageReader.mask_path` so COLMAP skips feature extraction in masked regions. Verify GLOMAP also honours the flag (it does; same flag).
4. **CRITICAL — nerfstudio integration**: COLMAP's mask_path **only blocks feature extraction**, NOT splatfacto's photometric loss. The hand mask must ALSO be written into `transforms.json` per the `mask_path` convention so splatfacto-big ignores hand pixels during training. (This is the gotcha the verification sweep specifically flagged.)
5. Existing scene-segmenter pipeline runs unchanged — it now sees a complete desk in the scene splat under each object's footprint.

**Effort:** 1 week plumbing + needs a fresh capture (user does the lift step on V32 desk).
**Acceptance:** the patched scene splat has real desk pixels under each object; no synthesis required.
**Kill criteria:** objects too heavy / wired to lift (e.g. desktop monitor) — fall back to Track 2 for those.

This and Track 2 are complementary, not competitive. Track 3 covers movable objects; Track 2 covers everything Track 3 can't reach.

### Track 4 — Unity hide-and-reveal + shadow plate (fallback only)

Only ship this if Tracks 2 AND 3 both fail visually in user testing.

GPT's "layered patch" runtime architecture:
- Per-object Unity prefab with states `{rest, grabbed, displaced}`.
- At `rest`: original scene splat unmodified; object collider hidden under the scene splat duplicate.
- At `grabbed`/`displaced`: hide the scene-splat Gaussians inside the object's footprint AABB (zero opacity), reveal a **shadow plate** sprite (cheap pre-baked AO disk) at the original rest pose.
- When user releases the object at a new pose: keep the shadow plate at original pose; allow them to "commit" the new rest pose if they want the scene to permanently update.

**Effort:** ~3 days Unity scripting.
**Trade-off:** ships fastest, looks worst on close inspection — but never breaks immersion catastrophically.

---

## 4 · Evaluation

Run the same eval against Tracks 2a, 2b, 3, 4 outputs.

| Tier | What | When applicable |
|---|---|---|
| **Synthetic (perfect GT)** | Replica / Hypersim scenes — render desk with-then-without object → perfect occluded-floor GT | Always |
| **Real-with-GT** | `360-USID` protocol on V32 desk — capture once with bottle, physically remove, recapture (tripod reference for alignment) | Once we run a controlled experiment on V32 |
| **Real-no-GT** | M-LPIPS + box-dilated FID (SPIn-NeRF defs) + Remove360 semantic residual probe on every shipped scene | Always |
| **Quest 3 on-device** | FPS + frame-time + memory measured on standalone Quest 3, not tethered. Target 72 FPS at the chosen budget | Mandatory before shipping |

**Per-route winners on the same metric set =** the comparison the experts kept asking for.

---

## 5 · What I'd actually do this week

| Day | Task |
|---|---|
| **Mon** | Track 1 — write `_unseen_core_map.py`, run on all three V32 objects, get the size statistic |
| **Tue** | Track 0 — plumb GLOMAP + SPZ export (low-risk, high-leverage) |
| **Wed** | Track 0 — wire EdgeAwareLogL1 + ToF-IR pre-clean into splatfacto-big run, kick off a V32 retrain |
| **Thu** | Track 2a — start writing M1 + M2 (mesh raycast + LaMa) — both standalone testable in a couple of hours each |
| **Fri** | Track 2b sanity check — clone GSFix3D, attempt WSL2 install, document blockers in PROGRESS_LOG.md |

End of week: we know (a) how big the hole really is, (b) whether GSFix3D is a real candidate, (c) Track 2a has its first two modules working on the bottle.

---

## 6 · Honest open questions

These are the ones I want the user (or another expert pass) to weigh in on:

1. **Faithful vs plausible.** If LaMa synthesizes a desk patch that *looks* plausible but isn't the real wood-grain pattern, is that OK? It will pass close visual inspection from any angle but it isn't the real desk under the bottle. (My read: yes for a digital twin, but the user should explicitly buy in.)

2. **Capture cost ceiling.** Track 3 only works if the user is willing to do the lift step. Is a 30-second extra capture step acceptable UX in ScanGuide?

3. **Per-object vs per-scene fill.** Should the fill be baked into the scene splat at packaging time (one combined patched scene splat), or layered as separate per-object patches that the Unity runtime composites? GPT advocated the latter; Gemini/Claude advocated the former. Trade-off is asset-shipping complexity vs runtime cost. (My read: bake at packaging, simpler runtime, no state machine.)

4. **Reflective monitor blob.** Tracks 0's ToF-IR pre-clean attacks the source. If it still persists, do we recapture monitors-off (the experts' consensus "best ROI") or live with the V32 blob?

5. **Quest 3 deployment surface.** aras-p UnityGaussianSplatting + SPZ is the verified path. Are we OK shipping a Unity APK (vs Meta Spatial SDK or WebXR)? Affects asset budget ceilings.

---

## 7 · Verified bibliography (the methods that survived)

**Use today**
- LaMa — Suvorov et al. "Resolution-robust Large Mask Inpainting with Fourier Convolutions" — [code](https://github.com/advimman/lama) Apache-2.0; runtime via [simple-lama-inpainting](https://pypi.org/project/simple-lama-inpainting/) or [IOPaint](https://github.com/Sanster/IOPaint).
- GLOMAP — Pan et al. ECCV 2024 — now in [COLMAP](https://github.com/colmap/colmap) 4.0 as the global mapper. BSD-3.
- Niantic SPZ — [github.com/nianticlabs/spz](https://github.com/nianticlabs/spz). MIT, v3.0.0 May 2026.
- aras-p UnityGaussianSplatting — [github.com/aras-p/UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting). MIT. Has ninjamode's VR fixes upstreamed.
- SAM3 — Meta, [facebookresearch/sam3](https://github.com/facebookresearch/sam3). Apache-2.0 code + Meta SAM license weights.

**Test on V32**
- GSFix3D — [github.com/GSFix3D/GSFix3D](https://github.com/GSFix3D/GSFix3D). Apache-2.0 code + RAIL++-M weights. Replica PSNR 26.49 / ScanNet++ 25.63.
- 3DGIC cross-view reprojection logic only — [github.com/peterjohnsonhuang/3dgic](https://github.com/peterjohnsonhuang/3dgic). "Suboptimal version" warning in README.

**Borrow the idea, not the codebase**
- InFusion's "depth as scaffold, unproject Gaussians" — [arxiv 2404.11613](https://arxiv.org/abs/2404.11613).
- ToF-Splatting's L1-quantile outlier filter — [arxiv 2504.16545](https://arxiv.org/abs/2504.16545). No code.

**Reference / benchmark only**
- AuraFusion360 — [arxiv 2502.05176](https://arxiv.org/abs/2502.05176) + [code](https://github.com/kkennethwu/AuraFusion360_official). 360-USID PSNR 17.90 / LPIPS 0.424.
- SPIn-NeRF evaluation protocol — [code](https://github.com/SamsungLabs/SPIn-NeRF).
- Remove360 semantic-residual benchmark — [arxiv 2508.11431](https://arxiv.org/abs/2508.11431).

**Avoid (verified)**
- FLUX.1 Fill (non-commercial weight license)
- InFusion as a codebase (stale, non-swappable depth net)
- SplatFill (no code, A100-only)
- Gaussian Grouping's removal mechanism (multi-view inconsistent baseline)
- GScream (no license file, sm_86, Scaffold-GS dependency)
- Mobile-GS (phone numbers, no Quest path)
- ninjamode VR fork (frozen, work upstreamed into aras-p)

---

*End of plan. We commit to evidence as we run, not now. Tracks 0 + 1 start Monday.*
