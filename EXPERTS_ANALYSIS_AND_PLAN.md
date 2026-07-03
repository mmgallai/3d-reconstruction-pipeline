# Reading the experts + the comprehensive plan — analysis log

*2026-06-18.* On 2026-06-17 the user asked three independent LLMs — Gemini, Claude, and GPT — for opinions on the unresolved scene-vs-object hole problem in this project (when a grabbable object is lifted in VR, what should appear under it?). I read all three opinions carefully, three times each, in order to internalise rather than skim them. I then spawned a 6-agent verification sweep that web-checked every load-bearing claim, every repository, and every license referenced across the three reviews, and ran a sixth synthesiser to reconcile the results. The actionable output of that work is `COMPREHENSIVE_PLAN.md` (a 5-track plan, Tracks 0–4). This document is the analysis companion: it captures the reading, the verification, and the reasoning that produced the plan, so the plan itself can stay short.

> **How to use this document.** If you want the actionable plan, read `COMPREHENSIVE_PLAN.md`. If you want to understand HOW we got there — what each expert contributed, where they disagreed, and what the verification sweep surfaced — read this.

## Table of contents

- [Reading expert_Gem.txt (Gemini)](#reading-expert_gemtxt-gemini)
- [Reading expert_Cl.md (Claude)](#reading-expert_clmd-claude)
- [Reading Expert_GP.docx (GPT)](#reading-expert_gpdocx-gpt)
- [Where the three experts agreed, disagreed, and where the evidence landed](#where-the-three-experts-agreed-disagreed-and-where-the-evidence-landed)
- [What the 6-agent verification sweep actually surfaced](#what-the-6-agent-verification-sweep-actually-surfaced)
- [Summary of the resulting plan](#summary-of-the-resulting-plan)
- [Honest meta-notes on this analysis process](#honest-meta-notes-on-this-analysis-process)

## Reading expert_Gem.txt (Gemini)

Gemini's one-line headline is **Depth-Anchored Splat Inpainting**: use the OpenMVS Poisson mesh as a geometric oracle for the occluded desk, then borrow a 2D inpainter (LaMa) to hallucinate texture, then lift those pixels into Gaussians constrained by mesh normals. It is the cleanest architectural write-up of the three experts and the only one that explicitly framed our advantage as "we already solved the geometry; we only need to guess the color."

### The 4-module architecture

| Module | What it does | Tool |
| --- | --- | --- |
| 1. Mesh depth raycasting | Pick 1-3 anchor views, raycast against the OpenMVS mesh to produce a perfect depth map of the occluded desk surface | Open3D raycaster |
| 2. Sparse 2D inpainting | Run LaMa (preferred for repetitive wood/desk textures) or SDXL (for complex grain/semantics) on those anchor RGBs | `lama-cleaner` or `diffusers` |
| 3. Guided unprojection | Lift inpainted pixels into 3D using Module 1's depth; spawn new Gaussians flat on the desk plane with mesh-normal-aligned scales | Custom |
| 4. Localized optimization | Freeze original scene Gaussians, run ~500 splatfacto iterations only on the new Gaussians with a TV/smoothness loss | nerfstudio/splatfacto |

The framing — "guarantees the patch is planar, avoids the multi-view smear that plagues pure 2D propagation, avoids the cloudy patch that plagues pure 3DGS inpainting" — is the most defensible argument any of the three experts made.

### Three alternatives Gemini offered

1. **Disocclusion Sweep** — at frame ~120 of the capture, the user reaches in, removes the objects, keeps recording the empty desk for ~20 frames; SAM3/YOLO masks the hand, SfM ignores the moving arm. Capture-side fix, zero generative AI. Gemini correctly flagged it works for bottles/boxes, fails for immovable furniture.
2. **2DGS Migration** — swap splatfacto for 2D Gaussian Splatting so the desk is mathematically planar by construction.
3. **Diminished Reality** — runtime Unity trick: swap object splat for scene splat on grab, hide the hole's edge with a baked shadow projector. Zero compute; fails if the user puts the bottle somewhere else on the desk.

### Risks Gemini called out

- **Poisson bowing under the object** — the mesh prior may not be perfectly planar. Mitigation: RANSAC plane-fit on hole-adjacent vertices before raycasting.
- **Lighting bake** — AO/shadows are burned into the original GS, so removing the object leaves a ghost shadow. Mitigation: SDXL prompt to "remove shadows and match ambient lighting."
- **Reflective monitor blob (the known V32 issue)** — could corrupt the localized optimization. Mitigation: mask the loss so gradients only touch new Gaussians.

### What Gemini got right

- The module breakdown is sharp and each module has a defined input/output — easiest of the three to translate into actual tickets.
- Picking **LaMa over SDXL** for repetitive desk textures is the right call; SDXL's semantic priors are a liability on featureless wood grain, and LaMa's Fourier-convolution prior is empirically better for structured texture continuation.
- The **Disocclusion Sweep** appeared independently in Claude's and GPT's write-ups too, which is a strong convergence signal. Gemini surfaced it cleanly as a capture-side bypass to the entire generative-AI rabbit hole.

### What Gemini left thin or wrong

- **No specific 3DGS-inpainting repos named.** Gemini cites "GS-Inpaint / Inpaint3D" generically in the bibliography but never gives a GitHub URL, commit, or recency check. The verification sweep could not confirm either project has a maintained, usable codebase as of mid-2026 — so this "alternative" is effectively unactionable from Gemini's text alone.
- **The 8-week timeline is optimistic.** Phases 1-4 assume coordinate-system glue between COLMAP, OpenMVS, and nerfstudio works on first try; Gemini's own Risk #1 (Poisson bowing) implies at least one debugging round Phase 3 doesn't budget for.
- **Quest 3 standalone FPS is unaddressed.** Gemini mentions 72 FPS once as an evaluation criterion but never analyzes whether added Gaussians will fit the standalone budget (Quest 3 has no PCVR fallback in our deployment).
- **SfM/pose tooling is silent.** The whole pipeline assumes our COLMAP poses are correct; no discussion of pose drift, ToF-RGB calibration, or what happens if the Disocclusion Sweep frames register poorly.
- **The 2DGS migration alternative is high-cost, low-leverage.** Rewriting the renderer to dodge one hole is not a real option and Gemini should have demoted it.

### Final read

Gemini delivered the cleanest architecture and the most defensible "why" — but it is an architecture document, not a feasibility document, and it needed Claude's repo-level grounding and GPT's deployment realism to become a plan.

## Reading expert_Cl.md (Claude)

Claude's review is the longest of the three (479 lines, 13 sections, a full annotated bibliography) and the only one whose headline recommendation is built specifically around an asset this project already owns: a metric-scale TSDF/Poisson mesh that *interpolates geometry under the occluder*. The single-sentence headline is **"metric-mesh-seeded amodal fill"** — render that mesh under the SAM3 footprint, unproject the resulting depth in the InFusion mechanism to seed Gaussians at *known* metric positions, and use a 2D inpainter (LaMa / FLUX.2-Klein) only as a texture stage on locked geometry. The argument is that every published 2024–2026 inpainter (InFusion, GScream, AuraFusion360, 3DGIC, SplatFill, GSFix3D) burns its budget on hallucinating *depth in the hole*, and this project skips that sub-problem entirely.

**Breadth.** Section 3 covers nine SOTA categories — 3DGS inpainting, surface-aligned GS/mesh, VR rendering/compression/LOD, object decomposition, ToF/reflections, datasets, industry, feed-forward recon — explicitly tallying ~107 systems and citing 70+ refs in an annotated bibliography. No other expert produced a field map of comparable depth.

**Verified low-risk switches Claude says "do now":**

| Switch | Replaces | Verification status |
|---|---|---|
| **GLOMAP** (in COLMAP 4.0) | incremental COLMAP | Confirmed: official MIT release, drop-in CLI, ECCV 2024 |
| **Niantic `.spz`** | NanoGS-only output | Confirmed: MIT, format used by Meta Spatial SDK + PlayCanvas |
| **EdgeAwareLogL1** on splatfacto-big's **native gsplat 1.5.3 densifier** | DN-Splatter's gsplat-1.0 densifier (V25 cause) | Confirmed by re-reading the V25 post-mortem; root cause matches |

**The Quest-3 FPS reality check.** This is where Claude does the most work the other two experts skipped. Gemini and GPT both cite "72 FPS Quest 3" splat papers (VRSplat, VR-Splatting) as if they were standalone numbers; Claude flags that those numbers are **tethered to an RTX 4090** and that the genuine standalone ceiling is roughly **~150k Gaussians on the Meta Spatial SDK** (one splat, SH=0) and **~150–400k on the [ninjamode VR fork](https://github.com/aras-p/UnityGaussianSplatting) of aras-p**, with **SqueezeMe** (Meta RL) being the only *verified* on-device datapoint at ~180k @72 FPS. He correctly notes that *Meta itself streams Hyperscape* rather than render room-scale on-device — a useful sanity anchor.

**Reflective-monitor blob — "remove don't model."** Claude flatly rejects 3DGS-DR / Ref-Gaussian / Mirror-3DGS / VoD-3DGS as wrong-goal for a *grabbable* twin: a content-bearing monitor isn't a clean far-field mirror, modelling it correctly is expensive, and a digital twin user wants the floater *gone*, not physically simulated. The recommended stack is SpotLessSplats utilization-based pruning + SAM3 screen masks + scale-filter prune + a ToF-Splatting-style multi-view outlier filter on IR ghosts *pre-init*. This is consistent with the V33 negative result (SAM3 alone didn't kill the blob — the ToF init ghosts are the missing piece).

**Four alternative architectures** are explicitly named: (A) off-the-shelf inpainter — AuraFusion360 or GSFix3D `--dual_input` — as a baseline; (B) capture-side first via FisherRF re-sweeps; (C) generative-prior twin (TRELLIS, Amodal3R) — rejected as primary because "hallucinated geometry is unacceptable for a faithful twin"; (D) unified surface-GS deliverable via RaDe-GS/MILo. He correctly identifies (A) as the *baseline to beat*, (B) as the parallel track and gold-standard eval, and (D) as a medium-term pilot, not a near-term switch.

**What Claude got right.** Most thorough field map. He flagged **GSFix3D's `--dual_input`** as the closest off-the-shelf match for the mesh-seeded approach, and verification confirmed the [GSFix3D repo](https://github.com/GSFix3D/GSFix3D) does fuse a rendered mesh + 3DGS as a diffusion prior. He flagged the standalone Quest FPS folklore, which the verification sweep independently corroborated. He flagged license landmines (FLUX.1, Hunyuan3D, NanoGS CC-BY-NC) that the other reviews glossed.

**What Claude got wrong or oversold.** Three notable cases:

1. **Ninjamode fork status.** Claude treats the ninjamode VR fork as if it were a live production renderer. Verification found the fork is **frozen** — its VR optimisations have already been **upstreamed into aras-p/UnityGaussianSplatting**, so the right dependency is the upstream repo, not the fork. Minor but important for the build plan.
2. **AuraFusion360 PSNR.** Claude quotes 17.66 / 0.388 on 360-USID; the paper actually reports **17.903**. Trivial delta, but illustrates a pattern of numbers cited to two decimals from a verification sweep he hadn't personally run.
3. **FisherRF "70 FPS active view selection."** Independent reports measure **5–10 seconds per candidate view** in real deployments — the paper's headline number is a best-case microbenchmark, not the field experience. Claude cites it without that qualifier.

Throughout, several numerical claims are sourced from "the verification sweep" he had not actually executed in his own context — a structural over-claim that the cross-check phase had to clean up.

**What only Claude added.** An explicit **"biggest uncertainty" framing**: the claim that a metric mesh prior *cures* multi-view inconsistency is theoretical, not empirically proven at this regime. A publishability angle ("Embodied Scene Editing via Cross-Modal Geometry Priors") with the **tabletop near-planar single-sweep occlusion benchmark** as a genuine field gap. And the evaluation rigour — **360-USID** physical-removal protocol + **Remove360** semantic residual probe + **MuSHRoom** as the external comparison — that neither other expert specified.

**One-sentence read.** Claude is the most exhaustive and the most useful as a *map of the field*, but he overstates the production-readiness of a few load-bearing dependencies (ninjamode, FisherRF) and quotes verification-sweep numbers as if first-hand — treat the architecture as primary, the citations as second-pass.

## Reading Expert_GP.docx (GPT)

GPT delivers the cleanest **reframe** of the three reports: the unsolved problem is not inpainting, it is **layered compositing with object-state awareness**. Its strongest contribution is a Unity-side runtime architecture the other two experts ignored, paired with a phased rollout that defers research-grade splat inpainting until after a shippable demo exists.

### The headline reframe

GPT opens by refusing the framing it was given: "this is no longer primarily an object-extraction problem. It is a disocclusion and layered-compositing problem." That single sentence redirects effort away from yet another segmentation tweak and toward the runtime question of what should appear under a lifted object. The supporting observation is correct and was verified against the project files: V32 per-object splats and meshes are already clean, so further mask tuning will not recover pixels that were never observed.

### Layered Disocclusion Patches

The architectural proposal is a per-object **support-surface patch** generated from the existing OpenMVS mesh and ToF depth, exported alongside `object_visual_splat.ply` and `object_physics_mesh.glb` as a sibling asset (`disocclusion_patch_splat.ply` + `disocclusion_patch_mesh.glb`). The patch is treated as a separate runtime layer that is hidden while the object sits at rest and revealed when the object is displaced. This cleanly avoids both failure modes the project hit: static duplicates (if the scene splat is untouched) and bare holes (if the object's Gaussians are subtracted).

### Unity runtime state machine

GPT is the only expert to specify runtime behavior in concrete states: **rest / grabbed / displaced / returned**. At rest, the patch is occluded by the object and not rendered. On grab or displacement past a threshold, the patch becomes visible under the original footprint. After placement elsewhere, the patch stays visible (the object does not snap back), with optional rest-pose rebake on user confirmation. This is the missing link between "we have a clean patch asset" and "the VR user does not see a glitch."

### Five-phase implementation roadmap

| Phase | Deliverable | Intent |
|---|---|---|
| 0 | Disocclusion benchmark + state spec | Measure patch reveal quality, not mask quality |
| 1 | Textured patch mesh + Unity demo | Ship a working lift interaction first |
| 2 | Mesh-guided local patch Gaussians | Replace flat texture with optimized splat layer |
| 3 | D-lite capture mode | Use real observed pixels when user cooperates |
| 4 | Benchmark vs GScream / 3DGIC / SplatFill / 2D-inpaint+retrain | Only after baseline ships |

Crucially, the 3DGS inpainting methods that dominate the other experts' write-ups are pushed to Phase 4 here, framed as a "research comparison" rather than the product path.

### D-lite capture

GPT proposes asking the user to shift movable objects 2-5 cm or briefly remove them for 10-20 frames, registering the micro-pass into the original scan to harvest real background pixels. Worth flagging: **Gemini independently proposed the same protocol**, which raises confidence that this is the highest-ceiling option for users willing to cooperate.

### What GPT got right

- Pragmatic "ship the demo first" sequencing — the textured-patch-mesh Phase 1 unblocks Unity work while splat inpainting is still being evaluated.
- The runtime state machine is genuinely useful and **neither Gemini nor Claude addressed it**; both stopped at asset generation.
- Honest about cross-view inconsistency risk for the full 2D-inpaint-and-retrain route ("can create cross-view texture disagreement and training smears"), which the verification sweep also flagged.
- The layered architecture cleanly separates the visual splat, the physics mesh, and the disocclusion layer — three concerns the other reports tended to entangle.

### What GPT got wrong or thin

- **Weakest engagement with the SOTA landscape.** The bibliography lists GScream, 3DGIC, SplatFill, NeRFiller, SPIn-NeRF as a generic menu without ever stating which has runnable code, which is license-clean for an open-source release, or which has been independently reproduced. Gemini and Claude both went deeper here.
- **No license or repo-level diligence at all** — the methods are cited by paper, not by integration cost.
- **Phase 1 textured patch mesh is too low-fidelity to actually ship for VR.** A flat-shaded patch under a lifted bottle will read as obviously baked on Quest 3 at arm's length; the demo will land, but the polish gap to Phase 2 is larger than GPT implies.
- **D-lite is hand-waved on the hard parts**: how the pipeline detects that the user successfully moved the object, how the micro-pass is registered into the original COLMAP frame (incremental SfM? ICP against ToF? pose anchors?), and what happens if the brief pass fails to converge. These are the questions that decide whether D-lite ships.

### Final read

GPT contributes the runtime architecture and sequencing discipline the other two experts left out, at the cost of being the shallowest on which specific 3DGS inpainting method to actually pick.

## Where the three experts agreed, disagreed, and where the evidence landed

Three independent reviews (Gemini, Claude, GPT) converged on a surprisingly narrow technical core: **use the metric mesh/ToF you already have as the depth scaffold, synthesize only RGB, and never run an unconstrained per-view diffusion loop**. A 6-agent verification sweep then confirmed that the field's 2024-26 work (AuraFusion360, GSFix3D, 3DGIC, InFusion, GScream) is built on exactly that pattern, that only one repo (GSFix3D, Nov 2025) implements it cleanly off-the-shelf, and that several heavier alternatives the experts hedged toward (FLUX.1 Fill, SplatFill, InFusion) failed verification on license, code availability, or framework grounds. The cleanest summary: Gemini wrote the minimum viable architecture, Claude wrote the field map and pipeline upgrades, GPT wrote the runtime/Unity layer, and verification said all three were right about different parts of the same elephant.

### 1. Consensus vs. disagreement matrix

| Claim | Gemini | Claude | GPT | Verification verdict |
|---|---|---|---|---|
| Mesh/ToF is the depth scaffold; only RGB needs synthesis | Yes (Module 1 raycaster) | Yes ("skip the hardest published sub-problem") | Yes ("mesh/ToF prior gives reliable support geometry") | **Confirmed** — GSFix3D `--dual_input` consumes mesh-render + 3DGS-render exactly as prescribed |
| Naive 2D-inpaint-all-views + retrain is multi-view inconsistent | Yes ("texture-soup") | Yes ("naive Gaussian-Grouping loop") | Yes ("smear") | **Confirmed** — Gaussian Grouping's removal mechanism is literally LaMa-per-view + finetune; SplatFill within-mask PSNR 15.67 shows this regime is hard for everyone |
| LaMa beats SDXL/FLUX for repetitive desk textures | Yes (explicit) | Yes (via "LaMa / FLUX.2-Klein" preference) | Silent | **Confirmed** — Apache-2.0 weights, deterministic Fourier convs, ~4 GB VRAM, sm_120-compatible |
| Hide-and-reveal alone is emergency UX only | Yes (Alt 3 caveat) | Yes (option-C critique) | Yes ("Keep as fallback only") | **Confirmed** by their own logic — fails on object relocation |
| Capture-side disocclusion sweep is underrated | Yes (Section 6, Alt 1) | Yes (FisherRF + 360-USID protocol) | Yes (D-lite) | **Confirmed** — SAM3 + COLMAP `--ImageReader.mask_path` is plumbing, not research; RoMo automates the SAM-to-COLMAP step |
| Full-scene generative regeneration is unsuitable for a "faithful twin" | Yes (Alt C dismissed implicitly) | Yes ("hallucinated; unsuitable for faithful twin") | Yes (avoid 2D-inpaint-all-views) | **Confirmed** — TRELLIS / Marble produce plausible, not faithful, content |

### 2. Where exactly two of three converged

| Notable pair | Claim | Comment |
|---|---|---|
| Gemini + GPT | Object-shift micro-capture (Disocclusion Sweep / D-lite) | Independently proposed within hours of each other; Claude folded it under "FisherRF + 360-USID protocol" but didn't name the shift trick |
| Claude + GPT | Unity collider must stay separate from visual splat | Claude calls out convex MeshColliders (≤255 tris); GPT writes the explicit `object_collider.json` bundle. Gemini treats colliders as a given |
| Gemini + Claude | LaMa as the 2D inpainter of choice | GPT was silent on the inpainter — a gap in their write-up, not a disagreement |

### 3. Where they diverged, and which side the evidence supported

| Divergence | Verification's verdict |
|---|---|
| Claude wanted broad pipeline upgrades (GLOMAP, `.spz`, surface-GS pilot, on-device Quest benchmark) layered on top of the fix | **Supported.** GLOMAP merged into COLMAP 4.0 (BSD-3-Clause, drop-in); SPZ v4 (May 2026, MIT, ~10x smaller than PLY) is read natively by aras-p UnityGaussianSplatting. Both are orthogonal low-risk wins |
| GPT framed it as a Unity runtime-layered patch (state machine, layered bundle, reveal-on-grab) | **Partially supported.** Runtime layering is a valid UX layer and a useful fallback, but the verified primary path bakes the fill into the asset (mesh-seeded Gaussians) rather than papering over it at runtime. Verification's `use_now` list is asset-baked, not runtime-toggled |
| Gemini's minimalist 4-module architecture (raycast → 2D inpaint → unproject → localized optimize) | **Most directly buildable today.** Heavier alternatives Claude mentioned (InFusion: stale repo, non-swappable depth net; SplatFill: no code, A100-class) and the FLUX option all failed verification. Gemini's modules map 1:1 to what verification says actually works |

### 4. Things only one expert flagged that we should preserve

| Expert | Unique contribution | Why keep it |
|---|---|---|
| Claude | Quest 3 **standalone**-FPS folklore problem — every "72 FPS" paper is tethered RTX 4090 | Verification confirms: even SqueezeMe (Meta) caps at ~180k @72 FPS on XR2 Gen2; ninjamode's 400k number is author-claimed, never independently re-measured |
| Claude | Reflective monitor blob is a "remove, don't model" problem (3DGS-DR / Ref-Gaussian / Mirror-3DGS solve the wrong goal) | Verification confirms: physical-reflection methods assume far-field clean mirrors; SpotLessSplats + SAM3 + scale-prune is the practical path |
| Claude | Novelty/publishability angle ("Embodied Scene Editing via Cross-Modal Geometry Priors") | Gemini hinted at this; Claude wrote the venue pitch and benchmark gap |
| GPT | Runtime state machine for object {rest, grabbed, displaced, returned} | The other two assumed grab semantics; GPT made them explicit. Necessary for the Unity build regardless of which fill path wins |
| GPT | Ship a textured-patch-mesh demo **first** before any optimization | Lowest-risk MVP; lets the team validate the UX before betting weeks on diffusion-fill integration |
| Gemini | RANSAC plane-fit around the hole before raycasting, to enforce strict planarity | Specific risk mitigation against Poisson bowing under the object — neither Claude nor GPT named this concrete failure |

### 5. The load-bearing claim nobody could verify

The mesh-seeded amodal fill plan rests on a single empirical assumption: **that supplying a metric mesh as the depth target eliminates the multi-view inconsistency that forces every published method into iterative refinement on a tabletop near-planar small-hole regime.** None of the three experts produced evidence for it (all argued by analogy from InFusion / GSFix3D). The verification sweep was explicit about this:

> "GSFix3D's `--dual_input` flag can accept an ARBITRARY user-supplied mesh (our V32 OpenMVS reconstruction) and inpaint a small near-planar desk patch WITHOUT scene-specific finetuning. The repo confirms the flag exists [...] but the SOTA Replica PSNR 26.49 / ScanNet++ PSNR 25.63 numbers come from in-distribution datasets the model was trained on. Whether the diffusion prior generalizes zero-shot to a Femto-Mega capture of an arbitrary desk [...] — none of this is verified. The entire mesh-seeded amodal fill plan hinges on this load-bearing claim."

This is the single experiment Phase 1 must run before anything else.

### 6. Wrap-up

The convergence is real but it isn't ours — it's the field's. By mid-2026 every serious 3DGS inpainter (AuraFusion360, GSFix3D, 3DGIC, InFusion, GScream) has settled on depth-guided placement + 2D-inpaint-for-texture + multi-view-consistency refinement; the three experts independently rediscovered that shape because that's where the open-source center of gravity is. What each contributed uniquely is the project-specific framing: Gemini gave the cleanest module decomposition and the only concrete RANSAC mitigation; Claude mapped the broader pipeline (GLOMAP, SPZ, surface-GS, on-device measurement) and named the publishable gap; GPT translated the fix into a Unity-shaped deliverable with explicit state semantics. The plan that beats any one of them is the union: **Gemini's architecture, Claude's plumbing, GPT's runtime contract, anchored by a Phase 1 experiment that resolves the one claim none of them could verify.**

## What the 6-agent verification sweep actually surfaced

Five verifier agents searched the web in parallel across five orthogonal axes — repo status, license, framework compat (gsplat 1.5.3 / nerfstudio 1.1.5 / sm_120 Blackwell / Windows), paper claims, and tabletop-planar fit — then a sixth synthesizer reconciled them. Total wall-clock was about 3.5 minutes. The sweep is what flipped several of the experts' confident recommendations from "go" to "don't" and surfaced two integration gotchas that all three experts missed.

### USE NOW — methods we can plug in today

| Method | License | What it gives us | One key gotcha verified |
|---|---|---|---|
| LaMa via [simple-lama-inpainting](https://github.com/advimman/lama) / [IOPaint](https://github.com/Sanster/IOPaint) | Apache-2.0 (code + Big-LaMa weights) | Deterministic, Fourier-conv RGB fill purpose-built for wood-grain / laminate; ~4 GB VRAM at 1024px, <1 s per call on 5070 Ti | Upstream repo is effectively frozen (last substantive commit ~2022); use the community wrappers, not master |
| GLOMAP via [COLMAP 4.0](https://github.com/colmap/colmap) global mapper | BSD-3-Clause | Drop-in global SfM, same sparse-model format as incremental COLMAP | Standalone `glomap` repo was archived 2026-03-09; nerfstudio's `ns-process-data` still doesn't invoke it — you must run `colmap mapper` yourself then `ns-process-data --skip-colmap --colmap-model-path sparse/0` |
| [Niantic SPZ v4](https://github.com/nianticlabs/spz) | MIT | ~10x smaller than PLY at no perceptible quality loss; 10M-point cap removed in v3.0.0 (May 2026) | Quest path is via aras-p, not a Meta SDK — convert at packaging time and ship `.spz` |
| [aras-p UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) main | MIT | Quest 3 runtime with AMD FFX radix sort + native SPZ import | ninjamode VR fork's work is already upstreamed; build against `main`, not the fork |
| SAM3 + COLMAP `--ImageReader.mask_path` + nerfstudio `transforms.json` `mask_path` | SAM3 Apache-2.0 (model under Meta SAM license) / COLMAP BSD-new | Text-prompted ("arm", "hand") hand/arm masks for disocclusion sweeps | COLMAP's `mask_path` ONLY blocks feature extraction — see gotcha (a) below |
| EdgeAwareLogL1 on native gsplat 1.5.3 densifier | Apache-2.0 (gsplat) | Depth-supervised loss term without DN-Splatter's broken densifier patches | Stays inside splatfacto defaults; no third-party densifier monkey-patches |

### TEST — focused experiments with explicit kill criteria

| Method | What we want to find out | Day-1 sanity check | Kill criteria |
|---|---|---|---|
| [GSFix3D](https://github.com/GSFix3D/GSFix3D) `--dual_input` (Nov 2025) | Whether the diffusion prior generalizes zero-shot from Replica/ScanNet++ training to a V32 OpenMVS mesh of our desk | Run pretrained checkpoint on one V32 (mesh-render, splat-render, mask) triple at 512px latent | Fill is worse than LaMa baseline at the desk patch, or VRAM > 16 GB even with gradient checkpointing + reduced latent res, or SD2 weights unobtainable after HF removal |
| 3DGIC cross-view reprojection logic (ported standalone) | Whether we can fill from frames that actually saw the patch BEFORE any hallucination | Reimplement only the visibility-warp + mask-shrink step; do not pull rest of repo (README itself says released code is the "suboptimal version") | Fewer than ~30% of hole pixels resolved from real captured pixels (i.e. the patch is genuinely fully occluded — fall through to 2D inpaint) |
| ToF-Splatting L1-quantile filter (reimplemented, ~50–100 LOC) | Whether the monitor IR-ghost blob (v33) gets eliminated by dropping ToF samples where \|D_DA3 − D_ToF\| > 75th percentile before splatfacto init | Run on one frame batch; visualize dropped ToF points overlaid on monitor region | Blob persists at q=0.5–0.9 sweep, or > 25% of ToF points dropped (means DA3, not ToF, is the outlier) |
| SDXL-Inpaint as LaMa fallback | Whether prompt-conditioned diffusion beats LaMa on textured-corner / semantic-occluder edge cases | Locked seed + identical prompt across 3 anchor views; visually compare cross-view consistency | Cross-view wood-grain mismatch worse than LaMa even with locked seeds; default back |

### AVOID — methods verification killed

| Method | Verified failure reason |
|---|---|
| [InFusion](https://github.com/ali-vilab/Infusion) (2024) | Stale (last commit Jul 2024); vanilla 3DGS not gsplat 1.5.3; depth-completion network is the core contribution and is NOT swappable for our ToF/OpenMVS depth |
| SplatFill (2025) | No code released; A100-80GB training; only +0.01 PSNR over GScream — not actionable |
| [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping) (inpainting path) | Removal pipeline IS the naive LaMa-per-view + finetune baseline that all three experts ruled out as multi-view inconsistent |
| FLUX.1 Fill-dev | FLUX.1 [dev] Non-Commercial License v2.0 on weights blocks redistribution in an open-source pipeline — see gotcha (b) below |
| Mobile-GS | Quest 3 numbers don't exist (Snapdragon 8 Gen 3 phone only); mobile Vulkan code closed-source by company policy |
| ToF-Splatting as a dependency | No code release as of Jun 2026; filter is baked into a full SLAM pipeline (DoD module + tracking frontend), not extractable — borrow the IDEA, not the repo |
| [VR-GS](https://arxiv.org/abs/2401.16663) | No code, tethered desktop demo, soft-body PBD coupling we don't need |
| DN-Splatter densifier on gsplat 1.5+ | Densification is patched off on modern gsplat; produces ~50× fewer Gaussians (verified by our own v25 regression) |
| FreeSplat / MVSplat as final splat | Feed-forward inference quality is below per-scene-optimized splatfacto on Femto data |

### WATCH — interesting but not ready

| Method | Why interesting | Why not yet |
|---|---|---|
| [AuraFusion360](https://github.com/kkennethwu/AuraFusion360_official) (CVPR 2025) | Apache-2.0, public 360-USID benchmark, reference-based depth-aware fill | 360-tuned architecture over-hallucinates on a small near-planar desk patch |
| [GScream](https://github.com/W-Ted/GScream) (ECCV 2024) | Depth-prior + cross-attention coherence | NO LICENSE FILE (legal risk); Scaffold-GS dependency; targets sm_86 |
| FisherRF | Active NBV for ScanGuide v2 | 70 FPS view-selection claim contested by independent reports of 5–10 s/view; nerfstudio fork has 2 stars |
| ninjamode VR fork | Historical context for the Quest sort fix | Frozen (5 commits, Oct 2024); README itself redirects to aras-p upstream |
| Octree-GS / Hierarchical-3DGS | LOD streaming for room-scale scenes | Mobile decode cost on Quest 3 unverified |
| MILo | Promising fix-the-mesh-and-splat-together formulation | Code not yet released |
| GPGS / InstaInpaint / Inpaint360GS | Paper-promising depth-aware inpainters | Paper-only or repos unverified |

### Load-bearing unverified claim

The entire mesh-seeded amodal fill route hinges on a single claim that the sweep could NOT confirm: that **GSFix3D's `--dual_input` flag generalises zero-shot from its Replica/ScanNet++ training distribution to (i) our V32 OpenMVS mesh of an arbitrary desk, (ii) a small near-planar patch (not a room-scale hole), (iii) Windows rather than Ubuntu 22.04, (iv) sm_120 Blackwell rather than RTX 4500 Ada sm_89, (v) 16 GB VRAM rather than their 24 GB testbed, and (vi) with SD2 weights sourced from a third-party mirror after Hugging Face's removal.** Every other USE/TEST item is independently valuable; this one is the single point of failure for the diffusion-fill route and needs the GSFix3D day-1 sanity check above before any downstream plan commits to it.

### Two gotchas the experts missed

**(a) COLMAP `--ImageReader.mask_path` only blocks feature extraction.** It does NOT propagate to bundle-adjustment outlier rejection and — critically — does NOT mask splatfacto's photometric loss. Both Gemini and GPT proposed the disocclusion-sweep + COLMAP-mask recipe; Claude proposed it too. None of them flagged that the SAM3 mask must ALSO be written into the nerfstudio `transforms.json` `mask_path` field, or the splat will keep optimising Gaussian colour against hand pixels in every frame they appear. This is a one-line dataparser fix but a silent failure mode if missed.

**(b) FLUX.1 Fill ships under FLUX.1 [dev] Non-Commercial License v2.0 on the weights.** The Claude expert opinion mentioned "FLUX.2-Klein" as a candidate without flagging the licence. Outputs are commercially usable, but the weights themselves cannot be redistributed or served in a commercial / open-source-pipeline distribution without a paid licence from bfl.ai. For a pipeline whose entire premise is "open-source consumer-scan-to-VR", this is a hard blocker independent of any quality argument. LaMa (Apache-2.0 on both code and weights) and SDXL-Inpaint (CreativeML-OpenRAIL-M, commercial OK) are the only RGB-fill options that survive the licence filter.

## Summary of the resulting plan

`COMPREHENSIVE_PLAN.md` resolves the three reviews into five parallel tracks, each owning a different risk profile so progress in one track is not gated by experiments in another.

- **Track 0 — Plumbing wins (this week).** The "USE NOW" rows the verification sweep confirmed: GLOMAP via COLMAP 4.0, Niantic `.spz` export, native gsplat-1.5.3 densifier with EdgeAwareLogL1, the SAM3 → `transforms.json` `mask_path` fix for gotcha (a). Pure orthogonal upgrades. Ship them regardless of which fill path wins.
- **Track 1 — Capture-side disocclusion sweep (1–2 weeks).** Gemini's "Disocclusion Sweep" and GPT's "D-lite," fused: SAM3-masked hand frames + a 10–20 frame object-shifted micro-pass registered into the original COLMAP frame. If the user cooperates, this harvests *real* pixels and skips the diffusion question entirely. Highest ceiling, highest user-effort cost.
- **Track 2 — Mesh-seeded amodal fill (the headline experiment).** Gemini's 4-module architecture as the buildable spine. Phase 1 is a single GSFix3D `--dual_input` sanity check on one V32 desk patch — the experiment that resolves the load-bearing unverified claim. If GSFix3D generalises zero-shot, productise it; if it doesn't, fall back to a LaMa-only Module 2 with Gemini's RANSAC plane-fit + freeze-and-optimise localised step.
- **Track 3 — Runtime contract (parallel, Unity team).** GPT's rest/grabbed/displaced/returned state machine and layered asset bundle (`disocclusion_patch_splat.ply` + `disocclusion_patch_mesh.glb`). Unblocks the VR demo independent of which fill path wins Track 2.
- **Track 4 — Monitor-blob cleanup.** Claude's "remove don't model" stack: SpotLessSplats utilisation pruning + SAM3 screen masks + ToF-Splatting-style quantile filter on the ToF init.

**Escalate** to Track 2 only if Track 1 fails to register the micro-pass; escalate from LaMa to GSFix3D only if the Track 2 day-1 sanity check passes. **This week:** ship Track 0, kick off Tracks 1 and 3 in parallel, run the GSFix3D sanity check on Track 2.

## Honest meta-notes on this analysis process

- The synthesiser writing this analysis *is* Claude. "Reading expert_Cl.md" is therefore partly self-review, with all the over-claim risk that implies — I tried to flag Claude's weakest claims (ninjamode status, AuraFusion360 PSNR digits, FisherRF FPS, citing the verification sweep before it existed) explicitly, but a reader should weight my critique of the Claude review as the least independent of the three readings.
- The 6-agent verification sweep is itself web search. It can be wrong: a paper's repo may have changed since the sweep ran, a license may be re-issued, a "no code released" verdict can flip overnight. Treat every USE/TEST/AVOID row as a snapshot, not a fact.
- The single biggest load-bearing unverified claim is **GSFix3D `--dual_input` generalising zero-shot** from Replica/ScanNet++ to our V32 mesh on Windows + sm_120 + 16 GB VRAM. The entire diffusion-fill route depends on it. Treat the Track 2 day-1 sanity check as the gating experiment for everything downstream of LaMa.
- The plan commits to *evidence on the way*, not now. Track 0 is the only set of changes I'd ship without further data; Tracks 1–4 each have explicit kill criteria so we find out cheaply rather than carrying load-bearing assumptions for weeks.
- Three independent LLMs converging on "mesh as depth scaffold, RGB-only synthesis, never unconstrained per-view diffusion" is a useful signal, but it is *not* proof. All three were trained on overlapping literature dominated by the same 2024–2026 papers (InFusion, GScream, AuraFusion360, GSFix3D), so consensus may reflect a shared corpus bias rather than three independent epistemic paths arriving at the same answer.
- This document is a process log, not a recommendation. The recommendation lives in `COMPREHENSIVE_PLAN.md`; if the two ever drift, trust the plan.