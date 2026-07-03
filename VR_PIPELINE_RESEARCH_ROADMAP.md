# Room-Scale VR Digital Twins from Consumer ToF Scans — Research & Solution Roadmap

*Senior research-scientist / principal-engineer review of the open-source Femto-ToF → Meta Quest 3 digital-twin pipeline (splat + physics mesh), with a deep SOTA sweep (2024–2026) and adversarial verification.*

**Date:** 2026-06-18 · **Author of review:** automated deep-research + design sweep (9 category finders, 6 adversarial verifiers, 1 synthesis), curated. · **Primary source docs:** `VR_Pipeline_Review_Paper.docx`, `VR_Pipeline_Lab_Report.docx`, `V5_Technical_Brief.docx`, `SCENE_VS_OBJECT_QUESTION.md`, `VERSIONS.md`, `SCENE_SEGMENTER_NOTES.md`, `FUTURE_WORK.md`.

> **Coverage honesty.** This sweep ran ~15 web-research agents across academia (CVPR/ICCV/ECCV/SIGGRAPH/NeurIPS/ICLR 2024–2026 + arXiv), open-source repos, and industry. Coverage is strong on: 3DGS object-removal/inpainting, surface-aligned GS/mesh, VR/mobile splat rendering, ToF/reflection handling, feed-forward recon, and industry systems. Coverage is **thinner** on: (a) hard on-device Quest-3 FPS numbers for room-scale (the field publishes almost none — most "72 FPS Quest 3" claims are PC-tethered), (b) tabletop near-planar occlusion completion (no benchmark exists), and (c) Orbbec-Femto-specific ToF/IR artifacts (no public dataset). Those three gaps are flagged throughout and are themselves opportunities.

---

## 1. Executive Summary

**The problem.** Convert a consumer ToF scan (Orbbec Femto Mega, ~129–263 frames, desk-to-room scale) into a Meta Quest 3 *digital twin* with **two paired deliverables**: a photorealistic 3D Gaussian Splat (visual, ≥72 FPS standalone) and a textured mesh + physics colliders (Unity rigid-body grab-and-place). The pipeline is open-source, Dockerized, single consumer GPU (RTX 5070 Ti, 16 GB, Blackwell sm_120). The **headline live blocker** (stated today in `SCENE_VS_OBJECT_QUESTION.md`): when a grabbable object is removed from the scene splat/mesh, the desk surface that was occluded *in every frame* leaves a **hole**; compositing the object back on top leaves a baked static duplicate. Secondary open issues: view-dependent monitor-reflection "blob" (+ ToF IR ghosts), depth-supervised-splat regression (gsplat 1.5 densifier), no 2DGS in nerfstudio 1.1.5, and standalone-Quest room-scale FPS.

**Best proposed solution (headline problem).** Build a **metric-mesh-seeded amodal fill** as the primary path, and use published 3DGS inpainters only as references/benchmarks. The entire field has converged on *depth-guided* hole-filling (InFusion, GScream, AuraFusion360, 3DGIC, SplatFill, GSFix3D) — every method's hardest sub-problem is estimating *where* the new Gaussians sit, i.e. hallucinating depth in the hole. **This project already owns that depth**: the OpenMVS/Poisson/TSDF mesh interpolates geometry under the object, and per-frame ToF depth is metric. Render that mesh to a depth map under the SAM3 object footprint, unproject to seed correctly-placed Gaussians (InFusion's mechanism, with *known* depth → no diffusion needed for placement), and fill appearance by nearest-neighbour copy from surrounding desk Gaussians (a near-planar, low-texture patch is the *easiest* case) — escalating to a single 2D inpaint (LaMa / FLUX.2-Klein) only if texture is insufficient. Because geometry is locked to the mesh, the team's standing fear that "2D-inpaint-then-retrain is multi-view-inconsistent" is largely **neutralised** — that inconsistency comes from inconsistent *estimated* depth, which the mesh eliminates.

**Highest-leverage pipeline switches (low risk, do now).** (1) **GLOMAP** (in COLMAP 4.0) in place of/alongside COLMAP — identical output format, 1–2 orders faster, +8% recall. (2) **`.spz`** (Niantic, MIT) as the interchange/compression format — the de-facto industry standard and what Meta's native Quest path reads. (3) Fix depth supervision the right way: add DN-Splatter's **EdgeAwareLogL1** loss on top of splatfacto-big's *native* gsplat densifier (not DN-Splatter's gsplat-1.0 densifier — the exact cause of the V25 650k→13k collapse).

**Main risks.** (a) *Standalone room-scale FPS is unproven publicly* — the room background (227k–745k Gaussians) straddles the ~150k (Meta Spatial SDK) to ~400k (ninjamode Unity fork) standalone ceiling; per-object 5k is trivially fine. Even Meta *streams* Hyperscape rather than render room-scale locally. (b) *No ground truth under the object* for single-sweep capture — must build eval via synthetic (Replica/Hypersim) + a physical-removal re-capture (360-USID protocol). (c) *Reflective-monitor blob is unsolved by everyone, including Meta* — the practical answer is *remove* the floater (SpotLessSplats-style robust masking + SAM3 + scale-prune + ToF-IR-ghost cleanup), not *model* it.

**Where the project can be genuinely novel (publishable).** Metric-mesh/ToF-anchored amodal completion (no published method renders a pre-existing metric mesh as the fill's depth target); a **tabletop near-planar single-sweep occlusion benchmark** (a real gap — every benchmark measures large 360/forward-facing removals); and a reproducible Dockerized *measured-on-device* nerfstudio→prune→`.spz`→standalone-Quest recipe (currently folklore-grade).

**Bottom line.** The occlusion problem is solved-in-principle and the project's metric ToF mesh is a genuine, unexploited edge that lets it *skip the hardest published sub-problem*. Keep splatfacto-big and the dual splat+mesh architecture (both validated as field-standard), switch GLOMAP + `.spz` now, build the mesh-seeded fill MVP in days, and benchmark with the 360-USID/SPIn-NeRF protocol plus a synthetic tier.

---

## 2. Project Understanding (Precise Restatement)

### 2.1 True goal
A reproducible, open-source, single-GPU pipeline producing, **per captured scene**, a *co-registered, metric-scale pair*:
- **D1 — Visual splat:** 3DGS rendering of the room/desk, ≥72 FPS on **standalone** Quest 3 (Snapdragon XR2 Gen2), per-object budget ≈ 5k Gaussians / ≈ 30 MB, room background today 227k–745k Gaussians.
- **D2 — Physics asset:** textured mesh + collider metadata for Unity rigid-body grab-and-place (per-object OBJ/MTL/PNG + collider JSON; scene mesh for the floor/desk).

### 2.2 Inputs available
- Femto Mega: RGB (1920×1080), metric ToF depth (float32 m), 850 nm IR; vendor intrinsics (fx≈1125.36, fy≈1125.07, cx≈966.03, cy≈519.94). ~129–263 frames, single handheld sweep, ToF→COLMAP scale factor stored per scene (e.g. 8.7986 on V32 desk; 23.98 elsewhere). iPhone RealityScan branch also supported (HEIC + ARKit).
- Per-object SAM3 text-prompt masks (cached, 140 views × N prompts).
- A scene mesh that **interpolates geometry under occluders** (OpenMVS Poisson-style fill; V26 Open3D TSDF).

### 2.3 Current system (V32/V33, baseline to beat)
COLMAP SfM (intrinsics locked) → ToF back-projection 270–500k colored init → **splatfacto-big** (nerfstudio 1.1.5, gsplat 1.5.3, 30k iters, 2× downscale) → prune (opacity>0.15 + CC) → NanoGS compression. **OpenMVS** Densify→Reconstruct→Refine→TextureMesh (8K UV atlas, 1.3–1.8M faces). Scene segmenter v9a_fp_v2: SAM3 → Open3D BVH rasterized 3-way vote → seed faces (≥0.7) → spatial X-Z footprint crop → per-object mesh + per-object splat (`v6_cleaned` = Clean-GS-pruned). Collider JSON (AABB/OBB/convex hull + recommended Unity collider).

### 2.4 Explicit requirements
Open-source only · Dockerized/reproducible · single consumer GPU · paired & co-registered splat+mesh · metric scale · Quest-3 frame & size budgets · text-prompt per-object extraction · grab-and-place interaction.

### 2.5 Implicit requirements / hidden dependencies
- **Co-registration**: the per-object and scene representations must share one metric frame (already satisfied).
- **Amodal completion**: D1 and D2 both need the *unobserved* desk patch under each grabbable object — neither pruning nor better masks can synthesise it (the live blocker).
- **Standalone, not tethered**: the FPS budget is for the Quest mobile GPU, not a desktop. This invalidates most "VR 3DGS" paper numbers.
- **"Remove, don't model" reflections**: for a grabbable twin, the monitor blob must disappear, not be physically simulated.
- **gsplat 1.5 / Blackwell sm_120 compatibility**: many SOTA repos are CUDA-11.x forks and will not co-exist with the current env.

### 2.6 Ambiguities / missing information (carry into Phase 0)
1. **Scale target**: desk-only forever, or true room-scale? Determines whether LOD (Octree-GS/Hierarchical-3DGS) and a streaming fallback are required.
2. **Deployment surface**: native Unity (aras-p/ninjamode) vs Meta Spatial SDK (native, ≤150k, one splat) vs WebXR (Spark/PlayCanvas). Each has different budgets and decoder costs.
3. **Acceptable hole UX**: is the "hide-and-reveal" transient hole (option C in the source doc) acceptable, or must the desk be permanently complete?
4. **Faithful vs plausible fill**: is a *plausible* (possibly hallucinated) desk patch acceptable, or must it be *faithful* (recovered from other frames only)? This sets whether generative priors are allowed.
5. **Eval rigor**: can a few canonical scenes be captured with-and-without objects (and on a tripod) to obtain real GT-under-object?

### 2.7 Known failure modes (from the project's own negative results — do not relitigate)
DA3 monocular init scale-mismatch (V5); DN-Splatter densifier broken on gsplat 1.5 → sparse splat (V25); depth_lambda=0.2 over-constraint regressed visuals (V30); TSDF+UV-atlas blocked by OpenMVS TextureMesh SEGV (V27/V31); SAM3 screen masking alone did *not* kill the blob (V33); many algorithm tweaks failed to beat vanilla V23 → **data/capture quality is the ceiling on the current scene.**

---

## 3. State-of-the-Art Landscape

Nine categories, ~107 systems reviewed. Per-category state, full item lists, gaps, and picks are preserved in the research dump; the decision-relevant comparison follows.

### 3.1 Headline comparison — 3DGS scene inpainting / object removal (the live problem)

| Method | Venue | License / code | GPU & time | Benchmark (360-USID PSNR↑/LPIPS↓) | Fit to *tabletop near-planar small hole* | Rec. |
|---|---|---|---|---|---|---|
| **AuraFusion360** | CVPR 2025 | Apache-2.0, [code](https://github.com/kkennethwu/AuraFusion360_official) | RTX 4090, ~1 min | **17.66 / 0.388** (best) | Built for "occluded in *all* views"; depth-aware *unseen-mask* + Adaptive Guided Depth Diffusion; ships **360-USID** GT benchmark. 360-tuned; admits floaters near heavy occlusion. | **use** (reference + benchmark) |
| **InFusion** | 2024 (arXiv 2404.11613) | MIT, [code](https://github.com/ali-vilab/Infusion) | single GPU, ~40 s | 14.42 / 0.484 | *Depth-completion → unproject Gaussians*; the canonical "depth-as-scaffold". Replace its learned depth with **your metric ToF mesh** → skips its failure mode. | **adapt** (core mechanism) |
| **GSFix3D** | arXiv 2508.14717 (code 2025-11-18) | [code](https://github.com/GSFix3D/GSFix3D) | single 24 GB | fine-tune 2–4 h | **Explicitly fuses rendered mesh + 3DGS** as a diffusion prior (`--dual_input`); "fills large holes"; ScanNet++ 25.63 PSNR. Custom data must be Replica format; verify 16 GB fit. | **test** (off-the-shelf mesh-prior) |
| **3DGIC** | CVPR 2025 | MIT, [code](https://github.com/peterjohnsonhuang/3dgic) | RTX 3090, 5000 iters | beats GScream/GG on SPIn-NeRF (FID 36.4) | Cross-view **mask-shrinking**: fill desk from *other frames first*, hallucinate only the residual. Best forward-facing/table fit. **Public repo is an admitted "suboptimal version."** | **adapt** (the mask-shrink idea) |
| **Gaussian Grouping** | ECCV 2024 | Apache, [code](https://github.com/lkeab/gaussian-grouping) | single GPU | 16.07 / 0.480 | render→LaMa→finetune = the *naive multi-view-inconsistent* path the team already distrusts. Good backbone reference. | adapt (study only) |
| **GScream** | ECCV 2024 | [code](https://github.com/W-Ted/GScream) | RTX 3090 | 14.76 / 0.514 | Depth-prior + cross-attn coherence; online depth-align becomes trivial with ToF. Scaffold-GS base. | adapt (geometry side) |
| **Inpaint360GS** | arXiv 2511.06457 | paper-only | ~24 min | 24.40 PSNR / 0.837 SSIM | Virtual cameras expose the hole + remove occluders first. Idea is portable. | watch |
| **GPGS** | AAAI 2026 | paper-only | — | claims SOTA geom restoration | Only method that conditions the fill on a **point-cloud completion** model — architecturally ideal for your 270–500k ToF cloud. | watch |
| **SplatFill** | arXiv 2509.07809 | **no code, A100-80 GB** | A100, 34 min | SPIn-NeRF 20.46 / 0.25 | "Only re-touch the inconsistent hole" philosophy is ideal, but fails open-source + consumer-GPU. | **avoid** (as build target) |
| **InstaInpaint** | NeurIPS 2025 | [code](https://github.com/dhmbb2/InstaInpaint) | ~0.4 s feed-forward | claims 1000× speedup | GS-LRM-based feed-forward fill; generalization to Femto data unverified. | watch |
| **NeRFiller** | CVPR 2024 | [code](https://github.com/ethanweber/nerfiller) | nerfstudio-native | — | Grid-of-views diffusion trick for multi-view consistency; NeRF (not Quest-deployable). | watch (idea) |
| **CAT3D / ReconFusion / Bolt3D** | NeurIPS/CVPR 2024–25 | **closed (Google)** | heavy | strong few-view NVS | Conceptual north star; not open. | avoid |

**Field consensus:** depth-guided placement + 2D-inpaint-for-texture + multi-view-consistency refinement. **Field gap:** nobody isolates the small near-planar regime and nobody ingests a *real metric mesh* prior — both are the project's openings.

### 3.2 Surface-aligned GS & mesh extraction (D2 geometry)

Quality ladder (DTU Chamfer mm ↓ / Tanks&Temples F1 ↑): 3DGS 1.96/0.09 ≪ SuGaR 1.33/0.19 < **2DGS 0.80/0.30** < **GOF 0.74/0.34** < **RaDe-GS 0.69/0.37** < **PGSR ~0.49/0.50** < Neuralangelo 0.61/0.50 (but ~100× slower). **MILo** (TOG 2026) extracts a *differentiable, watertight, low-poly* mesh in-loop, 10× fewer vertices, T&T F 0.47–0.49 — the most on-target for the paired deliverable, but code-availability unverified.

| Method | Code / status | Why it matters here | Rec. |
|---|---|---|---|
| **2DGS** | [code](https://github.com/hbb1/2d-gaussian-splatting); **native in gsplat 1.5** (`simple_trainer_2dgs.py`) — but **not** in nerfstudio splatfacto ([open issue #3677](https://github.com/nerfstudio-project/nerfstudio/issues/3677)) | Lowest-integration surface-GS on the *current* gsplat; run standalone trainer on existing COLMAP poses, ~4 GB VRAM. A/B vs V26 TSDF. | **use** (A/B) |
| **RaDe-GS** | [code](https://github.com/BaowenZ/RaDe-GS) | Best accuracy/speed of *available* code (DTU 0.69, ~18 min); splat+mesh from one run. CUDA-fork → separate Docker, sm_120 risk. | adapt (pilot) |
| **PGSR** | [code](https://github.com/zju3dv/PGSR) | Quality leader (0.49 mm planar geometry) → strong *amodal scaffold*; but TSDF depth-filter leaves holes (bad for watertight, fine for convex collider). | adapt (geometry source) |
| **GOF** | [code](https://github.com/autonomousvision/gaussian-opacity-fields) | Best for unbounded backgrounds; CUDA 11.3 pin = Blackwell pain. | watch |
| **MILo** | [project](https://anttwo.github.io/milo/), code TBD | Differentiable watertight low-poly mesh + splat in one run — eventual winner. | **watch** |
| **DN-Splatter / AGS-Mesh** | [code](https://github.com/maturk/dn-splatter) | ToF-native loss defs (reuse), but densifier is gsplat-1.0 (the V25 regression). | **avoid** densifier; reuse loss |
| **GS-Verse / Mani-GS / SuGaR-Frosting** | [GS-Verse](https://github.com/Anastasiya999/GS-Verse), [Mani-GS](https://github.com/gaoxiangjun/Mani-GS), [SuGaR](https://github.com/Anttwo/SuGaR) | Mesh↔splat *binding* for grab (anchor splat to collider) — the Unity blueprint. | adapt |

**Verified verdict:** *watertightness is a red herring* — Unity rigid-body grab uses **convex MeshColliders** (≤255 tris, auto-hulled) that tolerate messy input. **Keep OpenMVS/TSDF for colliders.** Surface-GS's real payoff is cleaner geometry for the amodal scaffold, not the collider.

### 3.3 VR rendering / compression / LOD — standalone Quest 3

| Component | Evidence | Use |
|---|---|---|
| **UnityGaussianSplatting (aras-p)** + **ninjamode VR fork** | community-verified **~72 FPS to ~400k Gaussians standalone** Quest 3 (needs VR-optimized radix sort; stock Unity chokes at ~50k) | **use** (runtime) |
| **Meta Spatial SDK native splat** | official: **avoid >150k**, *one splat at a time*, **SH degree 0**, no LOD | watch (native fallback) |
| **SqueezeMe** (Meta RL) | *verified standalone* custom Vulkan on XR2 Gen2: 3×~60k = **~180k @72 FPS** | proof point |
| **Mobile-GS** | **sort-free** order-independent: 74 FPS steady @0.47M on Snapdragon 8 Gen 3 phone | **test** (room-scale FPS lever) |
| **Niantic `.spz`/SPZ4** | MIT, ~10× smaller than PLY, Unity RUF built-in, the format Meta reads | **use** (format) |
| **NanoGS** (current) | best **PSNR-per-Gaussian** (25.81 @90% prune); **count-reduction only** (not a format/viewer/LOD) | keep, but pair |
| **LightGaussian** | image-supervised prune + SH distillation, **decoder-free**, cuts *render* cost (139→215 FPS) | **test** (you have the images → likely beats NanoGS) |
| **SOG/SOGS** | ~20–40× size; web/PlayCanvas; decode-cheap | adapt (web path) |
| **Octree-GS / Hierarchical-3DGS / LODGE** | real runtime LOD (the room-scale lever) — but **mobile decode cost unverified** | research-later |
| **VRSplat / VR-Splatting** | "72 FPS Quest 3" — **tethered RTX 4090**, not standalone; port StopThePop + optimal-projection tricks | adapt (tricks) |

**Verified verdict:** a Quest deliverable needs a **3-part stack** — count reduction (NanoGS/LightGaussian) **+** format (`.spz`/SOG) **+** viewer (ninjamode / Mobile-GS). Per-object 5k is trivial; the **room background is the constraint**. "72 FPS Quest 3" papers are tethered until measured on-device.

### 3.4 Object-centric decomposition & physics

Two-stage field standard: lift 2D SAM masks → per-Gaussian identity/feature; drive physics via a **mesh/tet proxy**. **Physics-on-Gaussians is refuted as mature/standalone:** VR-GS ([repo is just the Nerfies website](https://github.com/YingJiang96/VR-GS) — no code, tethered, soft-body, 19–40 FPS multi-object on a 4090); GS-Verse ([open](https://github.com/Anastasiya999/GS-Verse), mesh-face anchoring, tethered, small traction); all use a mesh in the loop. **Dual splat+mesh is the correct 2026 standard.** Segmentation alternatives to the current SAM3+BVH vote (OpenGaussian, InstanceGaussian, SAGA) offer cleaner boundaries but only marginal gains. For rigid grab: **parent the per-object splat to its mesh collider's RigidBody transform** — no XPBD needed.

### 3.5 ToF depth supervision & reflections

Depth supervision is a solved regularizer (**EdgeAwareLogL1**, depth_lambda~0.2) — the V25 collapse was DN-Splatter's gsplat-1.0 densifier, not depth supervision itself. Reflections split into *model* (3DGS-DR, Ref-Gaussian, Spec-Gaussian, VoD-3DGS, Mirror-3DGS — far-field/clean-mirror assumptions, wrong goal/cost) vs *remove* (**SpotLessSplats** utilization-based pruning + masked loss, WildGaussians). **No open method cleanly removes near-field screen-reflection floaters while keeping the planar surface at mobile cost.** Practical answer: SpotLessSplats-style robust masking + SAM3 screen masks + scale-filter prune + **ToF-Splatting**-style multi-view outlier filter to clean IR ghosts *before* back-projection init. **FisherRF** active-capture maps can guide a re-sweep of occluded/glossy regions (capture-side lever — consistent with "data is the ceiling").

### 3.6 Datasets, benchmarks, evaluation

No single benchmark covers the full stack. Use **MuSHRoom** (consumer RGB-D + Faro GT, joint mesh+NVS, *same group* as AGS-Mesh) as the external comparison and capture template; **360-USID** (AuraFusion360) physical-removal-then-recapture protocol for real GT-under-object; **SPIn-NeRF** metric defs (M-LPIPS, box-FID); **Replica/Hypersim** for *perfect* synthetic occluded-floor GT (controlled ablation); **Remove360** to verify no semantic residual/duplicate; **3DGS.zip** compression leaderboard; **VRSplat protocol + FovVideoVDP (JOD) + FLIP** for VR floater eval; **Chamfer/F-score/normal-consistency** (DTU/T&T) for mesh. **Gap = the project's opportunity:** no tabletop near-planar single-sweep occlusion benchmark exists.

### 3.7 Industry & big-lab systems

| System | What it is | Open? | Lesson / use |
|---|---|---|---|
| **Meta Hyperscape Capture** | Quest 3 room scan → ~4 h cloud reconstruction → on-device render (Horizon Engine) | **closed, no export, no API** | The *experience bar*; **shares the exact under-furniture occlusion + reflection failure** → validates the project's problem is unsolved even at Meta. Benchmark, don't depend. |
| **Niantic `.spz`/SPZ4** | "JPG for splats" | **MIT** | **Adopt as format.** |
| **Scaniverse / Into the Scaniverse** | free, **on-device**, `.spz`, Quest viewer | app closed, format open | iPhone-branch quality bar + interop. |
| **Spark (World Labs)** | web/three.js renderer fusing **splat + mesh**, LOD/streaming 100M+ | **MIT** | Architectural blueprint for splat+mesh compositing; WebXR option. |
| **PlayCanvas SuperSplat / SplatTransform / SOGS** | splat editor + CLI convert + ~20× compress | **open** | Dockerizable PLY↔SPZ↔SOG conversion + manual blob cleanup. |
| **OpenQuestCapture** | Quest 3 stereo+depth → COLMAP | **MIT** | Alternative open capture front-end. |
| **NVIDIA 3DGUT / gsplat / fVDB** | optics-aware (unscented transform) Gaussian rasterization | open | **Test 3DGUT vs the monitor blob.** |
| **Marble (World Labs), Tripo/Meshy** | generative worlds/objects, export splat+mesh | closed/freemium | Generative = hallucinated; **unsuitable for a faithful twin** (idea-only). |
| **Polycam / Luma / Varjo Teleport** | cloud capture, splat+mesh export | paid/cloud | Benchmark only; Luma/Polycam mesh export *also* holes/floaters → industry-wide unsolved. |

### 3.8 Feed-forward recon & pose

- **GLOMAP** (ECCV 2024, in COLMAP 4.0): drop-in, identical format, 1–2 orders faster, +8% recall/AUC. **Switch now.**
- **VGGT** (CVPR 2025 Best Paper, [code](https://github.com/facebookresearch/vggt)) / **Pi3** (ICLR 2026, [code](https://github.com/yyfz/Pi3), more robust: RE10K AUC 85.9 vs 77.6): feed-forward poses for textureless/reflective *small* scans where COLMAP is fragile → export COLMAP format, keep ToF for metric scale. Use VGGT-X BFloat16+chunking to fit 16 GB; 129–263 frames is near the attention limit.
- **Honest ceiling (VGGT-X):** feed-forward poses + refine still trail COLMAP-init per-scene 3DGS by **~1 PSNR** (26.40 vs 27.39 MipNeRF360) and overfit. **Keep splatfacto-big as the final deliverable.**
- **FreeSplat/MVSplat domain gap confirmed** (and the project already saw FreeSplat underperform) → avoid as final method.

---

## 4. Key Insights from Literature & Industry

1. **The occlusion problem is solved-in-principle, and depth is the lever.** Every 2024–2026 3DGS inpainter delegates truly-unseen pixels to a 2D inpaint and fights for 3D consistency *via depth*. The project owns the depth the others must hallucinate. → *Skip the hardest published sub-problem.*
2. **The team's "2D-inpaint is multi-view-inconsistent" fear is half-right.** It's true for the *naive* Gaussian-Grouping render→LaMa→finetune loop, but 2025 SOTA fixes it with depth-guided mask-shrinking (3DGIC) and unseen-mask + depth-anchored placement (AuraFusion360). With geometry locked to a metric mesh, the inconsistency source is removed.
3. **Most of the desk is *seen* at some sweep angle.** 3DGIC's cross-view mask-shrinking implies the *truly* amodal core is small — quantify it; you may need to hallucinate very little.
4. **Watertightness ≠ collider-grade.** Convex MeshColliders tolerate messy meshes; benchmark "F-score leaders" are often *worse* (holes). Don't rip out OpenMVS for D2.
5. **"72 FPS Quest 3" is mostly tethered.** Genuine standalone evidence caps at ~150k (Meta SDK) to ~400k (ninjamode). Even Meta *streams* room-scale. The room background, not per-object splats, is the constraint.
6. **Physics-on-Gaussians is not real for standalone** — and always uses a mesh proxy anyway. The dual splat+mesh architecture is correct.
7. **Reflections: remove, don't model.** Physical-reflection methods assume far-field/clean-mirror; near-field content-bearing monitors break them. Robust masking + ToF-IR-ghost cleanup is the practical path.
8. **Feed-forward is for poses/priors, not the final splat.** GLOMAP/VGGT/Pi3 improve robustness; per-scene splatfacto remains the deliverable (~1 PSNR gap).
9. **Capture quality is the ceiling** — independently corroborated by the project's own V23-beats-everything finding *and* by EmbodiedSplat/VGGT-X domain-gap results. FisherRF-guided re-capture is a real lever.
10. **Format convergence on `.spz`** across Niantic/Meta/PlayCanvas/World Labs is the cheapest interop win.

**Still unsolved (by anyone, incl. Meta):** faithful amodal completion of small near-planar occluded regions; near-field reflective-floater removal at mobile cost; standalone room-scale 72 FPS without quality compromise or streaming; a tabletop occlusion benchmark.

---

## 5. Recommended Solution Architecture

End-to-end, modular, Dockerized; **bold = change from current**.

```
Capture (Femto Mega RGB+ToF+IR, ~140 frames)  ──►  [ToF-IR-ghost pre-clean]  (ToF-Splatting-style multi-view outlier filter, esp. glossy screens)
        │
        ▼
Pose  ──►  GLOMAP (COLMAP 4.0) default; **VGGT/Pi3 → COLMAP-export fallback** for textureless/reflective scans.  ToF gives metric scale.
        │
        ▼
Init  ──►  ToF back-projection 270–500k colored cloud (keep)
        │
   ┌───────────────────────────────────────────────┬───────────────────────────────────────────────┐
   ▼ VISUAL (D1)                                     ▼ PHYSICS (D2)
splatfacto-big (nerfstudio 1.1.5 / gsplat 1.5.3)     OpenMVS Densify→Reconstruct→Refine→TextureMesh (keep, 8K atlas)
  + **EdgeAwareLogL1 depth loss (λ~0.2) on the         + V26 Open3D TSDF (coverage)
    NATIVE gsplat densifier**                          + **pilot RaDe-GS / 2DGS-gsplat / MILo** (cleaner geometry,
  + **SpotLessSplats robust masking + SAM3 screen        amodal scaffold) in a separate Docker
    masks + scale-filter prune** (kill monitor blob)
        │                                                     │
        ▼                                                     ▼
Scene splat                                           Scene mesh (+ interpolated geometry under occluders)
        │                                                     │
        └──────────────────────────┬──────────────────────────┘
                                    ▼
        OBJECT DECOMPOSITION  (keep: SAM3 → BVH 3-way vote → X-Z footprint crop → Clean-GS v6_cleaned)
                                    │
                                    ▼
        ★ AMODAL FILL (NEW — the headline fix) ★
        For each removed object footprint:
          1. Render scene mesh (TSDF/Poisson fill) to per-view depth, masked to SAM3 footprint.
          2. **3DGIC-style cross-view reprojection**: fill desk from frames that DID see it; isolate the tiny truly-unseen core.
          3. **InFusion-style unproject** the masked mesh-depth → seed new Gaussians at metric positions (NO depth diffusion).
          4. Appearance: nearest-neighbour copy/propagate from surrounding desk Gaussians;  escalate to a single 2D inpaint
             (LaMa / FLUX.2-Klein) on the FIXED geometry only if texture insufficient (GScream/GSFix3D style).
          5. Repeat for the scene MESH (seed faces on the mesh's interpolated surface; bake texture from the fill).
          6. Remove360 residual check: confirm no baked duplicate remains.
                                    │
                                    ▼
        COMPRESS + PACKAGE
          Splat: NanoGS / **LightGaussian** count-reduce (≤400k room, ≤5k object) → **.spz / SOG export**
          Mesh: decimate → convex-hull / VHACD collider (≤255 tris) + collider JSON (keep)
                                    │
                                    ▼
        UNITY / QUEST 3 (standalone)
          Renderer: **UnityGaussianSplatting + ninjamode VR fork** (or Mobile-GS sort-free for room-scale)
          Grab: **per-object splat parented to mesh-collider RigidBody transform** (rigid; no XPBD)
          (room-scale fallback: LOD via Octree-GS, or cloud/edge streaming if local FPS fails)
```

### 5.1 Module choices & rationale

| Module | Recommended | Alternatives | Why chosen | Risk / fallback |
|---|---|---|---|---|
| **ToF pre-clean** | ToF-Splatting-style multi-view outlier filter on depth/IR before init | None mature | Attacks the blob/ghost *at the source*; cheap | Paper-only; reimplement the filter (project already does outlier clipping) |
| **Pose** | **GLOMAP** (COLMAP 4.0) | VGGT/Pi3 export; MASt3R-SfM; keep COLMAP | Drop-in, same format, faster, more robust; zero domain risk | GLOMAP fails on a minority of low-overlap scenes → fall back to incremental COLMAP |
| **Init** | ToF back-projection (keep) | hybrid ToF+DA3 for >ToF-range | Already optimal in-range (V23/V24) | — |
| **Splat** | **splatfacto-big + EdgeAwareLogL1 on native densifier + robust masking** | 2DGS-gsplat; AnySplat (preview) | Validated baseline; correct depth-loss path; blob suppression | Depth loss can over-constrain (V30) → tune λ, ablate; keep vanilla as control |
| **Mesh (collider)** | **OpenMVS (keep)** + TSDF coverage | RaDe-GS/PGSR/MILo geometry | Convex collider makes watertightness moot; OpenMVS already good | Pilot RaDe-GS only as geometry source |
| **Amodal fill** | **Metric-mesh-seeded fill (InFusion mechanism + 3DGIC mask-shrink, known depth)** | GSFix3D `--dual_input`; AuraFusion360 | Exploits the project's unique metric mesh; sidesteps depth hallucination → multi-view-consistent | If NN-copy texture weak → 2D inpaint on fixed geometry; if all fails → hide-and-reveal UX |
| **Blob removal** | **SpotLessSplats UBP + masked loss + SAM3 + scale-prune** | 3DGUT (test); recapture monitors-off | Removes (not models) the floater; cuts Gaussians for budget | May mask real screen → tune; ultimate fix = recapture |
| **Decomposition** | SAM3 + BVH vote (keep) | OpenGaussian / InstanceGaussian | Already production; field-aligned | Swap only if boundaries fail on a new scene |
| **Compression** | **NanoGS/LightGaussian + `.spz`/SOG** | HAC++ (archival) | Count + format + decoder-free for mobile | HAC++ decode unproven on mobile → keep off runtime path |
| **Viewer** | **ninjamode Unity VR fork**; Mobile-GS (room-scale) | Meta Spatial SDK (≤150k); Spark (WebXR) | Only open renderer with verified ~400k standalone | Measure on-device; streaming fallback for room-scale |
| **Physics** | Per-object splat parented to mesh-collider RigidBody | GS-Verse / Mani-GS binding | Rigid grab needs only a transform; dual rep is field standard | — |

---

## 6. Alternative Architectures (≥3)

### Alt A — "Off-the-shelf inpainter" (fastest to a published baseline)
Run **AuraFusion360** (or **GSFix3D** with `--dual_input` fed the project mesh) directly on each extracted-object hole; keep everything else as-is.
- **Pros:** least new code; immediately benchmarkable against published numbers; AuraFusion360 ships GT (360-USID).
- **Cons:** 360-tuned, may over-hallucinate a small planar patch and produce floaters near heavy occlusion (its own admitted failure); research-grade repos vs gsplat 1.5.3; doesn't exploit the metric mesh.
- **Best as:** the *reference baseline* the recommended mesh-seeded fill must beat (and it likely will, since it supplies the depth AuraFusion360 must guess).

### Alt B — "Capture-side first" (lowest algorithmic risk)
Solve the hole and blob with **better data**: FisherRF-guided multi-pass capture (re-sweep occluded under-object regions + glossy monitors), shift-the-object micro-passes (the source doc's option D/Q5), and recapture monitors-off. Minimal new modeling.
- **Pros:** consistent with the project's own "data is the ceiling" finding; faithful (no hallucination); fixes blob *and* hole at the source.
- **Cons:** capture burden; doesn't generalise to immovable objects (sofa, wired lamp); needs a guidance UI in the ScanGuide app; coordinate alignment across passes.
- **Best as:** a parallel track + the gold-standard *eval* capture (with/without object) — and the right answer for movable objects.

### Alt C — "Generative-prior twin" (highest ceiling, lowest faithfulness)
Use **TRELLIS** (MIT) to regenerate clean watertight per-object splat+mesh from SAM3 crops (ICP-register to metric scene), and a scene-level generative completion (NeRFiller grid-prior, or Marble-style) for the desk patch.
- **Pros:** clean grabbable assets with complete unseen sides; handles partially-occluded objects (Amodal3R).
- **Cons:** hallucinated geometry/texture — *unacceptable for a faithful digital twin* of a real desk; canonical-pose registration overhead; license landmines (Hunyuan3D/SF3D/FLUX.1); still needs Clean-GS/NanoGS for budget.
- **Best as:** a *fallback for totally-unseen objects/regions only*, never the primary faithful path.

### Alt D — "Unified surface-GS deliverable" (architectural simplification)
Replace the split (splatfacto + OpenMVS) with a single **RaDe-GS / MILo** run yielding splat + watertight low-poly mesh together; bind via SuGaR-Frosting/Mani-GS for grab.
- **Pros:** one optimization, co-registered by construction, MILo's 10× fewer vertices help the Quest budget; cleaner amodal scaffold.
- **Cons:** CUDA-fork/sm_120 Blackwell build risk; surface-GS underperforms indoors/on reflective surfaces; MILo code unverified; loses OpenMVS's 8K UV atlas (texture fidelity the user values — cf. V26/V31 rejections).
- **Best as:** a medium-term pilot once MILo code lands; not a near-term switch.

**Trade-off summary:** Recommended (mesh-seeded fill) = best faithfulness × low risk × exploits a unique asset. Alt A = fastest baseline. Alt B = most faithful, capture cost. Alt C = highest visual ceiling, not faithful. Alt D = cleanest architecture, highest integration risk.

---

## 7. Implementation Roadmap (Phased)

### Phase 0 — Clarify & set up (≈1 week)
- **Tasks:** resolve §2.6 ambiguities (scale target; deployment surface; faithful-vs-plausible; hole-UX tolerance; eval-capture feasibility). Stand up Docker images: COLMAP 4.0 (GLOMAP), a `.spz`/SplatTransform tool container, and an *isolated* surface-GS container (RaDe-GS/2DGS). Add `.spz` export.
- **Deliverables:** decision memo; GLOMAP swapped in; `.spz` export working; SplatTransform in the pipeline.
- **Success:** GLOMAP reproduces V32 poses (same format) faster; PLY↔`.spz`↔SOG round-trips; one V32 splat renders in ninjamode fork on a Quest 3.
- **Risk:** GLOMAP fails on a scene → keep incremental COLMAP fallback.

### Phase 1 — Baseline prototype: the mesh-seeded amodal fill MVP (≈2–3 weeks)
- **Tasks:** (1) render scene TSDF/Poisson mesh → per-view depth under SAM3 footprint; (2) 3DGIC-style cross-view reprojection to isolate the truly-unseen core (and *measure its size*); (3) InFusion-style unproject → seed Gaussians at metric depth; (4) NN-copy appearance from surrounding desk; (5) same for the mesh (seed faces + bake texture); (6) Remove360 residual probe. Few-hundred LOC on the existing gsplat pipeline.
- **Deliverables:** filled scene splat + mesh for the V32 desk (bottle/box/lobster); side-by-side vs hole and vs baked-duplicate.
- **Success:** no visible hole when an object is lifted; no static duplicate; multi-view-consistent on the held-out trajectory; truly-unseen core quantified.
- **Risk:** NN-copy texture too blurry → escalate to single 2D inpaint on fixed geometry (Phase 2).

### Phase 2 — Improved research prototype (≈3–5 weeks)
- **Tasks:** add 2D-inpaint texture stage (LaMa / FLUX.2-Klein) on the locked geometry; depth-supervised splat done right (EdgeAwareLogL1 on native densifier, λ-ablation to avoid the V30 over-constraint); SpotLessSplats robust masking + ToF-IR-ghost pre-clean for the blob; LightGaussian vs NanoGS head-to-head; pilot RaDe-GS / 2DGS-gsplat / 3DGUT in the isolated container.
- **Deliverables:** blob-reduced V32; depth-supervised splat that *stays dense*; compression Pareto (NanoGS vs LightGaussian vs SOG/`.spz`); surface-GS A/B report.
- **Success:** blob visibly reduced without scene-quality loss; depth-supervised splat ≥ vanilla V23 visually; chosen compressor hits ≤400k room / ≤5k object at acceptable quality.
- **Risk:** depth supervision regresses again → keep vanilla as control; blob persists → recapture monitors-off (accepted ROI).

### Phase 3 — Evaluation & benchmarking (≈3 weeks, overlaps P2)
- **Tasks:** build the 3-tier eval (§8); capture ~3–5 desk scenes *with and without* objects (360-USID protocol, tripod reference); synthetic Replica/Hypersim ablation; run against MuSHRoom; measure **on-device** Quest-3 FPS/ms.
- **Deliverables:** masked-region PSNR/SSIM/LPIPS/FID for fills; Chamfer/F-score for mesh; FovVideoVDP/FLIP for blob; on-device FPS table; comparison vs AuraFusion360/3DGIC/GSFix3D and vs Hyperscape (experiential).
- **Success:** fill beats AuraFusion360/GSFix3D on masked-region metrics on the tabletop scenes; documented standalone FPS at the chosen budget.
- **Risk:** no GT for most scenes → rely on synthetic tier + plausibility (FID/M-LPIPS) + Remove360.

### Phase 4 — Production-ready system (≈4–6 weeks)
- **Tasks:** end-to-end Dockerized recipe (capture→pose→splat+mesh→decompose→fill→compress→`.spz`→Unity); Unity grab-and-place (per-object splat parented to collider RigidBody) via ninjamode fork; room-scale LOD or streaming fallback decision; idempotent stage checkpoints (extend existing).
- **Deliverables:** one-command pipeline; Quest 3 build with grabbable objects + complete desk; reproducibility doc.
- **Success:** a non-expert reproduces a grabbable twin on a fresh capture; ≥72 FPS standalone at target budget.
- **Risk:** room-scale FPS misses → ship desk-scale + streaming fallback (Nebula-style shared-stereo idea).

### Phase 5 — Advanced research extensions (ongoing)
- MILo unified watertight-mesh+splat when code lands; GPGS point-cloud-completion-conditioned fill; FisherRF-guided ScanGuide re-capture UI; multi-pass loop-closure for large scenes (already scaffolded in `refine_poses_icp.py`/`register_multipass.py`); IR-channel supervision; the tabletop occlusion **benchmark + paper**.

---

## 8. Evaluation Plan

### 8.1 Metrics
- **Splat NVS:** PSNR / SSIM / LPIPS (every-8th-image hold-out).
- **Hole-fill (masked region):** M-LPIPS + box-dilated FID (SPIn-NeRF defs); masked PSNR/SSIM where GT exists; Chamfer in the masked region (geometry).
- **Removal residual:** Remove360 semantic-residual probe (confirm no baked duplicate).
- **Mesh:** Chamfer (DTU protocol) + F-score (T&T) + normal consistency vs Faro/synthetic GT; convex-collider penetration/plausibility for grab.
- **VR fidelity / blob:** FovVideoVDP (JOD, temporal, across a trajectory) + NVIDIA FLIP error maps; **on-device** Quest-3 FPS + per-frame ms (report standalone, not tethered).
- **Compression:** PSNR/SSIM/LPIPS vs MB (3DGS.zip frontier) + Gaussian count vs FPS on-device.
- **Segmentation:** mIoU / PQ (ScanNet++ conventions) for per-object extraction.

### 8.2 Datasets / capture
- **External:** MuSHRoom (joint mesh+NVS), ScanNet++ v2 (NVS/seg conventions), DTU/T&T (mesh), SPIn-NeRF + 360-USID (removal).
- **Synthetic (Tier-1 GT):** Replica/Hypersim — render desk *with then without* an object for *perfect* occluded-floor GT.
- **Real (Tier-2 GT):** 360-USID protocol on ~3–5 desk scenes — capture with object, physically remove, re-sweep + tripod reference.
- **Tier-3 (no GT):** all other scenes — FID/M-LPIPS vs a clean reference desk patch + Remove360 residual check.

### 8.3 Benchmark comparisons
Fill vs AuraFusion360, 3DGIC, GSFix3D, Gaussian Grouping (naive), InFusion. Pose vs COLMAP/GLOMAP/VGGT/Pi3. Compression: NanoGS vs LightGaussian vs SOG vs `.spz`. Pipeline vs MuSHRoom splatfacto baseline; *experiential* vs Meta Hyperscape, Scaniverse.

### 8.4 Ablations
(1) mesh-seeded depth vs estimated mono-depth (the core hypothesis); (2) NN-copy vs 2D-inpaint texture; (3) with/without 3DGIC cross-view reprojection (how small is the truly-unseen core?); (4) depth_lambda sweep (avoid V30); (5) SpotLessSplats vs SAM3-mask-only vs scale-prune for the blob; (6) ToF-IR-ghost pre-clean on/off; (7) compressor × viewer × budget for FPS.

### 8.5 Human eval & failure-case analysis
Small VR user study (grab realism, hole visibility, blob salience) or FovVideoVDP proxy. Catalogue failure cases: floaters near heavy occlusion (AuraFusion360's mode), texture smear on large unseen cores, blob redistribution (V33), reflective-surface depth corruption, off-rest-pose re-placement (hide-and-reveal limit).

---

## 9. Risk Register

| # | Risk | Type | Likelihood | Impact | Mitigation |
|---|---|---|---|---|---|
| R1 | Truly-unseen desk core too large → fill must hallucinate, smears | Technical | Med | High | 3DGIC reprojection to shrink core; quantify it Phase 1; fall back to 2D inpaint / hide-and-reveal UX |
| R2 | Standalone room-scale < 72 FPS | Product | Med-High | High | LOD (Octree-GS), aggressive prune to ≤400k, Mobile-GS sort-free, streaming fallback; keep per-object high-detail |
| R3 | Depth supervision regresses again (V25/V30) | Technical | Med | Med | EdgeAwareLogL1 on *native* densifier; λ-ablation; vanilla control retained |
| R4 | Monitor blob persists | Technical | Med-High | Med | SpotLessSplats + SAM3 + scale-prune + ToF-IR pre-clean; **recapture monitors-off** (best ROI) |
| R5 | Surface-GS repos won't build on Blackwell sm_120 | Technical/infra | Med | Low-Med | Isolated Docker; prefer 2DGS-in-gsplat (already builds); keep OpenMVS |
| R6 | No GT-under-object → can't prove fill quality | Data/eval | High | Med | Synthetic Tier-1 + physical-recapture Tier-2 + plausibility Tier-3 |
| R7 | 16 GB VRAM too tight for GSFix3D/VGGT | Compute | Med | Low | 2× downscale; BFloat16+chunking (VGGT-X recipe); the MVP avoids diffusion entirely |
| R8 | License landmines (Hunyuan3D/SF3D/FLUX.1/NanoGS CC-BY-NC) | Licensing | Med | Med | Prefer MIT/Apache (TRELLIS, Hi3DGen, LaMa, FLUX.2-Klein, `.spz`, GLOMAP, AuraFusion360); legal review before any release |
| R9 | Repos are "suboptimal"/paper-only (3DGIC, SplatFill, MILo, GPGS, Inpaint360GS) | Research | Med | Low-Med | Treat as *ideas*; build own MVP; don't block on unreleased code |
| R10 | Meta Hyperscape sets an unreachable visual bar | Product | Low | Low | Differentiate on open/exportable/grabbable/physics; Meta shares the same occlusion+reflection weaknesses |
| R11 | Feed-forward pose noise degrades splat on larger scenes | Technical | Med | Med | GLOMAP first; VGGT/Pi3 only where COLMAP fragile; keep COLMAP fallback |
| R12 | Generative fill hallucinates wrong desk content | Technical/product | Med (if Alt C) | High | Keep generative as fallback-only; faithful mesh-seeded path is primary |

---

## 10. Resources Needed

- **Models / repos (open):** GLOMAP (COLMAP 4.0); VGGT / Pi3; AuraFusion360, 3DGIC, InFusion, GSFix3D (fill references); 2DGS-gsplat / RaDe-GS / (MILo when released); DN-Splatter (loss defs only); SpotLessSplats; ToF-Splatting (reimplement); LightGaussian / NanoGS; Niantic `.spz`, PlayCanvas SplatTransform/SOGS; UnityGaussianSplatting + ninjamode fork / Mobile-GS; LaMa / FLUX.2-Klein (2D inpaint); TRELLIS / Amodal3R (Alt C only).
- **Datasets:** MuSHRoom, ScanNet++ v2, SPIn-NeRF, 360-USID, Replica, Hypersim, DTU, Tanks&Temples; FovVideoVDP + FLIP tools.
- **Compute:** current RTX 5070 Ti (16 GB) suffices for the MVP (no diffusion) and per-scene optimization; one 24 GB GPU desirable for GSFix3D/diffusion-fill pilots; Quest 3 + Quest 3S for on-device FPS.
- **Software/infra:** Docker (multi-image, incl. an isolated CUDA-11.x surface-GS image), Unity + Meta XR/Spatial SDK, Open3D, xatlas, gsplat 1.5.3, nerfstudio 1.1.5.
- **Engineering skills:** 3DGS/gsplat internals; CUDA/Docker build (Blackwell); Unity XR + rendering; geometry processing (mesh→depth render, ICP, plane fit); eval/benchmarking; (optional) diffusion inpainting.
- **Team roles (indicative):** 1 research engineer (fill + depth supervision + blob), 1 graphics/Unity engineer (viewer, compression, grab, on-device perf), 0.5 capture/eval engineer (protocol, datasets, benchmarks), 0.25 PI/writing (benchmark + paper).
- **Timeline:** Phases 0–4 ≈ 3–4 months with the above; Phase 5 ongoing.

---

## 11. Research Gaps & Innovation Opportunities

1. **Metric-mesh/ToF-anchored amodal completion (primary, publishable).** No published method renders a *pre-existing metric mesh* as the fill's depth target — all estimate (Depth-Anything) or complete (InFusion diffusion) depth. Contribution: *supply the metric scaffold, hallucinate only RGB; quantify how a metric prior cures the multi-view-inconsistency that forces every prior method into iterative refinement.* Low novelty-risk, high motivation.
2. **Tabletop near-planar single-sweep occlusion benchmark.** Every benchmark (SPIn-NeRF, 360-USID, Remove360) measures *large* removals in 360/forward-facing scenes. A small (~10–30 cm) near-planar desk-patch benchmark — synthetic (Replica/Hypersim, free GT) + real physical-removal re-capture — fills a genuine gap and lets the project report credible masked-region metrics.
3. **Reproducible, Dockerized, *measured-on-device* Quest pipeline.** The ~300–400k room / ~5k object budgets are folklore. A nerfstudio→prune→`.spz`→ninjamode/Mobile-GS recipe with on-device ms is a real systems contribution.
4. **Femto ToF-IR-ghost cleanup before init.** No dataset models 850 nm IR ghosting on glossy screens; a Femto-specific multi-view outlier filter attacks the blob at the source — practical + novel for this hardware.
5. **Practical differentiators vs industry:** open + exportable + on-device-capable + ToF-metric + *paired physics mesh + grab interaction* + *working amodal fill* — exactly where Hyperscape (view-only, closed, cloud, occlusion-weak) and Scaniverse (no mesh/physics) fall short.

---

## 12. Final Refined Roadmap

### Build immediately (low-risk, high-leverage)
1. **GLOMAP** (COLMAP 4.0) as the pose default — drop-in, faster, more robust.
2. **`.spz`/SOG export** + SplatTransform in the pipeline (interop + Quest budget).
3. **Mesh-seeded amodal-fill MVP** (Phase 1): render mesh→depth under SAM3 footprint, 3DGIC reprojection, InFusion unproject, NN-copy texture. *Tests the core hypothesis in days.*
4. **Depth supervision done right:** EdgeAwareLogL1 on splatfacto-big's *native* gsplat densifier (λ-ablated).
5. **Standalone-Quest viewer benchmark** on the ninjamode fork with the project's own scenes — measure on-device.

### Test experimentally (pilot, decide on evidence)
- **GSFix3D `--dual_input`** fed the project mesh (off-the-shelf comparison); **AuraFusion360** on one hole (check its admitted floaters).
- **RaDe-GS / 2DGS-gsplat** in an isolated Docker (A/B mesh vs OpenMVS/TSDF; verify sm_120).
- **SpotLessSplats + ToF-IR pre-clean + NVIDIA 3DGUT** for the monitor blob.
- **Mobile-GS** sort-free renderer for room-scale FPS; **LightGaussian vs NanoGS** head-to-head.
- **VGGT/Pi3 → COLMAP-export** for textureless/reflective scans.

### Research later (promising / immature)
MILo (watertight low-poly mesh+splat in one run, when released); GPGS (point-cloud-completion-conditioned fill); Octree-GS/Hierarchical-3DGS/LODGE runtime LOD (verify mobile decode cost); FisherRF-guided ScanGuide re-capture UI; multi-pass loop-closure; IR-channel supervision; the tabletop occlusion **benchmark + paper**; CUT3R/NoPoSplat virtual-view priors for the amodal core.

### Avoid
SplatFill as a build target (no code, A100-only); DN-Splatter's densifier on gsplat 1.5+; porting VR-GS (no code, tethered, soft-body, sub-72 FPS); 3DGS-DR/Ref-Gaussian/Mirror-3DGS (model reflections; wrong goal; Ref-Gaussian needs 2DGS); closed/cloud SaaS as dependencies (Hyperscape/Luma/Polycam/Varjo — benchmark only); generative-from-scratch fills (Marble/TripoSR) for the *faithful* desk patch; FreeSplat/MVSplat as the final splat.

---

## 13. Annotated Bibliography

**Occlusion / 3DGS inpainting (headline)**
- **AuraFusion360** — Wu et al., NVIDIA/NYCU, CVPR 2025. [arXiv 2502.05176](https://arxiv.org/abs/2502.05176) · [code](https://github.com/kkennethwu/AuraFusion360_official) · [360-USID](https://huggingface.co/datasets/kkennethwu/360-USID). 360 object removal with depth-aware unseen-mask + Adaptive Guided Depth Diffusion; best open benchmark (17.66/0.388). *Use as reference + eval.*
- **InFusion** — Liu et al., Alibaba/HUST, 2024. [arXiv 2404.11613](https://arxiv.org/abs/2404.11613) · [code](https://github.com/ali-vilab/Infusion). Depth-completion → unproject Gaussians; *adapt with known ToF depth.*
- **3DGIC** — Huang et al., NVIDIA/NTU, CVPR 2025. [arXiv 2502.11801](https://arxiv.org/abs/2502.11801) · [code](https://github.com/peterjohnsonhuang/3dgic). Cross-view depth-guided mask-shrinking; *adapt the idea* (repo is "suboptimal version").
- **GSFix3D** — 2025. [arXiv 2508.14717](https://arxiv.org/abs/2508.14717) · [code](https://github.com/GSFix3D/GSFix3D) (2025-11-18). **Mesh + 3DGS fused as a diffusion prior** (`--dual_input`); *test off-the-shelf.*
- **GScream** — Wang et al., ECCV 2024. [arXiv 2404.13679](https://arxiv.org/abs/2404.13679) · [code](https://github.com/W-Ted/GScream). Depth-prior removal; online depth-align trivial with ToF.
- **Gaussian Grouping** — Ye et al., ECCV 2024. [code](https://github.com/lkeab/gaussian-grouping). Identity-encoding decomposition + render→LaMa→finetune (the naive path).
- **Inpaint360GS** — 2025. [arXiv 2511.06457](https://arxiv.org/abs/2511.06457). Virtual-camera exposure of the hole. *Watch.*
- **GPGS** — AAAI 2026. Point-cloud-completion-conditioned removal — fits the project's ToF cloud. *Watch (paper-only).*
- **SplatFill** — 2025. [arXiv 2509.07809](https://arxiv.org/abs/2509.07809). Selective consistency refinement; *avoid as build target* (no code, A100).
- **InstaInpaint** — NeurIPS 2025. [code](https://github.com/dhmbb2/InstaInpaint). Feed-forward 0.4 s fill. *Watch.*
- **NeRFiller** — Weber et al., CVPR 2024. [code](https://github.com/ethanweber/nerfiller). Grid-of-views diffusion consistency (NeRF). *Idea.*
- **SPIn-NeRF** — Mirzaei et al., CVPR 2023. [code](https://github.com/SamsungLabs/SPIn-NeRF). The reference removal benchmark + M-LPIPS/box-FID protocol. *Use protocol.*
- **Remove360** — 2025. [arXiv 2508.11431](https://arxiv.org/abs/2508.11431). Semantic-residual-after-removal benchmark. *Use as residual check.*

**Surface-aligned GS & mesh**
- **2DGS** — Huang et al., SIGGRAPH 2024. [code](https://github.com/hbb1/2d-gaussian-splatting). Native in gsplat 1.5, not in nerfstudio. DTU 0.80 / T&T 0.30.
- **RaDe-GS** — Zhang et al., TOG 2025. [code](https://github.com/BaowenZ/RaDe-GS). DTU 0.69, ~18 min. *Pilot.*
- **PGSR** — Chen et al., TVCG 2024. [code](https://github.com/zju3dv/PGSR). DTU ~0.49 / T&T 0.50; planar geometry = amodal scaffold.
- **GOF** — Yu et al., SIGGRAPH Asia 2024. [code](https://github.com/autonomousvision/gaussian-opacity-fields). Unbounded; CUDA 11.3.
- **MILo** — Guédon et al., TOG 2026. [project](https://anttwo.github.io/milo/). Differentiable watertight low-poly mesh + splat. *Watch.*
- **SuGaR / Frosting** — Guédon & Lepetit, CVPR/ECCV 2024. [code](https://github.com/Anttwo/SuGaR). Mesh+splat binding + Blender tooling.
- **DN-Splatter / AGS-Mesh** — Turkulainen et al., WACV 2025. [code](https://github.com/maturk/dn-splatter). ToF-native losses (reuse EdgeAwareLogL1); densifier broke V25.

**VR rendering / compression / LOD**
- **UnityGaussianSplatting** — Pranckevičius. [code](https://github.com/aras-p/UnityGaussianSplatting) + **ninjamode VR fork** (~400k @72 FPS standalone). *Use.*
- **Optimizing 3DGS for Mobile/Standalone VR** — SIGGRAPH Labs 2025. [DOI](https://dl.acm.org/doi/10.1145/3721251.3734056). Target <400k standalone.
- **SqueezeMe** — Meta RL, 2024. [arXiv 2412.15171](https://arxiv.org/abs/2412.15171). *Verified standalone* ~180k @72 FPS.
- **Mobile-GS** — 2026. [code](https://github.com/xiaobiaodu/mobile-gs). Sort-free, 74 FPS @0.47M on 8 Gen 3. *Test.*
- **Niantic spz / SPZ4** — [code](https://github.com/nianticlabs/spz). MIT splat format. *Use.*
- **NanoGS** — 2026. [arXiv 2603.16103](https://arxiv.org/abs/2603.16103). Training-free count reduction (best PSNR/Gaussian).
- **LightGaussian** — Fan et al., NeurIPS 2024. [code](https://github.com/VITA-Group/LightGaussian). Image-supervised prune + SH distill (decoder-free).
- **SOG/SOGS** — Fraunhofer HHI, ECCV 2024 / PlayCanvas. [code](https://github.com/playcanvas/sogs). ~20–40× compression.
- **Octree-GS** — Ren et al., TPAMI 2025. [code](https://github.com/city-super/Octree-GS). Runtime LOD. *Research-later.*
- **VRSplat** — 2025. [arXiv 2505.10144](https://arxiv.org/abs/2505.10144) · [code](https://github.com/Cekavis/VRSplat). VR fixes (StopThePop, optimal projection) — *tethered numbers.*
- **3DGS.zip** — CGF 2025. [leaderboard](https://w-m.github.io/3dgs-compression-survey/). Compression Pareto.

**Object-centric & physics**
- **VR-GS** — Jiang et al., SIGGRAPH 2024. [arXiv 2401.16663](https://arxiv.org/abs/2401.16663) (*repo = website only*). Tethered, soft-body. *Avoid porting.*
- **GS-Verse** — 2025. [arXiv 2510.11878](https://arxiv.org/abs/2510.11878) · [code](https://github.com/Anastasiya999/GS-Verse). Mesh-face splat anchoring (open). *Binding blueprint.*
- **PhysGaussian** — Xie et al., CVPR 2024. [code](https://github.com/XPandora/PhysGaussian). MPM on Gaussians (offline). *Cite for dual-rep justification.*
- **SAGA / OpenGaussian / InstanceGaussian** — 2024–25. Segmentation alternatives. *Watch.*

**ToF / depth / reflections**
- **SpotLessSplats** — Sabour et al., TOG 2025. [arXiv 2406.20055](https://arxiv.org/abs/2406.20055). Robust transient masking + utilization pruning. *Adapt for blob.*
- **ToF-Splatting** — ICCV 2025. [arXiv 2504.16545](https://arxiv.org/abs/2504.16545). Multi-view ToF outlier filter. *Adapt for IR ghosts.*
- **FisherRF** — Jiang et al., ECCV 2024. [project](https://jiangwenpl.github.io/FisherRF/). Active next-best-view. *Capture-side lever.*
- **3DGS-DR / Ref-Gaussian / Mirror-3DGS** — SIGGRAPH 2024 / ICLR 2025 / ECCV 2024. *Avoid (model, not remove).*

**Datasets / eval**
- **MuSHRoom** — Ren et al., WACV 2024 + AGS-Mesh 3DV 2025. [project](https://xuqianren.github.io/publications/MuSHRoom/). Consumer RGB-D + Faro GT, joint mesh+NVS. *Primary external.*
- **ScanNet++ v2** — Yeshwanth et al., ICCV 2023 (+v2 2024). [site](https://kaldir.vc.in.tum.de/scannetpp/). *Seg/NVS conventions.*
- **Replica / Hypersim** — [Replica](https://github.com/facebookresearch/Replica-Dataset) / Hypersim ICCV 2021. *Synthetic occlusion GT.*
- **FovVideoVDP / FLIP** — SIGGRAPH 2021 / HPG 2020. [FovVideoVDP](https://github.com/gfxdisp/FovVideoVDP). *VR floater metrics.*
- **DTU / Tanks&Temples** — [T&T](https://www.tanksandtemples.org/). *Mesh metrics.*

**Feed-forward / pose**
- **GLOMAP** — Pan et al., ECCV 2024. [code](https://github.com/colmap/glomap). Global SfM in COLMAP 4.0. *Use.*
- **VGGT** — Wang et al., CVPR 2025 (Best Paper). [code](https://github.com/facebookresearch/vggt). Feed-forward poses+pointmaps.
- **Pi3** — 2025–26, ICLR 2026. [code](https://github.com/yyfz/Pi3). More robust VGGT successor.
- **VGGT-X** — 2025. [arXiv 2509.25191](https://arxiv.org/abs/2509.25191). Honest ~1 PSNR gap vs COLMAP-init. *Keep splatfacto.*
- **MASt3R-SfM / CUT3R / AnySplat / NoPoSplat / DepthSplat** — 2024–25. Pose/prior options; domain gap on tabletop.
- **E3D-Bench + Feed-Forward 3D survey** — 2025. [arXiv 2506.01933](https://arxiv.org/pdf/2506.01933). Objective backbone map.

**Generative priors (Alt C / fallback)**
- **TRELLIS** — Microsoft, CVPR 2025. [code](https://github.com/microsoft/TRELLIS). MIT; SLAT → mesh + Gaussians. *Per-object upgrade fallback.*
- **Amodal3R** — 2025. [arXiv 2503.13439](https://arxiv.org/abs/2503.13439) · [weights](https://huggingface.co/Sm0kyWu/Amodal3R). Occlusion-aware TRELLIS finetune.
- **Hi3DGen / Hunyuan3D 2.1 / SF3D** — 2025. Geometry/PBR object generators (license-check).

**Industry**
- **Meta Hyperscape** — [UploadVR](https://www.uploadvr.com/meta-horizon-hyperscape-photorealistic-scene-capture-quest-3/). Closed, cloud-recon, no export; shares the occlusion+reflection weakness. *Benchmark only.*
- **Scaniverse / Spark / SuperSplat / OpenQuestCapture / NVIDIA NuRec+3DGUT** — open ecosystem to interoperate with (`.spz`, SOG, Quest capture, optics-aware rasterization).

---

*End of roadmap. Companion raw research (9 category dumps, 6 adversarial verifications, synthesis memo) was generated by the deep-research sweep on 2026-06-18 and is the evidentiary basis for every claim above; where a method is paper-only or a repo is degraded, this document says so explicitly.*
