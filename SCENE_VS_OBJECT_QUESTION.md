# Scene-vs-object conflict in a VR digital-twin pipeline — looking for expert input

We are building an open-source pipeline that turns consumer scans into
**Meta Quest 3 VR digital twins**, and we've hit a structural problem
we'd like a second opinion on. This document lays out the goal, what we
have, and the specific conflict we're stuck on.

---

## 1. Project objective

Take a desk-scale or room-scale physical scene, capture it with a
consumer device (iPhone or an Orbbec Femto Mega ToF camera), and
produce a **VR-ready digital twin** the user can step into on Meta
Quest 3 and **physically interact with**.

By "physically interact" we mean: the user reaches out, grabs a water
bottle on the desk with the Quest 3 controllers, lifts it, and puts it
back. The objects on the desk are real, persistent things in the VR
scene with mass, colliders, and grab affordances — not flat photos.

The deliverable for each scene is therefore **two paired representations**:

| Representation | Why we need it |
|---|---|
| **Gaussian Splat** (visual) | Sharp, view-dependent visual fidelity at 72+ FPS on the Quest 3, much better than a baked-texture mesh at the same poly budget. This is what the user *sees*. |
| **Textured mesh + colliders** (physics) | Unity's rigid-body / grab-interaction system needs actual geometry to ray-cast against. This is what the user *feels and grabs*. |

These are **paired deliverables, not alternatives**. The splat gives us
the look; the mesh gives us the physics.

---

## 2. Our pipeline (current state, V32, 2026-06-18)

Capture device: **Orbbec Femto Mega** — RGB + native ToF depth, ~140
frames per scene. (We also support iPhone capture for the visual splat
path; the geometry pipeline is Femto-specific because we need real
metric depth for the physics mesh.)

```
Femto Mega capture (RGB + ToF, ~140 frames)
   │
   v
COLMAP SfM (vendor intrinsic prior, lock during BA)
   │
   v
ToF back-projection → 500k-point coloured init cloud
   │
   v
┌───────────────────────────┬─────────────────────────────────────┐
│ splatfacto-big training   │ OpenMVS Densify→Reconstruct→Refine  │
│ (init from ToF cloud)     │ →TextureMesh (8K UV atlas)          │
│       │                   │       │                             │
│ scene Gaussian Splat      │ scene textured mesh                 │
│ (V32, ~227k Gaussians     │ (V32, ~1.3M faces, 8K atlas)        │
│ after pruning)            │                                     │
└───────────────────────────┴─────────────────────────────────────┘
```

So far so good: V32 produces both a clean scene-level splat (the visual
deliverable) and a clean scene-level mesh (the physics deliverable),
sharing the same coordinate frame and metric scale.

---

## 3. Per-object extraction (what we do for interaction)

For each object the user should be able to grab — e.g. "white water
bottle", "blue box", "red lobster figurine" — we run an automated
text-prompt-driven extraction:

```
SAM3 text-prompt masking      (140 views × N prompts, cached)
   │
   v
Rasterised SAM3 voting        (Open3D BVH raycast → per-pixel FaceID
                              + 3-way evidence vote per face)
   │
   v
Seeds (faces with score ≥ 0.7) + spatial voxel CC filter
   │
   v
2D X-Z footprint + 0.5 cm dilation + 0.5 cm Y margin
   │
   v
┌──────────────────────────────┬──────────────────────────────────┐
│ Spatial crop of scene mesh   │ Spatial crop of scene splat      │
│ (v9a_fp_v2)                  │ → SH-viewport colour filter      │
│                              │ → Clean-GS prune (v6_cleaned)    │
│                              │                                  │
│ per-object textured mesh     │ per-object Gaussian splat        │
│ (OBJ + MTL + PNG +           │ (nerfstudio PLY schema)          │
│ collider JSON: AABB / OBB /  │                                  │
│ convex hull / recommended    │                                  │
│ Unity collider type)         │                                  │
└──────────────────────────────┴──────────────────────────────────┘
```

We've spent a lot of effort getting both halves to look right:

- **Per-object splat**: ~5k Gaussians, no halos, opacity confidence
  matches the scene splat, bounding box matches the object's real 3D
  extent within 0.5 cm. Renders cleanly when placed in isolation in
  Unity.
- **Per-object mesh**: tight footprint following the object outline
  (not just AABB — the lobster's claws are followed by the spatial-CC
  + 2D-footprint approach), with a collider JSON that recommends Box /
  Capsule / Convex based on aspect ratios.

So **at the per-object level we are happy with what comes out**.

---

## 4. The problem — the scene has holes where the objects were

Now we try to assemble the VR scene in Unity. The naïve assembly is:

```
Unity scene:
  ┌─ V32 scene splat               ← background (everything that was on the table)
  │   minus the bottle/box/lobster
  ├─ V32 scene mesh                ← physics ground / desk surface
  │   minus the bottle/box/lobster
  ├─ per-object bottle splat       ← grabbable object 1 (visual)
  ├─ per-object bottle mesh        ← grabbable object 1 (physics)
  ├─ per-object box splat          ← grabbable object 2
  ├─ per-object box mesh           ← grabbable object 2
  └─ per-object lobster splat / mesh
```

Either we composite the objects ON TOP of the unmodified scene splat,
in which case picking one up reveals an **identical static copy
underneath** that was baked into the scene splat (the bottle appears
to "stay on the desk" when grabbed) — OR we subtract the objects from
the scene splat / mesh first, in which case there is a **visible hole**
in the desk surface where the object used to be. Neither composition
is acceptable for the VR experience we want.

Concretely: the desk under each object has **no Gaussians for the desk
surface that was occluded by the object during capture** — the bottle
sat there during all 140 frames, so the trainer never saw the desk
patch underneath. After we crop the bottle out of the scene splat, that
patch is just empty space: when the user lifts the bottle the desk
surface visibly does not exist where the bottle base used to be. Same
for the mesh — there are no faces for the occluded desk patch, so the
collider has a hole too.

This is a **structural problem with our current approach**, not a tuning
issue. We can prune halos and improve per-object fidelity all day, but
neither pruning nor better masks can synthesise the desk patch that
was never observed by the camera.

---

## 5. What we've considered (looking for expert input)

We're listing what we've thought about so far, neutrally — we're not
attached to any of these, we want to hear what we're missing.

### (A) 2D inpainting → re-train the scene splat without the objects

Render the trained scene splat from each of the 140 training views,
mask out the objects, run a 2D inpainter (Stable Diffusion / SDXL
inpaint / LaMa) to fill the desk patch behind each object in image
space, then **re-train splatfacto** using the inpainted images as the
photometric supervision. Result is a scene splat where the objects
were "never there" — no hole when we composite the per-object splats
on top.

**Concern**: 2D inpainting is multi-view inconsistent — each view's
inpainted desk patch will disagree with the others, so the re-trained
splat will likely produce a low-frequency smear / texture-soup where
the holes were. Multi-view-consistent 3D-aware inpainting (e.g.
SPIn-NeRF, NeRFiller, Gaussian-Splatting-Editing variants) is more
promising on paper but the open-source implementations we've seen are
either slow, depth-only, or weren't designed for high-resolution
texture work on a tabletop scale.

### (B) 3D-aware Gaussian inpainting

Methods that operate directly on the trained 3DGS (GaussianCutter,
GS-Inpaint, GScream, Infusion) and synthesise new Gaussians inside the
removed region using cross-view priors. These are more recent and we
have not evaluated any of them yet on our data.

**Concern**: most of these target NeRF-like room scenes with
multi-metre baselines. Our capture is a tabletop with sub-metre
baselines and reflective objects (a monitor in the back of frame
already gives V32 trouble with view-dependent blob artefacts). We're
not sure how robust the published methods are at our scale.

### (C) Hide-and-reveal trick (no inpainting)

Leave the V32 scene splat unmodified. At runtime, when the user grabs
an object, **hide the matching region of the scene splat** (e.g. by
zeroing opacity on the Gaussians inside the object's bounding box) and
**reveal the per-object splat** in the user's hand. When they put the
object back down, reverse the swap. The hole only appears DURING the
grab, while the user is looking at the object in their hand, not at
the desk — possibly acceptable UX.

**Concern**: the hole is still there, just transient. If the user puts
the object back in a different pose / location, the scene is now in a
permanently inconsistent state. Putting a transparent / blurred
"shadow plate" in the hole helps but doesn't fully sell the illusion
on close inspection.

### (D) Capture protocol fix — multi-pass

Re-capture the scene **twice**: once with all the objects on the desk
(gives us the object splats), and once **with the desk empty**
(gives us the clean background splat). Composite at runtime.

**Concern**: doubles the capture burden on the user, doesn't
generalise to objects the user can't move (sofa, lamp wired to the
wall), and requires perfect coordinate alignment between the two
captures.

### (E) Hybrid: splat as visual, mesh as physics, accept the splat hole

The mesh side already produces a closed surface under the object
(OpenMVS Poisson-style reconstruction fills the occluded desk patch
with some best-guess geometry, even if it has no real texture for it).
So **physics works**: the desk is solid under the object. Only the
visual splat has the hole. If we composite the per-object splat back
on top of its original location in the scene splat **as the resting
state**, the only visible discrepancy is when the object is held away
from its rest pose — and we could lazily render a soft splat patch
behind it.

**Concern**: relies on the mesh's interpolated geometry under the
object being accurate enough that the splat hole doesn't show. Worst
case: the desk patch in the mesh has the wrong colour / albedo so when
the object is lifted, the under-side reveal looks wrong.

---

## 6. Constraints / things to know about our setup

- **Capture**: ~140 Femto Mega frames per scene, RGB + ToF, ~1 m
  tabletop scale, single sweep around the desk. We have COLMAP poses
  in OpenGL convention and metric ToF depth per frame.
- **Scene splat**: V32 splatfacto-big, ~227k Gaussians after pruning,
  ~54 MB PLY. Reflective monitor in the back of frame produces a
  known view-dependent "blob" artefact we have not solved (SAM3 monitor
  masking helped but only -13 % Gaussians, blob persists).
- **Scene mesh**: V32 OpenMVS, ~1.3M faces, 8K UV atlas.
- **Target device**: Meta Quest 3 / Quest Pro. We have a ~5k-Gaussian /
  ~30 MB / Quest3-streamable per-object budget. NanoGS (post-hoc
  Gaussian merging) gets us a 50 % size cut on the scene splat with
  acceptable quality loss.
- **Per-object pipeline** (the part we're happy with): text-prompt-
  driven, fully automated, SAM3 → rasterised 3-way evidence vote →
  spatial voxel CC seeds → 2D X-Z footprint → Clean-GS prune
  (`v9a_fp_v2` mesh + `v6_cleaned` splat — see `SCENE_SEGMENTER_NOTES.md`
  in the repo for the full design).
- **Test data**: V32 desk scene with `white_water_bottle`, `blue_box`,
  `red_lobster_figurine`. Outputs at
  `output/segmented_v32_data3_v9a_fp_v2/<prompt>/` (mesh) and
  `output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned/` (splat).
- **What we are NOT trying to solve**: dynamic relighting, articulated
  objects, deformable bodies. Rigid-body grab-and-place only.

---

## 7. The specific questions we'd like input on

1. Of approaches (A)-(E) above, which has the best track record for
   a tabletop / desk-scale 3DGS capture? Are there other approaches we
   should be considering?
2. For 3D-aware Gaussian inpainting (B), are there published methods
   that handle the case where the occluded region is **small and on a
   nearly-planar surface** (the desk under the object)? Most published
   results we've seen target larger free-space removals.
3. Is there a way to use the V32 scene **mesh** (which DOES interpolate
   under the object via Poisson reconstruction) to inform the scene
   splat's inpainting — e.g. as a depth prior or a colour prior for a
   3D-aware inpainter?
4. The hide-and-reveal trick (C) is the cheapest option and only fails
   for "object placed off-original-spot" cases. Is there VR-UX prior
   art on what's acceptable to a user here?
5. For (D), are there capture-side tricks (e.g. asking the user to
   shift the bottle slightly between two sub-passes) that would let us
   recover the occluded patch without forcing them to do a fully
   empty-scene pass?

We're an open-source project; if there's a method we should try, we
have the time + GPU to implement and benchmark it. Mainly we want to
make sure we're not about to implement something a published paper has
already shown doesn't work at our scale.

Thanks for any input.
