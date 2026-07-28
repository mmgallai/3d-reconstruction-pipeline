# Room 3-object extraction — July 2026

## Objective
Extract 3 objects each from the Mip-NeRF 360 room + garden scenes as VR-ready assets:
- **Full scene splat** (baseline)
- **Scene without objects** (with hole fill)
- **Per-object splats** (isolated for placement/interaction)

## Room progression (v1 → v6)

| Version | Method | Objects | Best result | Key change |
|---|---|---|---|---|
| v1 | Legacy pipeline (`_pipeline_full.py`) | 2/3 (coffee table too fragmented) | BROKEN | coord transform mismatch; extraction returned wrong Gaussians |
| v2 | Direct projection extraction (new) | 3/3 | armchair ✓, coffee table ✓, subwoofer over-includes speakers | bypass mesh pipeline; use working renderer's camera setup |
| v3 | v2 + teddy bear union + hole-fill + spike filter | 3/3 | chair+teddy ✓, coffee table amputated, scene_without clean | mask union for chair+teddy; K-blend fill |
| v4a | v3 with anisotropy=60 (spike filter tighter) | 3/3 | same as v3 | control candidate |
| v4b | ottoman as 3rd object | 2/3 (ottoman failed) | SAM3 couldn't segment ottoman | prompt tuning |
| v4c | audio_speakers as 3rd object | 3/3 | includes TV cabinet too | broader group; less useful |
| v5 | v3 + `--keep-thresh 0.20` + `--fill-outer 0.50` + K-blend | 3/3 | coffee table INTACT (huge win) but scene_without has dark bushy artifacts | lower threshold catches thin table legs; K-blend darkens fills |
| **v6** | v5 tweaks + `--fill-mode nearest` + `--fill-outer 0.35` | 3/3 | **CANONICAL** — chair+teddy ✓, full coffee table ✓, subwoofer OK, scene_without much cleaner | nearest-donor fill avoids the median-darkening artifact |
| v7 | v6 + `--keep-largest-cc 0.10` (spatial coherence) | 3/3 | marginal improvement over v6; subwoofer's 3-speaker grouping is a SAM3 semantic issue not solvable with spatial filter | K-NN union-find largest connected component |

### Room known issues (accepted for v6):
- **"Black subwoofer"** SAM3 mask semantically groups all 3 black audio devices (subwoofer + floor speaker + center speaker) — extraction produces a coherent "audio equipment" cluster instead of one isolated speaker. Fixing requires SAM3 point-prompting or per-instance segmentation (out of scope).
- Some spike artifacts on cropped object edges (Gaussians whose supporting neighbors were removed by the mask). Reducing anisotropy threshold further hurts fidelity — accepted trade-off.
- Scene_without_objects has visible slipper/shoe residue on the rug (small items not in any of the 3 target prompts, kept as scene).

## Final room outputs (v6)
- `room_v6/splat/scene_full.ply` — original full splat (161 MB, 680k Gaussians)
- `room_v6/splat/scene_without_objects.ply` — raw removal (608k Gaussians)
- `room_v6/splat/scene_without_objects_filled.ply` — with K-NN hole fill (654k Gaussians)
- `room_v6/splat/objects/blue_armchair.ply` — chair + teddy (15.5k Gaussians)
- `room_v6/splat/objects/wooden_coffee_table.ply` — table + items on top (22.4k)
- `room_v6/splat/objects/black_subwoofer.ply` — subwoofer + adjacent speakers (34.0k)
- `room_v6/qa/*.png` — 5-view render grids of all 4 splats above

## Key technical findings
1. **splatfacto-big with `nerfstudio-data` stores Gaussian positions in COLMAP-world, NOT dp-normalized space.** The existing extraction pipeline (`_pipeline_full.py`, `_seed_desk_patch.py`) assumes dp-normalized, which produces 0 donor Gaussians on this data. Documented in memory `splat_position_space_finding.md`.
2. **Direct projection extraction** (`_extract_by_projection.py`) bypasses all mesh-based coord conversion. It uses the working renderer's camera pipeline (dp_transform + dp_scale on cameras) to project each Gaussian and score against cached SAM3 masks. Simple and reliable.
3. **Union masks** for cases where SAM3 can't reliably merge related items (chair + teddy) work well — generate separate masks and union at extraction time via `+` syntax in `--prompts`.
4. **Hole-fill with nearest-donor** > K-blend median for these scenes — median darkens because donors span a wide range of colors.

## Renderer
`_render_splat_views.py` runs inside `nerfstudio-blackwell` docker (gsplat compiled).
Wrapper: `_render_qa_docker.sh <ply> <indices> <out.png> [<transforms>] [<dataparser>]`. Sets `MSYS_NO_PATHCONV=1` to prevent git-bash path mangling. Custom transforms/dataparser args useful when the active `colmap/` and `nerfstudio/` belong to a different scene.

---

## Garden progression (v1 → v4)

| Version | Objects | Best result | Key change |
|---|---|---|---|
| v1 | ceramic_pot, green_ball, wooden_garden_table | table ✓, pot ⚠️ (some sphere over-include), **ball ✗ (0 gaussians — mask too sparse)** | initial direct-projection run |
| v2 | ceramic_pot, wooden_table (2 objects, wider fill) | tested wider `--fill-shell-outer-m 1.00` — helps somewhat but centerpiece occlusion limits fill quality | dropped ball, tested wider fill |
| v3 | ceramic_pot, wooden_table, potted_plant + `--keep-largest-cc 0.15` | over-aggressive CC filter dropped most of pot + plant — REJECTED | too-tight spatial coherence for garden's dispersed objects |
| **v4** | ceramic_pot, wooden_table, **potted_plant** (no CC filter) | **CANONICAL** — 3 non-zero objects, table beautiful, pot + plant with some multi-instance grouping | swap failed ball for potted_plant, drop CC filter |

## Final garden outputs (v4)
- `garden_v4/splat/scene_full.ply` — original 1.13M-gaussian splat
- `garden_v4/splat/scene_without_objects.ply` — 748k after removal
- `garden_v4/splat/scene_without_objects_filled.ply` — 1.10M with K-NN nearest fill
- `garden_v4/splat/objects/wooden_table.ply` — 331k gaussians (large, includes some surrounding grass)
- `garden_v4/splat/objects/ceramic_pot.ply` — 6.4k gaussians (some silver spheres over-included)
- `garden_v4/splat/objects/potted_plant.ply` — 41.6k gaussians (includes multiple pots since garden has several)

### Garden known issues (accepted for v4)
- **Centerpiece occlusion**: the central wooden table blocks the ground/paving beneath it in every view — no Gaussians exist for that region. Hole fill has to interpolate from surrounding grass which produces obvious dark/blotchy artifacts in the removed region. This is a fundamental limitation for scenes with a large centerpiece object.
- **Multi-instance semantic grouping**: garden has many similar objects (multiple pots, silver spheres, potted plants). SAM3's text prompts don't distinguish instances, so "ceramic pot" catches SPHERE-shaped objects too, "potted plant" catches all plants. Would need per-instance point-prompting to isolate one.
- **Green soccer ball fails** — too small + occluded in most views (0.2% mean coverage). Not extractable at prompt-only level.

### Scene contents (garden)
- Central round wooden garden table on hexagonal paving
- Ceramic pot with dried palm frond on the table
- Green soccer ball under the table (couldn't extract)
- Grass lawn, ivy-covered wall, hedges
- Black wooden door on left, brick wall
- Several potted plants around perimeter (silver reflective garden spheres, dark pots with cypress)
- Building visible in background
