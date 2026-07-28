# Changes since last push (initial commit `e1971e7`)

**Push date:** 2026-07-28
**Baseline:** `e1971e7 Initial commit: 3D reconstruction pipeline (RealityScan -> Gaussian Splat) with DA3 init`
**This push includes** 10 local commits + this session's untracked files.

---

## Session summary (2026-07-27 → 2026-07-28)

### Goal
Extract 3 objects each from two Mip-NeRF 360 scenes (room, garden) as VR-ready splats: full scene, scene-without-objects (with hole fill), plus per-object splats. Iterate until visually acceptable.

### Outcome
- **Room v6** (`output/july/room_v6/`) = canonical: `blue_armchair.ply` (chair + teddy bear), `wooden_coffee_table.ply`, `black_subwoofer.ply` (subwoofer + adjacent speakers), plus `scene_without_objects_filled.ply` with K-NN hole fill.
- **Garden v4** (`output/july/garden_v4/`) = canonical: `wooden_table.ply`, `ceramic_pot.ply`, `potted_plant.ply` + filled scene-without.
- All v1..vN iterations preserved under `output/july/` for the trail.
- Full detail in [`JULY_RESULTS_README.md`](JULY_RESULTS_README.md).

### Key technical finding
Splatfacto-big with `nerfstudio-data` stores Gaussian positions in **COLMAP-world space**, NOT dp-normalized space. The legacy extraction pipeline (`_pipeline_full.py`, `_seed_desk_patch.py`) assumed dp-normalized → 0 donor Gaussians on this data. Workaround: use a copy of `dataparser_transforms.json` with `scale: 1.0` for extraction (kept as `nerfstudio/dataparser_transforms_extraction.json`, gitignored via `nerfstudio/`). Renderer still uses the true scale=0.176 dataparser.

---

## New scripts (in this push)

### Extraction — direct projection (bypasses legacy mesh-based pipeline)
- **`_extract_by_projection.py`** — NEW. Projects each Gaussian center through every view using the same camera math as the renderer, votes against cached SAM3 masks, classifies each Gaussian to one object (or none). Includes:
  - `+` prompt syntax to UNION multiple SAM3 masks per object (e.g. `blue_armchair+teddy_bear`).
  - `--drop-anisotropy` spike filter (removes needle-shaped Gaussians when their neighbors are cropped away).
  - `--keep-largest-cc` K-NN connected-component filter (drops isolated floaters).
  - `--fill-holes` K-NN clone-in-place hole fill (nearest-donor or K-blend median).
  - `--fill-shell-inner-m/--fill-shell-outer-m` shell bounds for donor validity.

### Rendering — native gsplat via docker
- **`_render_splat_views.py`** — NEW. Loads INRIA PLY (SH degree 3), applies dp_transform + dp_scale to cameras, rasterizes N views to a grid PNG.
- **`_render_qa_docker.sh`** — NEW. Wraps `_render_splat_views.py` in the `nerfstudio-blackwell` docker image (gsplat is pre-compiled there). Accepts optional custom transforms/dataparser paths for cross-scene rendering. Sets `MSYS_NO_PATHCONV=1` to prevent git-bash mangling.

### SAM3 wrappers (per scene / per prompt set)
- `_run_sam3_room.py` — 3 room prompts (blue armchair, wooden coffee table, black subwoofer).
- `_run_sam3_room_teddy.py` — supplemental teddy_bear mask for union with chair.
- `_run_sam3_room_v3.py` — experiment with compound prompts (failed — compound prompts hurt SAM3).
- `_run_sam3_room_ottoman.py` — experiment with alternate 3rd object (ottoman + audio_speakers).
- `_run_sam3_garden.py` — 4 garden prompts (ceramic_pot, dried_leaves, green_ball, wooden_table).
- `_run_sam3_garden_extras.py` — alternate garden 3rd-object prompts (silver_garden_sphere, black_door, potted_plant).

### Reconstruction + extraction chains
- `_run_test_scenes_garden_and_room.sh` — reconstructs garden + room back-to-back from Mip-NeRF 360 source dirs, with backup/restore of data4 workspace.
- `_run_garden_reconstruct.sh` — reconstruct garden only, with proper `.JPG → .jpg` rename step (empirical fix for case-sensitive glob issue).
- `_run_garden_extract.sh` — direct-projection garden extraction template.
- `_run_room_v1_chain.sh` — first-attempt legacy-pipeline room extraction (documented failure mode; superseded by direct-projection).

### Older work from prior sessions (also being pushed for the first time)
- `_iterative_refine.py`, `_sam3_residue.py`, `_surgical_repair.py` — iterative refinement loop with SAM3-residue detection + surgical repair (from 2026-07-19/20).
- `_run_data4_chain.sh`, `_run_data4_chain_resume4.sh`, `_run_data4_chain_resume5.sh` — data4 scene end-to-end chain wrappers (from 2026-07-13).

## Modified

- **`_pipeline_full.py`** — small fix in `_build_footprint_from_predicate`: `effective_buffer_m = max(buffer_m, crop_dist_m + 0.04)` so the search rectangle scales with `crop_dist_m`. Without this, large `--crop-dist-m` values silently produced patches narrower than the crop hole → visible gaps beside removed objects.

## New docs

- `CHANGES_SINCE_LAST_PUSH.md` — this file.
- `JULY_RESULTS_README.md` — full room + garden extraction progression (v1..vN, techniques, known issues, technical findings). Mirror of `output/july/README.md`.
- `CAPTURE_HARDWARE_NOTES.md` — pre-existing from 2026-07-12, being pushed now.

---

## Deliberately NOT pushed

- **All `output/` contents** (excluded by `.gitignore`) — this is where the actual PLYs live (many GB per scene). Referenced by paths in the docs.
- Diagnostic / one-off investigation scripts kept locally: `_check_desk_proj*.py`, `_diag_remnants*.py`, `_investigate_desk_plane*.py`, `_reviewer_*.py`.
- Backup workspaces (`colmap.bak*/`, `nerfstudio.room_*_saved_*/`, `nerfstudio_data.*/`) — added to `.gitignore` in this push.
- Stub HEIC scan dirs used to bypass the pipeline's HEIC precondition (`_stub_heic_scan_*/`) — added to `.gitignore`.

---

## Next planned work (as of this push)

Full-quality reconstruction of the room scene at native resolution (`--downscale-factor 1`) for a quality comparison against the current 2x-downscaled result. See `FULL_QUALITY_RUN_NOTES.md` (added separately when the run kicks off).
