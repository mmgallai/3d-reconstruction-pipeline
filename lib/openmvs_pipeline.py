"""
OpenMVS textured-mesh reconstruction pipeline (parallel to the 3DGS path).

Takes the existing COLMAP dense workspace (cameras + undistorted images +
fused.ply dense point cloud) and produces a textured triangle mesh comparable
to commercial photogrammetry tools (RealityScan / RealityCapture, Metashape).

CANONICAL RECIPE (a.k.a. "V17 quality") — the configuration that produces
the best visual quality. To reproduce, run from project root:

    python reconstruct_realityscan.py --skip-mvs --skip-training --no-splat --refine-mesh

This uses all defaults below: DensifyPointCloud level 1, OpenMVS Delaunay
graph-cut surface reconstruction, pymeshlab v2 hole closing, RefineMesh
photometric refinement, TextureMesh with full-res images, scale calibration,
LOD decimation, and per-vertex AO.

Stages:
  1. InterfaceCOLMAP    — convert COLMAP workspace to OpenMVS .mvs
  2. DensifyPointCloud  — OpenMVS dense reconstruction (level-1: 2× downscale)
  3. ReconstructMesh    — OpenMVS Delaunay graph-cut surface
                           (`use_poisson=True` swaps in Open3D Poisson — EXPERIMENTAL,
                            produces noisier surfaces; not recommended after V20 testing)
  3b. PostProcess       — pymeshlab close_holes + Open3D orient_triangles
  4. RefineMesh         — (optional but recommended) photometric refinement
  5. TextureMesh        — multi-view texture mapping → final textured PLY
  5b. Scale calibration — apply DA3 scale factor → vertices in meters

Outputs to: <project_root>/openmvs/scene_textured.ply  (and intermediates)

All OpenMVS steps run inside the yeicor/openmvs-ubuntu-cuda:v2.3.0 image.
Post-process / Poisson / scale steps run in the host Python (da3 conda env).
"""

import logging
import subprocess
import sys
from pathlib import Path

import config as _cfg
from lib.docker_runner import run

logger = logging.getLogger(__name__)

_OPENMVS_DIR_REL = "openmvs"

# OpenMVS binaries live in /usr/local/bin/OpenMVS/ in yeicor/openmvs-ubuntu-cuda
# but the container's $PATH is empty, so we prepend the dir before every command.
_OPENMVS_PATH_PREFIX = "export PATH=/usr/local/bin/OpenMVS:$PATH && "


def _vol(project_root: Path) -> str:
    return f'"{project_root}:/workspace"'


def _C(name):
    return getattr(_cfg, name)


def ensure_workspace(project_root: Path) -> Path:
    out = project_root / _OPENMVS_DIR_REL
    out.mkdir(parents=True, exist_ok=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — InterfaceCOLMAP
# ─────────────────────────────────────────────────────────────────────────────
def interface_colmap(project_root: Path,
                     colmap_dense_rel: str = "colmap/dense") -> bool:
    """
    Convert COLMAP dense workspace → OpenMVS .mvs.
    Output: openmvs/scene.mvs + per-image depth0XXX.dmap files.
    """
    logger.info("=== OpenMVS [1/5]  InterfaceCOLMAP — converting COLMAP → .mvs ===")
    out_rel = _OPENMVS_DIR_REL
    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} '
        f'{_C("DOCKER_OPENMVS")} bash -c "'
        f'{_OPENMVS_PATH_PREFIX}'
        f'cd /workspace && '
        f'mkdir -p {out_rel} && '
        f'InterfaceCOLMAP '
        f'  -i /workspace/{colmap_dense_rel} '
        f'  -o /workspace/{out_rel}/scene.mvs '
        f'  --image-folder /workspace/{colmap_dense_rel}/images '
        f'  --working-folder /workspace/{out_rel}'
        f'"'
    )
    return run(cmd, allow_fail=False)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — DensifyPointCloud (NEW in Tier 1)
# ─────────────────────────────────────────────────────────────────────────────
def densify_pointcloud(project_root: Path) -> bool:
    """
    OpenMVS's own dense reconstruction. Produces scene_dense.mvs with
    per-point normals — much better input for Poisson surface reconstruction
    than COLMAP's fused.ply (which lacks normals).

    Important: deletes any cached .dmap files from a prior InterfaceCOLMAP
    run before launching. Those depth maps were imported from COLMAP's
    PatchMatchStereo and use a confidence layout that DensifyPointCloud's
    filter rejects (we observed it discarding 100% of depths). Forcing a
    clean recompute makes DensifyPointCloud build its own internally
    consistent depth maps.

    --resolution-level 1: 2× downscale (good memory/quality balance).
        Reverted from Tier-4 level-0 experiment which was 4× slower (56 min)
        without improving mesh density (algorithm-bound at ~7.7M faces).
    --max-resolution 2560: cap input resolution
    --number-views 8 / --number-views-fuse 3: 8-view depth, 3-view fusion
    --estimate-normals 2: compute per-point normals
    """
    omvs = project_root / _OPENMVS_DIR_REL
    stale_dmaps = list(omvs.glob("depth*.dmap"))
    if stale_dmaps:
        logger.info(f"  Deleting {len(stale_dmaps)} cached COLMAP depth maps "
                    "before DensifyPointCloud (incompatible filter format).")
        for p in stale_dmaps:
            p.unlink()

    logger.info("=== OpenMVS [2/5]  DensifyPointCloud — dense cloud + normals (resolution-level 1) ===")
    out_rel = _OPENMVS_DIR_REL
    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} '
        f'{_C("DOCKER_OPENMVS")} bash -c "'
        f'{_OPENMVS_PATH_PREFIX}'
        f'cd /workspace/{out_rel} && '
        f'DensifyPointCloud '
        f'  -i scene.mvs '
        f'  -o scene_dense.mvs '
        f'  --working-folder . '
        f'  --resolution-level 1 '
        f'  --max-resolution 2560 '
        f'  --number-views 8 '
        f'  --number-views-fuse 3 '
        f'  --estimate-normals 2 '
        f'  --remove-dmaps 0'
        f'"'
    )
    return run(cmd, allow_fail=False)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — ReconstructMesh (Tier 1: tuned params)
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_mesh(project_root: Path,
                     dense: bool = True) -> bool:
    """
    Surface reconstruction via OpenMVS's Delaunay graph-cut.

    Tier-1 tuning:
      - Drop --remove-spikes (was over-pruning, fragmenting mesh)
      - Raise --close-holes 30 → 60 (fill larger gaps before post-process)
      - Keep --smooth 2

    NOTE: this algorithm caps mesh density at the scene's surface complexity
    (~7.7M faces for our scene) regardless of input point count. For denser
    output, use `poisson_reconstruct_mesh()` instead.
    """
    src = "scene_dense.mvs" if dense else "scene.mvs"
    logger.info(f"=== OpenMVS [3/5]  ReconstructMesh — Delaunay graph-cut from {src} ===")
    out_rel = _OPENMVS_DIR_REL
    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} '
        f'{_C("DOCKER_OPENMVS")} bash -c "'
        f'{_OPENMVS_PATH_PREFIX}'
        f'cd /workspace/{out_rel} && '
        f'ReconstructMesh '
        f'  -i {src} '
        f'  -o scene_mesh.mvs '
        f'  --working-folder . '
        f'  --close-holes 60 '
        f'  --smooth 2'
        f'"'
    )
    return run(cmd, allow_fail=False)


def poisson_reconstruct_mesh(project_root: Path, depth: int = 11,
                             density_threshold: float = 0.05) -> bool:
    """
    Alternative to `reconstruct_mesh()`: screened Poisson reconstruction
    via Open3D, fed by the scene_dense.ply that DensifyPointCloud writes
    alongside scene_dense.mvs.

    Output: openmvs/scene_mesh.ply (same name as the OpenMVS path so
            downstream stages — PostProcess, RefineMesh, TextureMesh —
            don't need to change).

    `depth=11` targets ~3-8M faces. Use 12 for ~10-20M (more RAM).
    """
    omvs = project_root / _OPENMVS_DIR_REL
    dense_ply = omvs / "scene_dense.ply"
    if not dense_ply.exists():
        logger.error(f"  Poisson: scene_dense.ply not found at {dense_ply}")
        return False

    logger.info(f"=== OpenMVS [3/5]  Poisson reconstruct (Open3D, depth={depth}) — alternative to Delaunay ===")
    script = project_root / "lib" / "poisson_mesh.py"
    out_ply = omvs / "scene_mesh.ply"
    res = subprocess.run(
        ["conda", "run", "-n", "da3", "--no-capture-output",
         "python", str(script), str(dense_ply), str(out_ply),
         "--depth", str(depth),
         "--density-threshold", str(density_threshold)],
        capture_output=True, text=True,
    )
    if res.stdout:
        for line in res.stdout.splitlines():
            logger.info(line)
    if res.returncode != 0:
        logger.error(f"Poisson reconstruct failed (exit {res.returncode})")
        if res.stderr:
            for line in res.stderr.splitlines()[:20]:
                logger.error(f"  stderr: {line}")
        return False
    return out_ply.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3b — PostProcess (NEW in Tier 1, host Python)
# ─────────────────────────────────────────────────────────────────────────────
def mesh_postprocess(project_root: Path, min_component_faces: int = 100,
                     src_name: str = "scene_mesh.ply", dst_name: str = "scene_mesh_clean.ply") -> bool:
    """
    Clean mesh output before texturing:
      • Drop connected components smaller than `min_component_faces`
        (kills the thousands of tiny floating fragments).
      • Fix winding (uniform outward normals — recovers a positive volume).
      • Fill small triangle/quad holes (pymeshlab close_holes).

    Reads:  openmvs/<src_name>
    Writes: openmvs/<dst_name>
    """
    logger.info(f"=== OpenMVS PostProcess {src_name} -> {dst_name} (v2: pymeshlab close_holes) ===")
    src = project_root / _OPENMVS_DIR_REL / src_name
    dst = project_root / _OPENMVS_DIR_REL / dst_name

    # mesh_postprocess_v2.py: pymeshlab close_holes (handles arbitrary hole sizes)
    # + Open3D orient_triangles (multi-shell winding).  Falls back to v1 if missing.
    script_v2 = project_root / "lib" / "mesh_postprocess_v2.py"
    script = script_v2 if script_v2.exists() else (project_root / "lib" / "mesh_postprocess.py")
    
    # Run using the robust direct da3 env python if available to bypass Conda run CLI issues on Windows
    import os
    python_exe = "C:\\Users\\mgallai\\AppData\\Local\\miniconda3\\envs\\da3\\python.exe"
    if os.path.exists(python_exe):
        cmd = [python_exe, str(script), str(src), str(dst), str(min_component_faces), "300"]
    else:
        cmd = ["conda", "run", "-n", "da3", "--no-capture-output",
               "python", str(script), str(src), str(dst), str(min_component_faces), "300"]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        for line in res.stdout.splitlines():
            logger.info(line)
    if res.returncode != 0:
        logger.error(f"PostProcess failed (exit {res.returncode})")
        if res.stderr:
            for line in res.stderr.splitlines()[:20]:
                logger.error(f"  stderr: {line}")
        return False
    return dst.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — RefineMesh (optional, slow)
# ─────────────────────────────────────────────────────────────────────────────
def refine_mesh(project_root: Path, src_mesh: str = "scene_mesh_clean.ply",
                resolution_level: int = 2) -> bool:
    """
    Photometric refinement: deforms the mesh so projections match image
    edges. Tightens silhouettes and recovers fine geometric detail.

    Critical: --decimate 1 DISABLES auto-decimation. Default (=0) means
    "auto" which silently halves the face count before refinement —
    not what we want when feeding it a hand-tuned dense mesh.
    --ensure-edge-size 0 disables edge-size remeshing for the same reason.

    resolution_level=2 ≈ 5 min on 685K faces; level 1 ≈ 30-60 min.
    """
    logger.info(f"=== OpenMVS [4/5]  RefineMesh — photometric refinement of {src_mesh} (res-level {resolution_level}, decimate disabled) ===")
    out_rel = _OPENMVS_DIR_REL
    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} '
        f'{_C("DOCKER_OPENMVS")} bash -c "'
        f'{_OPENMVS_PATH_PREFIX}'
        f'cd /workspace/{out_rel} && '
        f'RefineMesh '
        f'  -i scene.mvs '
        f'  -m {src_mesh} '
        f'  -o scene_mesh_refine.mvs '
        f'  --working-folder . '
        f'  --resolution-level {resolution_level} '
        f'  --decimate 1 '
        f'  --ensure-edge-size 0 '
        f'  --max-face-area 32'
        f'"'
    )
    return run(cmd, allow_fail=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5b — Scale calibration (Tier-2, host Python)
# ─────────────────────────────────────────────────────────────────────────────
def scale_calibrate(project_root: Path,
                    in_ply: str = "scene_textured.ply",
                    out_ply: str = "scene_textured.ply") -> bool:
    """
    Convert mesh vertex coords from COLMAP arbitrary units → meters using
    a measured scale factor. Prefers tof_bounds.json (Femto ToF — hardware-
    measured, more accurate) and falls back to da3_bounds.json (DA3 learned-
    metric estimate, ~7% off vs ToF on tested scenes).
    Idempotent: in_ply and out_ply may be the same file (default).
    Returns True if scaled or if no bounds file (silent skip).
    """
    omvs = project_root / _OPENMVS_DIR_REL
    tof_bounds = project_root / "colmap" / "dense" / "tof_bounds.json"
    da3_bounds = project_root / "colmap" / "dense" / "da3_bounds.json"
    if tof_bounds.exists():
        bounds = tof_bounds
        logger.info(f"  Scale source: {bounds.relative_to(project_root)} (Femto ToF, measured)")
    elif da3_bounds.exists():
        bounds = da3_bounds
        logger.info(f"  Scale source: {bounds.relative_to(project_root)} (DA3 estimate)")
    else:
        logger.info("  Scale calibration skipped — no bounds JSON found")
        return True

    logger.info("=== OpenMVS [5b/5]  Scale calibration — COLMAP units → meters ===")
    script = project_root / "lib" / "mesh_scale_calibrate.py"
    res = subprocess.run(
        ["conda", "run", "-n", "da3", "--no-capture-output",
         "python", str(script), str(omvs / in_ply), str(omvs / out_ply), str(bounds)],
        capture_output=True, text=True,
    )
    if res.stdout:
        for line in res.stdout.splitlines():
            logger.info(line)
    if res.returncode != 0:
        logger.error(f"Scale calibration failed (exit {res.returncode})")
        if res.stderr:
            for line in res.stderr.splitlines()[:10]:
                logger.error(f"  stderr: {line}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — TextureMesh
# ─────────────────────────────────────────────────────────────────────────────
def texture_mesh(project_root: Path, mesh_ply: str = "scene_mesh_clean.ply",
                 resolution_level: int = 0) -> bool:
    """
    Multi-view texture mapping: apply image colors to mesh surface.

    Uses the original scene.mvs (cameras+images) plus the mesh PLY.

    --resolution-level 0 = full image resolution. Switch to 1 (half-res
    images, ~1/4 memory) when the mesh has >10M faces — at that scale the
    atlas-packing stage with full-res images crashes with SIGSEGV.
    """
    logger.info(f"=== OpenMVS [5/5]  TextureMesh — applying textures using mesh={mesh_ply} (res-level {resolution_level}) ===")
    out_rel = _OPENMVS_DIR_REL
    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} '
        f'{_C("DOCKER_OPENMVS")} bash -c "'
        f'{_OPENMVS_PATH_PREFIX}'
        f'cd /workspace/{out_rel} && '
        f'TextureMesh '
        f'  -i scene.mvs '
        f'  -m {mesh_ply} '
        f'  -o scene_textured.mvs '
        f'  --working-folder . '
        f'  --resolution-level {resolution_level} '
        f'  --export-type ply'
        f'"'
    )
    return run(cmd, allow_fail=False)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_full_mesh_pipeline(project_root: Path,
                           colmap_dense_rel: str = "colmap/dense",
                           pointcloud_rel:  str | None = None,  # legacy, unused now
                           do_refine: bool = False,
                           refine_resolution_level: int = 2,
                           force_refine: bool = False,
                           do_scale_calibrate: bool = True,
                           use_poisson: bool = False,
                           poisson_depth: int = 11) -> Path | None:
    """
    Orchestrate Stages 1 → 5 with caching of the slow steps.

    Caching policy:
      scene.mvs       — cached if newer than COLMAP cameras.bin (skips ~26 min)
      scene_dense.mvs — cached if newer than scene.mvs           (skips ~30-60 min)
      scene_mesh.ply  — always re-run (cheap, ~2 min)
      scene_mesh_clean.ply — always re-run (very cheap, <1 min)
      scene_textured  — always re-run (~5 min)
    """
    ensure_workspace(project_root)
    omvs = project_root / _OPENMVS_DIR_REL

    # ── Stage 1: InterfaceCOLMAP ────────────────────────────────────────────
    scene_mvs = omvs / "scene.mvs"
    cameras_bin = project_root / colmap_dense_rel / "sparse" / "cameras.bin"
    if (scene_mvs.exists() and cameras_bin.exists()
            and scene_mvs.stat().st_mtime > cameras_bin.stat().st_mtime):
        logger.info("=== OpenMVS [1/5]  scene.mvs cached — skipping InterfaceCOLMAP ===")
    else:
        if not interface_colmap(project_root, colmap_dense_rel):
            logger.error("OpenMVS: InterfaceCOLMAP failed.")
            return None

    # ── Stage 2: DensifyPointCloud ──────────────────────────────────────────
    scene_dense = omvs / "scene_dense.mvs"
    if (scene_dense.exists()
            and scene_dense.stat().st_mtime > scene_mvs.stat().st_mtime):
        logger.info("=== OpenMVS [2/5]  scene_dense.mvs cached — skipping DensifyPointCloud ===")
    else:
        if not densify_pointcloud(project_root):
            logger.error("OpenMVS: DensifyPointCloud failed.")
            return None

    # ── Stage 3: ReconstructMesh ────────────────────────────────────────────
    scene_mesh = omvs / "scene_mesh.ply"
    if (scene_mesh.exists()
            and scene_dense.exists()
            and scene_mesh.stat().st_mtime > scene_dense.stat().st_mtime):
        logger.info("=== OpenMVS [3/5]  scene_mesh.ply cached — skipping reconstruct ===")
    else:
        if use_poisson:
            if not poisson_reconstruct_mesh(project_root, depth=poisson_depth):
                logger.error("OpenMVS: Poisson reconstruct failed.")
                return None
        else:
            if not reconstruct_mesh(project_root, dense=True):
                logger.error("OpenMVS: ReconstructMesh failed.")
                return None

    # ── Stage 3b: PostProcess ───────────────────────────────────────────────
    scene_mesh_clean = omvs / "scene_mesh_clean.ply"
    if (scene_mesh_clean.exists()
            and scene_mesh.exists()
            and scene_mesh_clean.stat().st_mtime > scene_mesh.stat().st_mtime):
        logger.info("=== OpenMVS [3b/5]  scene_mesh_clean.ply cached — skipping PostProcess ===")
        textured_input = "scene_mesh_clean.ply"
    elif not mesh_postprocess(project_root):
        logger.warning("OpenMVS: PostProcess failed — texturing the raw mesh instead.")
        textured_input = "scene_mesh.ply"
    else:
        textured_input = "scene_mesh_clean.ply"

    # ── Stage 4: RefineMesh (optional) ──────────────────────────────────────
    scene_mesh_refine = omvs / "scene_mesh_refine.ply"
    if do_refine:
        # force_refine=True (user passed --mesh-quality high/best) invalidates
        # the cache so the higher-resolution refine actually runs.
        if force_refine and scene_mesh_refine.exists():
            logger.info(f"  force_refine=True - removing cached scene_mesh_refine.ply "
                        f"to re-run at res-level {refine_resolution_level}")
            scene_mesh_refine.unlink()
        if (scene_mesh_refine.exists()
                and scene_mesh_clean.exists()
                and scene_mesh_refine.stat().st_mtime > scene_mesh_clean.stat().st_mtime):
            logger.info("=== OpenMVS [4/5]  scene_mesh_refine.ply cached — skipping RefineMesh ===")
            textured_input = "scene_mesh_refine.ply"
        elif refine_mesh(project_root, src_mesh=textured_input,
                         resolution_level=refine_resolution_level):
            textured_input = "scene_mesh_refine.ply"
        else:
            logger.warning("OpenMVS: RefineMesh failed — using un-refined mesh.")

    # ── Stage 4b: Post-process refined mesh (heals RefineMesh fragmentation) ──
    if do_refine and textured_input == "scene_mesh_refine.ply":
        scene_mesh_refine_clean = omvs / "scene_mesh_refine_clean.ply"
        # Overwrite if forced or if clean refined mesh is missing/outdated
        need_repost = (not scene_mesh_refine_clean.exists() 
                       or force_refine
                       or scene_mesh_refine_clean.stat().st_mtime < (omvs / "scene_mesh_refine.ply").stat().st_mtime)
        if need_repost:
            logger.info("=== OpenMVS [4b/5] PostProcess refined mesh (heals RefineMesh fragmentation) ===")
            # Use min_component_faces=250 to aggressively prune refinement-induced floaters
            if mesh_postprocess(project_root, min_component_faces=250, 
                                src_name="scene_mesh_refine.ply", dst_name="scene_mesh_refine_clean.ply"):
                textured_input = "scene_mesh_refine_clean.ply"
            else:
                logger.warning("OpenMVS: PostProcess on refined mesh failed — texturing un-cleaned refined mesh instead.")
        else:
            logger.info("=== OpenMVS [4b/5] scene_mesh_refine_clean.ply cached — skipping post-refine PostProcess ===")
            textured_input = "scene_mesh_refine_clean.ply"

    # ── Stage 5: TextureMesh ────────────────────────────────────────────────
    # Auto-pick image resolution level: dense meshes (>10M faces) crash the
    # atlas packer at full image resolution (we hit SIGSEGV at 12M faces with
    # res-level 0). Drop to res-level 1 (half-res images, ~1/4 memory) above
    # the threshold.
    tex_input_path = omvs / textured_input
    tex_res_level = 0
    if tex_input_path.exists():
        try:
            import open3d as _o3d
            _m = _o3d.io.read_triangle_mesh(str(tex_input_path))
            n_f = len(_m.triangles)
            if n_f > 10_000_000:
                tex_res_level = 1
                logger.info(f"  Mesh has {n_f:,} faces (>10M) → using TextureMesh res-level 1 (half-res images)")
        except Exception as e:
            logger.warning(f"  could not probe mesh face count: {e}")

    if not texture_mesh(project_root, mesh_ply=textured_input, resolution_level=tex_res_level):
        logger.error("OpenMVS: TextureMesh failed.")
        return None

    final = omvs / "scene_textured.ply"
    if not final.exists():
        logger.error(f"OpenMVS: expected final output {final} not found.")
        return None

    # ── Stage 5b: Scale calibration (Tier-2) ────────────────────────────────
    if do_scale_calibrate:
        if not scale_calibrate(project_root,
                               in_ply="scene_textured.ply",
                               out_ply="scene_textured.ply"):
            logger.warning("OpenMVS: scale calibration failed — leaving mesh in COLMAP units.")

    size_mb = final.stat().st_size / 1_048_576
    logger.info(f"OpenMVS textured mesh ready: {final}  ({size_mb:.1f} MB)")
    return final
