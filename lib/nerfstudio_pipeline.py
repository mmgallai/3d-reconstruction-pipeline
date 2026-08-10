"""
Nerfstudio training and export — Gaussian Splatting.
"""

import logging
import math
import os
from pathlib import Path

import config as _cfg
from lib.docker_runner import run, image_exists, build_image

# Read config values at call time (not import time) so --quick overrides work.
def _C(name):
    return getattr(_cfg, name)

logger = logging.getLogger(__name__)

# Patch needed because nerfstudio uses torch.load without weights_only=True
TORCH_LOAD_PATCH = """\
from pathlib import Path
OLD = 'torch.load(load_path, map_location="cpu")'
NEW = 'torch.load(load_path, map_location="cpu", weights_only=False)'
files = [
    '/usr/local/lib/python3.10/dist-packages/nerfstudio/utils/eval_utils.py',
    '/usr/local/lib/python3.10/dist-packages/nerfstudio/engine/trainer.py',
]
for path in files:
    p = Path(path)
    if not p.exists():
        print(f'[WARN] Not found: {path}')
        continue
    t = p.read_text()
    if OLD in t:
        p.write_text(t.replace(OLD, NEW))
        print(f'[PATCH] Patched: {p.name}')
    elif NEW in t:
        print(f'[INFO]  Already patched: {p.name}')
    else:
        print(f'[WARN]  Pattern not found in: {p.name}')
"""


def _vol(project_root: Path) -> str:
    return f'"{project_root}:/workspace"'


def ensure_image(dockerfile_dir: Path) -> None:
    """Build the nerfstudio-blackwell image if it does not exist yet."""
    if not image_exists(_C("DOCKER_NERFSTUDIO")):
        logger.info(f"Docker image '{_C("DOCKER_NERFSTUDIO")}' not found — building now (this takes ~10 min)...")
        build_image(dockerfile_dir, _C("DOCKER_NERFSTUDIO"))
    else:
        logger.info(f"Docker image '{_C("DOCKER_NERFSTUDIO")}' found.")


def write_patch_file(project_root: Path) -> Path:
    """Write the torch.load patch script to disk so Docker can execute it."""
    patch_path = project_root / "fix_weights.py"
    patch_path.write_text(TORCH_LOAD_PATCH)
    return patch_path


def train(project_root: Path, sparse_model_rel: str = "colmap/sparse/0") -> None:
    """
    Train a Gaussian Splat model with Nerfstudio splatfacto.
    Only passes --masks-path if a masks directory actually exists.
    """
    logger.info(f"=== Nerfstudio Training: {_C("NERF_TRAIN_METHOD")} ({_C("NERF_MAX_ITERATIONS")} iters) ===")
    masks_dir = project_root / "colmap" / "dense" / "masks"
    masks_arg = "--masks-path masks " if masks_dir.exists() and any(masks_dir.iterdir()) else ""
    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} {_C("DOCKER_NERFSTUDIO")} bash -c "'
        f'yes | ns-train {_C("NERF_TRAIN_METHOD")} '
        f'  --data /workspace/colmap/dense '
        f'  --output-dir /workspace/nerfstudio '
        f'  --vis tensorboard '
        f'  --max-num-iterations {_C("NERF_MAX_ITERATIONS")} '
        f'  colmap '
        f'  --colmap-path /workspace/{sparse_model_rel} '
        f'  {masks_arg}'
        f'  --downscale-factor {_C("NERF_DOWNSCALE_FACTOR")}'
        f'"'
    )
    run(cmd)


def pregenerate_downscaled_images(data_dir: Path, factor: int) -> None:
    """
    Pre-generate images_{factor}/ from images/ using PIL so Nerfstudio's
    nerfstudio-data dataparser can find the downscaled images on the first run.
    Skips images that already exist (idempotent).
    """
    if factor <= 1:
        return
    src_dir = data_dir / "images"
    dst_dir = data_dir / f"images_{factor}"
    if not src_dir.exists():
        logger.warning(f"pregenerate_downscaled_images: {src_dir} not found, skipping.")
        return

    from PIL import Image as PILImage

    dst_dir.mkdir(parents=True, exist_ok=True)
    jpegs = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.jpeg")) + sorted(src_dir.glob("*.png"))
    skipped = 0
    for src in jpegs:
        dst = dst_dir / src.name
        if dst.exists():
            skipped += 1
            continue
        img = PILImage.open(src)
        w, h = img.size
        new_w, new_h = max(1, math.ceil(w / factor)), max(1, math.ceil(h / factor))
        img_small = img.resize((new_w, new_h), PILImage.LANCZOS)
        img_small.save(dst, format="JPEG", quality=92)

    total = len(jpegs)
    generated = total - skipped
    logger.info(f"Downscaled images_{factor}/: {generated} generated, {skipped} already existed ({total} total).")


def resume_training(project_root: Path, data_rel: str = "colmap/dense") -> None:
    """
    Resume the most recent interrupted training run from its latest checkpoint.
    Finds the run with the most recent checkpoint that is NOT at max iterations.
    """
    method = _C("NERF_TRAIN_METHOD")
    iters  = _C("NERF_MAX_ITERATIONS")
    scale  = _C("NERF_DOWNSCALE_FACTOR")
    nerf_base = project_root / "nerfstudio"

    # Find all runs and pick the one with the latest checkpoint that isn't finished
    candidates = list(nerf_base.rglob("nerfstudio_models/step-*.ckpt"))
    if not candidates:
        logger.error("No checkpoints found to resume from.")
        return

    # Group by run directory, find latest checkpoint per run
    from collections import defaultdict
    runs = defaultdict(list)
    for ckpt in candidates:
        runs[ckpt.parent].append(ckpt)

    # Find the most recently modified incomplete run
    incomplete = []
    for ckpt_dir, ckpts in runs.items():
        latest = max(ckpts, key=os.path.getmtime)
        step = int(latest.stem.split("-")[-1])
        if step < iters - 1:
            incomplete.append((latest.stat().st_mtime, ckpt_dir, step))

    if not incomplete:
        logger.info("No incomplete runs found — all runs reached max iterations.")
        return

    incomplete.sort(reverse=True)
    _, ckpt_dir, step = incomplete[0]
    ckpt_dir_rel = ckpt_dir.relative_to(project_root).as_posix()

    logger.info(f"=== Resuming training from step {step} ({ckpt_dir_rel}) ===")
    write_patch_file(project_root)
    # Mirror the VRAM-fit flags used at scale=1 in train_nerfstudio_format —
    # without these, resuming a native-res run would try to cache all images
    # in GPU VRAM and OOM on a 16 GB card.
    extra_resume_args = ''
    if scale == 1:
        extra_resume_args = (
            '  --pipeline.datamanager.cache-images cpu '
            '  --pipeline.model.stop-split-at 12000 '
        )
        logger.info("Full-quality resume (scale=1): +cache-images=cpu, +stop-split-at=12000")
    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} '
        f'-e TORCH_HOME=/workspace/torch_cache '
        f'{_C("DOCKER_NERFSTUDIO")} bash -c "'
        f'python3 /workspace/fix_weights.py && '
        f'yes | ns-train {method} '
        f'  --load-dir /workspace/{ckpt_dir_rel} '
        f'  --data /workspace/{data_rel} '
        f'  --output-dir /workspace/nerfstudio '
        f'  --vis tensorboard '
        f'  --max-num-iterations {iters} '
        f'{extra_resume_args}'
        f'  nerfstudio-data '
        f'  --downscale-factor {scale}'
        f'"'
    )
    run(cmd)


def train_nerfstudio_format(
    project_root: Path,
    data_rel:     str   = "nerfstudio_data",
) -> None:
    """
    Train using a transforms.json (nerfstudio-data dataparser).

    data_rel : path relative to project_root containing transforms.json + images/.
    If transforms.json contains ply_file_path, the model uses it as Gaussian
    initialisation automatically — no extra flags required.
    Camera optimizer is enabled (SO3xR3) to fine-tune COLMAP poses during training.
    """
    method = _C("NERF_TRAIN_METHOD")
    iters  = _C("NERF_MAX_ITERATIONS")
    scale  = _C("NERF_DOWNSCALE_FACTOR")
    logger.info(f"=== Nerfstudio Training ({method}, {iters} iters, {scale}x downscale) ===")
    # V22: when training method is dn-splatter / ags-mesh, install the patched
    # dn-splatter into the container at runtime (--no-deps preserves the
    # nerfstudio/gsplat versions already in the image). Apply the source patches
    # from patches/dn_splatter_for_gsplat_15.patch to your local dn-splatter/
    # clone BEFORE running, otherwise the entry-point load fails on gsplat 1.5+.
    is_dn  = method.startswith(("dn-splatter", "dn_splatter", "ags-mesh", "ags_mesh"))
    is_tof = method.startswith(("splatfacto-tof", "splatfacto_tof"))
    if is_dn:
        extra_model_args = (
            '  --pipeline.model.use-depth-loss True '
            '  --pipeline.model.depth-lambda 0.2 '
            '  --pipeline.model.use-normal-loss True '
            '  --pipeline.model.use-normal-tv-loss True '
            '  --pipeline.model.normal-supervision depth '
            '  --pipeline.model.cull-alpha-thresh 0.01 '
        )
        install_cmd = (
            "pip install --upgrade -q 'setuptools>=61' wheel && "
            "pip install -q --no-build-isolation --no-deps --force-reinstall "
            "  /workspace/dn-splatter && "
            "pip install -q natsort geffnet rerun-sdk pytorch-lightning "
            "  omnidata-tools vdbfusion PyMCubes && "
        )
    elif is_tof:
        # splatfacto-tof uses splat_tof plugin which implements real depth supervision on modern gsplat 1.5
        extra_model_args = '  '
        install_cmd = (
            "pip install --upgrade -q 'setuptools>=61' wheel && "
            "pip install -q --no-build-isolation --no-deps --force-reinstall "
            "  /workspace/splat_tof && "
        )
    else:
        extra_model_args = '  --pipeline.model.cull-alpha-thresh 0.01 '
        install_cmd = ""

    # Full-quality run: at native resolution (--downscale-factor 1) with 311+ high-res
    # images, the default cache-images=gpu behavior eats 6-10 GB of VRAM just for the
    # image cache. When training at native, offload images to CPU RAM so we still fit
    # on a 16 GB card. Also stop densification a bit earlier (12k of 30k iters, default
    # is 15k) to keep Gaussian growth in check — the existing --cull-alpha-thresh 0.01
    # already does most of the work.
    #
    # NOTE: nerfstudio 1.1.5 (the version in nerfstudio-blackwell) does NOT have a
    # --pipeline.model.max-gauss-num flag; density is controlled indirectly via
    # cull-alpha-thresh + stop-split-at.
    if scale == 1 and not (is_dn or is_tof):
        extra_model_args += '  --pipeline.datamanager.cache-images cpu '
        extra_model_args += '  --pipeline.model.stop-split-at 12000 '
        logger.info("Full-quality mode (scale=1): +cache-images=cpu, +stop-split-at=12000")

    dataparser_args = f'  --downscale-factor {scale}'
    if is_dn or is_tof:
        # Our sensor depth maps are float32 metres; override the default 0.001 (mm convention)
        dataparser_args += '  --depth-unit-scale-factor 1.0 '
        
    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} '
        f'-e TORCH_HOME=/workspace/torch_cache '
        f'{_C("DOCKER_NERFSTUDIO")} bash -c "'
        f'{install_cmd}'
        f'yes | ns-train {method} '
        f'  --data /workspace/{data_rel} '
        f'  --output-dir /workspace/nerfstudio '
        f'  --vis tensorboard '
        f'  --max-num-iterations {iters} '
        f'{extra_model_args}'
        f'  nerfstudio-data '
        f'{dataparser_args}'
        f'"'
    )
    run(cmd)


def export_mesh(project_root: Path, method: str = "poisson") -> bool:
    """
    Export a mesh from the latest 2DGS training run using Poisson or TSDF.

    method : "poisson" (default, best for 2DGS surfaces) or "tsdf"
    Returns True if the export succeeded.
    """
    nerf_base    = project_root / "nerfstudio"
    train_method = _C("NERF_TRAIN_METHOD")
    candidates   = list(nerf_base.rglob(f"{train_method}/*/config.yml"))
    if not candidates:
        candidates = list(nerf_base.rglob("config.yml"))
    if not candidates:
        logger.error(f"No config.yml found under {nerf_base}")
        return False

    latest_cfg = max(candidates, key=os.path.getmtime)
    config_rel = latest_cfg.relative_to(project_root).as_posix()
    logger.info(f"Exporting mesh ({method}) from: {config_rel}")

    write_patch_file(project_root)
    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} '
        f'-e TORCH_HOME=/workspace/torch_cache '
        f'{_C("DOCKER_NERFSTUDIO")} bash -c "'
        f'python3 /workspace/fix_weights.py && '
        f'ns-export {method} '
        f'  --load-config /workspace/{config_rel} '
        f'  --output-dir /workspace/output '
        f'  --target-num-faces {_C("MESH_NUM_FACES")}'
        f'"'
    )
    return run(cmd, allow_fail=True)


def export_splat(project_root: Path) -> bool:
    """
    Export the trained Gaussian Splat to splat.ply.
    Automatically finds the latest training run directory.
    Returns True if export succeeded.
    """
    nerf_base = project_root / "nerfstudio"
    method    = _C("NERF_TRAIN_METHOD")
    # nerfstudio may shorten method name in folder (e.g. splatfacto-big → splatfacto)
    candidates = list(nerf_base.rglob(f"{method}/*/config.yml"))
    if not candidates:
        # fallback: search all config.yml under nerfstudio/
        candidates = list(nerf_base.rglob("config.yml"))
    if not candidates:
        logger.error(f"No config.yml found under {nerf_base}")
        return False

    latest_cfg = max(candidates, key=os.path.getmtime)
    config_rel = latest_cfg.relative_to(project_root).as_posix()
    logger.info(f"Exporting from: {config_rel}")

    write_patch_file(project_root)

    is_dn  = method.startswith(("dn-splatter", "dn_splatter", "ags-mesh", "ags_mesh"))
    is_tof = method.startswith(("splatfacto-tof", "splatfacto_tof"))
    install_cmd = ""
    if is_dn:
        install_cmd = (
            "pip install --upgrade -q 'setuptools>=61' wheel && "
            "pip install -q --no-build-isolation --no-deps --force-reinstall "
            "  /workspace/dn-splatter && "
            "pip install -q natsort geffnet rerun-sdk pytorch-lightning "
            "  omnidata-tools vdbfusion PyMCubes && "
        )
    elif is_tof:
        install_cmd = (
            "pip install --upgrade -q 'setuptools>=61' wheel && "
            "pip install -q --no-build-isolation --no-deps --force-reinstall "
            "  /workspace/splat_tof && "
        )

    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} '
        f'-e TORCH_HOME=/workspace/torch_cache '
        f'{_C("DOCKER_NERFSTUDIO")} bash -c "'
        f'{install_cmd}'
        f'python3 /workspace/fix_weights.py && '
        f'ns-export gaussian-splat '
        f'  --load-config /workspace/{config_rel} '
        f'  --output-dir /workspace/output'
        f'"'
    )
    return run(cmd, allow_fail=True)
