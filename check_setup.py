"""
Environment validation script.
Run this before your first reconstruction to catch setup problems early.

Usage:
    python check_setup.py
"""

import shutil
import subprocess
import sys
import importlib.util
from pathlib import Path

import config

PASS  = "[PASS]"
FAIL  = "[FAIL]"
WARN  = "[WARN]"
INFO  = "[INFO]"
SEP   = "-" * 50


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def check_docker():
    section("Docker Engine")

    if shutil.which("docker") is None:
        print(f"{FAIL} docker not in PATH. Install Docker Desktop.")
        return

    try:
        result = subprocess.run("docker --version", shell=True,
                                capture_output=True, text=True, check=True)
        print(f"{PASS} {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print(f"{FAIL} Docker not running. Start Docker Desktop and retry.")
        return

    # GPU passthrough
    print(f"{INFO} Testing GPU passthrough (needs 'colmap/colmap:latest')...")
    gpu_check = subprocess.run(
        "docker run --rm --gpus all colmap/colmap:latest nvidia-smi",
        shell=True, capture_output=True, text=True
    )
    if gpu_check.returncode == 0:
        gpu_name = next((l for l in gpu_check.stdout.splitlines() if "RTX" in l or "GTX" in l), "")
        print(f"{PASS} Docker GPU access OK.  {gpu_name.strip()}")
    elif "Unable to find image" in gpu_check.stderr:
        print(f"{WARN} colmap/colmap image not pulled yet — skipping GPU test.")
    else:
        print(f"{FAIL} Docker cannot access GPU.")
        print(f"       {gpu_check.stderr[:300]}")


def check_docker_images():
    section("Docker Images")

    required = [
        (config.DOCKER_COLMAP,      "COLMAP"),
        (config.DOCKER_OPENMVS,     "OpenMVS (optional, for mesh)"),
        (config.DOCKER_NERFSTUDIO,  "Nerfstudio Blackwell (local build)"),
    ]

    result = subprocess.run(
        "docker images --format {{.Repository}}:{{.Tag}}",
        shell=True, capture_output=True, text=True
    )
    installed = set(result.stdout.splitlines())

    for tag, label in required:
        if tag in installed:
            print(f"{PASS} {label}: {tag}")
        else:
            if "nerfstudio-blackwell" in tag:
                print(f"{WARN} {label}: NOT FOUND")
                print(f"       Build it with:  docker build -t {tag} .")
            else:
                print(f"{FAIL} {label}: NOT FOUND")
                print(f"       Pull with:      docker pull {tag}")


def check_python():
    section("Python Environment")

    # Torch + CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        device  = torch.cuda.get_device_name(0) if cuda_ok else "CPU only"
        status  = PASS if cuda_ok else WARN
        print(f"{status} PyTorch {torch.__version__}  |  CUDA: {cuda_ok}  |  Device: {device}")
    except ImportError:
        print(f"{FAIL} PyTorch not installed.  Run: pip install torch --index-url https://download.pytorch.org/whl/cu128")

    # Numpy + Pillow
    for pkg in ["numpy", "PIL"]:
        spec = importlib.util.find_spec(pkg)
        print(f"{PASS if spec else FAIL} {pkg}")


def check_sam3():
    section("SAM3")

    repo = config.SAM3_REPO_PATH
    ckpt = config.SAM3_CHECKPOINT

    if repo.exists():
        print(f"{PASS} SAM3 repo:        {repo}")
    else:
        print(f"{FAIL} SAM3 repo not found: {repo}")
        print(f"       Set SAM3_REPO_PATH in config.py or via environment variable.")

    if ckpt.exists():
        size_mb = ckpt.stat().st_size / 1024 / 1024
        print(f"{PASS} SAM3 checkpoint:  {ckpt}  ({size_mb:.0f} MB)")
    else:
        print(f"{FAIL} SAM3 checkpoint not found: {ckpt}")
        print(f"       Download sam3.pt and place it at: {ckpt}")


def check_config():
    section("config.py Summary")
    print(f"  SAM3_REPO_PATH       = {config.SAM3_REPO_PATH}")
    print(f"  SAM3_CHECKPOINT      = {config.SAM3_CHECKPOINT}")
    print(f"  NERF_MAX_ITERATIONS  = {config.NERF_MAX_ITERATIONS}")
    print(f"  NERF_DOWNSCALE       = {config.NERF_DOWNSCALE_FACTOR}")
    print(f"  EXPORT_MESH          = {config.EXPORT_MESH}")
    print(f"  EXPORT_SPLAT         = {config.EXPORT_SPLAT}")


if __name__ == "__main__":
    print("\n3D Reconstruction — Environment Check")
    check_docker()
    check_docker_images()
    check_python()
    check_sam3()
    check_config()
    print(f"\n{SEP}\n  Done. Fix any [FAIL] items before running reconstruct.py\n{SEP}\n")
