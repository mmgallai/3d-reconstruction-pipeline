"""
Docker command runner with logging, retry, and clear error messages.
All Docker calls in the pipeline go through here.
"""

import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class DockerRunError(RuntimeError):
    pass


def run(cmd: str, allow_fail: bool = False, retries: int = 0) -> bool:
    """
    Run a shell command (typically a docker run ...).

    Parameters
    ----------
    cmd         : Full shell command string.
    allow_fail  : If True, log the failure but continue instead of raising.
    retries     : Number of automatic retries on failure (0 = no retry).

    Returns True on success, False if allow_fail and command failed.
    Raises DockerRunError if not allow_fail and command failed.
    """
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            logger.info(f"Retry {attempt - 1}/{retries} ...")
            time.sleep(5)

        logger.info(f"Running:\n  {cmd}\n")
        result = subprocess.run(cmd, shell=True)

        if result.returncode == 0:
            return True

        msg = f"Command failed (exit {result.returncode}):\n  {cmd}"
        if attempt < attempts:
            logger.warning(msg)
        elif allow_fail:
            logger.warning(f"{msg}\n  Continuing anyway (allow_fail=True).")
            return False
        else:
            logger.error(msg)
            raise DockerRunError(msg)

    return False


def build_image(dockerfile_dir: Path, tag: str) -> None:
    """Build a Docker image from a Dockerfile directory."""
    cmd = f'docker build -t {tag} "{dockerfile_dir}"'
    logger.info(f"Building Docker image '{tag}' ...")
    run(cmd)
    logger.info(f"Image '{tag}' built successfully.")


def image_exists(tag: str) -> bool:
    """Return True if a Docker image with the given tag is present locally."""
    result = subprocess.run(
        f'docker images -q {tag}',
        shell=True, capture_output=True, text=True
    )
    return bool(result.stdout.strip())
