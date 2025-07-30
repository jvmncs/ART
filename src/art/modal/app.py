"""
Modal app for ART backend with proper Modal API usage.

This module creates a Modal app that can be deployed with different:
- GPU types and counts (H100:2, A100:4, etc.)
- Volume configurations
- App names
- Environment variables
- Image configurations with ART backend dependencies
"""

import pathlib
import os
import sys
from typing import Dict

import modal

from art.cli import create_local_backend_app
from art import local
from art.modal.util import ModalGPUConfig


def create_modal_app(
    app_name: str = "art-backend",
    volume_name: str = "art-volume",
    gpu_config: str | ModalGPUConfig | list[str | ModalGPUConfig] = "H100:1",
    timeout: int = 3600,
    memory: int = 32000,
    keep_warm: int = 0,
    env_vars: Dict[str, str] | None = None,
    image: modal.Image | None = None,
    image_packages: list[str] | None = None,
    art_version: str | pathlib.Path | None = None,
) -> modal.App:
    """
    Create a Modal app for ART backend.

    Args:
        app_name: Name for the Modal app
        volume_name: Name for the Modal volume (created if missing)
        gpu_config: GPU specification (e.g., "H100:2", "A100:4", "T4:1")
        timeout: Function timeout in seconds
        memory: Memory allocation in MB
        keep_warm: Number of containers to keep warm
        env_vars: Environment variables to forward to Modal
        image: Base Modal image to use
        image_packages: Additional packages to install in the image
        art_version: ART version to install (semver string or local path)

    Returns:
        Configured Modal app instance
    """
    # Create Modal app with configurable name
    app = modal.App(app_name)

    # Create configurable volume
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)

    gpu_configs = []
    if not isinstance(gpu_config, list):
        gpu_config = [gpu_config]

    for cfg in gpu_config:
        if not isinstance(cfg, ModalGPUConfig):
            gpu_configs.append(ModalGPUConfig(cfg))
        else:
            gpu_configs.append(cfg)

    # Validate that only one of image or image_packages is provided
    if image is not None and image_packages is not None:
        raise ValueError(
            "Only one of 'image' or 'image_packages' can be provided, not both"
        )

    if image is None:
        # Get current Python version for base image
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Use debian_slim as base image with current Python version
        image = modal.Image.debian_slim(python_version=python_version)

        # Base packages for backend functionality
        base_packages = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "httpx",
            "aiofiles",
        ]

        # Extend with additional packages if provided
        if image_packages:
            base_packages.extend(image_packages)

        # Install base packages
        image = image.uv_pip_install(base_packages)

        # Handle ART version installation
        if art_version is not None:
            if isinstance(art_version, pathlib.Path) or (
                isinstance(art_version, str) and os.path.exists(art_version)
            ):
                # Install from local path
                image = image.uv_pip_install(f"{art_version}[backend]")
            else:
                # Install from PyPI with specific version
                image = image.uv_pip_install(f"openpipe-art[backend]=={art_version}")
        else:
            # Install latest from PyPI
            image = image.uv_pip_install("openpipe-art[backend]")

    # Prepare environment variables and secrets
    modal_env = {}
    if env_vars:
        modal_env.update(env_vars)

    # Create Modal Secret from environment variables
    secrets = []
    if modal_env:
        secrets.append(modal.Secret.from_dict(modal_env))

    # Configure Modal function
    function_config = {
        "image": image,
        "gpu": gpu_configs,
        "memory": memory,
        "timeout": timeout,
        "volumes": {"/art/": volume},
        "secrets": secrets,
        "keep_warm": keep_warm,
        "serialized": True,
    }

    @app.function(**function_config)
    @modal.asgi_app()
    def fastapi_app():
        """Create and configure the FastAPI app for ART backend."""
        backend = local.LocalBackend(path="/art/")
        return create_local_backend_app(backend)

    return app
