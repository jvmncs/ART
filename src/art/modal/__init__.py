"""
Modal backend implementation for ART.

This package provides a Modal-based backend that can run inference and training
on Modal's serverless cloud platform with dynamic GPU provisioning.

Key components:
- ModalBackend: Main backend implementation that handles Modal app deployment
- ModalServiceManager: Service management for different model types
- Modal utilities: Deployment helpers and resource management
- FastAPI app: HTTP server that exposes ART backend API endpoints
"""

from .backend import ModalBackend

# Note: ModalServiceManager is imported lazily to avoid dependency issues
# It can be imported directly with: from art.modal.service import ModalServiceManager
from .utils import (
    ModalDeploymentConfig,
    ModalResourceConfig,
    ModalError,
    ModalDeploymentError,
    ModalURLError,
    ModalResourceError,
    ModalServiceError,
)

__all__ = [
    "ModalBackend",
    # "ModalServiceManager",  # Available via direct import to avoid dependencies
    "ModalDeploymentConfig",
    "ModalResourceConfig",
    "ModalError",
    "ModalDeploymentError",
    "ModalURLError",
    "ModalResourceError",
    "ModalServiceError",
]
