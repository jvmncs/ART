"""
ART Modal Backend - Serverless AI Infrastructure

The Modal backend provides a clean, simple interface to Modal's serverless infrastructure
for ART training and inference workloads.

## Quick Start

```python
from art.modal import ModalBackend

# Initialize with desired configuration
backend = await ModalBackend.initialize_cluster(
    app_name="my-art-app",
    gpu_type="A10G",
    gpu_count=1,
    memory=16000,
    keep_warm=1,
)

# Use with ART models (same API as other backends)
await backend.register(model)
await backend.train(model, config)
```

## Core Features

- **Serverless**: No cluster management, automatic scaling
- **Pay-per-use**: Only pay for actual compute time
- **GPU Support**: Easy access to A10G, A100, H100, and other GPUs
- **Fast deployment**: Deploy in seconds, not minutes
- **Built-in volumes**: Persistent storage for models and data

## Configuration Options

- `gpu_type`: GPU type ("A10G", "A100", "H100", "T4")
- `gpu_count`: Number of GPUs (1, 2, 4, 8)
- `memory`: Memory allocation in MB
- `keep_warm`: Number of containers to keep warm
- `timeout`: Function timeout in seconds
- `environment`: Modal environment name
"""

# Core backend and app creation
from .backend import ModalBackend
from .app import create_modal_app

# Configuration utilities
from .util import (
    ModalGPUType,
    ModalGPUConfig,
    ModalClusterConfig,
)

# Export everything for convenient access
__all__ = [
    # Core Components
    "ModalBackend",
    "create_modal_app",
    # Configuration
    "ModalGPUType",
    "ModalGPUConfig",
    "ModalClusterConfig",
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "ART Modal Team"
__description__ = "Simple Modal backend for ART training and inference"
