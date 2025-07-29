"""
Modal service integration logic that composes ModalBackend with existing ModelService implementations.

This module handles:
1. Service selection logic similar to LocalBackend._get_service()
2. Service lifecycle management in Modal's containerized environment
3. File system abstraction for services that expect local directories
4. Communication bridging between Modal functions and existing service interfaces
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Optional, Type, TYPE_CHECKING
from pathlib import Path

from .. import dev
from .. import types

# Import types only for typing, actual imports happen at runtime
if TYPE_CHECKING:
    from ..local.service import ModelService
    from ..local.pack import DiskPackedTensors
from ..model import TrainableModel
from ..dev.get_model_config import get_model_config


@dataclass
class ModalServiceManager:
    """
    Manages ModelService instances in Modal's containerized environment.

    Handles service selection, lifecycle, file system abstraction, and
    communication bridging between Modal functions and existing services.
    """

    # Service registry and state
    _services: Dict[str, "ModelService"] = field(default_factory=dict)
    _temp_dirs: Dict[str, str] = field(default_factory=dict)
    _service_configs: Dict[str, dev.InternalModelConfig] = field(default_factory=dict)

    # Modal-specific configuration
    modal_workspace_path: str = "/tmp/art_workspace"
    enable_file_sync: bool = True
    verbose: bool = False

    async def get_service(
        self, model: TrainableModel, force_recreate: bool = False
    ) -> "ModelService":
        """
        Get or create a ModelService instance for the given model.

        This method replicates LocalBackend._get_service() logic but adapts it
        for Modal's containerized environment.

        Args:
            model: The trainable model to get a service for
            force_recreate: Force recreation of the service even if it exists

        Returns:
            ModelService instance (UnslothService, TorchtuneService, or DecoupledUnslothService)
        """
        if model.name not in self._services or force_recreate:
            if self.verbose:
                print(f"Creating new service for model: {model.name}")

            # Prepare the output directory in Modal's file system
            output_dir = await self._prepare_model_directory(model)

            # Get model configuration (similar to LocalBackend logic)
            config = get_model_config(
                base_model=model.base_model,
                output_dir=output_dir,
                config=model._internal_config,
            )

            # Store config for potential later use
            self._service_configs[model.name] = config

            # Determine service class based on configuration
            service_class = self._select_service_class(config)

            if self.verbose:
                print(f"Selected service class: {service_class.__name__}")

            # Create the service instance
            service = service_class(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=output_dir,
            )

            # Wrap the service for Modal compatibility
            wrapped_service = self._wrap_service_for_modal(service, model)

            self._services[model.name] = wrapped_service

        return self._services[model.name]

    def _select_service_class(
        self, config: dev.InternalModelConfig
    ) -> Type["ModelService"]:
        """
        Select the appropriate service class based on configuration.

        Replicates the selection logic from LocalBackend._get_service().
        """
        # Import here to avoid circular imports and Modal environment issues
        from ..torchtune.service import TorchtuneService
        from ..unsloth.service import UnslothService
        from ..unsloth.decoupled_service import DecoupledUnslothService

        if config.get("torchtune_args") is not None:
            return TorchtuneService
        elif config.get("_decouple_vllm_and_unsloth", False):
            return DecoupledUnslothService
        else:
            return UnslothService

    async def _prepare_model_directory(self, model: TrainableModel) -> str:
        """
        Prepare and return the model directory in Modal's file system.

        This handles file system abstraction for services that expect local directories.
        In Modal, we create temporary directories and optionally sync with persistent storage.
        """
        model_key = f"{model.project}_{model.name}"

        if model_key not in self._temp_dirs:
            # Create model directory structure in Modal's temp space
            base_dir = Path(self.modal_workspace_path)
            model_dir = base_dir / model.project / model.name
            model_dir.mkdir(parents=True, exist_ok=True)

            # Create required subdirectories
            for subdir in ["checkpoints", "logs", "trajectories", "tensors"]:
                (model_dir / subdir).mkdir(exist_ok=True)

            self._temp_dirs[model_key] = str(model_dir)

            if self.verbose:
                print(f"Created model directory: {model_dir}")

            # If file sync is enabled, attempt to pull from persistent storage
            if self.enable_file_sync:
                await self._sync_model_files_from_storage(model, str(model_dir))

        return self._temp_dirs[model_key]

    async def _sync_model_files_from_storage(
        self, model: TrainableModel, local_dir: str
    ) -> None:
        """
        Sync model files from persistent storage to local Modal directory.

        This could be extended to work with S3, GCS, or other storage backends.
        """
        if self.verbose:
            print(f"Syncing files for model {model.name} to {local_dir}")

        # Write model metadata
        model_json_path = Path(local_dir) / "model.json"
        with open(model_json_path, "w") as f:
            json.dump(model.model_dump(), f)

        # TODO: Add actual file syncing logic here
        # This would typically involve downloading checkpoints, logs, etc.
        # from persistent storage (S3, GCS, etc.)

    def _wrap_service_for_modal(
        self, service: "ModelService", model: TrainableModel
    ) -> "ModelService":
        """
        Wrap a service instance to make it compatible with Modal's environment.

        This handles any Modal-specific adaptations needed for the service to work
        in the containerized environment.
        """
        # Create a wrapper that preserves the original interface
        # but adds Modal-specific behavior
        return ModalServiceWrapper(service=service, model=model, manager=self)

    async def cleanup_service(self, model_name: str) -> None:
        """
        Clean up resources for a specific model service.
        """
        if model_name in self._services:
            service = self._services[model_name]

            # Call close method if available
            if hasattr(service, "close") and callable(getattr(service, "close")):
                if asyncio.iscoroutinefunction(service.close):
                    await service.close()
                else:
                    service.close()

            del self._services[model_name]

        # Clean up temporary directory
        model_keys_to_remove = [
            k for k in self._temp_dirs.keys() if k.endswith(f"_{model_name}")
        ]
        for key in model_keys_to_remove:
            del self._temp_dirs[key]

        if self.verbose:
            print(f"Cleaned up service for model: {model_name}")

    async def cleanup_all_services(self) -> None:
        """
        Clean up all service resources.
        """
        for model_name in list(self._services.keys()):
            await self.cleanup_service(model_name)

        self._service_configs.clear()


class ModalServiceWrapper:
    """
    Wrapper class that adapts existing ModelService implementations for Modal.

    This class preserves the ModelService interface while adding Modal-specific
    functionality like file system abstraction and communication bridging.
    """

    def __init__(
        self,
        service: "ModelService",
        model: TrainableModel,
        manager: ModalServiceManager,
    ):
        self._service = service
        self._model = model
        self._manager = manager

    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None:
        """
        Start the OpenAI server with Modal-specific adaptations.
        """
        if self._manager.verbose:
            print(f"Starting OpenAI server for model {self._model.name} in Modal")

        # Ensure any required files are synced
        await self._ensure_model_files_ready()

        # Delegate to the wrapped service
        await self._service.start_openai_server(config)

        if self._manager.verbose:
            print(f"OpenAI server started for model {self._model.name}")

    async def train(
        self,
        disk_packed_tensors: "DiskPackedTensors",
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """
        Train the model with Modal-specific adaptations.
        """
        if self._manager.verbose or verbose:
            print(f"Starting training for model {self._model.name} in Modal")

        # Ensure training data and model files are ready
        await self._ensure_training_data_ready(disk_packed_tensors)

        # Delegate to the wrapped service
        async for result in self._service.train(
            disk_packed_tensors, config, _config, verbose
        ):
            # Optionally add Modal-specific metrics or transformations
            yield self._augment_training_result(result)

        if self._manager.verbose or verbose:
            print(f"Training completed for model {self._model.name}")

        # Optionally sync results back to persistent storage
        if self._manager.enable_file_sync:
            await self._sync_results_to_storage()

    async def _ensure_model_files_ready(self) -> None:
        """
        Ensure all required model files are available in the Modal environment.
        """
        # This could involve downloading checkpoints, configuration files, etc.
        pass

    async def _ensure_training_data_ready(
        self, disk_packed_tensors: "DiskPackedTensors"
    ) -> None:
        """
        Ensure training data is available in the Modal environment.
        """
        # Verify that the tensor files exist and are accessible
        tensor_dir = disk_packed_tensors.get("tensor_dir")
        if tensor_dir and not os.path.exists(tensor_dir):
            if self._manager.verbose:
                print(f"Warning: Tensor directory not found: {tensor_dir}")

    def _augment_training_result(self, result: dict[str, float]) -> dict[str, float]:
        """
        Augment training results with Modal-specific information.
        """
        # Add Modal-specific metrics if needed
        augmented = result.copy()
        augmented["modal_backend"] = True
        return augmented

    async def _sync_results_to_storage(self) -> None:
        """
        Sync training results back to persistent storage.
        """
        if self._manager.verbose:
            print(f"Syncing results for model {self._model.name} to persistent storage")

        # TODO: Implement syncing of checkpoints, logs, etc. to persistent storage
        pass

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped service.
        """
        return getattr(self._service, name)


# Modal function utilities for integrating with the service manager
def create_modal_service_manager(
    workspace_path: str = "/tmp/art_workspace",
    enable_file_sync: bool = True,
    verbose: bool = False,
) -> ModalServiceManager:
    """
    Factory function to create a ModalServiceManager with common configurations.
    """
    return ModalServiceManager(
        modal_workspace_path=workspace_path,
        enable_file_sync=enable_file_sync,
        verbose=verbose,
    )


async def get_or_create_service_for_modal(
    model: TrainableModel, manager: Optional[ModalServiceManager] = None, **kwargs
) -> "ModelService":
    """
    Convenience function to get or create a service for a model in Modal.

    This function can be used directly in Modal functions to quickly get
    a service instance without managing the ModalServiceManager manually.
    """
    if manager is None:
        manager = create_modal_service_manager(**kwargs)

    return await manager.get_service(model)
