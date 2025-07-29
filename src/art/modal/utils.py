"""Modal utility functions for deployment and resource management."""

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, List, Optional, TypeVar
from dataclasses import dataclass
from enum import Enum
import time

from .. import dev

if TYPE_CHECKING:
    import modal

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModalAppStatus(Enum):
    """Modal app deployment status."""

    UNKNOWN = "unknown"
    DEPLOYING = "deploying"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ModalResourceConfig:
    """Configuration for Modal resources."""

    gpu_type: str = "H100"
    gpu_count: int = 1
    memory_gb: int = 32
    cpu_count: float = 4.0
    timeout_seconds: int = 3600
    allow_concurrent_inputs: int = 10
    keep_warm: Optional[int] = None
    container_idle_timeout: int = 300


@dataclass
class ModalDeploymentConfig:
    """Configuration for Modal app deployment."""

    app_name: str
    environment: str = "dev"
    workspace: Optional[str] = None
    secrets: List[str] = None
    volumes: Dict[str, str] = None
    image_registry: Optional[str] = None

    def __post_init__(self):
        if self.secrets is None:
            self.secrets = []
        if self.volumes is None:
            self.volumes = {}


class ModalError(Exception):
    """Base exception for Modal-related errors."""

    pass


class ModalDeploymentError(ModalError):
    """Exception raised during Modal app deployment."""

    pass


class ModalURLError(ModalError):
    """Exception raised when unable to get Modal URL."""

    pass


class ModalResourceError(ModalError):
    """Exception raised for Modal resource configuration issues."""

    pass


class ModalServiceError(ModalError):
    """Exception raised for Modal service integration issues."""

    pass


# Modal App Deployment and Management


async def deploy_modal_app(
    app_name: str,
    deployment_config: ModalDeploymentConfig,
    resource_config: ModalResourceConfig,
    env_vars: Optional[Dict[str, str]] = None,
) -> "modal.App":
    """
    Deploy a Modal app with the specified configuration.

    Args:
        app_name: Name of the Modal app
        deployment_config: Deployment configuration
        resource_config: Resource configuration
        env_vars: Environment variables to set

    Returns:
        Deployed Modal app instance

    Raises:
        ModalDeploymentError: If deployment fails
    """
    try:
        import modal
    except ImportError:
        raise ModalDeploymentError(
            "Modal is required. Install it with: pip install modal"
        )

    try:
        # Create app
        app = modal.App(app_name)

        logger.info(f"Deploying Modal app: {app_name}")

        # Note: Modal app configuration is done when functions are defined
        # This function returns the app for further configuration
        return app

    except Exception as e:
        raise ModalDeploymentError(f"Failed to deploy Modal app {app_name}: {e}")


def create_modal_image(config: ModalDeploymentConfig) -> "modal.Image":
    """
    Create a Modal image with ART dependencies.

    Args:
        config: Deployment configuration

    Returns:
        Configured Modal image
    """
    import modal

    # Start with base image
    image = modal.Image.debian_slim()

    # Install system dependencies
    image = image.run_commands(
        [
            "apt-get update",
            "apt-get install -y curl git build-essential",
        ]
    )

    # Install Python dependencies
    image = image.pip_install(
        [
            "openpipe-art[backend]",
            "httpx",
            "asyncio",
            "tqdm",
        ]
    )

    # Configure custom registry if specified
    if config.image_registry:
        # Modal doesn't directly support custom registries in the same way,
        # but we can configure the image differently based on requirements
        pass

    return image


def get_gpu_config(resource_config: ModalResourceConfig) -> "modal.gpu.GpuConfig":
    """
    Get GPU configuration for Modal.

    Args:
        resource_config: Resource configuration

    Returns:
        Modal GPU configuration

    Raises:
        ModalResourceError: If GPU type is not supported
    """
    import modal

    gpu_type = resource_config.gpu_type.upper()
    count = resource_config.gpu_count

    if gpu_type == "H100":
        return modal.gpu.H100(count=count)
    elif gpu_type == "A100":
        return modal.gpu.A100(count=count, size="80GB")
    elif gpu_type == "A10G":
        return modal.gpu.A10G(count=count)
    elif gpu_type == "T4":
        return modal.gpu.T4(count=count)
    elif gpu_type == "V100":
        return modal.gpu.V100(count=count)
    else:
        raise ModalResourceError(f"Unsupported GPU type: {gpu_type}")


async def get_app_status(app: "modal.App") -> ModalAppStatus:
    """
    Get the status of a Modal app.

    Args:
        app: Modal app instance

    Returns:
        App status
    """
    try:
        # Modal doesn't have a direct status API, so we check by trying to access it
        # This is a simplified approach - in practice, you'd implement more robust checks
        if hasattr(app, "_deployed") and app._deployed:
            return ModalAppStatus.READY
        else:
            return ModalAppStatus.UNKNOWN
    except Exception:
        return ModalAppStatus.ERROR


async def stop_modal_app(app: "modal.App") -> None:
    """
    Stop a Modal app and clean up resources.

    Args:
        app: Modal app to stop
    """
    try:
        # Modal apps don't need explicit stopping as they're serverless
        # But we can trigger cleanup hooks if needed
        logger.info(f"Stopping Modal app: {app.name}")

        # In practice, Modal functions stop automatically when not in use
        # This would be where you'd implement any custom cleanup logic

    except Exception as e:
        logger.error(f"Error stopping Modal app: {e}")


# URL Generation and Access


async def get_modal_web_url(
    app: "modal.App", function_name: str, timeout_seconds: int = 30
) -> str:
    """
    Get the web URL for a Modal function.

    Args:
        app: Modal app instance
        function_name: Name of the function to get URL for
        timeout_seconds: Timeout for URL retrieval

    Returns:
        Web URL for the function

    Raises:
        ModalURLError: If unable to get URL
    """
    try:
        # Get the function from the app
        function = getattr(app, function_name, None)
        if function is None:
            raise ModalURLError(f"Function {function_name} not found in app")

        # Wait for function to be deployed and get web URL
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                if hasattr(function, "web_url") and function.web_url:
                    return function.web_url
                await asyncio.sleep(1)
            except Exception:
                await asyncio.sleep(1)

        raise ModalURLError(
            f"Failed to get web URL for function {function_name} within {timeout_seconds}s"
        )

    except Exception as e:
        raise ModalURLError(f"Error getting Modal web URL: {e}")


def construct_modal_url(
    workspace: str, app_name: str, function_name: str, path: str = ""
) -> str:
    """
    Construct a Modal URL based on naming conventions.

    Args:
        workspace: Modal workspace name
        app_name: App name
        function_name: Function name
        path: Optional path to append

    Returns:
        Constructed Modal URL
    """
    # Modal URL format: https://{workspace}--{app-name}-{function-name}.modal.run/{path}
    base_url = f"https://{workspace}--{app_name}-{function_name}.modal.run"
    if path:
        if not path.startswith("/"):
            path = "/" + path
        return base_url + path
    return base_url


async def wait_for_url_ready(url: str, timeout_seconds: int = 60) -> bool:
    """
    Wait for a Modal URL to become ready and responsive.

    Args:
        url: URL to check
        timeout_seconds: Timeout in seconds

    Returns:
        True if URL is ready, False if timeout
    """
    import httpx

    start_time = time.time()
    async with httpx.AsyncClient(timeout=10) as client:
        while time.time() - start_time < timeout_seconds:
            try:
                response = await client.get(f"{url}/healthcheck")
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(2)

    return False


# Modal Function Deployment Patterns


def create_art_server_function(
    app: "modal.App",
    image: "modal.Image",
    resource_config: ModalResourceConfig,
    secrets: List["modal.Secret"] = None,
    volumes: Dict[str, "modal.Volume"] = None,
) -> Callable:
    """
    Create a Modal function for running the ART server.

    Args:
        app: Modal app instance
        image: Modal image to use
        resource_config: Resource configuration
        secrets: List of Modal secrets
        volumes: Dictionary of volume mounts

    Returns:
        Decorated Modal function
    """
    import modal

    if secrets is None:
        secrets = []
    if volumes is None:
        volumes = {}

    gpu_config = get_gpu_config(resource_config)

    @app.function(
        image=image,
        gpu=gpu_config,
        memory=resource_config.memory_gb * 1024,  # Convert GB to MB
        cpu=resource_config.cpu_count,
        timeout=resource_config.timeout_seconds,
        allow_concurrent_inputs=resource_config.allow_concurrent_inputs,
        keep_warm=resource_config.keep_warm,
        container_idle_timeout=resource_config.container_idle_timeout,
        secrets=secrets,
        volumes=volumes,
    )
    @modal.asgi_app()
    def art_server():
        """Run the ART server as an ASGI app."""
        from art.serve import create_app

        return create_app()

    return art_server


def create_training_function(
    app: "modal.App",
    image: "modal.Image",
    resource_config: ModalResourceConfig,
    secrets: List["modal.Secret"] = None,
    volumes: Dict[str, "modal.Volume"] = None,
) -> Callable:
    """
    Create a Modal function for model training.

    Args:
        app: Modal app instance
        image: Modal image to use
        resource_config: Resource configuration
        secrets: List of Modal secrets
        volumes: Dictionary of volume mounts

    Returns:
        Decorated Modal function for training
    """

    if secrets is None:
        secrets = []
    if volumes is None:
        volumes = {}

    gpu_config = get_gpu_config(resource_config)

    @app.function(
        image=image,
        gpu=gpu_config,
        memory=resource_config.memory_gb * 1024,
        cpu=resource_config.cpu_count,
        timeout=resource_config.timeout_seconds,
        secrets=secrets,
        volumes=volumes,
    )
    async def train_model(
        model_config: dict, train_config: dict, data_path: str
    ) -> dict:
        """Train a model on Modal."""
        # Training implementation would go here
        # This is a placeholder that would integrate with the appropriate service
        # based on model_config and train_config
        return {
            "status": "completed",
            "model_path": f"/models/{model_config['name']}/trained",
        }

    return train_model


# Container and Resource Configuration Helpers


def get_memory_config(base_memory_gb: int, model_size: str = "7B") -> int:
    """
    Calculate memory requirements based on model size.

    Args:
        base_memory_gb: Base memory requirement
        model_size: Model size (e.g., "7B", "13B", "70B")

    Returns:
        Recommended memory in GB
    """
    size_multipliers = {
        "7B": 1.0,
        "13B": 1.5,
        "30B": 2.5,
        "70B": 4.0,
        "180B": 8.0,
    }

    multiplier = size_multipliers.get(model_size, 1.0)
    return max(base_memory_gb, int(base_memory_gb * multiplier))


def get_optimal_gpu_config(
    model_size: str, task_type: str = "inference"
) -> ModalResourceConfig:
    """
    Get optimal GPU configuration for a given model size and task.

    Args:
        model_size: Model size (e.g., "7B", "13B", "70B")
        task_type: Type of task ("inference" or "training")

    Returns:
        Optimal resource configuration
    """
    # Base configurations for different model sizes
    configs = {
        "7B": {
            "inference": ModalResourceConfig(
                gpu_type="A10G", gpu_count=1, memory_gb=32
            ),
            "training": ModalResourceConfig(gpu_type="H100", gpu_count=1, memory_gb=80),
        },
        "13B": {
            "inference": ModalResourceConfig(
                gpu_type="A100", gpu_count=1, memory_gb=64
            ),
            "training": ModalResourceConfig(
                gpu_type="H100", gpu_count=2, memory_gb=160
            ),
        },
        "70B": {
            "inference": ModalResourceConfig(
                gpu_type="H100", gpu_count=2, memory_gb=160
            ),
            "training": ModalResourceConfig(
                gpu_type="H100", gpu_count=4, memory_gb=320
            ),
        },
    }

    return configs.get(model_size, {}).get(
        task_type,
        ModalResourceConfig(),  # Default config
    )


def validate_resource_config(config: ModalResourceConfig) -> None:
    """
    Validate Modal resource configuration.

    Args:
        config: Resource configuration to validate

    Raises:
        ModalResourceError: If configuration is invalid
    """
    if config.gpu_count < 1:
        raise ModalResourceError("GPU count must be at least 1")

    if config.memory_gb < 1:
        raise ModalResourceError("Memory must be at least 1 GB")

    if config.timeout_seconds < 60:
        raise ModalResourceError("Timeout must be at least 60 seconds")

    # Check GPU type support
    supported_gpus = ["H100", "A100", "A10G", "T4", "V100"]
    if config.gpu_type not in supported_gpus:
        raise ModalResourceError(f"Unsupported GPU type: {config.gpu_type}")


# Service Composition Helpers


class ModalModelService:
    """Modal-based model service that integrates with existing ModelService implementations."""

    def __init__(
        self,
        modal_app: "modal.App",
        service_type: str,
        model_name: str,
        base_model: str,
        config: dev.InternalModelConfig,
        output_dir: str,
    ):
        self.modal_app = modal_app
        self.service_type = service_type
        self.model_name = model_name
        self.base_model = base_model
        self.config = config
        self.output_dir = output_dir
        self._deployed_service = None

    async def deploy_service(self, resource_config: ModalResourceConfig) -> str:
        """
        Deploy the model service on Modal.

        Returns:
            URL of the deployed service
        """
        try:
            # Create service-specific function based on service type
            if self.service_type == "torchtune":
                service_function = self._create_torchtune_service(resource_config)
            elif self.service_type == "unsloth":
                service_function = self._create_unsloth_service(resource_config)
            elif self.service_type == "decoupled_unsloth":
                service_function = self._create_decoupled_unsloth_service(
                    resource_config
                )
            else:
                raise ModalServiceError(
                    f"Unsupported service type: {self.service_type}"
                )

            # Get the service URL
            url = await get_modal_web_url(self.modal_app, service_function.__name__)
            self._deployed_service = service_function

            return url

        except Exception as e:
            raise ModalServiceError(f"Failed to deploy service: {e}")

    def _create_torchtune_service(
        self, resource_config: ModalResourceConfig
    ) -> Callable:
        """Create Modal function for TorchTune service."""

        image = create_modal_image(ModalDeploymentConfig(app_name=self.model_name))
        gpu_config = get_gpu_config(resource_config)

        @self.modal_app.function(
            image=image,
            gpu=gpu_config,
            memory=resource_config.memory_gb * 1024,
            timeout=resource_config.timeout_seconds,
        )
        async def torchtune_service(request: dict) -> dict:
            """TorchTune service function."""
            from art.torchtune.service import TorchtuneService

            service = TorchtuneService(
                model_name=self.model_name,
                base_model=self.base_model,
                config=self.config,
                output_dir=self.output_dir,
            )

            # Handle different request types
            if request["action"] == "start_openai_server":
                await service.start_openai_server(request.get("config"))
                return {"status": "server_started"}
            elif request["action"] == "train":
                # This would need to be adapted for async iteration
                return {"status": "training_started"}
            else:
                raise ValueError(f"Unknown action: {request['action']}")

        return torchtune_service

    def _create_unsloth_service(self, resource_config: ModalResourceConfig) -> Callable:
        """Create Modal function for Unsloth service."""

        image = create_modal_image(ModalDeploymentConfig(app_name=self.model_name))
        gpu_config = get_gpu_config(resource_config)

        @self.modal_app.function(
            image=image,
            gpu=gpu_config,
            memory=resource_config.memory_gb * 1024,
            timeout=resource_config.timeout_seconds,
        )
        async def unsloth_service(request: dict) -> dict:
            """Unsloth service function."""
            from art.unsloth.service import UnslothService

            service = UnslothService(
                model_name=self.model_name,
                base_model=self.base_model,
                config=self.config,
                output_dir=self.output_dir,
            )

            # Handle different request types
            if request["action"] == "start_openai_server":
                await service.start_openai_server(request.get("config"))
                return {"status": "server_started"}
            elif request["action"] == "train":
                return {"status": "training_started"}
            else:
                raise ValueError(f"Unknown action: {request['action']}")

        return unsloth_service

    def _create_decoupled_unsloth_service(
        self, resource_config: ModalResourceConfig
    ) -> Callable:
        """Create Modal function for Decoupled Unsloth service."""

        image = create_modal_image(ModalDeploymentConfig(app_name=self.model_name))
        gpu_config = get_gpu_config(resource_config)

        @self.modal_app.function(
            image=image,
            gpu=gpu_config,
            memory=resource_config.memory_gb * 1024,
            timeout=resource_config.timeout_seconds,
        )
        async def decoupled_unsloth_service(request: dict) -> dict:
            """Decoupled Unsloth service function."""
            from art.unsloth.decoupled_service import DecoupledUnslothService

            service = DecoupledUnslothService(
                model_name=self.model_name,
                base_model=self.base_model,
                config=self.config,
                output_dir=self.output_dir,
            )

            # Handle different request types
            if request["action"] == "start_openai_server":
                await service.start_openai_server(request.get("config"))
                return {"status": "server_started"}
            elif request["action"] == "train":
                return {"status": "training_started"}
            else:
                raise ValueError(f"Unknown action: {request['action']}")

        return decoupled_unsloth_service


def create_service_factory(
    modal_app: "modal.App",
    deployment_config: ModalDeploymentConfig,
) -> Callable[[str, str, str, dev.InternalModelConfig, str], ModalModelService]:
    """
    Create a service factory for Modal-based model services.

    Args:
        modal_app: Modal app instance
        deployment_config: Deployment configuration

    Returns:
        Service factory function
    """

    def service_factory(
        service_type: str,
        model_name: str,
        base_model: str,
        config: dev.InternalModelConfig,
        output_dir: str,
    ) -> ModalModelService:
        return ModalModelService(
            modal_app=modal_app,
            service_type=service_type,
            model_name=model_name,
            base_model=base_model,
            config=config,
            output_dir=output_dir,
        )

    return service_factory


async def run_with_modal_retry(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
) -> T:
    """
    Run a function with exponential backoff retry for Modal operations.

    Args:
        func: Async function to run
        max_retries: Maximum number of retries
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiplier for delay on each retry

    Returns:
        Function result

    Raises:
        ModalError: If all retries fail
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = delay_seconds * (backoff_multiplier**attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} attempts failed")

    raise ModalError(f"Operation failed after {max_retries + 1} attempts: {last_error}")


# Cleanup and resource management


async def cleanup_modal_resources(app: "modal.App", timeout_seconds: int = 30) -> None:
    """
    Clean up Modal app resources.

    Args:
        app: Modal app to clean up
        timeout_seconds: Timeout for cleanup operations
    """
    try:
        logger.info(f"Cleaning up Modal app: {app.name}")

        # Modal apps are serverless and clean up automatically
        # Any custom cleanup logic would go here

        logger.info("Modal cleanup completed")

    except Exception as e:
        logger.error(f"Error during Modal cleanup: {e}")
        raise ModalError(f"Cleanup failed: {e}")
