from typing import TYPE_CHECKING, AsyncIterator, Optional, Literal

from ..backend import Backend
from .. import dev
from ..trajectories import TrajectoryGroup
from ..types import TrainConfig
from ..utils.deploy_model import LoRADeploymentJob, LoRADeploymentProvider
from .utils import (
    ModalDeploymentConfig,
    ModalResourceConfig,
    ModalError,
    create_modal_image,
    get_gpu_config,
    wait_for_url_ready,
    cleanup_modal_resources,
    run_with_modal_retry,
)

if TYPE_CHECKING:
    from ..model import Model, TrainableModel
    from .service import ModalServiceManager
    import modal


class ModalBackend(Backend):
    """
    A Backend implementation that runs inference and training on Modal.

    Modal is a serverless cloud platform that can dynamically provision GPU resources
    for ML workloads. This backend handles Modal app deployment and lifecycle management.
    """

    def __init__(
        self,
        *,
        gpu_type: str = "H100",
        gpu_count: int = 1,
        memory_gb: int = 32,
        timeout_seconds: int = 3600,
        app_name: str = "art-backend",
        enable_file_sync: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the Modal backend.

        Args:
            gpu_type: The type of GPU to request (e.g., "H100", "A100", "T4").
            gpu_count: Number of GPUs to request.
            memory_gb: Amount of memory in GB to request.
            timeout_seconds: Timeout for Modal functions in seconds.
            app_name: Name for the Modal app.
            enable_file_sync: Enable file synchronization with persistent storage.
            verbose: Enable verbose logging.
        """
        # Resource configuration
        self._resource_config = ModalResourceConfig(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            memory_gb=memory_gb,
            timeout_seconds=timeout_seconds,
        )

        # Deployment configuration
        self._deployment_config = ModalDeploymentConfig(
            app_name=app_name,
            environment="production",
        )

        # Modal app and function references
        self._modal_app: Optional["modal.App"] = None
        self._modal_url: Optional[str] = None
        self._is_deployed = False
        self._service_manager: Optional["ModalServiceManager"] = None

        # Configuration
        self._enable_file_sync = enable_file_sync
        self._verbose = verbose

        # Initialize the base Backend with a placeholder URL
        # This will be updated once the Modal app is deployed
        super().__init__(base_url="http://localhost:7999")

    async def __aenter__(self):
        """
        Async context manager entry. Deploys the Modal app and starts the backend.
        """
        await self._deploy_modal_app()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit. Cleans up Modal resources.
        """
        await self._cleanup_modal_app()

    async def _deploy_modal_app(self) -> None:
        """
        Deploy the Modal app with the specified GPU and memory configuration.

        This method creates and deploys a Modal app that runs the ART backend server,
        then updates the base_url to point to the deployed Modal endpoint.
        """
        try:
            # Import modal here to avoid requiring it unless this backend is used
            import modal  # noqa: F401
        except ImportError:
            raise ModalError(
                "Modal is required to use ModalBackend. Install it with: pip install modal"
            )

        try:
            await run_with_modal_retry(self._create_and_deploy_app, max_retries=3)
        except Exception as e:
            raise ModalError(f"Failed to deploy Modal app: {e}")

    async def _create_and_deploy_app(self) -> None:
        """Create and deploy the Modal app."""
        import modal
        from .app import fastapi_app

        if self._verbose:
            print(f"Creating Modal app: {self._deployment_config.app_name}")

        # Create Modal app
        self._modal_app = modal.App(self._deployment_config.app_name)

        # Create Modal image with ART dependencies
        image = create_modal_image(self._deployment_config)
        gpu_config = get_gpu_config(self._resource_config)

        # Deploy the FastAPI app from app.py as a Modal function
        @self._modal_app.function(
            image=image,
            gpu=gpu_config,
            memory=self._resource_config.memory_gb * 1024,  # Convert GB to MB
            timeout=self._resource_config.timeout_seconds,
            allow_concurrent_inputs=self._resource_config.allow_concurrent_inputs,
            keep_warm=self._resource_config.keep_warm,
            container_idle_timeout=self._resource_config.container_idle_timeout,
        )
        @modal.asgi_app()
        def art_web_app():
            """Deploy the ART FastAPI application to Modal."""
            return fastapi_app

        if self._verbose:
            print(
                f"Deploying Modal app '{self._deployment_config.app_name}' with {self._resource_config.gpu_count}x{self._resource_config.gpu_type} GPU(s)..."
            )

        # Deploy the app using Modal's deployment mechanism
        try:
            with self._modal_app.run():
                # Get the web URL for the deployed function
                self._modal_url = art_web_app.web_url
                self._is_deployed = True

                # Update the base client URL to point to Modal
                self._base_url = self._modal_url

                if self._verbose:
                    print(f"Modal app deployed successfully at: {self._modal_url}")

                # Initialize service manager (lazy import)
                from .service import ModalServiceManager

                self._service_manager = ModalServiceManager(
                    modal_workspace_path="/tmp/art_workspace",
                    enable_file_sync=self._enable_file_sync,
                    verbose=self._verbose,
                )

                # Wait for the URL to be ready
                if not await wait_for_url_ready(self._modal_url, timeout_seconds=60):
                    raise ModalError("Modal app deployed but URL is not responding")

        except Exception as e:
            self._is_deployed = False
            self._modal_url = None
            raise ModalError(f"Failed to deploy Modal app: {e}")

    async def _cleanup_modal_app(self) -> None:
        """
        Clean up Modal app resources.
        """
        if self._modal_app and self._is_deployed:
            if self._verbose:
                print(f"Cleaning up Modal app '{self._deployment_config.app_name}'...")

            try:
                # Clean up service manager
                if self._service_manager:
                    await self._service_manager.cleanup_all_services()
                    self._service_manager = None

                # Clean up Modal resources
                await cleanup_modal_resources(self._modal_app)

                self._is_deployed = False
                self._modal_url = None

                if self._verbose:
                    print("Modal app cleanup completed")
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Error during cleanup: {e}")

    async def _ensure_deployed(self) -> None:
        """
        Ensure the Modal app is deployed before making requests.
        """
        if not self._is_deployed:
            raise RuntimeError(
                "Modal app not deployed. Use 'async with ModalBackend() as backend:' "
                "or call await backend._deploy_modal_app() first."
            )

    # Backend interface implementation

    async def register(self, model: "Model") -> None:
        """
        Register a model with the Modal backend.

        Args:
            model: The model to register.
        """
        await self._ensure_deployed()
        await super().register(model)

    async def close(self) -> None:
        """
        Close the Modal backend and clean up resources.
        """
        await self._cleanup_modal_app()
        await super().close()

    async def _get_step(self, model: "TrainableModel") -> int:
        """Get the current training step for a model."""
        await self._ensure_deployed()
        return await super()._get_step(model)

    async def _delete_checkpoints(
        self,
        model: "TrainableModel",
        benchmark: str,
        benchmark_smoothing: float,
    ) -> None:
        """Delete model checkpoints based on benchmark criteria."""
        await self._ensure_deployed()
        await super()._delete_checkpoints(model, benchmark, benchmark_smoothing)

    async def _prepare_backend_for_training(
        self,
        model: "TrainableModel",
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, str]:
        """Prepare the backend for training and return connection details."""
        await self._ensure_deployed()
        return await super()._prepare_backend_for_training(model, config)

    async def _log(
        self,
        model: "Model",
        trajectory_groups: list[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        """Log trajectory groups for a model."""
        await self._ensure_deployed()
        await super()._log(model, trajectory_groups, split)

    async def _train_model(
        self,
        model: "TrainableModel",
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Train a model with streaming progress updates."""
        await self._ensure_deployed()

        if self._verbose or verbose:
            print(f"Starting training for model {model.name} on Modal")

        # Stream results from the parent implementation (which makes HTTP calls to Modal)
        async for result in super()._train_model(
            model, trajectory_groups, config, dev_config, verbose
        ):
            yield result

        if self._verbose or verbose:
            print(f"Training completed for model {model.name}")

    # Experimental S3 support methods

    async def _experimental_pull_from_s3(
        self,
        model: "Model",
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
        only_step: int | Literal["latest"] | None = None,
    ) -> None:
        """Download the model directory from S3 into file system where the Modal backend is running."""
        await self._ensure_deployed()
        await super()._experimental_pull_from_s3(
            model=model,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            delete=delete,
            only_step=only_step,
        )

    async def _experimental_push_to_s3(
        self,
        model: "Model",
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
    ) -> None:
        """Upload the model directory from the file system where the Modal backend is running to S3."""
        await self._ensure_deployed()
        await super()._experimental_push_to_s3(
            model=model,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            delete=delete,
        )

    async def _experimental_fork_checkpoint(
        self,
        model: "Model",
        from_model: str,
        from_project: str | None = None,
        from_s3_bucket: str | None = None,
        not_after_step: int | None = None,
        verbose: bool = False,
        prefix: str | None = None,
    ) -> None:
        """Fork a checkpoint from another model to initialize this model."""
        await self._ensure_deployed()
        await super()._experimental_fork_checkpoint(
            model=model,
            from_model=from_model,
            from_project=from_project,
            from_s3_bucket=from_s3_bucket,
            not_after_step=not_after_step,
            verbose=verbose,
            prefix=prefix,
        )

    async def _experimental_deploy(
        self,
        deploy_to: LoRADeploymentProvider,
        model: "Model",
        step: int | None = None,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        pull_s3: bool = True,
        wait_for_completion: bool = True,
    ) -> LoRADeploymentJob:
        """Deploy the model's latest checkpoint to a hosted inference endpoint."""
        await self._ensure_deployed()
        return await super()._experimental_deploy(
            deploy_to=deploy_to,
            model=model,
            step=step,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            pull_s3=pull_s3,
            wait_for_completion=wait_for_completion,
        )

    # Property accessors

    @property
    def modal_url(self) -> Optional[str]:
        """Get the deployed Modal app URL."""
        return self._modal_url

    @property
    def is_deployed(self) -> bool:
        """Check if the Modal app is currently deployed."""
        return self._is_deployed

    @property
    def gpu_config(self) -> dict:
        """Get the current GPU configuration."""
        return {
            "gpu_type": self._resource_config.gpu_type,
            "gpu_count": self._resource_config.gpu_count,
            "memory_gb": self._resource_config.memory_gb,
            "timeout_seconds": self._resource_config.timeout_seconds,
        }

    @property
    def deployment_config(self) -> ModalDeploymentConfig:
        """Get the deployment configuration."""
        return self._deployment_config

    @property
    def resource_config(self) -> ModalResourceConfig:
        """Get the resource configuration."""
        return self._resource_config

    @property
    def service_manager(self) -> Optional["ModalServiceManager"]:
        """Get the service manager instance."""
        return self._service_manager
