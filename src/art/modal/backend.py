import asyncio
import os
from typing import TYPE_CHECKING, Dict, Any

import modal
import httpx
from dotenv import dotenv_values

from .app import create_modal_app
from ..backend import Backend

if TYPE_CHECKING:
    from ..model import Model, TrainableModel


class ModalBackend(Backend):
    """
    Modal backend for ART with basic functionality and proper Modal API usage.

    Provides the core functionality needed for ART training and inference
    with Modal's serverless infrastructure.
    """

    def __init__(self, base_url: str):
        """Private constructor. Use initialize_cluster() class method instead."""
        super().__init__(base_url=base_url)
        self._modal_app = None
        self._function_handle = None
        self._deployment_url = base_url
        self._app_name = None
        self._env_vars = {}

    @classmethod
    async def initialize_cluster(
        cls,
        app_name: str = "art-backend",
        gpu_type: str = "A10G",
        gpu_count: int = 1,
        memory: int = 32000,
        timeout: int = 3600,
        keep_warm: int = 0,
        volume_name: str = "art-volume",
        image_packages: list[str] | None = None,
        environment: str = "main",
        env_file: str | None = None,
        force_rebuild: bool = False,
    ) -> "ModalBackend":
        """
        Initialize a Modal backend cluster.

        Args:
            app_name: Name for the Modal app
            gpu_type: GPU type (e.g., "A10G", "A100", "H100", "T4")
            gpu_count: Number of GPUs
            memory: Memory allocation in MB
            timeout: Function timeout in seconds
            keep_warm: Number of containers to keep warm
            volume_name: Name for the Modal volume
            image_packages: Additional packages to install
            environment: Modal environment name
            env_file: Path to environment file (.env)
            force_rebuild: Force rebuild of existing app

        Returns:
            Initialized ModalBackend instance
        """
        self = cls.__new__(cls)
        self._app_name = app_name
        self._env_vars = {}

        # Load environment variables from file if provided
        if env_file and os.path.exists(env_file):
            env_vars_from_file = dotenv_values(env_file)
            self._env_vars.update({k: v for k, v in env_vars_from_file.items() if v})

        # Construct GPU configuration string
        gpu_config = f"{gpu_type}:{gpu_count}"

        # Prepare image packages
        image_packages_final = image_packages or []

        try:
            # Check if app already exists
            existing_function = None
            try:
                existing_function = modal.Function.lookup(
                    app_name, "fastapi_app", environment_name=environment
                )
            except modal.exception.NotFoundError:
                pass

            if existing_function and not force_rebuild:
                print(f"App {app_name} already exists, using existing deployment...")
                base_url = existing_function.get_web_url()
                self._function_handle = existing_function
            else:
                # Deploy new app
                base_url = await self._deploy_new_app(
                    app_name,
                    gpu_config,
                    memory,
                    timeout,
                    keep_warm,
                    volume_name,
                    image_packages_final,
                    environment,
                )

        except Exception as e:
            print(f"Error with existing app, deploying new app: {e}")
            base_url = await self._deploy_new_app(
                app_name,
                gpu_config,
                memory,
                timeout,
                keep_warm,
                volume_name,
                image_packages_final,
                environment,
            )

        if not base_url:
            raise RuntimeError("Failed to get valid base URL from Modal deployment")

        print(f"Using base_url: {base_url}")

        # Initialize the backend with the URL
        super(cls, self).__init__(base_url=base_url)
        self._deployment_url = base_url

        # Wait for the app to be ready
        await self._wait_for_app_ready(base_url)

        return self

    async def _deploy_new_app(
        self,
        app_name: str,
        gpu_config: str,
        memory: int,
        timeout: int,
        keep_warm: int,
        volume_name: str,
        image_packages: list[str],
        environment: str,
    ) -> str:
        """Deploy a new Modal app and return its URL."""
        try:
            # Create the Modal app
            self._modal_app = create_modal_app(
                app_name=app_name,
                volume_name=volume_name,
                gpu_config=gpu_config,
                timeout=timeout,
                memory=memory,
                keep_warm=keep_warm,
                env_vars=self._env_vars,
                image_packages=image_packages,
            )

            # Deploy the app
            print("Deploying Modal app...")
            await asyncio.to_thread(
                lambda: self._modal_app.deploy(environment=environment)
            )

            # Get the function handle and URL
            self._function_handle = modal.Function.lookup(
                app_name, "fastapi_app", environment_name=environment
            )

            base_url = self._function_handle.get_web_url()
            print(f"App deployed successfully at: {base_url}")

            return base_url

        except Exception as e:
            print(f"Error deploying Modal app: {e}")
            raise RuntimeError(f"Failed to deploy Modal app: {e}")

    async def _wait_for_app_ready(
        self, base_url: str, max_retries: int = 30, delay: float = 10
    ) -> None:
        """Wait for the Modal app to be ready and responsive."""
        print("Waiting for app to be ready...")

        async with httpx.AsyncClient() as client:
            for attempt in range(max_retries):
                try:
                    response = await client.get(f"{base_url}/health", timeout=5)
                    if response.status_code == 200:
                        print("App is ready!")
                        return
                except (httpx.RequestError, httpx.TimeoutException):
                    pass

                if attempt < max_retries - 1:
                    print(
                        f"App not ready yet (attempt {attempt + 1}/{max_retries}), retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    # Try basic connectivity on final attempt
                    try:
                        response = await client.get(base_url, timeout=10)
                        if response.status_code in [
                            200,
                            404,
                            422,
                        ]:  # 404/422 are OK for FastAPI
                            print("App appears to be responding (no health endpoint)")
                            return
                    except Exception:
                        pass

                    print("Warning: App may not be fully ready, but proceeding...")

    async def register(self, model: "Model") -> None:
        """Register a model with the backend."""
        print("Registering model with Modal backend")
        print(f"App URL: {self._deployment_url}")
        await super().register(model)

    async def _prepare_backend_for_training(
        self,
        model: "TrainableModel",
        config: Any,
    ) -> tuple[str, str]:
        """Prepare backend for training and return base URL and API key."""
        response = await self._client.post(
            "/_prepare_backend_for_training",
            json={"model": model.model_dump(), "config": config},
            timeout=1200,
        )
        response.raise_for_status()
        result = response.json()

        # Return the same base URL since Modal serves everything from one endpoint
        return self._deployment_url, result[1] if isinstance(
            result, list
        ) else result.get("api_key", "")

    async def down(self) -> None:
        """Shutdown the Modal app deployment."""
        if self._function_handle:
            try:
                print(f"Modal app '{self._app_name}' is deployed.")
                print("Note: Modal functions cannot be stopped programmatically.")
                print(f"To stop the app, run: modal app stop {self._app_name}")
                # Reset internal state
                self._function_handle = None
                self._modal_app = None
                print("Backend state cleared")
            except Exception as e:
                print(f"Error clearing backend state: {e}")
                raise
        else:
            print("No active Modal function to stop")

    async def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Modal deployment."""
        if self._function_handle is None:
            return {"status": "not_deployed", "app_name": self._app_name}

        try:
            # Try to get basic info about the function
            stats = await asyncio.to_thread(
                lambda: self._function_handle.get_current_stats()
            )

            return {
                "status": "deployed",
                "app_name": self._app_name,
                "url": self._deployment_url,
                "stats": stats,
            }
        except Exception as e:
            return {"status": "error", "app_name": self._app_name, "error": str(e)}

    async def is_healthy(self) -> bool:
        """Check if the Modal backend is healthy and responsive."""
        if not self._deployment_url:
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self._deployment_url}/health", timeout=5)
                return response.status_code == 200
        except Exception:
            return False

    @property
    def app_name(self) -> str:
        """Get the app name."""
        return self._app_name

    @property
    def deployment_url(self) -> str | None:
        """Get the deployment URL."""
        return self._deployment_url

    @property
    def env_vars(self) -> Dict[str, str]:
        """Get the environment variables."""
        return self._env_vars.copy()
