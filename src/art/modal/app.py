import json
from typing import AsyncIterator
import modal
from fastapi import FastAPI, Body, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .. import dev
from ..local import LocalBackend
from ..model import Model, TrainableModel
from ..trajectories import TrajectoryGroup
from ..types import TrainConfig
from ..utils.deploy_model import LoRADeploymentProvider
from ..errors import ARTError


# Modal app configuration
app = modal.App("art-backend")

# GPU and image configuration
gpu_config = modal.gpu.H100(count=1)
image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "torch",
            "transformers",
            "unsloth",
            "torchtune",
            "fastapi",
            "uvicorn",
            "httpx",
            "pydantic",
            "tqdm",
            "openpipe-art[backend]",
        ]
    )
    .run_commands("apt-get update && apt-get install -y curl git")
)

# FastAPI app for ART endpoints
fastapi_app = FastAPI(title="ART Modal Backend", version="1.0.0")

# Global backend instance - will be initialized on first request
_backend_instance: LocalBackend | None = None


def get_backend() -> LocalBackend:
    """Get or create the backend instance."""
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = LocalBackend()
    return _backend_instance


# Exception handler for ARTError
@fastapi_app.exception_handler(ARTError)
async def art_error_handler(request: Request, exc: ARTError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@fastapi_app.get("/healthcheck")
async def healthcheck():
    """Service health monitoring."""
    return {"status": "ok"}


@fastapi_app.post("/close")
async def close():
    """Close the backend and clean up resources."""
    backend = get_backend()
    await backend.close()
    return {"status": "closed"}


@fastapi_app.post("/register")
async def register(model: Model):
    """Register a model with the backend for logging and/or training."""
    backend = get_backend()
    await backend.register(model)
    return {"status": "registered"}


@fastapi_app.post("/_get_step")
async def _get_step(model: TrainableModel):
    """Get the current training step for a model."""
    backend = get_backend()
    step = await backend._get_step(model)
    return step


@fastapi_app.post("/_delete_checkpoints")
async def _delete_checkpoints(
    model: TrainableModel,
    benchmark: str = Body(...),
    benchmark_smoothing: float = Body(...),
):
    """Delete model checkpoints based on benchmark criteria."""
    backend = get_backend()
    await backend._delete_checkpoints(model, benchmark, benchmark_smoothing)
    return {"status": "deleted"}


@fastapi_app.post("/_prepare_backend_for_training")
async def _prepare_backend_for_training(
    model: TrainableModel,
    config: dev.OpenAIServerConfig | None = Body(None),
):
    """Prepare the backend for training and return connection details."""
    backend = get_backend()
    result = await backend._prepare_backend_for_training(model, config)
    return result


@fastapi_app.post("/_log")
async def _log(
    model: Model,
    trajectory_groups: list[TrajectoryGroup],
    split: str = Body("val"),
):
    """Log trajectory groups for a model."""
    backend = get_backend()
    await backend._log(model, trajectory_groups, split)
    return {"status": "logged"}


@fastapi_app.post("/_train_model")
async def _train_model(
    model: TrainableModel,
    trajectory_groups: list[TrajectoryGroup],
    config: TrainConfig,
    dev_config: dev.TrainConfig,
    verbose: bool = Body(False),
) -> StreamingResponse:
    """Train a model with streaming progress updates."""
    backend = get_backend()

    async def stream() -> AsyncIterator[str]:
        async for result in backend._train_model(
            model, trajectory_groups, config, dev_config, verbose
        ):
            yield json.dumps(result) + "\n"

    return StreamingResponse(stream(), media_type="text/plain")


# S3 experimental endpoints
@fastapi_app.post("/_experimental_pull_from_s3")
async def _experimental_pull_from_s3(
    model: Model = Body(...),
    s3_bucket: str | None = Body(None),
    prefix: str | None = Body(None),
    verbose: bool = Body(False),
    delete: bool = Body(False),
    only_step: int | str | None = Body(None),
):
    """Download model directory from S3 to local file system."""
    backend = get_backend()
    await backend._experimental_pull_from_s3(
        model=model,
        s3_bucket=s3_bucket,
        prefix=prefix,
        verbose=verbose,
        delete=delete,
        only_step=only_step,
    )
    return {"status": "pulled"}


@fastapi_app.post("/_experimental_push_to_s3")
async def _experimental_push_to_s3(
    model: Model = Body(...),
    s3_bucket: str | None = Body(None),
    prefix: str | None = Body(None),
    verbose: bool = Body(False),
    delete: bool = Body(False),
):
    """Upload model directory from local file system to S3."""
    backend = get_backend()
    await backend._experimental_push_to_s3(
        model=model,
        s3_bucket=s3_bucket,
        prefix=prefix,
        verbose=verbose,
        delete=delete,
    )
    return {"status": "pushed"}


@fastapi_app.post("/_experimental_fork_checkpoint")
async def _experimental_fork_checkpoint(
    model: Model = Body(...),
    from_model: str = Body(...),
    from_project: str | None = Body(None),
    from_s3_bucket: str | None = Body(None),
    not_after_step: int | None = Body(None),
    verbose: bool = Body(False),
    prefix: str | None = Body(None),
):
    """Fork a checkpoint from another model to initialize this model."""
    backend = get_backend()
    await backend._experimental_fork_checkpoint(
        model=model,
        from_model=from_model,
        from_project=from_project,
        from_s3_bucket=from_s3_bucket,
        not_after_step=not_after_step,
        verbose=verbose,
        prefix=prefix,
    )
    return {"status": "forked"}


@fastapi_app.post("/_experimental_deploy")
async def _experimental_deploy(
    deploy_to: LoRADeploymentProvider = Body(...),
    model: TrainableModel = Body(...),
    step: int | None = Body(None),
    s3_bucket: str | None = Body(None),
    prefix: str | None = Body(None),
    verbose: bool = Body(False),
    pull_s3: bool = Body(True),
    wait_for_completion: bool = Body(True),
):
    """Deploy the model's latest checkpoint to a hosted inference endpoint."""
    backend = get_backend()
    result = await backend._experimental_deploy(
        deploy_to=deploy_to,
        model=model,
        step=step,
        s3_bucket=s3_bucket,
        prefix=prefix,
        verbose=verbose,
        pull_s3=pull_s3,
        wait_for_completion=wait_for_completion,
    )
    return result


# Deploy FastAPI app to Modal
@app.function(
    image=image,
    gpu=gpu_config,
    memory=32 * 1024,  # 32GB in MB
    timeout=3600,  # 1 hour timeout
    allow_concurrent_inputs=10,
    keep_warm=1,
)
@modal.asgi_app()
def art_web_app():
    """Deploy the ART FastAPI application to Modal."""
    return fastapi_app


# Utility function to get the deployed app URL programmatically
@app.function()
def get_app_url():
    """Get the deployed app URL programmatically."""
    return art_web_app.web_url


# Local development entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(fastapi_app, host="0.0.0.0", port=7999)
