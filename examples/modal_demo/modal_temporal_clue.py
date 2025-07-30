import art
import asyncio
import argparse
from dotenv import load_dotenv
import json
import random
import re
from typing import TypedDict
import os

load_dotenv()


class TemporalCluePuzzle(TypedDict):
    num_clues: int
    prompt: str
    solution: dict[str, str]


# Load puzzles from the data directory
puzzles_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "temporal-clue", "puzzles.json"
)

puzzles: list[TemporalCluePuzzle] = json.loads(open(puzzles_path).read())
val_puzzles = puzzles[:64]
test_puzzles = puzzles[64:128]
train_puzzles = puzzles[128:]
random.seed(42)
random.shuffle(train_puzzles)


async def rollout(model: art.Model, puzzle: TemporalCluePuzzle) -> art.Trajectory:
    """Execute a single rollout on a temporal clue puzzle."""
    messages: art.Messages = [{"role": "user", "content": puzzle["prompt"]}]
    client = model.openai_client()
    chat_completion = await client.chat.completions.create(
        messages=messages, model=model.name
    )
    choice = chat_completion.choices[0]
    content = choice.message.content
    assert isinstance(content, str)

    # Calculate accuracy by checking each solution key
    num_correct = 0
    for key, value in puzzle["solution"].items():
        if matches := re.findall(rf"{key}\. ([A-Za-z \.:-]+)", content):
            match = matches[-1]
            if match.strip().lower() == value.lower():
                num_correct += 1

    reward = acc = num_correct / len(puzzle["solution"])
    return art.Trajectory(
        messages_and_choices=[*messages, choice], reward=reward, metrics={"acc": acc}
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Train a model on temporal clue puzzles"
    )
    parser.add_argument(
        "--backend",
        choices=["modal", "local"],
        # FEEDBACK: modal should be default. it's useful to have both because it demonstrates what changes are needed (similar to SkypilotBackend)
        default="local",
        help="Backend to use for training (default: local)",
    )
    parser.add_argument(
        "--app-name",
        default="art-temporal-clue",
        help="Modal app name (only used with modal backend)",
    )
    parser.add_argument(
        "--gpu-type",
        # FEEDBACK: the default model is likely too big for this GPU
        default="A10G",
        help="GPU type for Modal backend (e.g., A10G, A100, H100)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of Modal app (modal backend only)",
    )
    args = parser.parse_args()

    # Initialize backend based on selection
    if args.backend == "modal":
        # Import Modal backend only when needed to avoid unnecessary dependencies
        from art.modal.backend import ModalBackend

        print(f"Initializing Modal backend with app name: {args.app_name}")
        backend = await ModalBackend.initialize_cluster(
            app_name=args.app_name,
            gpu_type=args.gpu_type,
            gpu_count=1,
            memory=32000,
            timeout=3600,
            keep_warm=0,
            volume_name="art-volume",
            environment="main",
            env_file=".env",
            force_rebuild=args.force_rebuild,
        )
        print("Modal backend initialized successfully")
    else:
        # Import Local backend only when needed
        from art.local.backend import LocalBackend

        print("Using local backend")
        backend = LocalBackend()

    # Create the trainable model
    model = art.TrainableModel(
        name="temporal-clue-001",
        project="temporal-clue",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        _internal_config={"init_args": {"gpu_memory_utilization": 0.775}},
    )

    # For Modal backend, we typically don't pull from S3 initially
    # For local backend, we can try to pull existing state
    if args.backend == "local":
        try:
            await backend._experimental_pull_from_s3(model)
            print("Pulled model state from S3")
        except Exception as e:
            print(f"Could not pull from S3 (this is normal for new models): {e}")

    print("Registering model with backend...")
    await model.register(backend)
    print("Model registered successfully")

    print("Starting training loop...")
    stride = 4

    try:
        for i in range(await model.get_step(), 1_000):
            print(f"Training step {i}")

            # Gather validation and training trajectories
            val_groups, train_groups = await asyncio.gather(
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(rollout(model, puzzle) for _ in range(2))
                        for puzzle in val_puzzles
                    ),
                    pbar_desc="val",
                ),
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(rollout(model, puzzle) for _ in range(50))
                        for puzzle in train_puzzles[i * stride : (i + 1) * stride]
                    ),
                    pbar_desc="train",
                ),
            )

            # Log validation results
            await model.log(val_groups)

            # Clean up old checkpoints to save space
            await model.delete_checkpoints()

            # Push model state to S3 for persistence
            try:
                # FEEDBACK: we shouldn't need to do this for Modal. if checkpointing isn't already happening in the Volume, we need to fix that (by mounting the checkpointing volume wherever the existing checkpoint logic in the various concrete ModelService subtypes put them)
                await backend._experimental_push_to_s3(model)
                print("Model state pushed to S3")
            except Exception as e:
                print(f"Could not push to S3: {e}")

            # Train the model on the gathered trajectories
            await model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=5e-5),
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        # Clean up Modal resources if using Modal backend
        if args.backend == "modal":
            print("Cleaning up Modal backend...")
            try:
                await backend.down()
                print("Modal backend cleanup completed")
            except Exception as e:
                print(f"Error during Modal cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())
