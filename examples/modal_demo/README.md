# Modal Backend Temporal Clue Example

This example demonstrates how to use the ModalBackend with ART to train a model on temporal clue puzzles. It's based on the temporal-clue example but adapted to show Modal integration patterns.

## Features

- **Backend Selection**: Choose between local and Modal backends via command line
- **Modal Integration**: Demonstrates proper ModalBackend initialization using the factory method
- **Error Handling**: Includes robust error handling for Modal deployment and S3 operations
- **Resource Management**: Proper cleanup of Modal resources after training
- **Configurable GPU**: Support for different GPU types (A10G, A100, H100, etc.)

## Prerequisites

1. **Modal Account**: Set up a Modal account and install the Modal CLI
2. **Environment Variables**: Configure your `.env` file with required API keys
3. **Dependencies**: Install ART with Modal dependencies

## Usage

### Local Backend (Default)
```bash
python modal_temporal_clue.py
```

### Modal Backend
```bash
python modal_temporal_clue.py --backend modal
```

### Custom Modal Configuration
```bash
python modal_temporal_clue.py --backend modal --app-name my-temporal-clue --gpu-type A100 --force-rebuild
```

## Command Line Options

- `--backend`: Choose backend (`modal` or `local`, default: `local`)
- `--app-name`: Modal app name (default: `art-temporal-clue`)
- `--gpu-type`: GPU type for Modal (default: `A10G`)
- `--force-rebuild`: Force rebuild of Modal app

## Key Modal Features Demonstrated

1. **Factory Method Initialization**: Uses `ModalBackend.initialize_cluster()` class method
2. **Conditional Imports**: Backend dependencies are imported only when needed
3. **App Configuration**: Configurable GPU types, memory, and timeout settings
4. **Health Checking**: Waits for Modal app to be ready before proceeding
5. **Resource Cleanup**: Proper cleanup of Modal resources on exit
6. **Error Handling**: Graceful handling of deployment and runtime errors

## Architecture

The example follows ART's established patterns:

- **Model Definition**: Creates a `TrainableModel` with appropriate configuration
- **Rollout Function**: Defines how to evaluate the model on temporal clue puzzles
- **Training Loop**: Gathers trajectories, logs metrics, and trains the model
- **State Persistence**: Uses S3 for model state persistence across runs

## Modal-Specific Considerations

- **Cold Starts**: Modal functions may have cold start delays
- **Volume Persistence**: Uses Modal volumes for persistent storage
- **Environment Variables**: Automatically loads environment variables from `.env`
- **GPU Selection**: Different GPU types available based on Modal's offerings
- **Concurrent Limits**: Modal has concurrent execution limits to be aware of

## Troubleshooting

- **Modal Deployment Issues**: Check Modal credentials and quota limits
- **GPU Availability**: Some GPU types may not be available in all regions
- **Memory Errors**: Adjust memory settings in the `initialize_cluster()` call
- **Timeout Issues**: Increase timeout values for long-running operations
