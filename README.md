# ML Training and Serving Pipeline

End-to-end machine learning pipeline for training and serving a DLRM recommendation model.

## Overview

This project implements a recommendation system using a simplified Deep Learning Recommendation Model (DLRM) trained on synthetic Amazon Electronics review data. It covers the full lifecycle: data generation, model training with PyTorch, artifact management, and a model loading/inference interface. An optional Azure ML integration supports submitting training jobs to the cloud with spot instance cost optimization.

## Architecture

The codebase is organized into three layers:

- **training/** -- Data processing, DLRM model definition, and training loop. Supports local execution and Azure ML job submission.
- **serving/** -- Model loader that reads training artifacts and exposes single prediction, batch prediction, and top-k recommendation methods.
- **shared/** -- Configuration dataclasses for training, serving, and Azure settings, plus data processing and logging utilities shared across layers.

## Project Structure

```
ml-training-serving-pipeline/
├── training/
│   ├── train_dlrm.py              # Main training entry point
│   ├── models/
│   │   └── simple_dlrm.py         # SimpleDLRM model definition
│   └── azure_integration/
│       └── submit_job.py          # Azure ML job submission
├── serving/
│   └── model_loader.py            # Model loading and prediction API
├── shared/
│   ├── config/
│   │   ├── training_config.py     # DLRMTrainingConfig dataclass
│   │   ├── serving_config.py      # ServingConfig dataclass
│   │   └── azure_config.py        # AzureConfig dataclass
│   └── utils/
│       ├── data_processing.py     # Dataset generation and processing
│       └── logging_utils.py       # Structured logging setup
├── examples/
│   └── end_to_end_demo.py         # Full train-load-predict demo
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_data_processing.py
│   ├── test_model.py
│   └── test_model_loader.py
├── docs/
│   └── azure_setup.md             # Azure ML setup guide
├── models/                        # Generated model artifacts (gitignored)
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
├── .env.example
└── LICENSE
```

## Setup

### Prerequisites

- Python 3.10 or later
- PyTorch 2.0 or later

### Installation

Install from `pyproject.toml` (recommended):

```bash
pip install -e .
```

Or install from the pinned requirements file:

```bash
pip install -r requirements.txt
```

For development tooling (pytest, black, ruff, mypy):

```bash
pip install -e ".[dev]"
```

For Azure ML support:

```bash
pip install -e ".[azure]"
```

### Configuration

Copy `.env.example` to `.env` and edit as needed. Key variables:

| Variable | Description | Default |
| --- | --- | --- |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription (optional, for cloud training) | -- |
| `AZURE_RESOURCE_GROUP` | Azure resource group | -- |
| `AZURE_WORKSPACE_NAME` | Azure ML workspace name | -- |
| `TRAINING_EPOCHS` | Number of training epochs | 20 |
| `TRAINING_BATCH_SIZE` | Training batch size | 256 |
| `TRAINING_SAMPLES` | Number of training samples | 10000 |
| `MODEL_DIR` | Output directory for model artifacts | `models` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

## Usage

### Training

Run local training via the CLI:

```bash
python training/train_dlrm.py --local --samples 10000 --epochs 5
```

This generates synthetic review data, trains a SimpleDLRM model, and writes artifacts to the `models/` directory:

- `amazon_dlrm_model.pth` -- serialized model weights
- `model_info.json` -- model metadata (parameter count, loss, architecture)
- `data_mappings.json` -- user/item ID-to-index mappings

Makefile shortcuts:

```bash
make train-local          # 10K samples, 5 epochs
make train-local-full     # 200K samples, 20 epochs
```

#### Azure ML Training (Optional)

Set the Azure environment variables (see Configuration above), then:

```bash
python training/train_dlrm.py --azure --samples 100000 --epochs 10
```

Or use the Makefile targets:

```bash
make train-azure-small    # 100K samples, 10 epochs
make train-azure-large    # 1M samples, 30 epochs
```

The Azure integration selects an instance type based on dataset size and enables spot instances by default for cost savings. See `docs/azure_setup.md` for detailed setup instructions.

### Serving / Inference

Load a trained model and generate predictions:

```python
from serving.model_loader import DLRMModelLoader

loader = DLRMModelLoader()
loader.load_from_training_output("models")

# Single prediction
score = loader.predict_single("user_id", "item_id", {"verified": 1})

# Batch predictions
scores = loader.predict_batch(
    ["user_a", "user_b"],
    ["item_x", "item_y"],
)

# Top-k recommendations
recs = loader.recommend_items("user_id", num_recommendations=10)
```

Unknown user or item IDs are handled with a cold-start fallback (returns 0.5 for predictions, or a default popular-item list for recommendations).

You can also run the model loader directly to verify that artifacts load correctly:

```bash
python -m serving.model_loader
```

### End-to-End Demo

Run the complete train-then-serve workflow:

```bash
python examples/end_to_end_demo.py
```

This trains a small model (5K samples, 3 epochs), loads it, and runs sample predictions and recommendations. A `--quick` flag skips training and tests an already-trained model:

```bash
python examples/end_to_end_demo.py --quick
```

## Docker

The Dockerfile defines two build stages: `training` and `serving`.

Build and run training:

```bash
docker build --target training -t dlrm-training .
docker run dlrm-training
```

Build and run serving:

```bash
docker build --target serving -t dlrm-serving .
docker run -p 8000:8000 dlrm-serving
```

## Testing

Run the test suite:

```bash
make test
```

Run with coverage:

```bash
make test-cov
```

Other development commands:

```bash
make format        # Format code with black
make lint          # Lint with ruff
make type-check    # Type-check with mypy
make clean         # Remove generated files and caches
```

## License

MIT -- see [LICENSE](LICENSE).
