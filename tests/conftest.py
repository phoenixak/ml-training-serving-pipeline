"""Shared fixtures for ML pipeline test suite."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from training.models.simple_dlrm import DLRMConfig, SimpleDLRM, create_model_from_mappings
from shared.config.serving_config import ServingConfig


@pytest.fixture
def sample_mappings() -> Dict[str, Any]:
    """Mapping dicts matching what DataProcessor / AmazonDatasetProcessor produce.

    Keys mirror the structure written by AmazonDatasetProcessor.save_processed_data:
    user_mapping, item_mapping (str -> int), plus num_users / num_items counts.
    """
    user_mapping = {f"user_{i}": i for i in range(50)}
    item_mapping = {f"item_{i}": i for i in range(100)}
    return {
        "user_mapping": user_mapping,
        "item_mapping": item_mapping,
        "num_users": len(user_mapping),
        "num_items": len(item_mapping),
    }


@pytest.fixture
def model_config() -> DLRMConfig:
    """A small DLRMConfig suitable for fast tests."""
    return DLRMConfig(
        num_users=50,
        num_items=100,
        embedding_dim=16,
        dense_feature_dim=6,
        bottom_mlp_dims=(32, 16),
        top_mlp_dims=(32, 1),
        dropout_rate=0.0,
    )


@pytest.fixture
def trained_model(model_config: DLRMConfig) -> SimpleDLRM:
    """A SimpleDLRM instance with random (Xavier-initialized) weights."""
    model = SimpleDLRM(config=model_config)
    model.eval()
    return model


@pytest.fixture
def temp_model_dir(
    tmp_path: Path,
    sample_mappings: Dict[str, Any],
) -> Path:
    """Temporary directory containing a saved model checkpoint and mappings JSON.

    Layout matches what DLRMModelLoader.load_from_training_output expects:
      - amazon_dlrm_model.pth
      - model_info.json
      - data_mappings.json

    The model is built using only num_users/num_items/embedding_dim kwargs
    (with default MLP dims), matching the architecture that
    DLRMModelLoader.load_from_training_output will reconstruct.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    num_users = sample_mappings["num_users"]
    num_items = sample_mappings["num_items"]
    embedding_dim = 16

    # Build the model the same way the loader does: kwargs only, default MLP dims
    loader_model = SimpleDLRM(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
    )

    # Save model weights
    torch.save(loader_model.state_dict(), model_dir / "amazon_dlrm_model.pth")

    # Save model info (matches schema used in train_dlrm.py)
    model_info = {
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": embedding_dim,
        "total_parameters": loader_model.count_parameters(),
        "final_loss": 0.42,
        "epochs_trained": 5,
        "model_type": "SimpleDLRM",
    }
    with open(model_dir / "model_info.json", "w") as f:
        json.dump(model_info, f)

    # Save data mappings
    data_mappings = {
        "user_mapping": sample_mappings["user_mapping"],
        "item_mapping": sample_mappings["item_mapping"],
    }
    with open(model_dir / "data_mappings.json", "w") as f:
        json.dump(data_mappings, f)

    return model_dir


@pytest.fixture
def sample_interaction() -> Dict[str, Any]:
    """A sample interaction dict matching what DLRMModelLoader.predict_single expects.

    Keys: user_id (str), item_id (str), context (dict of dense feature overrides).
    """
    return {
        "user_id": "user_0",
        "item_id": "item_0",
        "context": {
            "verified": 1,
            "review_length": 150,
            "has_summary": 1,
            "hour": 14,
            "day_of_week": 3,
            "month": 7,
        },
    }
