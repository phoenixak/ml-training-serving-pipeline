"""Tests for configuration dataclasses."""

import os

import pytest

from shared.config.training_config import DLRMTrainingConfig, TrainingConfig
from shared.config.serving_config import ServingConfig
from shared.config.azure_config import AzureConfig


# ---------------------------------------------------------------------------
# DLRMTrainingConfig / TrainingConfig tests
# ---------------------------------------------------------------------------


class TestDLRMTrainingConfig:
    """Tests for the training configuration dataclass."""

    def test_defaults(self) -> None:
        """DLRMTrainingConfig should have documented default values."""
        cfg = DLRMTrainingConfig()
        assert cfg.embedding_dim == 64
        assert cfg.dense_features == 6
        assert cfg.epochs == 20
        assert cfg.batch_size == 1024
        assert cfg.learning_rate == 0.001
        assert cfg.max_samples == 200000
        assert cfg.category == "Electronics"
        assert cfg.max_storage_gb == 5
        assert cfg.output_dir == "models"
        assert cfg.model_name == "amazon_dlrm_model.pth"
        assert cfg.use_azure is False
        assert cfg.compute_target == "cpu-cluster"
        assert cfg.experiment_name == "dlrm-training"

    def test_custom_values(self) -> None:
        """DLRMTrainingConfig should accept overridden values."""
        cfg = DLRMTrainingConfig(
            embedding_dim=32,
            epochs=5,
            batch_size=256,
            learning_rate=0.01,
            max_samples=1000,
            category="Books",
        )
        assert cfg.embedding_dim == 32
        assert cfg.epochs == 5
        assert cfg.batch_size == 256
        assert cfg.learning_rate == 0.01
        assert cfg.max_samples == 1000
        assert cfg.category == "Books"

    def test_alias_is_same_class(self) -> None:
        """TrainingConfig alias should be the same class as DLRMTrainingConfig."""
        assert TrainingConfig is DLRMTrainingConfig

    def test_validation_epochs_negative(self) -> None:
        """Negative epochs should raise ValueError."""
        with pytest.raises(ValueError, match="epochs must be non-negative"):
            DLRMTrainingConfig(epochs=-1)

    def test_validation_epochs_zero_is_valid(self) -> None:
        """Zero epochs should be accepted (non-negative)."""
        cfg = DLRMTrainingConfig(epochs=0)
        assert cfg.epochs == 0

    def test_validation_batch_size_zero(self) -> None:
        """Zero batch_size should raise ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DLRMTrainingConfig(batch_size=0)

    def test_validation_batch_size_negative(self) -> None:
        """Negative batch_size should raise ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DLRMTrainingConfig(batch_size=-10)

    def test_validation_learning_rate_zero(self) -> None:
        """Zero learning_rate should raise ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            DLRMTrainingConfig(learning_rate=0.0)

    def test_validation_learning_rate_negative(self) -> None:
        """Negative learning_rate should raise ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            DLRMTrainingConfig(learning_rate=-0.01)

    def test_get_model_path(self) -> None:
        """get_model_path should join output_dir and model_name."""
        cfg = DLRMTrainingConfig(output_dir="out", model_name="my_model.pth")
        assert cfg.get_model_path() == os.path.join("out", "my_model.pth")

    def test_get_model_info_path(self) -> None:
        """get_model_info_path should point to model_info.json in output_dir."""
        cfg = DLRMTrainingConfig(output_dir="results")
        assert cfg.get_model_info_path() == os.path.join("results", "model_info.json")

    def test_get_mappings_path(self) -> None:
        """get_mappings_path should point to data_mappings.json in output_dir."""
        cfg = DLRMTrainingConfig(output_dir="results")
        assert cfg.get_mappings_path() == os.path.join("results", "data_mappings.json")


# ---------------------------------------------------------------------------
# ServingConfig tests
# ---------------------------------------------------------------------------


class TestServingConfig:
    """Tests for the serving configuration dataclass."""

    def test_defaults(self) -> None:
        """ServingConfig should have documented default values."""
        cfg = ServingConfig()
        assert cfg.model_dir == "models"
        assert cfg.model_file == "amazon_dlrm_model.pth"
        assert cfg.device == "cpu"
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 3000
        assert cfg.workers == 1
        assert cfg.cache_ttl == 300
        assert cfg.cache_size == 1000
        assert cfg.timeout == 30
        assert cfg.max_batch_size == 100
        assert cfg.enable_fallback is True
        assert cfg.fallback_cache_ttl == 600

    def test_custom_values(self) -> None:
        """ServingConfig should accept overridden values."""
        cfg = ServingConfig(
            model_dir="/tmp/serving",
            port=8080,
            workers=4,
            device="cuda",
            max_batch_size=50,
        )
        assert cfg.model_dir == "/tmp/serving"
        assert cfg.port == 8080
        assert cfg.workers == 4
        assert cfg.device == "cuda"
        assert cfg.max_batch_size == 50

    def test_validation_port_zero(self) -> None:
        """Zero port should raise ValueError."""
        with pytest.raises(ValueError, match="port must be positive"):
            ServingConfig(port=0)

    def test_validation_port_negative(self) -> None:
        """Negative port should raise ValueError."""
        with pytest.raises(ValueError, match="port must be positive"):
            ServingConfig(port=-1)

    def test_validation_workers_zero(self) -> None:
        """Zero workers should raise ValueError."""
        with pytest.raises(ValueError, match="workers must be positive"):
            ServingConfig(workers=0)

    def test_validation_workers_negative(self) -> None:
        """Negative workers should raise ValueError."""
        with pytest.raises(ValueError, match="workers must be positive"):
            ServingConfig(workers=-2)

    def test_get_model_path(self) -> None:
        """get_model_path should join model_dir and model_file."""
        cfg = ServingConfig(model_dir="serve", model_file="best.pth")
        assert cfg.get_model_path() == os.path.join("serve", "best.pth")

    def test_get_model_info_path(self) -> None:
        """get_model_info_path should point to model_info.json in model_dir."""
        cfg = ServingConfig(model_dir="serve")
        assert cfg.get_model_info_path() == os.path.join("serve", "model_info.json")

    def test_get_mappings_path(self) -> None:
        """get_mappings_path should point to data_mappings.json in model_dir."""
        cfg = ServingConfig(model_dir="serve")
        assert cfg.get_mappings_path() == os.path.join("serve", "data_mappings.json")


# ---------------------------------------------------------------------------
# AzureConfig tests
# ---------------------------------------------------------------------------


class TestAzureConfig:
    """Tests for the Azure ML configuration dataclass."""

    def test_defaults(self) -> None:
        """AzureConfig should have documented default values."""
        cfg = AzureConfig()
        assert cfg.subscription_id is None
        assert cfg.resource_group is None
        assert cfg.workspace_name is None
        assert cfg.compute_target == "cpu-cluster"
        assert cfg.use_spot_instances is True
        assert cfg.max_duration_hours == 4
        assert isinstance(cfg.instance_types, dict)
        assert "small" in cfg.instance_types
        assert "medium" in cfg.instance_types
        assert "large" in cfg.instance_types
        assert "xlarge" in cfg.instance_types

    def test_get_instance_for_samples_small(self) -> None:
        """Fewer than 50k samples should map to the 'small' instance type."""
        cfg = AzureConfig()
        assert cfg.get_instance_for_samples(10_000) == cfg.instance_types["small"]

    def test_get_instance_for_samples_medium(self) -> None:
        """50k-500k samples should map to the 'medium' instance type."""
        cfg = AzureConfig()
        assert cfg.get_instance_for_samples(100_000) == cfg.instance_types["medium"]

    def test_get_instance_for_samples_large(self) -> None:
        """500k-2M samples should map to the 'large' instance type."""
        cfg = AzureConfig()
        assert cfg.get_instance_for_samples(1_000_000) == cfg.instance_types["large"]

    def test_get_instance_for_samples_xlarge(self) -> None:
        """2M+ samples should map to the 'xlarge' instance type."""
        cfg = AzureConfig()
        assert cfg.get_instance_for_samples(5_000_000) == cfg.instance_types["xlarge"]

    def test_get_duration_for_samples_minimum(self) -> None:
        """Duration should be at least 1 hour."""
        cfg = AzureConfig()
        duration = cfg.get_duration_for_samples(100, 1)
        assert duration >= 1

    def test_get_duration_for_samples_capped(self) -> None:
        """Duration should be capped at 12 hours."""
        cfg = AzureConfig()
        duration = cfg.get_duration_for_samples(10_000_000, 100)
        assert duration <= 12

    def test_get_duration_scales_with_samples(self) -> None:
        """More samples and epochs should produce a longer (or equal) duration."""
        cfg = AzureConfig()
        short = cfg.get_duration_for_samples(10_000, 5)
        long = cfg.get_duration_for_samples(500_000, 20)
        assert long >= short
