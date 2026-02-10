"""Tests for DLRMModelLoader serving predictions."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from serving.model_loader import DLRMModelLoader
from shared.config.serving_config import ServingConfig


class TestDLRMModelLoaderInit:
    """Tests for DLRMModelLoader instantiation."""

    def test_default_instantiation(self) -> None:
        """DLRMModelLoader should initialize with default ServingConfig."""
        loader = DLRMModelLoader()
        assert loader.config is not None
        assert isinstance(loader.config, ServingConfig)
        assert loader.model is None
        assert loader.model_info is None
        assert loader.user_mapping == {}
        assert loader.item_mapping == {}

    def test_custom_config(self) -> None:
        """DLRMModelLoader should accept a custom ServingConfig."""
        cfg = ServingConfig(port=9090, device="cpu")
        loader = DLRMModelLoader(config=cfg)
        assert loader.config.port == 9090
        assert loader.device == "cpu"

    def test_is_loaded_false_before_loading(self) -> None:
        """is_loaded should return False when no model is loaded."""
        loader = DLRMModelLoader()
        assert loader.is_loaded() is False

    def test_get_model_info_before_loading(self) -> None:
        """get_model_info should return a no-model status when nothing is loaded."""
        loader = DLRMModelLoader()
        info = loader.get_model_info()
        assert info["status"] == "no_model_loaded"


class TestDLRMModelLoaderLoad:
    """Tests for loading a model from a temp directory."""

    def test_load_from_training_output_success(self, temp_model_dir: Path) -> None:
        """load_from_training_output should return True with valid artifacts."""
        loader = DLRMModelLoader()
        success = loader.load_from_training_output(str(temp_model_dir))
        assert success is True
        assert loader.is_loaded() is True

    def test_load_sets_model_info(self, temp_model_dir: Path) -> None:
        """After loading, model_info should contain expected keys."""
        loader = DLRMModelLoader()
        loader.load_from_training_output(str(temp_model_dir))
        info = loader.get_model_info()

        assert info["status"] == "loaded"
        assert info["model_type"] == "SimpleDLRM"
        assert "num_users" in info
        assert "num_items" in info
        assert "embedding_dim" in info
        assert "total_parameters" in info
        assert "final_loss" in info
        assert "epochs_trained" in info

    def test_load_populates_mappings(
        self, temp_model_dir: Path, sample_mappings: Dict[str, Any]
    ) -> None:
        """After loading, user_mapping and item_mapping should be populated."""
        loader = DLRMModelLoader()
        loader.load_from_training_output(str(temp_model_dir))

        assert len(loader.user_mapping) == sample_mappings["num_users"]
        assert len(loader.item_mapping) == sample_mappings["num_items"]

    def test_load_creates_reverse_mappings(self, temp_model_dir: Path) -> None:
        """After loading, reverse mappings (int -> str) should be populated."""
        loader = DLRMModelLoader()
        loader.load_from_training_output(str(temp_model_dir))

        assert len(loader.reverse_user_mapping) == len(loader.user_mapping)
        assert len(loader.reverse_item_mapping) == len(loader.item_mapping)

    def test_load_missing_directory(self, tmp_path: Path) -> None:
        """load_from_training_output should return False for a missing directory."""
        loader = DLRMModelLoader()
        success = loader.load_from_training_output(str(tmp_path / "nonexistent"))
        assert success is False
        assert loader.is_loaded() is False

    def test_load_incomplete_artifacts(self, tmp_path: Path) -> None:
        """load_from_training_output should return False if files are missing."""
        incomplete_dir = tmp_path / "incomplete"
        incomplete_dir.mkdir()
        # Only write one of the three required files
        with open(incomplete_dir / "model_info.json", "w") as f:
            json.dump({"num_users": 1}, f)

        loader = DLRMModelLoader()
        success = loader.load_from_training_output(str(incomplete_dir))
        assert success is False


class TestDLRMModelLoaderPredict:
    """Tests for prediction methods."""

    @pytest.fixture(autouse=True)
    def _loaded_loader(self, temp_model_dir: Path) -> None:
        """Load the model before each test in this class."""
        self.loader = DLRMModelLoader()
        self.loader.load_from_training_output(str(temp_model_dir))

    def test_predict_single_returns_float(self, sample_interaction: Dict[str, Any]) -> None:
        """predict_single should return a Python float."""
        result = self.loader.predict_single(
            user_id=sample_interaction["user_id"],
            item_id=sample_interaction["item_id"],
            context=sample_interaction["context"],
        )
        assert isinstance(result, float)

    def test_predict_single_in_range(self, sample_interaction: Dict[str, Any]) -> None:
        """predict_single result should be in [0, 1]."""
        result = self.loader.predict_single(
            user_id=sample_interaction["user_id"],
            item_id=sample_interaction["item_id"],
            context=sample_interaction["context"],
        )
        assert 0.0 <= result <= 1.0

    def test_predict_single_cold_start_unknown_user(self) -> None:
        """Unknown user_id should return 0.5 (cold-start fallback)."""
        result = self.loader.predict_single(
            user_id="completely_unknown_user",
            item_id="item_0",
        )
        assert result == 0.5

    def test_predict_single_cold_start_unknown_item(self) -> None:
        """Unknown item_id should return 0.5 (cold-start fallback)."""
        result = self.loader.predict_single(
            user_id="user_0",
            item_id="completely_unknown_item",
        )
        assert result == 0.5

    def test_predict_single_cold_start_both_unknown(self) -> None:
        """Both unknown user and item should return 0.5."""
        result = self.loader.predict_single(
            user_id="no_such_user",
            item_id="no_such_item",
        )
        assert result == 0.5

    def test_predict_single_with_known_ids(self) -> None:
        """Known user/item IDs should produce a model prediction (not cold-start)."""
        result = self.loader.predict_single(
            user_id="user_0",
            item_id="item_0",
        )
        # Should be a real model prediction, not necessarily 0.5
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_single_no_context(self) -> None:
        """predict_single should work without a context argument."""
        result = self.loader.predict_single(user_id="user_1", item_id="item_1")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_batch_correct_length(self) -> None:
        """predict_batch should return a list with the same length as input."""
        user_ids = ["user_0", "user_1", "user_2"]
        item_ids = ["item_0", "item_1", "item_2"]

        results = self.loader.predict_batch(user_ids, item_ids)

        assert isinstance(results, list)
        assert len(results) == 3

    def test_predict_batch_values_in_range(self) -> None:
        """All batch predictions should be in [0, 1]."""
        user_ids = ["user_0", "user_1", "user_2", "user_3"]
        item_ids = ["item_0", "item_1", "item_2", "item_3"]

        results = self.loader.predict_batch(user_ids, item_ids)

        for score in results:
            assert 0.0 <= score <= 1.0

    def test_predict_batch_mixed_cold_start(self) -> None:
        """Batch with mixed known/unknown IDs should handle cold start per-item."""
        user_ids = ["user_0", "unknown_user"]
        item_ids = ["item_0", "unknown_item"]

        results = self.loader.predict_batch(user_ids, item_ids)

        assert len(results) == 2
        # Second pair is cold-start
        assert results[1] == 0.5
        # First pair is a real prediction
        assert 0.0 <= results[0] <= 1.0

    def test_predict_batch_all_cold_start(self) -> None:
        """Batch where all pairs are cold-start should return all 0.5."""
        user_ids = ["unknown_a", "unknown_b"]
        item_ids = ["unknown_x", "unknown_y"]

        results = self.loader.predict_batch(user_ids, item_ids)

        assert results == [0.5, 0.5]

    def test_predict_batch_mismatched_lengths(self) -> None:
        """Mismatched user_ids and item_ids lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            self.loader.predict_batch(["user_0"], ["item_0", "item_1"])


class TestDLRMModelLoaderNotLoaded:
    """Tests for error handling when model is not loaded."""

    def test_predict_single_raises_without_load(self) -> None:
        """predict_single should raise ValueError if model is not loaded."""
        loader = DLRMModelLoader()
        with pytest.raises(ValueError, match="Model not loaded"):
            loader.predict_single("user_0", "item_0")

    def test_predict_batch_raises_without_load(self) -> None:
        """predict_batch should raise ValueError if model is not loaded."""
        loader = DLRMModelLoader()
        with pytest.raises(ValueError, match="Model not loaded"):
            loader.predict_batch(["user_0"], ["item_0"])

    def test_recommend_items_raises_without_load(self) -> None:
        """recommend_items should raise ValueError if model is not loaded."""
        loader = DLRMModelLoader()
        with pytest.raises(ValueError, match="Model not loaded"):
            loader.recommend_items("user_0")
