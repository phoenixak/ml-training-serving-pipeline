"""Tests for DataProcessor and AmazonDatasetProcessor.

All tests avoid network access; data generation methods create synthetic data locally.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from shared.utils.data_processing import AmazonDatasetProcessor, DataProcessor


# ---------------------------------------------------------------------------
# DataProcessor tests
# ---------------------------------------------------------------------------


class TestDataProcessor:
    """Tests for the simplified DataProcessor class."""

    def test_instantiation_default_temp_dir(self) -> None:
        """DataProcessor should create a temp directory when none is provided."""
        processor = DataProcessor()
        assert processor.temp_dir.exists()

    def test_instantiation_custom_temp_dir(self, tmp_path: Path) -> None:
        """DataProcessor should use the specified temp directory."""
        custom_dir = tmp_path / "custom_data"
        custom_dir.mkdir()
        processor = DataProcessor(temp_dir=str(custom_dir))
        assert processor.temp_dir == custom_dir

    def test_download_method_is_callable(self) -> None:
        """download_amazon_sample should be a callable method."""
        processor = DataProcessor()
        assert callable(processor.download_amazon_sample)

    def test_download_returns_tuple_of_three(self, tmp_path: Path) -> None:
        """download_amazon_sample should return (reviews, user_mapping, item_mapping)."""
        processor = DataProcessor(temp_dir=str(tmp_path))
        result = processor.download_amazon_sample(category="Test", max_reviews=50)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_download_reviews_structure(self, tmp_path: Path) -> None:
        """Each review should have the expected keys."""
        processor = DataProcessor(temp_dir=str(tmp_path))
        reviews, _, _ = processor.download_amazon_sample(category="Test", max_reviews=10)

        assert len(reviews) == 10
        expected_keys = {"user_id", "item_id", "rating", "timestamp", "verified", "review_length"}
        for review in reviews:
            assert expected_keys.issubset(review.keys()), f"Missing keys in review: {review.keys()}"

    def test_download_mapping_structure(self, tmp_path: Path) -> None:
        """user_mapping and item_mapping should be str->int dicts."""
        processor = DataProcessor(temp_dir=str(tmp_path))
        _, user_mapping, item_mapping = processor.download_amazon_sample(
            category="Test", max_reviews=20
        )

        assert isinstance(user_mapping, dict)
        assert isinstance(item_mapping, dict)
        assert len(user_mapping) > 0
        assert len(item_mapping) > 0

        # All keys should be strings, all values ints
        for key, val in user_mapping.items():
            assert isinstance(key, str)
            assert isinstance(val, int)

    def test_download_mapping_values_are_contiguous(self, tmp_path: Path) -> None:
        """Mapping indices should be contiguous starting from 0."""
        processor = DataProcessor(temp_dir=str(tmp_path))
        _, user_mapping, _ = processor.download_amazon_sample(category="Test", max_reviews=100)

        indices = sorted(user_mapping.values())
        assert indices == list(range(len(indices)))


# ---------------------------------------------------------------------------
# AmazonDatasetProcessor tests
# ---------------------------------------------------------------------------


class TestAmazonDatasetProcessor:
    """Tests for the full AmazonDatasetProcessor class."""

    def test_instantiation(self, tmp_path: Path) -> None:
        """AmazonDatasetProcessor should initialize with temp_dir and storage limit."""
        processor = AmazonDatasetProcessor(temp_dir=str(tmp_path), max_storage_gb=1)
        assert processor.temp_dir == tmp_path
        assert processor.max_storage_bytes == 1 * 1024**3

    def test_download_returns_path(self, tmp_path: Path) -> None:
        """download_amazon_sample should return a Path to a compressed file."""
        processor = AmazonDatasetProcessor(temp_dir=str(tmp_path), max_storage_gb=1)
        result = processor.download_amazon_sample(category="TestCat", max_reviews=20)

        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == ".gz"

    def test_process_reviews_streaming(self, tmp_path: Path) -> None:
        """process_reviews_streaming should return (reviews, user_map, item_map)."""
        processor = AmazonDatasetProcessor(temp_dir=str(tmp_path), max_storage_gb=1)
        sample_file = processor.download_amazon_sample(category="Books", max_reviews=50)

        reviews, user_map, item_map = processor.process_reviews_streaming(
            sample_file, chunk_size=10
        )

        assert len(reviews) == 50
        assert len(user_map) > 0
        assert len(item_map) > 0

        # Processed reviews should have enriched features
        expected_keys = {
            "user_idx",
            "item_idx",
            "rating",
            "timestamp",
            "verified",
            "review_length",
            "has_summary",
            "hour",
            "day_of_week",
            "month",
            "rating_normalized",
            "log_review_length",
        }
        for review in reviews[:5]:
            assert expected_keys.issubset(review.keys())

    def test_cleanup_removes_temp_dir(self, tmp_path: Path) -> None:
        """cleanup should remove the temporary directory."""
        temp_dir = tmp_path / "to_clean"
        temp_dir.mkdir()
        processor = AmazonDatasetProcessor(temp_dir=str(temp_dir), max_storage_gb=1)
        processor.cleanup()

        assert not temp_dir.exists()
