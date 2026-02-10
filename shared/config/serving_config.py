#!/usr/bin/env python3
"""Serving configuration for model API."""

import os
from dataclasses import dataclass


@dataclass
class ServingConfig:
    """Configuration for model serving."""

    # Model settings
    model_dir: str = "models"
    model_file: str = "amazon_dlrm_model.pth"
    device: str = "cpu"

    # API settings
    host: str = "0.0.0.0"
    port: int = 3000
    workers: int = 1

    # Cache settings
    cache_ttl: int = 300  # 5 minutes
    cache_size: int = 1000

    # Performance settings
    timeout: int = 30
    max_batch_size: int = 100

    # Fallback settings
    enable_fallback: bool = True
    fallback_cache_ttl: int = 600  # 10 minutes

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.port <= 0:
            raise ValueError(f"port must be positive, got {self.port}")
        if self.workers <= 0:
            raise ValueError(f"workers must be positive, got {self.workers}")

    def get_model_path(self) -> str:
        """Get full path to model file."""
        return os.path.join(self.model_dir, self.model_file)

    def get_model_info_path(self) -> str:
        """Get full path to model info file."""
        return os.path.join(self.model_dir, "model_info.json")

    def get_mappings_path(self) -> str:
        """Get full path to data mappings file."""
        return os.path.join(self.model_dir, "data_mappings.json")
