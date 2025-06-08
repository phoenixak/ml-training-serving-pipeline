"""Configuration management for training and serving."""

from .training_config import DLRMTrainingConfig
from .serving_config import ServingConfig

__all__ = ["DLRMTrainingConfig", "ServingConfig"]