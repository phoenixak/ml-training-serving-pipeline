#!/usr/bin/env python3
"""Training configuration for DLRM model."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DLRMTrainingConfig:
    """Configuration for DLRM model training."""
    
    # Model architecture
    embedding_dim: int = 64
    dense_features: int = 6
    
    # Training parameters
    epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 0.001
    
    # Data parameters
    max_samples: int = 200000
    category: str = "Electronics"
    max_storage_gb: int = 5
    
    # Output parameters
    output_dir: str = "models"
    model_name: str = "amazon_dlrm_model.pth"
    
    # Azure configuration (optional)
    use_azure: bool = False
    compute_target: str = "cpu-cluster"
    experiment_name: str = "dlrm-training"
    azure_subscription_id: Optional[str] = None
    azure_resource_group: Optional[str] = None
    azure_workspace_name: Optional[str] = None
    
    def get_model_path(self) -> str:
        """Get full path to model file."""
        return f"{self.output_dir}/{self.model_name}"
    
    def get_model_info_path(self) -> str:
        """Get full path to model info file."""
        return f"{self.output_dir}/model_info.json"
    
    def get_mappings_path(self) -> str:
        """Get full path to data mappings file."""
        return f"{self.output_dir}/data_mappings.json"