#!/usr/bin/env python3
"""Azure ML configuration for cloud training."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AzureConfig:
    """Configuration for Azure ML integration."""
    
    # Required Azure settings (from environment variables)
    subscription_id: Optional[str] = None  # AZURE_SUBSCRIPTION_ID
    resource_group: Optional[str] = None   # AZURE_RESOURCE_GROUP  
    workspace_name: Optional[str] = None   # AZURE_WORKSPACE_NAME
    
    # Compute settings
    compute_target: str = "cpu-cluster"
    
    # Cost optimization
    use_spot_instances: bool = True
    max_duration_hours: int = 4
    
    # Instance types based on workload
    instance_types = {
        "small": "Standard_F2s_v2",    # 2 vCPUs, 4 GB RAM (~$0.10/hr)
        "medium": "Standard_F4s_v2",   # 4 vCPUs, 8 GB RAM (~$0.19/hr)  
        "large": "Standard_F8s_v2",    # 8 vCPUs, 16 GB RAM (~$0.39/hr)
        "xlarge": "Standard_D8s_v3"    # 8 vCPUs, 32 GB RAM (~$0.44/hr)
    }
    
    def get_instance_for_samples(self, num_samples: int) -> str:
        """Get appropriate instance type based on dataset size."""
        if num_samples < 50000:
            return self.instance_types["small"]
        elif num_samples < 500000:
            return self.instance_types["medium"]
        elif num_samples < 2000000:
            return self.instance_types["large"]
        else:
            return self.instance_types["xlarge"]
    
    def get_duration_for_samples(self, num_samples: int, epochs: int) -> int:
        """Estimate training duration based on dataset size."""
        # Rough estimates based on experience
        base_minutes = (num_samples / 10000) * epochs * 2  # 2 min per 10K samples per epoch
        hours = max(1, int(base_minutes / 60) + 1)  # Round up with minimum 1 hour
        return min(hours, 12)  # Cap at 12 hours for safety