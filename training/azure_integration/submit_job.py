#!/usr/bin/env python3
"""
Azure ML integration for cloud training.
Simplified version for ml-training-serving-pipeline.
"""

import os
import logging
from typing import Dict, Optional
from pathlib import Path

try:
    from azure.ai.ml import MLClient, command, Input
    from azure.ai.ml.entities import Environment, Job
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

from shared.utils.logging_utils import setup_logging
from shared.config.training_config import DLRMTrainingConfig

logger = setup_logging()


class AzureTrainingSubmitter:
    """Simplified Azure ML training job submission."""
    
    def __init__(self, config: DLRMTrainingConfig):
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure ML SDK not available. Install with:\n"
                "  pip install azure-ai-ml azure-identity"
            )
        
        self.config = config
        self.ml_client = self._setup_client()
        logger.info("✅ Azure ML client initialized")
    
    def _setup_client(self) -> MLClient:
        """Setup Azure ML client using environment variables."""
        
        # Check required environment variables
        required_vars = [
            'AZURE_SUBSCRIPTION_ID',
            'AZURE_RESOURCE_GROUP', 
            'AZURE_WORKSPACE_NAME'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {missing_vars}\n"
                "Set them with:\n"
                "  export AZURE_SUBSCRIPTION_ID='your-subscription-id'\n"
                "  export AZURE_RESOURCE_GROUP='your-resource-group'\n"
                "  export AZURE_WORKSPACE_NAME='your-workspace-name'"
            )
        
        credential = DefaultAzureCredential()
        
        return MLClient(
            credential=credential,
            subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
            resource_group_name=os.getenv('AZURE_RESOURCE_GROUP'),
            workspace_name=os.getenv('AZURE_WORKSPACE_NAME')
        )
    
    def create_environment(self) -> Environment:
        """Create training environment with our dependencies."""
        
        # Use our exact requirements for consistency
        requirements = """
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0
azure-ai-ml>=1.8.0
azure-identity>=1.12.0
prometheus-client>=0.17.0
pydantic>=2.0.0
"""
        
        environment = Environment(
            name="ml-pipeline-training",
            description="ML Pipeline training environment",
            conda_file={
                "name": "ml-pipeline",
                "channels": ["pytorch", "conda-forge"],
                "dependencies": [
                    "python=3.11",
                    "pip",
                    {"pip": requirements.strip().split('\n')}
                ]
            },
            image="mcr.microsoft.com/azureml/curated/pytorch-1.13-ubuntu20.04-py38-cpu-inference:latest"
        )
        
        try:
            env = self.ml_client.environments.create_or_update(environment)
            logger.info(f"✅ Environment '{environment.name}' ready")
            return env
        except Exception as e:
            logger.error(f"❌ Failed to create environment: {e}")
            raise
    
    def submit_training_job(
        self,
        instance_type: str = "Standard_F4s_v2",
        use_spot_instances: bool = True,
        max_duration_hours: int = 4
    ) -> Job:
        """
        Submit DLRM training job to Azure ML.
        
        Args:
            instance_type: Azure VM instance type
            use_spot_instances: Use spot instances for cost savings (60-90% cheaper)
            max_duration_hours: Maximum training duration in hours
        """
        
        logger.info("🚀 Submitting training job to Azure ML...")
        
        # Create environment
        environment = self.create_environment()
        
        # Define training command
        training_command = command(
            code=".",  # Current directory
            command=(
                "python training/train_dlrm.py "
                f"--local "
                f"--samples {self.config.max_samples} "
                f"--epochs {self.config.epochs}"
            ),
            environment=environment,
            compute=self.config.compute_target,
            experiment_name=self.config.experiment_name,
            display_name=f"DLRM Training - {self.config.max_samples:,} samples",
            description=f"DLRM training with {self.config.max_samples:,} samples for {self.config.epochs} epochs"
        )
        
        # Configure cost optimization with spot instances
        if use_spot_instances:
            training_command.resources = {
                "instance_type": instance_type,
                "instance_count": 1,
                "max_run_duration": f"PT{max_duration_hours}H",
                "priority": "Spot"  # 60-90% cost savings
            }
        else:
            training_command.resources = {
                "instance_type": instance_type,
                "instance_count": 1,
                "max_run_duration": f"PT{max_duration_hours}H"
            }
        
        # Submit job
        try:
            job = self.ml_client.jobs.create_or_update(training_command)
            
            logger.info("✅ Training job submitted successfully!")
            logger.info(f"   📊 Job name: {job.name}")
            logger.info(f"   🌐 Studio URL: {job.studio_url}")
            logger.info(f"   💻 Instance: {instance_type}")
            logger.info(f"   💰 Spot instances: {use_spot_instances}")
            logger.info(f"   ⏱️ Max duration: {max_duration_hours}h")
            
            # Estimate costs
            self._estimate_costs(instance_type, max_duration_hours, use_spot_instances)
            
            return job
            
        except Exception as e:
            logger.error(f"❌ Failed to submit job: {e}")
            raise
    
    def _estimate_costs(self, instance_type: str, hours: int, use_spot: bool):
        """Estimate training costs."""
        
        # Approximate Azure pricing (USD per hour)
        costs = {
            "Standard_F2s_v2": 0.10,   # 2 vCPUs, 4 GB RAM
            "Standard_F4s_v2": 0.19,   # 4 vCPUs, 8 GB RAM
            "Standard_F8s_v2": 0.39,   # 8 vCPUs, 16 GB RAM
            "Standard_D4s_v3": 0.22,   # 4 vCPUs, 16 GB RAM
            "Standard_D8s_v3": 0.44    # 8 vCPUs, 32 GB RAM
        }
        
        base_cost_per_hour = costs.get(instance_type, 0.19)
        base_total = base_cost_per_hour * hours
        spot_total = base_total * 0.3 if use_spot else base_total  # ~70% discount
        savings = base_total - spot_total
        
        logger.info("💰 Cost Estimates:")
        logger.info(f"   💻 Instance: {instance_type}")
        logger.info(f"   ⏱️ Duration: {hours} hours")
        logger.info(f"   💵 Regular cost: ${base_total:.2f}")
        logger.info(f"   🎯 Your cost: ${spot_total:.2f}")
        logger.info(f"   💚 Savings: ${savings:.2f} ({savings/base_total*100:.0f}%)")
        
        if spot_total > 20:
            logger.warning("⚠️ Estimated cost exceeds $20. Consider shorter duration.")
    
    def monitor_job(self, job_name: str):
        """Monitor training job progress."""
        try:
            job = self.ml_client.jobs.get(job_name)
            
            logger.info(f"📊 Job Status: {job.status}")
            logger.info(f"🌐 Studio URL: {job.studio_url}")
            
            return job
        except Exception as e:
            logger.error(f"❌ Failed to get job status: {e}")
            return None
    
    def download_model(self, job_name: str, local_path: str = "./models"):
        """Download trained model from Azure ML."""
        try:
            job = self.ml_client.jobs.get(job_name)
            
            if job.status == "Completed":
                self.ml_client.jobs.download(name=job_name, download_path=local_path)
                logger.info(f"✅ Model downloaded to {local_path}")
                return True
            else:
                logger.warning(f"⚠️ Job not completed. Status: {job.status}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            return False


def submit_training_job(config: DLRMTrainingConfig) -> Optional[Job]:
    """Submit training job to Azure ML (called from main training script)."""
    
    if not AZURE_AVAILABLE:
        logger.error("❌ Azure ML SDK not available")
        logger.info("Install with: pip install azure-ai-ml azure-identity")
        return None
    
    try:
        submitter = AzureTrainingSubmitter(config)
        
        # Use appropriate instance size based on dataset size
        if config.max_samples < 50000:
            instance_type = "Standard_F2s_v2"  # Small datasets
            max_hours = 2
        elif config.max_samples < 500000:
            instance_type = "Standard_F4s_v2"  # Medium datasets  
            max_hours = 4
        else:
            instance_type = "Standard_F8s_v2"  # Large datasets
            max_hours = 8
        
        job = submitter.submit_training_job(
            instance_type=instance_type,
            use_spot_instances=True,  # Always use spot for cost savings
            max_duration_hours=max_hours
        )
        
        logger.info("\n📋 Next steps:")
        logger.info(f"   1. Monitor: az ml job show -n {job.name}")
        logger.info(f"   2. View in studio: {job.studio_url}")
        logger.info(f"   3. Download when complete using job name: {job.name}")
        
        return job
        
    except Exception as e:
        logger.error(f"❌ Azure submission failed: {e}")
        return None


if __name__ == "__main__":
    """Test Azure integration."""
    
    # Test configuration
    config = DLRMTrainingConfig()
    config.max_samples = 10000
    config.epochs = 5
    config.use_azure = True
    
    logger.info("🧪 Testing Azure ML integration...")
    
    if AZURE_AVAILABLE:
        job = submit_training_job(config)
        if job:
            logger.info(f"✅ Test submission successful: {job.name}")
        else:
            logger.error("❌ Test submission failed")
    else:
        logger.error("❌ Azure ML SDK not available for testing")