#!/usr/bin/env python3
"""
Unified DLRM training script for local and Azure training.
Refactored from original MLOpsPipeline with real testing focus.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import pandas as pd

from training.models.simple_dlrm import SimpleDLRM, create_model_from_mappings
from shared.config.training_config import DLRMTrainingConfig
from shared.utils.data_processing import AmazonDatasetProcessor
from shared.utils.logging_utils import setup_logging

logger = setup_logging()


def train_model(data_file: str, mappings_file: str, config: DLRMTrainingConfig):
    """Train SimpleDLRM model with given data and configuration."""
    
    logger.info("🎯 Starting model training...")
    
    # Load data
    df = pd.read_parquet(data_file)
    with open(mappings_file) as f:
        mappings = json.load(f)
    
    logger.info(f"📊 Dataset: {len(df):,} samples")
    logger.info(f"👥 Users: {mappings['num_users']:,}")
    logger.info(f"📦 Items: {mappings['num_items']:,}")
    
    # Create model
    model = create_model_from_mappings(
        user_mapping=mappings['user_mapping'],
        item_mapping=mappings['item_mapping'],
        embedding_dim=config.embedding_dim
    )
    
    logger.info(f"🏗️ Model: {model.count_parameters():,} parameters")
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Convert to tensors
    user_ids = torch.tensor(df['user_idx'].values, dtype=torch.long)
    item_ids = torch.tensor(df['item_idx'].values, dtype=torch.long)
    
    # Dense features (6 features as defined in SimpleDLRM)
    dense_features = torch.tensor(df[[
        'verified', 'log_review_length', 'has_summary', 
        'hour', 'day_of_week', 'month'
    ]].values, dtype=torch.float32)
    
    ratings = torch.tensor(df['rating_normalized'].values, dtype=torch.float32).unsqueeze(1)
    
    # Training loop - adjust batch size for small datasets
    batch_size = min(config.batch_size, len(df))
    num_batches = max(1, len(df) // batch_size)
    
    logger.info(f"🚀 Training for {config.epochs} epochs, {num_batches} batches per epoch")
    logger.info(f"📦 Batch size: {batch_size}")
    
    for epoch in range(config.epochs):
        total_loss = 0
        model.train()
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            
            batch_users = user_ids[start_idx:end_idx]
            batch_items = item_ids[start_idx:end_idx]
            batch_dense = dense_features[start_idx:end_idx]
            batch_ratings = ratings[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_users, batch_items, batch_dense)
            loss = criterion(outputs, batch_ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{config.epochs} - Loss: {avg_loss:.6f}")
    
    # Save model and info
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / config.model_name
    torch.save(model.state_dict(), model_path)
    
    # Save model info
    model_info = {
        "num_users": mappings['num_users'],
        "num_items": mappings['num_items'],
        "embedding_dim": config.embedding_dim,
        "total_parameters": model.count_parameters(),
        "final_loss": avg_loss,
        "epochs_trained": config.epochs,
        "model_type": "SimpleDLRM"
    }
    
    model_info_path = output_dir / "model_info.json"
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"✅ Model saved to {model_path}")
    logger.info(f"📊 Model info: {model_info}")
    
    return model, model_info


def train_local(config: DLRMTrainingConfig):
    """Train model locally."""
    
    logger.info("🏠 Starting local training pipeline")
    logger.info("=" * 60)
    
    processor = None
    
    try:
        # 1. Initialize processor
        processor = AmazonDatasetProcessor(max_storage_gb=config.max_storage_gb)
        
        # 2. Download/create sample data
        logger.info("📥 Step 1: Data Acquisition")
        sample_file = processor.download_amazon_sample(
            category=config.category, 
            max_reviews=config.max_samples
        )
        
        # 3. Process data
        logger.info("🔄 Step 2: Data Processing")
        reviews, user_map, item_map = processor.process_reviews_streaming(sample_file)
        
        # 4. Save processed data
        logger.info("💾 Step 3: Data Storage")
        data_file, mappings_file = processor.save_processed_data(reviews, user_map, item_map)
        
        # 5. Train model
        logger.info("🎯 Step 4: Model Training")
        model, model_info = train_model(str(data_file), str(mappings_file), config)
        
        logger.info("✅ Local training completed successfully!")
        
        return {
            "status": "success",
            "data_file": str(data_file),
            "mappings_file": str(mappings_file),
            "model_path": config.get_model_path(),
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise
    
    finally:
        if processor:
            processor.cleanup()


def submit_to_azure(config: DLRMTrainingConfig):
    """Submit training job to Azure ML."""
    
    logger.info("☁️ Submitting job to Azure ML")
    
    # Import Azure integration
    try:
        from training.azure_integration.submit_job import submit_training_job
        
        # Configure Azure settings
        config.use_azure = True
        config.compute_target = "cpu-cluster"
        config.experiment_name = "dlrm-training"
        
        logger.info(f"📊 Azure training configuration:")
        logger.info(f"   Samples: {config.max_samples:,}")
        logger.info(f"   Epochs: {config.epochs}")
        logger.info(f"   Compute: {config.compute_target}")
        logger.info(f"   Experiment: {config.experiment_name}")
        
        return submit_training_job(config)
        
    except ImportError as e:
        logger.error("❌ Azure integration not available")
        logger.info("Install Azure dependencies with:")
        logger.info("  pip install azure-ai-ml azure-identity")
        return None
    except Exception as e:
        logger.error(f"❌ Azure submission failed: {e}")
        return None


def main():
    """Main training entry point."""
    
    parser = argparse.ArgumentParser(description="DLRM Training Pipeline")
    parser.add_argument("--local", action="store_true", help="Run training locally")
    parser.add_argument("--azure", action="store_true", help="Submit to Azure ML")
    parser.add_argument("--config", type=str, help="Config file path (optional)")
    parser.add_argument("--samples", type=int, default=50000, help="Number of samples for local training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Create configuration
    config = DLRMTrainingConfig()
    
    # Override with command line arguments
    if args.samples:
        config.max_samples = args.samples
    if args.epochs:
        config.epochs = args.epochs
    
    logger.info(f"🚀 DLRM Training Pipeline")
    logger.info(f"📊 Configuration: {args.samples:,} samples, {args.epochs} epochs")
    
    if args.local:
        result = train_local(config)
        logger.info(f"🎉 Training result: {result}")
        
    elif args.azure:
        result = submit_to_azure(config)
        logger.info(f"☁️ Azure submission result: {result}")
        
    else:
        logger.error("❌ Please specify --local or --azure")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()