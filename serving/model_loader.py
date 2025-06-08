#!/usr/bin/env python3
"""
Model loader for serving pipeline.
Loads trained SimpleDLRM models and provides prediction interface.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from training.models.simple_dlrm import SimpleDLRM
from shared.config.serving_config import ServingConfig
from shared.utils.logging_utils import setup_logging

logger = setup_logging()


class DLRMModelLoader:
    """Loads and serves DLRM models from training pipeline."""
    
    def __init__(self, config: Optional[ServingConfig] = None):
        self.config = config or ServingConfig()
        self.model = None
        self.model_info = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.device = self.config.device
        
    def load_from_training_output(self, model_dir: str = "models") -> bool:
        """Load model artifacts from training output directory."""
        
        model_path = Path(model_dir)
        
        # Check for required files
        model_file = model_path / "amazon_dlrm_model.pth"
        info_file = model_path / "model_info.json"
        mappings_file = model_path / "data_mappings.json"
        
        if not all([model_file.exists(), info_file.exists(), mappings_file.exists()]):
            logger.error(f"Missing required files in {model_dir}")
            logger.error(f"  Model file: {model_file.exists()}")
            logger.error(f"  Info file: {info_file.exists()}")
            logger.error(f"  Mappings file: {mappings_file.exists()}")
            return False
        
        try:
            # Load model info
            with open(info_file, 'r') as f:
                self.model_info = json.load(f)
            
            # Load data mappings
            with open(mappings_file, 'r') as f:
                mappings = json.load(f)
                self.user_mapping = mappings['user_mapping']
                self.item_mapping = mappings['item_mapping']
                
                # Create reverse mappings for recommendations
                self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
                self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
            
            # Initialize model
            self.model = SimpleDLRM(
                num_users=self.model_info['num_users'],
                num_items=self.model_info['num_items'],
                embedding_dim=self.model_info['embedding_dim']
            )
            
            # Load trained weights
            state_dict = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Successfully loaded DLRM model:")
            logger.info(f"   Users: {self.model_info['num_users']:,}")
            logger.info(f"   Items: {self.model_info['num_items']:,}")
            logger.info(f"   Parameters: {self.model_info['total_parameters']:,}")
            logger.info(f"   Final Loss: {self.model_info['final_loss']:.6f}")
            logger.info(f"   Device: {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None and self.model_info is not None
    
    def predict_single(self, user_id: str, item_id: str, context: Optional[Dict] = None) -> float:
        """Generate prediction for a single user-item pair."""
        
        if not self.is_loaded():
            raise ValueError("Model not loaded. Call load_from_training_output() first.")
        
        # Map user and item IDs
        user_idx = self.user_mapping.get(user_id)
        item_idx = self.item_mapping.get(item_id)
        
        if user_idx is None or item_idx is None:
            # Handle cold start with average prediction
            logger.warning(f"Cold start for user={user_id}, item={item_id}")
            return 0.5
        
        # Prepare dense features
        dense_features = self._extract_dense_features(context or {})
        
        # Convert to tensors
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
        item_tensor = torch.tensor([item_idx], dtype=torch.long, device=self.device)
        dense_tensor = torch.tensor([dense_features], dtype=torch.float32, device=self.device)
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(user_tensor, item_tensor, dense_tensor)
            return prediction.item()
    
    def predict_batch(self, user_ids: List[str], item_ids: List[str],
                     contexts: Optional[List[Dict]] = None) -> List[float]:
        """Generate predictions for multiple user-item pairs."""
        
        if not self.is_loaded():
            raise ValueError("Model not loaded. Call load_from_training_output() first.")
        
        if len(user_ids) != len(item_ids):
            raise ValueError("user_ids and item_ids must have same length")
        
        batch_size = len(user_ids)
        contexts = contexts or [{}] * batch_size
        
        # Prepare batch data
        user_indices = []
        item_indices = []
        dense_features_batch = []
        
        for user_id, item_id, context in zip(user_ids, item_ids, contexts):
            user_idx = self.user_mapping.get(user_id, 0)  # Default to user 0 for cold start
            item_idx = self.item_mapping.get(item_id, 0)  # Default to item 0 for cold start
            
            user_indices.append(user_idx)
            item_indices.append(item_idx)
            dense_features_batch.append(self._extract_dense_features(context))
        
        # Convert to tensors
        user_tensor = torch.tensor(user_indices, dtype=torch.long, device=self.device)
        item_tensor = torch.tensor(item_indices, dtype=torch.long, device=self.device)
        dense_tensor = torch.tensor(dense_features_batch, dtype=torch.float32, device=self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(user_tensor, item_tensor, dense_tensor)
            return predictions.squeeze().cpu().numpy().tolist()
    
    def recommend_items(self, user_id: str, num_recommendations: int = 10,
                       exclude_items: Optional[List[str]] = None,
                       context: Optional[Dict] = None) -> List[Dict]:
        """Generate top-k item recommendations for a user."""
        
        if not self.is_loaded():
            raise ValueError("Model not loaded. Call load_from_training_output() first.")
        
        user_idx = self.user_mapping.get(user_id)
        if user_idx is None:
            # Cold start: return popular items
            return self._get_popular_items(num_recommendations)
        
        exclude_items = exclude_items or []
        exclude_indices = {self.item_mapping.get(item_id) for item_id in exclude_items}
        exclude_indices.discard(None)
        
        # Get candidate items (limit to first 1000 for efficiency)
        all_item_indices = list(range(min(1000, self.model_info['num_items'])))
        candidate_indices = [idx for idx in all_item_indices if idx not in exclude_indices]
        
        if not candidate_indices:
            return []
        
        # Prepare batch prediction
        batch_size = len(candidate_indices)
        user_indices = [user_idx] * batch_size
        dense_features = [self._extract_dense_features(context or {})] * batch_size
        
        # Convert to tensors
        user_tensor = torch.tensor(user_indices, dtype=torch.long, device=self.device)
        item_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=self.device)
        dense_tensor = torch.tensor(dense_features, dtype=torch.float32, device=self.device)
        
        # Get predictions
        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor, dense_tensor).squeeze()
        
        # Get top-k recommendations
        if scores.dim() == 0:  # Single item case
            scores = scores.unsqueeze(0)
        
        top_k_indices = torch.topk(scores, min(num_recommendations, len(scores))).indices
        
        recommendations = []
        for i, idx in enumerate(top_k_indices):
            item_idx = candidate_indices[idx.item()]
            item_id = self.reverse_item_mapping.get(item_idx, f"item_{item_idx}")
            score = scores[idx].item()
            
            recommendations.append({
                "item_id": item_id,
                "score": score,
                "confidence": min(score * 2, 1.0),  # Simple confidence score
                "rank": i + 1
            })
        
        return recommendations
    
    def _extract_dense_features(self, context: Dict) -> List[float]:
        """Extract dense features from context (matching training format)."""
        
        # Default feature values (matching training pipeline)
        features = [
            context.get('verified', 0),  # verified
            np.log1p(context.get('review_length', 100)),  # log_review_length
            context.get('has_summary', 0),  # has_summary
            context.get('hour', 12),  # hour
            context.get('day_of_week', 1),  # day_of_week
            context.get('month', 6)  # month
        ]
        
        return features
    
    def _get_popular_items(self, num_items: int) -> List[Dict]:
        """Get popular items for cold start scenarios."""
        
        # Simple popular items (first N items as mock popular items)
        popular_items = []
        for i in range(min(num_items, 20)):
            item_id = self.reverse_item_mapping.get(i, f"item_{i}")
            popular_items.append({
                "item_id": item_id,
                "score": 0.8 - (i * 0.02),  # Decreasing popularity score
                "confidence": 0.6,
                "rank": i + 1,
                "reason": "popular_item"
            })
        
        return popular_items[:num_items]
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        
        if not self.is_loaded():
            return {"status": "no_model_loaded"}
        
        return {
            "model_type": "SimpleDLRM",
            "num_users": self.model_info['num_users'],
            "num_items": self.model_info['num_items'],
            "embedding_dim": self.model_info['embedding_dim'],
            "total_parameters": self.model_info['total_parameters'],
            "final_loss": self.model_info['final_loss'],
            "epochs_trained": self.model_info['epochs_trained'],
            "device": self.device,
            "status": "loaded"
        }


# Simple test function
if __name__ == "__main__":
    logger.info("🧪 Testing DLRM Model Loader...")
    
    loader = DLRMModelLoader()
    
    # Try to load from models directory
    if loader.load_from_training_output("models"):
        logger.info("✅ Model loaded successfully!")
        
        # Test single prediction
        prediction = loader.predict_single("A123", "B00001", {"verified": 1})
        logger.info(f"📊 Sample prediction: {prediction:.4f}")
        
        # Test recommendations
        recommendations = loader.recommend_items("A123", num_recommendations=5)
        logger.info(f"🎯 Sample recommendations: {recommendations}")
        
        # Show model info
        info = loader.get_model_info()
        logger.info(f"ℹ️ Model info: {info}")
        
    else:
        logger.error("❌ Failed to load model")
    
    logger.info("✅ Model Loader test completed")