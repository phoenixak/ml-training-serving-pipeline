#!/usr/bin/env python3
"""
Data processing utilities for Amazon Reviews dataset.
Extracted and optimized from original training pipeline.
"""

import os
import json
import gzip
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from .logging_utils import setup_logging

logger = setup_logging()


class AmazonDatasetProcessor:
    """Download and process Amazon Reviews with storage optimization."""
    
    def __init__(self, temp_dir=None, max_storage_gb=5):
        self.temp_dir = Path(temp_dir or tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)
        self.max_storage_bytes = max_storage_gb * 1024**3
        self.output_dir = Path("models")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Temp directory: {self.temp_dir}")
        logger.info(f"Storage limit: {max_storage_gb}GB")
    
    def download_amazon_sample(self, category="Electronics", max_reviews=100000):
        """Download Amazon Reviews sample data (creates realistic sample for demo)."""
        
        logger.info(f"Creating sample Amazon Reviews data for {category}...")
        
        sample_file = self.temp_dir / f"{category}_reviews.json.gz"
        
        # Generate realistic Amazon-style reviews
        sample_reviews = []
        for i in range(max_reviews):
            review = {
                "reviewerID": f"A{i % 10000}",  # 10K unique users
                "asin": f"B{category[0]}{i % 50000:05d}",  # 50K unique items
                "overall": int(np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])),
                "reviewText": f"This {category.lower()} product is {'great' if np.random.random() > 0.3 else 'okay'}. Review {i}.",
                "summary": f"{'Excellent' if np.random.random() > 0.5 else 'Good'} product",
                "unixReviewTime": int(time.time()) - int(np.random.randint(0, 365*24*3600)),
                "verified": bool(np.random.choice([True, False], p=[0.8, 0.2])),
                "category": [category]
            }
            sample_reviews.append(review)
        
        # Write compressed sample data
        with gzip.open(sample_file, 'wt') as f:
            for review in sample_reviews:
                f.write(json.dumps(review) + '\n')
        
        logger.info(f"Created {len(sample_reviews):,} sample reviews")
        logger.info(f"File size: {sample_file.stat().st_size / 1024**2:.1f} MB")
        
        return sample_file
    
    def process_reviews_streaming(self, file_path, chunk_size=10000) -> Tuple[List[Dict], Dict, Dict]:
        """Process reviews in chunks to manage memory."""
        
        logger.info(f"Processing {file_path} in chunks of {chunk_size:,}")
        
        processed_reviews = []
        user_map = {}
        item_map = {}
        next_user_id = 0
        next_item_id = 0
        
        with gzip.open(file_path, 'rt') as f:
            chunk = []
            
            for line_num, line in enumerate(f):
                try:
                    review = json.loads(line.strip())
                    
                    # Map users and items to integers
                    user_id = review['reviewerID']
                    item_id = review['asin']
                    
                    if user_id not in user_map:
                        user_map[user_id] = next_user_id
                        next_user_id += 1
                    
                    if item_id not in item_map:
                        item_map[item_id] = next_item_id
                        next_item_id += 1
                    
                    # Extract features
                    processed_review = {
                        'user_idx': user_map[user_id],
                        'item_idx': item_map[item_id],
                        'rating': float(review['overall']),
                        'timestamp': review['unixReviewTime'],
                        'verified': 1 if review.get('verified', False) else 0,
                        'review_length': len(review.get('reviewText', '')),
                        'has_summary': 1 if review.get('summary') else 0
                    }
                    
                    chunk.append(processed_review)
                    
                    # Process chunk when full
                    if len(chunk) >= chunk_size:
                        processed_reviews.extend(self._process_chunk(chunk))
                        chunk = []
                        
                        if line_num % 50000 == 0:
                            logger.info(f"Processed {line_num:,} lines...")
                
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue
            
            # Process final chunk
            if chunk:
                processed_reviews.extend(self._process_chunk(chunk))
        
        # Clean up file immediately
        os.remove(file_path)
        logger.info(f"Cleaned up {file_path}")
        
        logger.info(f"Final dataset: {len(processed_reviews):,} reviews")
        logger.info(f"Users: {len(user_map):,}, Items: {len(item_map):,}")
        
        return processed_reviews, user_map, item_map
    
    def _process_chunk(self, chunk):
        """Process a chunk of reviews."""
        # Add time-based features
        for review in chunk:
            timestamp = review['timestamp']
            review['hour'] = (timestamp % (24*3600)) // 3600
            review['day_of_week'] = (timestamp // (24*3600)) % 7
            review['month'] = (timestamp // (30*24*3600)) % 12
            
            # Normalize rating to 0-1
            review['rating_normalized'] = (review['rating'] - 1) / 4
            
            # Log transform review length
            review['log_review_length'] = np.log1p(review['review_length'])
        
        return chunk
    
    def save_processed_data(self, reviews, user_map, item_map):
        """Save processed data efficiently."""
        
        # Convert to DataFrame for efficient storage
        df = pd.DataFrame(reviews)
        
        # Save as parquet for efficiency
        data_file = self.output_dir / "amazon_reviews_processed.parquet"
        df.to_parquet(data_file, compression='snappy')
        
        # Save mappings
        mappings = {
            'user_mapping': user_map,
            'item_mapping': item_map,
            'num_users': len(user_map),
            'num_items': len(item_map),
            'num_interactions': len(reviews)
        }
        
        mappings_file = self.output_dir / "data_mappings.json"
        with open(mappings_file, 'w') as f:
            json.dump(mappings, f, indent=2)
        
        logger.info(f"Saved processed data to {data_file}")
        logger.info(f"Saved mappings to {mappings_file}")
        logger.info(f"Data file size: {data_file.stat().st_size / 1024**2:.1f} MB")
        
        return data_file, mappings_file
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")