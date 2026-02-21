"""
rl_plotter module for logging training metrics.
"""

import os
import json
from datetime import datetime


class Logger:
    """Simple logger for tracking metrics during training."""
    
    def __init__(self, exp_name, env_name, save_dir="./logs"):
        """
        Initialize the logger.
        
        Args:
            exp_name: Experiment name
            env_name: Environment/metric name (e.g., 'acc', 'auc', etc.)
            save_dir: Directory to save logs
        """
        self.exp_name = exp_name
        self.env_name = env_name
        self.save_dir = save_dir
        self.metrics = []
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Log file path
        self.log_file = os.path.join(save_dir, f"{exp_name}_{env_name}.json")
    
    def update(self, score, total_steps):
        """
        Update the logger with new metrics.
        
        Args:
            score: List of scores/metrics
            total_steps: Current training step/epoch
        """
        # Handle both single values and lists
        if isinstance(score, list):
            score_val = score[0] if score else 0.0
        else:
            score_val = score
        
        # Record the metric
        entry = {
            "step": total_steps,
            "score": float(score_val),
            "timestamp": datetime.now().isoformat()
        }
        self.metrics.append(entry)
        
        # Optionally save to file (for persistence)
        self._save_to_file()
    
    def _save_to_file(self):
        """Save metrics to a JSON file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metrics to {self.log_file}: {e}")
    
    def get_metrics(self):
        """Get all recorded metrics."""
        return self.metrics


logger = type('module', (), {'Logger': Logger})()
