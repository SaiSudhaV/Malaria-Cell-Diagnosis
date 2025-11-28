"""Monitoring with Evidently AI"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

# Note: Evidently AI integration requires additional setup
# For now, we provide basic monitoring utilities


class ModelMonitor:
    """Monitor model performance and data quality"""
    
    def __init__(self, output_dir: str = None):
        """Initialize monitor
        
        Args:
            output_dir: Directory for saving reports
        """
        if output_dir is None:
            from src.utils.config import Config
            output_dir = str(Config.EVIDENTLY_REPORTS_DIR)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def log_prediction(self, prediction: Dict, timestamp: datetime = None):
        """Log a prediction for monitoring
        
        Args:
            prediction: Prediction result dictionary
            timestamp: Timestamp of prediction
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            **prediction
        }
        
        # Save to CSV for later analysis
        log_file = self.output_dir / "predictions_log.csv"
        
        df = pd.DataFrame([log_entry])
        if log_file.exists():
            existing_df = pd.read_csv(log_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(log_file, index=False)
        logger.debug(f"Logged prediction at {timestamp}")
    
    def generate_report(self) -> Dict:
        """Generate monitoring report
        
        Returns:
            Dictionary with report metrics
        """
        log_file = self.output_dir / "predictions_log.csv"
        
        if not log_file.exists():
            logger.warning("No predictions logged yet")
            return {}
        
        df = pd.read_csv(log_file)
        
        report = {
            'total_predictions': len(df),
            'timestamp': datetime.now().isoformat(),
            'predictions_by_class': df['class_name'].value_counts().to_dict(),
            'avg_confidence': float(df['confidence'].mean()),
            'min_confidence': float(df['confidence'].min()),
            'max_confidence': float(df['confidence'].max()),
        }
        
        logger.info(f"Generated monitoring report: {len(df)} predictions")
        return report
    
    def check_data_drift(self, current_predictions: List[Dict],
                        reference_predictions: List[Dict] = None) -> Dict:
        """Check for data drift
        
        Args:
            current_predictions: Current batch of predictions
            reference_predictions: Reference batch for comparison
            
        Returns:
            Drift analysis report
        """
        current_df = pd.DataFrame(current_predictions)
        
        drift_report = {
            'current_class_distribution': current_df['class_name'].value_counts().to_dict(),
            'current_avg_confidence': float(current_df['confidence'].mean()),
        }
        
        if reference_predictions:
            ref_df = pd.DataFrame(reference_predictions)
            
            # Calculate drift metrics
            current_class_pct = current_df['class_name'].value_counts(normalize=True).to_dict()
            ref_class_pct = ref_df['class_name'].value_counts(normalize=True).to_dict()
            
            drift_report['reference_class_distribution'] = ref_class_pct
            drift_report['reference_avg_confidence'] = float(ref_df['confidence'].mean())
            
            # KL divergence (simplified)
            drift_report['class_distribution_change'] = {
                cls: abs(current_class_pct.get(cls, 0) - ref_class_pct.get(cls, 0))
                for cls in set(list(current_class_pct.keys()) + list(ref_class_pct.keys()))
            }
        
        logger.info("Data drift check completed")
        return drift_report
