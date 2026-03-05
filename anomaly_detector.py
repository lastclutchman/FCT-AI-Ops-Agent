"""
Anomaly Detection Agent - Identifies system anomalies in CPU, memory, and latency metrics.

Uses multiple detection strategies:
1. Statistical (Z-score, IQR) for baseline anomalies
2. Isolation Forest for multivariate anomaly detection
3. Exponential Weighted Moving Average (EWMA) for trend-based detection
"""

import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json
from datetime import datetime


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    metric_name: str
    severity: AnomalySeverity
    value: float
    threshold: float
    timestamp: str
    detection_method: str  # 'zscore', 'isolation_forest', 'ewma', 'iqr'
    context: Dict = None
    
    def to_dict(self):
        return {
            'metric': self.metric_name,
            'severity': self.severity.name,
            'value': round(self.value, 2),
            'threshold': round(self.threshold, 2),
            'timestamp': self.timestamp,
            'method': self.detection_method,
            'context': self.context or {}
        }


class AnomalyDetector:
    """
    Detects anomalies in system metrics using multiple AI/ML strategies.
    
    AI Techniques:
    - Statistical methods (Z-score, IQR) for single-variable detection
    - Isolation Forest: unsupervised ML for multivariate anomaly detection
    - EWMA: time-series based anomaly detection
    - Adaptive thresholds that learn from historical data
    """
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        """
        Initialize the anomaly detector.
        
        Args:
            window_size: Number of historical data points to maintain
            sensitivity: Controls anomaly detection sensitivity (higher = more sensitive)
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        
        # Historical data buffers for each metric
        self.history = {
            'cpu': deque(maxlen=window_size),
            'memory': deque(maxlen=window_size),
            'latency': deque(maxlen=window_size),
        }
        
        # Isolation Forest models (trained periodically)
        self.isolation_forests = {}
        self.last_training_size = {}
        
        # EWMA parameters for trend detection
        self.ewma_values = {}
        self.ewma_alpha = 0.3  # Smoothing factor
        
        # Learned thresholds (adaptive)
        self.thresholds = {
            'cpu': 80.0,
            'memory': 85.0,
            'latency': 200.0,  # milliseconds
        }
        
        # Baseline statistics (updated as we learn)
        self.baselines = {}
        
    def update_metric(self, metric_name: str, value: float) -> Optional[Anomaly]:
        """
        Add a new metric value and check for anomalies.
        
        Args:
            metric_name: 'cpu', 'memory', or 'latency'
            value: The metric value
            
        Returns:
            Anomaly object if anomaly detected, None otherwise
        """
        if metric_name not in self.history:
            return None
        
        # Store the value
        self.history[metric_name].append(value)
        
        # Update EWMA for trend detection
        if metric_name not in self.ewma_values:
            self.ewma_values[metric_name] = value
        else:
            self.ewma_values[metric_name] = (
                self.ewma_alpha * value + 
                (1 - self.ewma_alpha) * self.ewma_values[metric_name]
            )
        
        # Periodic model retraining (every 30 new points)
        if len(self.history[metric_name]) >= 30:
            if metric_name not in self.last_training_size or \
               len(self.history[metric_name]) - self.last_training_size[metric_name] >= 30:
                self._train_isolation_forest(metric_name)
        
        # Run anomaly detection checks
        anomaly = self._detect_anomaly(metric_name, value)
        
        # Update adaptive thresholds based on recent data
        if len(self.history[metric_name]) >= 10:
            self._update_adaptive_threshold(metric_name)
        
        return anomaly
    
    def _train_isolation_forest(self, metric_name: str):
        """Train Isolation Forest model on historical data."""
        data = np.array(list(self.history[metric_name])).reshape(-1, 1)
        
        if len(data) >= 10:
            self.isolation_forests[metric_name] = IsolationForest(
                contamination=0.1,  # Expect ~10% anomalies
                random_state=42,
                n_estimators=50
            )
            self.isolation_forests[metric_name].fit(data)
            self.last_training_size[metric_name] = len(self.history[metric_name])
    
    def _detect_anomaly(self, metric_name: str, value: float) -> Optional[Anomaly]:
        """
        Detect anomalies using multiple strategies.
        
        Returns the most severe anomaly detected, or None.
        """
        if len(self.history[metric_name]) < 5:
            return None  # Need baseline data
        
        anomalies = []
        
        # Strategy 1: Z-score based detection
        zscore_anomaly = self._zscore_detection(metric_name, value)
        if zscore_anomaly:
            anomalies.append(zscore_anomaly)
        
        # Strategy 2: IQR-based detection
        iqr_anomaly = self._iqr_detection(metric_name, value)
        if iqr_anomaly:
            anomalies.append(iqr_anomaly)
        
        # Strategy 3: Isolation Forest (multivariate)
        if metric_name in self.isolation_forests:
            iso_anomaly = self._isolation_forest_detection(metric_name, value)
            if iso_anomaly:
                anomalies.append(iso_anomaly)
        
        # Strategy 4: EWMA-based trend detection
        ewma_anomaly = self._ewma_detection(metric_name, value)
        if ewma_anomaly:
            anomalies.append(ewma_anomaly)
        
        # Return the most severe anomaly
        if anomalies:
            return max(anomalies, key=lambda x: x.severity.value)
        
        return None
    
    def _zscore_detection(self, metric_name: str, value: float) -> Optional[Anomaly]:
        """Z-score based anomaly detection."""
        data = np.array(list(self.history[metric_name]))
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return None
        
        zscore = abs((value - mean) / std)
        threshold = self.sensitivity  # Default 2.0 = ~95% confidence
        
        if zscore > threshold:
            severity = self._calculate_severity(zscore, threshold)
            return Anomaly(
                metric_name=metric_name,
                severity=severity,
                value=value,
                threshold=mean,
                timestamp=datetime.now().isoformat(),
                detection_method='zscore',
                context={'zscore': round(zscore, 2), 'std_devs_from_mean': round(zscore, 1)}
            )
        
        return None
    
    def _iqr_detection(self, metric_name: str, value: float) -> Optional[Anomaly]:
        """IQR (Interquartile Range) based detection."""
        data = np.array(list(self.history[metric_name]))
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        if value < lower_bound or value > upper_bound:
            severity = AnomalySeverity.MEDIUM if abs(value - (upper_bound if value > upper_bound else lower_bound)) < iqr else AnomalySeverity.HIGH
            return Anomaly(
                metric_name=metric_name,
                severity=severity,
                value=value,
                threshold=upper_bound,
                timestamp=datetime.now().isoformat(),
                detection_method='iqr',
                context={'lower_bound': round(lower_bound, 2), 'upper_bound': round(upper_bound, 2)}
            )
        
        return None
    
    def _isolation_forest_detection(self, metric_name: str, value: float) -> Optional[Anomaly]:
        """Isolation Forest based detection."""
        if metric_name not in self.isolation_forests:
            return None
        
        model = self.isolation_forests[metric_name]
        prediction = model.predict([[value]])[0]
        score = model.score_samples([[value]])[0]
        
        # prediction = -1 means anomaly, 1 means normal
        if prediction == -1:
            severity = AnomalySeverity.HIGH if score < -0.5 else AnomalySeverity.MEDIUM
            return Anomaly(
                metric_name=metric_name,
                severity=severity,
                value=value,
                threshold=np.mean(list(self.history[metric_name])),
                timestamp=datetime.now().isoformat(),
                detection_method='isolation_forest',
                context={'anomaly_score': round(score, 3)}
            )
        
        return None
    
    def _ewma_detection(self, metric_name: str, value: float) -> Optional[Anomaly]:
        """EWMA (Exponential Weighted Moving Average) trend detection."""
        if metric_name not in self.ewma_values or len(self.history[metric_name]) < 10:
            return None
        
        ewma = self.ewma_values[metric_name]
        data = np.array(list(self.history[metric_name]))
        std = np.std(data)
        
        # Detect if value deviates significantly from EWMA trend
        deviation = abs(value - ewma)
        threshold = 2.0 * std
        
        if deviation > threshold and std > 0:
            severity = AnomalySeverity.MEDIUM if deviation < 3 * std else AnomalySeverity.HIGH
            return Anomaly(
                metric_name=metric_name,
                severity=severity,
                value=value,
                threshold=ewma,
                timestamp=datetime.now().isoformat(),
                detection_method='ewma',
                context={'trend_value': round(ewma, 2), 'deviation': round(deviation, 2)}
            )
        
        return None
    
    def _calculate_severity(self, zscore: float, threshold: float) -> AnomalySeverity:
        """Calculate severity based on how far the anomaly is from threshold."""
        ratio = zscore / threshold
        if ratio < 1.5:
            return AnomalySeverity.LOW
        elif ratio < 2.5:
            return AnomalySeverity.MEDIUM
        elif ratio < 4.0:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL
    
    def _update_adaptive_threshold(self, metric_name: str):
        """Update thresholds based on recent data (learning)."""
        data = np.array(list(self.history[metric_name]))
        mean = np.mean(data)
        std = np.std(data)
        
        # Set threshold to mean + 2*std for normal operations
        if metric_name == 'cpu':
            self.thresholds[metric_name] = min(mean + 2 * std, 95)
        elif metric_name == 'memory':
            self.thresholds[metric_name] = min(mean + 2 * std, 95)
        elif metric_name == 'latency':
            self.thresholds[metric_name] = mean + 3 * std
        
        # Store baseline for reference
        self.baselines[metric_name] = {
            'mean': round(mean, 2),
            'std': round(std, 2),
            'threshold': round(self.thresholds[metric_name], 2)
        }
    
    def get_statistics(self) -> Dict:
        """Get current statistics about detected patterns."""
        stats = {}
        for metric_name, data in self.history.items():
            if len(data) > 0:
                stats[metric_name] = {
                    'count': len(data),
                    'mean': round(float(np.mean(data)), 2),
                    'std': round(float(np.std(data)), 2),
                    'min': round(float(np.min(data)), 2),
                    'max': round(float(np.max(data)), 2),
                    'threshold': self.thresholds.get(metric_name, 'N/A'),
                }
        return stats
