"""
Correlation Agent - Detects relationships between metrics and identifies systemic issues.

AI Features:
1. Multi-metric correlation analysis
2. Root cause inference
3. Cascade effect detection
4. Pattern recognition for systemic vs. isolated issues
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np


@dataclass
class MetricRelationship:
    """Describes a relationship between two metrics."""
    metric1: str
    metric2: str
    correlation_coefficient: float
    lag_time: int  # seconds before metric2 reacts to metric1
    strength: str  # 'weak', 'moderate', 'strong'
    inferred_cause: str  # Which metric likely causes the other


class CorrelationAgent:
    """
    Analyzes relationships between system metrics.
    
    AI Techniques:
    - Pearson correlation for detecting linear relationships
    - Time-lagged correlation for causal inference
    - Pattern matching to identify known cascading failures
    - Graph analysis to detect systemic issues
    """
    
    def __init__(self, window_size: int = 100):
        """Initialize correlation agent."""
        self.window_size = window_size
        self.metric_history: Dict[str, deque] = {
            'cpu': deque(maxlen=window_size),
            'memory': deque(maxlen=window_size),
            'latency': deque(maxlen=window_size),
        }
        
        # Learned correlations
        self.learned_relationships: List[MetricRelationship] = []
        
        # Anomaly event tracking for pattern detection
        self.recent_anomalies: List[Dict] = deque(maxlen=50)
        
    def update_metrics(self, metrics: Dict[str, float]):
        """Update all metrics for correlation analysis."""
        for metric_name, value in metrics.items():
            if metric_name in self.metric_history:
                self.metric_history[metric_name].append(value)
    
    def record_anomaly(self, anomaly):
        """Record an anomaly for correlation learning."""
        self.recent_anomalies.append({
            'metric': anomaly.metric_name,
            'severity': anomaly.severity.name,
            'value': anomaly.value,
            'timestamp': anomaly.timestamp
        })
    
    def analyze_current_correlations(self) -> List[MetricRelationship]:
        """
        Analyze current relationships between metrics.
        
        Returns list of detected correlations.
        """
        correlations = []
        metrics = list(self.metric_history.keys())
        
        # Check all pairs of metrics
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i+1:]:
                relationship = self._analyze_pair(metric1, metric2)
                if relationship and abs(relationship.correlation_coefficient) > 0.5:
                    correlations.append(relationship)
        
        return correlations
    
    def _analyze_pair(self, metric1: str, metric2: str) -> Optional[MetricRelationship]:
        """Analyze correlation between two metrics."""
        data1 = np.array(list(self.metric_history[metric1]))
        data2 = np.array(list(self.metric_history[metric2]))
        
        if len(data1) < 10 or len(data2) < 10:
            return None
        
        # Calculate Pearson correlation
        if len(data1) == len(data2) and np.std(data1) > 0 and np.std(data2) > 0:
            correlation = np.corrcoef(data1, data2)[0, 1]
            
            if np.isnan(correlation):
                return None
            
            # Infer causality using time lag analysis
            lag_time = self._find_optimal_lag(data1, data2)
            
            # Determine causal direction
            if abs(correlation) > 0.7:
                # Strong correlation - check which causes which
                lead_metric = metric1 if lag_time > 0 else metric2
                follow_metric = metric2 if lag_time > 0 else metric1
                inferred_cause = f"{lead_metric} spike leads to {follow_metric} increase"
            else:
                inferred_cause = "Concurrent anomalies (possible shared root cause)"
            
            # Classify strength
            abs_corr = abs(correlation)
            if abs_corr > 0.75:
                strength = 'strong'
            elif abs_corr > 0.5:
                strength = 'moderate'
            else:
                strength = 'weak'
            
            return MetricRelationship(
                metric1=metric1,
                metric2=metric2,
                correlation_coefficient=correlation,
                lag_time=lag_time,
                strength=strength,
                inferred_cause=inferred_cause
            )
        
        return None
    
    def _find_optimal_lag(self, data1: np.ndarray, data2: np.ndarray, max_lag: int = 10) -> int:
        """
        Find the time lag that maximizes correlation.
        
        Positive lag means data2 follows data1.
        """
        if len(data1) < max_lag + 2:
            return 0
        
        best_lag = 0
        best_corr = 0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                if np.std(data1) > 0 and np.std(data2) > 0:
                    corr = abs(np.corrcoef(data1, data2)[0, 1])
                else:
                    continue
            elif lag > 0:
                if len(data1) > lag and np.std(data1[:-lag]) > 0 and np.std(data2[lag:]) > 0:
                    corr = abs(np.corrcoef(data1[:-lag], data2[lag:])[0, 1])
                else:
                    continue
            else:  # lag < 0
                lag_abs = abs(lag)
                if len(data2) > lag_abs and np.std(data2[:-lag_abs]) > 0 and np.std(data1[lag_abs:]) > 0:
                    corr = abs(np.corrcoef(data2[:-lag_abs], data1[lag_abs:])[0, 1])
                else:
                    continue
            
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag
        
        return best_lag
    
    def detect_systemic_issues(self, current_anomalies: List) -> Optional[Dict]:
        """
        Detect if multiple anomalies indicate a systemic issue.
        
        Returns analysis of systemic issue or None.
        """
        if len(current_anomalies) < 2:
            return None
        
        # Check if anomalies are correlated in time
        anomaly_metrics = [a.metric_name for a in current_anomalies]
        
        # Analyze patterns
        pattern = self._identify_pattern(anomaly_metrics)
        
        if pattern['is_systemic']:
            return {
                'is_systemic': True,
                'pattern': pattern['pattern_name'],
                'affected_metrics': list(set(anomaly_metrics)),
                'likely_root_cause': pattern['likely_cause'],
                'recommendation': pattern['recommendation']
            }
        
        return None
    
    def _identify_pattern(self, anomaly_metrics: List[str]) -> Dict:
        """
        Identify known failure patterns.
        
        AI Pattern Library:
        - Resource Exhaustion: High CPU + High Memory
        - Performance Degradation: CPU + Latency
        - Cascading Failure: CPU -> Memory -> Latency
        - Bottleneck: Latency without CPU/Memory spike
        """
        
        metrics_set = set(anomaly_metrics)
        
        # Pattern 1: Resource Exhaustion
        if {'cpu', 'memory'} <= metrics_set:
            return {
                'is_systemic': True,
                'pattern_name': 'Resource Exhaustion',
                'likely_cause': 'System running out of capacity',
                'recommendation': 'Scale up resources or identify resource leak'
            }
        
        # Pattern 2: Performance Degradation
        if {'cpu', 'latency'} <= metrics_set and 'memory' not in metrics_set:
            return {
                'is_systemic': True,
                'pattern_name': 'Performance Degradation',
                'likely_cause': 'Computational bottleneck without memory issues',
                'recommendation': 'Add CPU capacity or optimize workload distribution'
            }
        
        # Pattern 3: Cascading Failure
        if len(metrics_set) == 3:
            return {
                'is_systemic': True,
                'pattern_name': 'Cascading Failure',
                'likely_cause': 'Initial resource constraint causing downstream effects',
                'recommendation': 'Address root cause at bottom of cascade'
            }
        
        # Pattern 4: Isolated Latency Issue
        if metrics_set == {'latency'}:
            return {
                'is_systemic': False,
                'pattern_name': 'Isolated Latency',
                'likely_cause': 'Network or application logic issue',
                'recommendation': 'Investigate network or specific service'
            }
        
        # Default
        return {
            'is_systemic': len(metrics_set) > 1,
            'pattern_name': 'Multi-metric Anomaly',
            'likely_cause': 'Possible correlated issues',
            'recommendation': 'Investigate metric correlations'
        }
    
    def predict_next_anomaly(self) -> Optional[Dict]:
        """
        Use learned patterns to predict likely next anomalies.
        
        ML Technique: Pattern-based prediction
        """
        if len(self.recent_anomalies) < 5:
            return None
        
        # Look at last few anomalies
        recent = list(self.recent_anomalies)[-5:]
        metrics_sequence = [a['metric'] for a in recent]
        
        # Check if we see a predictable pattern
        if len(set(metrics_sequence)) >= 2:
            # Multiple metrics involved - check for cascade
            if (metrics_sequence.count('cpu') >= 2 and 
                metrics_sequence.count('memory') >= 1):
                return {
                    'predicted_metric': 'latency',
                    'likelihood': 'high',
                    'reasoning': 'Resource exhaustion pattern detected - latency typically follows',
                    'recommended_preemptive_action': 'Prepare to scale latency-sensitive services'
                }
            
            if (metrics_sequence.count('latency') >= 2):
                return {
                    'predicted_metric': 'cpu',
                    'likelihood': 'medium',
                    'reasoning': 'Latency anomalies often preceded by CPU spikes',
                    'recommended_preemptive_action': 'Monitor CPU closely'
                }
        
        return None
    
    def get_correlation_summary(self) -> Dict:
        """Get summary of learned correlations."""
        correlations = self.analyze_current_correlations()
        
        return {
            'total_correlations': len(correlations),
            'strong_correlations': [
                {
                    'metrics': f"{c.metric1} <-> {c.metric2}",
                    'coefficient': round(c.correlation_coefficient, 3),
                    'strength': c.strength,
                    'cause': c.inferred_cause
                }
                for c in correlations if c.strength == 'strong'
            ],
            'recent_anomalies_count': len(self.recent_anomalies),
            'anomaly_patterns': self._summarize_patterns()
        }
    
    def _summarize_patterns(self) -> Dict:
        """Summarize patterns in recent anomalies."""
        if not self.recent_anomalies:
            return {}
        
        metrics_count = defaultdict(int)
        severity_count = defaultdict(int)
        
        for anomaly in self.recent_anomalies:
            metrics_count[anomaly['metric']] += 1
            severity_count[anomaly['severity']] += 1
        
        return {
            'by_metric': dict(metrics_count),
            'by_severity': dict(severity_count),
            'total': len(self.recent_anomalies)
        }
