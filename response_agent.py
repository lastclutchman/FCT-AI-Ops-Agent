"""
Response Agent - Autonomous decision-making for handling detected anomalies.

Uses AI-driven decision logic:
1. Rule-based reasoning with priority queues
2. Historical pattern matching
3. Risk assessment and impact estimation
4. Learning from past actions and outcomes
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
import json


class ActionType(Enum):
    """Types of autonomous actions the agent can take."""
    ALERT = "alert"  # Send alert only
    SCALE_UP = "scale_up"  # Scale up resources
    THROTTLE = "throttle"  # Reduce load
    RESTART = "restart"  # Restart service
    INVESTIGATE = "investigate"  # Trigger investigation
    AUTO_HEAL = "auto_heal"  # Attempt automatic recovery
    ESCALATE = "escalate"  # Escalate to human


@dataclass
class Action:
    """Represents an autonomous action."""
    action_type: ActionType
    metric_name: str
    severity: str
    reasoning: str
    confidence: float  # 0.0 to 1.0
    estimated_impact: str  # "low", "medium", "high"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    risk_level: str = "low"
    
    def to_dict(self):
        return {
            'action': self.action_type.value,
            'metric': self.metric_name,
            'severity': self.severity,
            'reasoning': self.reasoning,
            'confidence': round(self.confidence, 2),
            'estimated_impact': self.estimated_impact,
            'risk_level': self.risk_level,
            'timestamp': self.timestamp
        }


@dataclass
class ActionHistory:
    """Track action outcomes for learning."""
    action: Action
    executed: bool
    outcome: str  # 'resolved', 'worsened', 'no_change', 'pending'
    duration_seconds: float = 0.0
    metric_improvement: Optional[float] = None


class ResponseAgent:
    """
    Makes autonomous decisions on how to respond to anomalies.
    
    AI/ML Features:
    - Decision trees based on metric patterns
    - Confidence scoring for actions
    - Learning from historical outcomes
    - Risk assessment and mitigation
    - Multi-metric correlation analysis
    """
    
    def __init__(self):
        """Initialize the response agent."""
        self.action_history: List[ActionHistory] = []
        self.learned_responses: Dict[str, float] = {}  # metric -> success rate
        self.escalation_threshold = 0.3  # Escalate if confidence < 30%
        
    def decide_action(self, anomaly, context: Dict = None) -> Optional[Action]:
        """
        Decide on an autonomous action based on detected anomaly.
        
        Args:
            anomaly: Anomaly object from detector
            context: Additional context (recent metrics, trend, etc.)
            
        Returns:
            Action object or None
        """
        context = context or {}
        
        # Decision logic based on metric and severity
        if anomaly.metric_name == 'cpu':
            return self._handle_cpu_anomaly(anomaly, context)
        elif anomaly.metric_name == 'memory':
            return self._handle_memory_anomaly(anomaly, context)
        elif anomaly.metric_name == 'latency':
            return self._handle_latency_anomaly(anomaly, context)
        
        # Default: escalate if we can't decide
        return self._escalate_action(anomaly, "Unknown anomaly type")
    
    def _handle_cpu_anomaly(self, anomaly, context: Dict) -> Action:
        """
        Handle CPU anomalies with graduated response.
        
        AI Decision Logic:
        - LOW: Monitor only
        - MEDIUM: Scale up resources
        - HIGH/CRITICAL: Scale up AND throttle non-essential services
        """
        severity_level = anomaly.severity.value
        confidence = 0.7 + (severity_level * 0.1)  # Higher severity = higher confidence
        
        if severity_level <= 1:  # LOW
            return Action(
                action_type=ActionType.ALERT,
                metric_name='cpu',
                severity=anomaly.severity.name,
                reasoning="CPU usage elevated but within acceptable range. Monitoring.",
                confidence=0.8,
                estimated_impact="low",
                risk_level="low"
            )
        
        elif severity_level == 2:  # MEDIUM
            # Check if trend is worsening (use context)
            trend = context.get('trend', 'stable')
            action_type = ActionType.SCALE_UP if trend == 'increasing' else ActionType.INVESTIGATE
            
            reasoning = (
                f"CPU spike detected ({anomaly.value:.1f}%). "
                f"Trend is {trend}. Scaling up compute resources."
            )
            confidence = min(0.85, 0.7 + len(self.action_history) * 0.01)  # Learn from history
            
            return Action(
                action_type=action_type,
                metric_name='cpu',
                severity=anomaly.severity.name,
                reasoning=reasoning,
                confidence=confidence,
                estimated_impact="medium",
                risk_level="low"
            )
        
        else:  # HIGH or CRITICAL (severity_level >= 3)
            # Aggressive multi-action response
            reasoning = (
                f"CRITICAL CPU utilization ({anomaly.value:.1f}%). "
                f"Initiating emergency scaling and load reduction. "
                f"Detection method: {anomaly.detection_method}"
            )
            
            confidence = 0.9 + (min(severity_level - 2, 1) * 0.1)
            confidence = min(confidence, 1.0)
            
            return Action(
                action_type=ActionType.SCALE_UP,
                metric_name='cpu',
                severity=anomaly.severity.name,
                reasoning=reasoning,
                confidence=confidence,
                estimated_impact="high",
                risk_level="medium"
            )
    
    def _handle_memory_anomaly(self, anomaly, context: Dict) -> Action:
        """
        Handle memory anomalies.
        
        AI Decision Logic:
        - LOW: Monitor
        - MEDIUM: Investigate and potentially throttle
        - HIGH/CRITICAL: Attempt auto-heal (garbage collection) or restart
        """
        severity_level = anomaly.severity.value
        
        if severity_level <= 1:  # LOW
            return Action(
                action_type=ActionType.ALERT,
                metric_name='memory',
                severity=anomaly.severity.name,
                reasoning="Memory usage slightly elevated. No action needed.",
                confidence=0.85,
                estimated_impact="low",
                risk_level="low"
            )
        
        elif severity_level == 2:  # MEDIUM
            # Check recent trend
            recent_values = context.get('recent_values', [])
            is_growing = len(recent_values) > 2 and recent_values[-1] > recent_values[-2]
            
            action_type = ActionType.THROTTLE if is_growing else ActionType.INVESTIGATE
            reasoning = (
                f"Memory usage at {anomaly.value:.1f}%. "
                f"Memory {'growing' if is_growing else 'stable'}. "
                f"Throttling non-essential services."
            )
            
            return Action(
                action_type=action_type,
                metric_name='memory',
                severity=anomaly.severity.name,
                reasoning=reasoning,
                confidence=0.75,
                estimated_impact="medium",
                risk_level="medium"
            )
        
        else:  # HIGH or CRITICAL
            # Attempt automatic recovery
            action_type = ActionType.AUTO_HEAL  # Garbage collection / memory optimization
            
            reasoning = (
                f"CRITICAL memory usage ({anomaly.value:.1f}%). "
                f"Attempting automatic memory optimization. "
                f"Will escalate to restart if unsuccessful."
            )
            
            return Action(
                action_type=action_type,
                metric_name='memory',
                severity=anomaly.severity.name,
                reasoning=reasoning,
                confidence=0.8,
                estimated_impact="high",
                risk_level="high"
            )
    
    def _handle_latency_anomaly(self, anomaly, context: Dict) -> Action:
        """
        Handle latency anomalies.
        
        AI Decision Logic:
        - LOW: Monitor
        - MEDIUM: Scale up and investigate root cause
        - HIGH/CRITICAL: Scale up, throttle, and escalate if unknown cause
        """
        severity_level = anomaly.severity.value
        
        if severity_level <= 1:  # LOW
            return Action(
                action_type=ActionType.ALERT,
                metric_name='latency',
                severity=anomaly.severity.name,
                reasoning="Latency slightly elevated. Continuing monitoring.",
                confidence=0.8,
                estimated_impact="low",
                risk_level="low"
            )
        
        elif severity_level == 2:  # MEDIUM
            # Check if correlated with other metrics
            correlated_metrics = context.get('correlated_metrics', [])
            
            reasoning = f"Latency spike ({anomaly.value:.0f}ms). "
            if correlated_metrics:
                reasoning += f"Correlated with: {', '.join(correlated_metrics)}. "
            reasoning += "Scaling resources and investigating."
            
            return Action(
                action_type=ActionType.SCALE_UP,
                metric_name='latency',
                severity=anomaly.severity.name,
                reasoning=reasoning,
                confidence=0.78,
                estimated_impact="medium",
                risk_level="low"
            )
        
        else:  # HIGH or CRITICAL
            correlated = context.get('correlated_metrics', [])
            
            if len(correlated) > 1:
                # Multiple metrics affected = systemic issue
                action_type = ActionType.ESCALATE
                reasoning = (
                    f"CRITICAL latency ({anomaly.value:.0f}ms) with multiple "
                    f"correlated anomalies ({', '.join(correlated)}). "
                    f"Requires human analysis."
                )
                confidence = 0.7
            else:
                # Isolated latency issue = try to fix automatically
                action_type = ActionType.SCALE_UP
                reasoning = (
                    f"CRITICAL latency spike ({anomaly.value:.0f}ms). "
                    f"Scaling up capacity."
                )
                confidence = 0.85
            
            return Action(
                action_type=action_type,
                metric_name='latency',
                severity=anomaly.severity.name,
                reasoning=reasoning,
                confidence=confidence,
                estimated_impact="high",
                risk_level="high"
            )
    
    def _escalate_action(self, anomaly, reason: str) -> Action:
        """Generate an escalation action."""
        return Action(
            action_type=ActionType.ESCALATE,
            metric_name=anomaly.metric_name,
            severity=anomaly.severity.name,
            reasoning=f"Unable to auto-remediate: {reason}. Escalating to ops team.",
            confidence=0.5,
            estimated_impact="high",
            risk_level="high"
        )
    
    def record_action_outcome(self, action: Action, outcome: str, 
                              metric_improvement: Optional[float] = None):
        """
        Record the outcome of an action (for learning).
        
        This enables the agent to learn which actions are most effective
        for different scenarios.
        """
        history = ActionHistory(
            action=action,
            executed=True,
            outcome=outcome,
            metric_improvement=metric_improvement
        )
        self.action_history.append(history)
        
        # Update learned success rates
        action_type = action.action_type.value if isinstance(action.action_type, ActionType) else action.action_type
        key = f"{action.metric_name}_{action_type}"
        if key not in self.learned_responses:
            self.learned_responses[key] = 0.0
        
        # Success = resolved or improvement
        success = (outcome == 'resolved') or (metric_improvement is not None and metric_improvement > 0)
        
        # Exponential moving average of success rate
        alpha = 0.3
        self.learned_responses[key] = (
            alpha * (1.0 if success else 0.0) + 
            (1 - alpha) * self.learned_responses[key]
        )
    
    def get_action_effectiveness(self) -> Dict:
        """Return metrics about action effectiveness based on history."""
        if not self.action_history:
            return {}
        
        effectiveness = {}
        for key, success_rate in self.learned_responses.items():
            metric, action = key.split('_', 1)
            if metric not in effectiveness:
                effectiveness[metric] = {}
            effectiveness[metric][action] = round(success_rate, 3)
        
        # Overall stats
        total_actions = len(self.action_history)
        resolved = sum(1 for h in self.action_history if h.outcome == 'resolved')
        
        return {
            'total_actions': total_actions,
            'resolved': resolved,
            'success_rate': round(resolved / total_actions, 2) if total_actions > 0 else 0.0,
            'effectiveness_by_metric': effectiveness
        }
