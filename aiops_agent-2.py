"""
AIOps Agent Orchestrator - Main entry point
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from anomaly_detector import AnomalyDetector, Anomaly
from response_agent import ResponseAgent
from correlation_agent import CorrelationAgent


@dataclass
class IncidentReport:
    """Incident report"""
    incident_id: str
    timestamp: str
    anomalies: List[Dict]
    actions_taken: List[Dict]
    correlations: List[Dict]
    systemic_issue: Optional[Dict] = None
    prediction: Optional[Dict] = None
    
    def to_dict(self):
        return {
            'incident_id': self.incident_id,
            'timestamp': self.timestamp,
            'anomalies': self.anomalies,
            'actions_taken': self.actions_taken,
            'correlations': self.correlations,
            'systemic_issue': self.systemic_issue,
            'prediction': self.prediction
        }


class AIOpsAgent:
    """Main orchestrator for the AIOps system"""
    
    def __init__(self):
        """Initialize the AIOps agent"""
        self.anomaly_detector = AnomalyDetector(window_size=100, sensitivity=2.0)
        self.response_agent = ResponseAgent()
        self.correlation_agent = CorrelationAgent(window_size=100)
        
        self.incident_counter = 0
        self.incident_history: List[IncidentReport] = []
        self.active_incidents: Dict[str, Dict] = {}
        
    def ingest_metrics(self, metrics: Dict[str, float]) -> Optional[IncidentReport]:
        """Ingest system metrics and process through all agents"""
        self.correlation_agent.update_metrics(metrics)
        
        # Detect anomalies
        anomalies = []
        for metric_name, value in metrics.items():
            anomaly = self.anomaly_detector.update_metric(metric_name, value)
            if anomaly:
                anomalies.append(anomaly)
                self.correlation_agent.record_anomaly(anomaly)
        
        # Handle anomalies
        if anomalies:
            return self._handle_anomalies(anomalies, metrics)
        
        return None
    
    def _handle_anomalies(self, anomalies: List[Anomaly], metrics: Dict[str, float]) -> IncidentReport:
        """Handle detected anomalies"""
        self.incident_counter += 1
        incident_id = f"INC-{self.incident_counter:05d}"
        timestamp = datetime.now().isoformat()
        
        anomaly_dicts = [a.to_dict() for a in anomalies]
        
        correlations = self.correlation_agent.analyze_current_correlations()
        correlation_dicts = [
            {
                'metric1': c.metric1,
                'metric2': c.metric2,
                'correlation': round(c.correlation_coefficient, 3),
                'strength': c.strength,
                'cause': c.inferred_cause
            }
            for c in correlations
        ]
        
        systemic_issue = self.correlation_agent.detect_systemic_issues(anomalies)
        prediction = self.correlation_agent.predict_next_anomaly()
        
        actions_taken = []
        for anomaly in anomalies:
            context = {
                'correlated_metrics': [a.metric_name for a in anomalies 
                                      if a.metric_name != anomaly.metric_name],
                'systemic': systemic_issue is not None,
                'trend': self._detect_trend(anomaly.metric_name),
                'recent_values': list(self.anomaly_detector.history[anomaly.metric_name])[-5:],
                'current_metrics': metrics
            }
            
            action = self.response_agent.decide_action(anomaly, context)
            
            if action:
                actions_taken.append(action.to_dict())
        
        report = IncidentReport(
            incident_id=incident_id,
            timestamp=timestamp,
            anomalies=anomaly_dicts,
            actions_taken=actions_taken,
            correlations=correlation_dicts,
            systemic_issue=systemic_issue,
            prediction=prediction
        )
        
        self.incident_history.append(report)
        self.active_incidents[incident_id] = {
            'report': report,
            'status': 'open',
            'created_at': timestamp
        }
        
        return report
    
    def _detect_trend(self, metric_name: str) -> str:
        """Detect if metric is trending up or down"""
        history = list(self.anomaly_detector.history[metric_name])
        
        if len(history) < 10:
            return 'unknown'
        
        recent = history[-5:]
        older = history[-10:-5]
        
        if len(older) == 0:
            return 'unknown'
        
        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)
        
        if older_mean == 0:
            return 'unknown'
        
        if recent_mean > older_mean * 1.1:
            return 'increasing'
        elif recent_mean < older_mean * 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    def resolve_incident(self, incident_id: str, outcome: str, metric_improvement: Optional[float] = None):
        """Mark incident as resolved"""
        if incident_id not in self.active_incidents:
            return
        
        incident = self.active_incidents[incident_id]
        incident['status'] = 'resolved'
        incident['resolved_at'] = datetime.now().isoformat()
        
        for action_dict in incident['report'].actions_taken:
            action = type('Action', (), {
                'action_type': type('ActionType', (), {'value': action_dict['action']})(),
                'metric_name': action_dict['metric'],
                'severity': action_dict['severity'],
                'reasoning': action_dict['reasoning'],
                'confidence': action_dict['confidence'],
                'estimated_impact': action_dict['estimated_impact'],
                'risk_level': action_dict['risk_level']
            })()
            
            self.response_agent.record_action_outcome(action, outcome, metric_improvement)
    
    def get_system_health_report(self) -> Dict:
        """Get comprehensive system health"""
        return {
            'timestamp': datetime.now().isoformat(),
            'incident_statistics': {
                'total_incidents': len(self.incident_history),
                'active_incidents': len([i for i in self.active_incidents.values() if i['status'] == 'open']),
                'resolved_incidents': len([i for i in self.active_incidents.values() if i['status'] == 'resolved']),
            },
            'metric_statistics': self.anomaly_detector.get_statistics(),
        }
    
    def get_incident_report(self, incident_id: str) -> Optional[Dict]:
        """Get detailed report for a specific incident"""
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            return {
                'incident': incident['report'].to_dict(),
                'status': incident['status'],
                'created_at': incident['created_at'],
                'resolved_at': incident.get('resolved_at'),
            }
        return None
    
    def get_recent_incidents(self, limit: int = 10) -> List[Dict]:
        """Get recent incident reports"""
        return [
            {
                'incident': report.to_dict(),
                'status': self.active_incidents.get(report.incident_id, {}).get('status', 'unknown')
            }
            for report in self.incident_history[-limit:]
        ]
    
    def export_incidents(self, filepath: str = 'incidents.json'):
        """Export incident history"""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_incidents': len(self.incident_history),
            'incidents': [report.to_dict() for report in self.incident_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def get_learning_summary(self) -> Dict:
        """Get summary of what the agent has learned"""
        return {
            'total_incidents': len(self.incident_history),
            'total_actions': sum(len(incident.actions_taken) for incident in self.incident_history)
        }
