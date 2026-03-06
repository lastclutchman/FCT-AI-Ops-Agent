"""
Interactive Demo - Simulates realistic system metrics
"""

import random
import json
from datetime import datetime
from aiops_agent import AIOpsAgent
import time


class SystemSimulator:
    """Simulates realistic system behavior"""
    
    def __init__(self, scenario: str = 'normal'):
        self.scenario = scenario
        self.iteration = 0
        self.baseline_cpu = 30
        self.baseline_memory = 45
        self.baseline_latency = 100
        
    def get_metrics(self) -> dict:
        """Generate next set of metrics"""
        self.iteration += 1
        
        if self.scenario == 'normal':
            return self._normal_metrics()
        elif self.scenario == 'cpu_spike':
            return self._cpu_spike_metrics()
        elif self.scenario == 'memory_leak':
            return self._memory_leak_metrics()
        elif self.scenario == 'cascading':
            return self._cascading_failure_metrics()
        elif self.scenario == 'isolated_latency':
            return self._isolated_latency_metrics()
        else:
            return self._normal_metrics()
    
    def _normal_metrics(self) -> dict:
        cpu = self.baseline_cpu + random.uniform(-5, 5)
        memory = self.baseline_memory + random.uniform(-5, 5)
        latency = self.baseline_latency + random.uniform(-20, 20)
        return {
            'cpu': max(0, min(100, cpu)),
            'memory': max(0, min(100, memory)),
            'latency': max(1, latency)
        }
    
    def _cpu_spike_metrics(self) -> dict:
        cpu_trend = min(95, self.baseline_cpu + (self.iteration * 0.5))
        cpu = cpu_trend + random.uniform(-3, 3)
        memory = self.baseline_memory + random.uniform(-5, 5)
        latency = self.baseline_latency + random.uniform(-20, 20)
        return {
            'cpu': max(0, min(100, cpu)),
            'memory': max(0, min(100, memory)),
            'latency': max(1, latency)
        }
    
    def _memory_leak_metrics(self) -> dict:
        cpu = self.baseline_cpu + random.uniform(-5, 5)
        memory_trend = self.baseline_memory + (self.iteration * 0.3)
        memory = memory_trend + random.uniform(-2, 2)
        latency = self.baseline_latency + random.uniform(-20, 20)
        return {
            'cpu': max(0, min(100, cpu)),
            'memory': max(0, min(100, memory)),
            'latency': max(1, latency)
        }
    
    def _cascading_failure_metrics(self) -> dict:
        cpu_base = min(90, self.baseline_cpu + (self.iteration * 0.6))
        cpu = cpu_base + random.uniform(-2, 2)
        
        if self.iteration > 10:
            memory_base = self.baseline_memory + ((self.iteration - 10) * 0.5)
        else:
            memory_base = self.baseline_memory
        memory = memory_base + random.uniform(-2, 2)
        
        if self.iteration > 15:
            latency_base = self.baseline_latency + ((self.iteration - 15) * 2)
        else:
            latency_base = self.baseline_latency
        latency = latency_base + random.uniform(-5, 5)
        
        return {
            'cpu': max(0, min(100, cpu)),
            'memory': max(0, min(100, memory)),
            'latency': max(1, latency)
        }
    
    def _isolated_latency_metrics(self) -> dict:
        cpu = self.baseline_cpu + random.uniform(-5, 5)
        memory = self.baseline_memory + random.uniform(-5, 5)
        
        if self.iteration > 5 and self.iteration < 15:
            latency = 300 + random.uniform(-20, 20)
        else:
            latency = self.baseline_latency + random.uniform(-20, 20)
        
        return {
            'cpu': max(0, min(100, cpu)),
            'memory': max(0, min(100, memory)),
            'latency': max(1, latency)
        }


def run_demo_scenario(scenario: str, iterations: int = 30):
    """Run a complete demo scenario"""
    print(f"\n\n{'#'*70}")
    print(f"# AIOPS AGENT DEMO - Scenario: {scenario.upper()}")
    print(f"# Running {iterations} iterations")
    print(f"{'#'*70}\n")
    
    agent = AIOpsAgent()
    simulator = SystemSimulator(scenario=scenario)
    
    incident_reports = []
    
    for i in range(iterations):
        metrics = simulator.get_metrics()
        
        print(f"[Iteration {i+1}] Metrics: CPU={metrics['cpu']:.1f}%, " +
              f"Memory={metrics['memory']:.1f}%, Latency={metrics['latency']:.0f}ms")
        
        report = agent.ingest_metrics(metrics)
        
        if report:
            incident_reports.append(report)
            print(f"  ⚠️  INCIDENT DETECTED: {report.incident_id}")
            
            for anomaly in report.anomalies:
                print(f"     - {anomaly['metric'].upper()}: {anomaly['severity']} " +
                      f"(value={anomaly['value']}, method={anomaly['method']})")
            
            for action in report.actions_taken:
                print(f"     → ACTION: {action['action']} " +
                      f"(confidence={action['confidence']}, " +
                      f"impact={action['estimated_impact']})")
            
            if report.correlations:
                for corr in report.correlations:
                    print(f"     ~ Correlation: {corr['metric1']} <-> {corr['metric2']} " +
                          f"({corr['strength']})")
            
            if report.systemic_issue:
                print(f"     🔴 SYSTEMIC ISSUE: {report.systemic_issue['pattern']}")
                print(f"        Cause: {report.systemic_issue['likely_root_cause']}")
            
            if report.prediction:
                print(f"     🔮 PREDICTION: {report.prediction['predicted_metric']} " +
                      f"likely next ({report.prediction['likelihood']})")
        
        if i > 10 and incident_reports:
            oldest = incident_reports[0]
            if oldest.incident_id in agent.active_incidents:
                if agent.active_incidents[oldest.incident_id]['status'] == 'open':
                    agent.resolve_incident(
                        oldest.incident_id,
                        outcome='resolved',
                        metric_improvement=random.uniform(5, 15)
                    )
                    incident_reports.pop(0)
    
    print(f"\n\n{'='*70}")
    print("SCENARIO SUMMARY")
    print(f"{'='*70}")
    
    health = agent.get_system_health_report()
    print(json.dumps(health, indent=2, default=str))
    
    learning = agent.get_learning_summary()
    print(f"\nLearning Summary: {learning}")
    
    if incident_reports:
        print(f"\nRecent Incidents: {len(incident_reports)}")
    
    return agent, health, learning


def main():
    """Run all demo scenarios"""
    print("\n" + "="*70)
    print("AIOPS AGENT - Interactive Demonstration")
    print("="*70)
    print("\nThis demo showcases the AI-driven AIOps agent handling various")
    print("system scenarios with multiple specialized agents:")
    print("  • Anomaly Detector: Multi-method anomaly detection")
    print("  • Correlation Agent: Pattern recognition and root cause analysis")
    print("  • Response Agent: Autonomous decision-making and learning")
    print("="*70)
    
    scenarios = [
        ('normal', 15),
        ('cpu_spike', 25),
        ('memory_leak', 30),
        ('cascading', 30),
        ('isolated_latency', 20)
    ]
    
    for scenario, iterations in scenarios:
        agent, health, learning = run_demo_scenario(scenario, iterations)
        agent.export_incidents(f'incidents_{scenario}.json')
        print(f"\n✓ Incidents exported to incidents_{scenario}.json")
        time.sleep(0.5)
    
    print(f"\n\n{'='*70}")
    print("DEMO COMPLETE!")
    print(f"{'='*70}")
    print("Check the incidents_*.json files for detailed results")


if __name__ == '__main__':
    main()
