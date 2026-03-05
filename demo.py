"""
Interactive Demo - Simulates realistic system metrics and demonstrates agent behavior.

Scenarios:
1. Normal operation
2. Gradual CPU increase (controlled scaling scenario)
3. Memory leak (cascading failure)
4. Network latency spike (isolated issue)
5. System recovery
"""

import random
import json
from datetime import datetime
from aiops_agent import AIOpsAgent
import time


class SystemSimulator:
    """Simulates realistic system behavior with configurable scenarios."""
    
    def __init__(self, scenario: str = 'normal'):
        """
        Initialize simulator.
        
        Scenarios:
        - 'normal': Stable operation
        - 'cpu_spike': CPU usage increases
        - 'memory_leak': Memory gradually increases
        - 'cascading': CPU leads to memory then latency
        - 'isolated_latency': Latency spike without resource issues
        """
        self.scenario = scenario
        self.iteration = 0
        self.baseline_cpu = 30
        self.baseline_memory = 45
        self.baseline_latency = 100
        
    def get_metrics(self) -> dict:
        """Generate next set of metrics based on scenario."""
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
        """Normal operation with slight random variation."""
        cpu = self.baseline_cpu + random.uniform(-5, 5)
        memory = self.baseline_memory + random.uniform(-5, 5)
        latency = self.baseline_latency + random.uniform(-20, 20)
        
        return {
            'cpu': max(0, min(100, cpu)),
            'memory': max(0, min(100, memory)),
            'latency': max(1, latency)
        }
    
    def _cpu_spike_metrics(self) -> dict:
        """Gradual CPU spike scenario."""
        # Gradual increase over time
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
        """Memory leak scenario - gradual memory increase."""
        cpu = self.baseline_cpu + random.uniform(-5, 5)
        
        # Gradual memory increase (leak)
        memory_trend = self.baseline_memory + (self.iteration * 0.3)
        memory = memory_trend + random.uniform(-2, 2)
        
        # Latency stays normal initially
        latency = self.baseline_latency + random.uniform(-20, 20)
        
        return {
            'cpu': max(0, min(100, cpu)),
            'memory': max(0, min(100, memory)),
            'latency': max(1, latency)
        }
    
    def _cascading_failure_metrics(self) -> dict:
        """Cascading failure: CPU spike -> memory pressure -> latency."""
        # CPU increases first
        cpu_base = min(90, self.baseline_cpu + (self.iteration * 0.6))
        cpu = cpu_base + random.uniform(-2, 2)
        
        # Memory responds to CPU (lag effect)
        if self.iteration > 10:
            memory_base = self.baseline_memory + ((self.iteration - 10) * 0.5)
        else:
            memory_base = self.baseline_memory
        memory = memory_base + random.uniform(-2, 2)
        
        # Latency responds to both (lag effect)
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
        """Isolated latency spike without resource issues."""
        cpu = self.baseline_cpu + random.uniform(-5, 5)
        memory = self.baseline_memory + random.uniform(-5, 5)
        
        # Latency spike
        if self.iteration > 5 and self.iteration < 15:
            latency = 300 + random.uniform(-20, 20)
        else:
            latency = self.baseline_latency + random.uniform(-20, 20)
        
        return {
            'cpu': max(0, min(100, cpu)),
            'memory': max(0, min(100, memory)),
            'latency': max(1, latency)
        }


def print_formatted_output(title: str, data: dict, indent: int = 0):
    """Pretty print output."""
    prefix = "  " * indent
    print(f"\n{prefix}{'='*70}")
    print(f"{prefix}{title}")
    print(f"{prefix}{'='*70}")
    print(json.dumps(data, indent=2, default=str))


def run_demo_scenario(scenario: str, iterations: int = 30):
    """
    Run a complete demo scenario.
    
    Shows:
    1. Metric ingestion
    2. Anomaly detection
    3. Decision making
    4. Learning outcomes
    """
    print(f"\n\n{'#'*70}")
    print(f"# AIOPS AGENT DEMO - Scenario: {scenario.upper()}")
    print(f"# Running {iterations} iterations")
    print(f"{'#'*70}\n")
    
    # Initialize
    agent = AIOpsAgent()
    simulator = SystemSimulator(scenario=scenario)
    
    incident_reports = []
    
    # Simulation loop
    for i in range(iterations):
        # Get simulated metrics
        metrics = simulator.get_metrics()
        
        print(f"\n[Iteration {i+1}] Metrics: CPU={metrics['cpu']:.1f}%, " +
              f"Memory={metrics['memory']:.1f}%, Latency={metrics['latency']:.0f}ms")
        
        # Ingest metrics into agent
        report = agent.ingest_metrics(metrics)
        
        # If anomaly detected, show incident details
        if report:
            incident_reports.append(report)
            print(f"  ⚠️  INCIDENT DETECTED: {report.incident_id}")
            
            # Show anomalies
            for anomaly in report.anomalies:
                print(f"     - {anomaly['metric'].upper()}: {anomaly['severity']} " +
                      f"(value={anomaly['value']}, method={anomaly['method']})")
            
            # Show actions
            for action in report.actions_taken:
                print(f"     → ACTION: {action['action']} " +
                      f"(confidence={action['confidence']}, " +
                      f"impact={action['estimated_impact']})")
            
            # Show correlations if found
            if report.correlations:
                for corr in report.correlations:
                    print(f"     ~ Correlation: {corr['metric1']} <-> {corr['metric2']} " +
                          f"({corr['strength']})")
            
            # Show systemic issue if detected
            if report.systemic_issue:
                print(f"     🔴 SYSTEMIC ISSUE: {report.systemic_issue['pattern']}")
                print(f"        Cause: {report.systemic_issue['likely_root_cause']}")
            
            # Show prediction if available
            if report.prediction:
                print(f"     🔮 PREDICTION: {report.prediction['predicted_metric']} " +
                      f"likely next ({report.prediction['likelihood']})")
        
        # Simulate learning: mark resolved incidents
        if i > 10 and incident_reports:
            # Mark oldest incident as resolved
            oldest = incident_reports[0]
            if oldest.incident_id in agent.active_incidents:
                if agent.active_incidents[oldest.incident_id]['status'] == 'open':
                    agent.resolve_incident(
                        oldest.incident_id,
                        outcome='resolved',
                        metric_improvement=random.uniform(5, 15)
                    )
                    incident_reports.pop(0)
    
    # Final summary
    print(f"\n\n{'='*70}")
    print("SCENARIO SUMMARY")
    print(f"{'='*70}")
    
    health = agent.get_system_health_report()
    print_formatted_output("System Health Report", health)
    
    learning = agent.get_learning_summary()
    print_formatted_output("Agent Learning Summary", learning)
    
    if incident_reports:
        print_formatted_output(
            "Recent Incidents",
            [r.to_dict() for r in incident_reports[-3:]]
        )
    
    return agent, health, learning


def main():
    """Run all demo scenarios."""
    print("\n" + "="*70)
    print("AIOPS AGENT - Interactive Demonstration")
    print("="*70)
    print("\nThis demo showcases the AI-driven AIOps agent handling various")
    print("system scenarios with multiple specialized agents:")
    print("  • Anomaly Detector: Multi-method anomaly detection")
    print("  • Correlation Agent: Pattern recognition and root cause analysis")
    print("  • Response Agent: Autonomous decision-making and learning")
    print("="*70)
    
    # Run selected scenarios
    scenarios = [
        ('normal', 15),           # Baseline
        ('cpu_spike', 25),        # Gradual problem
        ('memory_leak', 30),      # Resource issue
        ('cascading', 30),        # Systemic failure
        ('isolated_latency', 20)  # Single metric issue
    ]
    
    results = {}
    
    for scenario, iterations in scenarios:
        agent, health, learning = run_demo_scenario(scenario, iterations)
        results[scenario] = {
            'health': health,
            'learning': learning
        }
        
        # Save incidents for reference
        agent.export_incidents(f'incidents_{scenario}.json')
        print(f"\n✓ Incidents exported to incidents_{scenario}.json")
        
        # Brief pause between scenarios
        time.sleep(0.5)
    
    # Comparative summary
    print(f"\n\n{'='*70}")
    print("COMPARATIVE ANALYSIS ACROSS SCENARIOS")
    print(f"{'='*70}")
    
    for scenario in scenarios:
        scenario_name = scenario[0]
        learning = results[scenario_name]['learning']
        incidents = learning['total_learning_examples']
        avg_confidence = learning['confidence_in_decisions']
        
        print(f"\n{scenario_name.upper()}:")
        print(f"  Total Incidents: {incidents}")
        print(f"  Avg Decision Confidence: {avg_confidence:.1%}")
        if learning['action_effectiveness']:
            print(f"  Success Rate: {learning['action_effectiveness'].get('success_rate', 'N/A')}")


if __name__ == '__main__':
    main()
