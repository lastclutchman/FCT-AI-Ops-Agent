from src.aiops_agent import AIOpsAgent
from src.simulator import TelemetrySimulator


def run_demo() -> None:
    simulator = TelemetrySimulator(seed=11)
    agent = AIOpsAgent(auto_confidence_threshold=0.67)

    healthy_data = simulator.healthy_window(300)
    agent.fit(healthy_data)
    print("✅ Baseline model trained on healthy telemetry")

    scenarios = ["cpu_spike_db_contention", "bad_release", "memory_leak"]

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario} ---")
        point = simulator.incident_point(scenario)
        decision = agent.process(point)

        print(f"anomaly={decision.is_anomaly}, severity={decision.severity:.2f}")
        print(f"root_cause={decision.root_cause}")
        print(f"action={decision.action}, confidence={decision.confidence:.2f}")
        print(f"explanation={decision.explanation}")

        if agent.should_auto_remediate(decision):
            outcome = simulator.apply_action(scenario, decision.action)
            agent.register_outcome(decision.action, outcome.reward)
            print(f"auto_remediation=YES | success={outcome.success} | reward={outcome.reward:.2f}")
            print(f"outcome_note={outcome.note}")
        else:
            print("auto_remediation=NO | escalated to operator")


if __name__ == "__main__":
    run_demo()
