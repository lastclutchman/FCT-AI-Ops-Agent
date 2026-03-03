from src.aiops_agent import AIOpsAgent
from src.simulator import TelemetrySimulator


def test_detects_incident_and_selects_action():
    sim = TelemetrySimulator(seed=100)
    agent = AIOpsAgent(auto_confidence_threshold=0.67)
    agent.fit(sim.healthy_window(220))

    point = sim.incident_point("cpu_spike_db_contention")
    decision = agent.process(point)

    assert decision.is_anomaly is True
    assert decision.action != "none"
    assert decision.root_cause != "healthy"


def test_healthy_point_mostly_not_anomalous():
    sim = TelemetrySimulator(seed=123)
    agent = AIOpsAgent(auto_confidence_threshold=0.67)
    healthy = sim.healthy_window(250)
    agent.fit(healthy)

    decision = agent.process(healthy[0])
    assert decision.severity < 0.75
