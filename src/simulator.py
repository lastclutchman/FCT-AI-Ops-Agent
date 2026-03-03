from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ActionOutcome:
    success: bool
    reward: float
    note: str


class TelemetrySimulator:
    """Generates healthy telemetry and incident telemetry for demo/testing."""

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def healthy_window(self, n: int = 240) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = []
        for _ in range(n):
            points.append(
                {
                    "cpu": self._clip(self.rng.gauss(48, 8), 20, 75),
                    "memory": self._clip(self.rng.gauss(62, 7), 30, 82),
                    "latency_ms": self._clip(self.rng.gauss(120, 20), 70, 190),
                    "error_rate": self._clip(self.rng.gauss(0.015, 0.008), 0.0, 0.05),
                    "queue_depth": self._clip(self.rng.gauss(32, 14), 3, 90),
                }
            )
        return points

    def incident_point(self, scenario: str) -> Dict[str, float]:
        base = self.healthy_window(1)[0]
        if scenario == "cpu_spike_db_contention":
            base.update({"cpu": 93.0, "latency_ms": 430.0, "queue_depth": 165.0, "error_rate": 0.07})
        elif scenario == "bad_release":
            base.update({"error_rate": 0.18, "latency_ms": 320.0, "cpu": 79.0, "queue_depth": 95.0})
        elif scenario == "memory_leak":
            base.update({"memory": 96.0, "latency_ms": 245.0, "error_rate": 0.06})
        else:
            base.update({"latency_ms": 230.0, "queue_depth": 110.0, "error_rate": 0.08})
        return base

    def apply_action(self, scenario: str, action: str) -> ActionOutcome:
        action_table = {
            "cpu_spike_db_contention": {
                "scale_out_service": (True, 0.95, "Horizontal scale reduced queue pressure."),
                "shed_noncritical_load": (True, 0.70, "Latency improved with reduced workload."),
                "restart_unhealthy_pod": (False, -0.20, "Restart caused temporary disruption only."),
            },
            "bad_release": {
                "rollback_last_release": (True, 0.96, "Error rate normalized after rollback."),
                "switch_to_fallback_dependency": (True, 0.65, "Fallback reduced errors but increased cost."),
                "restart_unhealthy_pod": (False, -0.15, "Issue persisted after restart."),
            },
            "memory_leak": {
                "restart_unhealthy_pod": (True, 0.75, "Restart recovered memory temporarily."),
                "scale_out_service": (True, 0.60, "Scale-out diluted pressure."),
            },
        }
        default = (False, -0.1, "Action had limited impact.")
        success, reward, note = action_table.get(scenario, {}).get(action, default)
        return ActionOutcome(success=success, reward=reward, note=note)

    @staticmethod
    def _clip(value: float, min_value: float, max_value: float) -> float:
        return float(max(min_value, min(max_value, value)))
