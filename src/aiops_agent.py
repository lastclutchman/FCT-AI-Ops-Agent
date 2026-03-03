from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from statistics import mean, pstdev
from typing import Dict, List, Tuple


METRIC_ORDER = ["cpu", "memory", "latency_ms", "error_rate", "queue_depth"]


@dataclass
class IncidentDecision:
    is_anomaly: bool
    severity: float
    root_cause: str
    action: str
    confidence: float
    explanation: str


class AIOpsAgent:
    """Simple AI-driven AIOps agent with detect-diagnose-decide-act loop."""

    def __init__(self, auto_confidence_threshold: float = 0.65):
        self.auto_confidence_threshold = auto_confidence_threshold
        self.feature_mean: Dict[str, float] = {}
        self.feature_std: Dict[str, float] = {}
        self.memory_rewards: Dict[str, List[float]] = {}

    def fit(self, healthy_window: List[Dict[str, float]]) -> None:
        for metric in METRIC_ORDER:
            values = [p[metric] for p in healthy_window]
            self.feature_mean[metric] = mean(values)
            self.feature_std[metric] = pstdev(values) + 1e-6

    def process(self, point: Dict[str, float]) -> IncidentDecision:
        if not self.feature_mean:
            raise RuntimeError("Agent must be fit on healthy telemetry before processing incidents.")

        z_scores = {
            metric: abs((point[metric] - self.feature_mean[metric]) / self.feature_std[metric])
            for metric in METRIC_ORDER
        }
        stat_score = mean(list(z_scores.values()))

        # Multivariate distance as a simple unsupervised outlier score.
        distance = sqrt(sum(z**2 for z in z_scores.values())) / len(METRIC_ORDER)
        severity = self._sigmoid(1.1 * distance + 0.6 * stat_score - 1.2)
        is_anomaly = severity > 0.55

        if not is_anomaly:
            return IncidentDecision(
                is_anomaly=False,
                severity=severity,
                root_cause="healthy",
                action="none",
                confidence=1.0 - severity,
                explanation="Telemetry appears within healthy baseline envelope.",
            )

        root_cause, rc_conf = self._infer_root_cause(point)
        action, action_conf = self._choose_action(root_cause, severity)

        confidence = min(0.99, 0.5 * severity + 0.3 * rc_conf + 0.2 * action_conf)
        explanation = (
            f"Anomaly detected (severity={severity:.2f}). "
            f"Root cause hypothesis: {root_cause} (confidence={rc_conf:.2f}). "
            f"Selected action: {action} (policy_score={action_conf:.2f})."
        )

        return IncidentDecision(
            is_anomaly=True,
            severity=severity,
            root_cause=root_cause,
            action=action,
            confidence=confidence,
            explanation=explanation,
        )

    def should_auto_remediate(self, decision: IncidentDecision) -> bool:
        return decision.is_anomaly and decision.confidence >= self.auto_confidence_threshold

    def register_outcome(self, action: str, reward: float) -> None:
        self.memory_rewards.setdefault(action, []).append(reward)

    def _infer_root_cause(self, point: Dict[str, float]) -> Tuple[str, float]:
        cpu = point["cpu"]
        latency = point["latency_ms"]
        err = point["error_rate"]
        queue = point["queue_depth"]
        mem = point["memory"]

        if cpu > 85 and latency > 300 and queue > 100:
            return "compute_saturation_or_db_contention", 0.84
        if err > 0.09 and latency > 250:
            return "failing_dependency_or_bad_release", 0.80
        if mem > 90 and latency > 180:
            return "memory_pressure_or_leak", 0.76
        if queue > 120 and cpu < 70:
            return "downstream_throttling", 0.73
        return "unknown_transient_or_multi_factor", 0.55

    def _choose_action(self, root_cause: str, severity: float) -> Tuple[str, float]:
        playbooks = {
            "compute_saturation_or_db_contention": ["scale_out_service", "shed_noncritical_load", "restart_unhealthy_pod"],
            "failing_dependency_or_bad_release": ["rollback_last_release", "switch_to_fallback_dependency", "restart_unhealthy_pod"],
            "memory_pressure_or_leak": ["restart_unhealthy_pod", "scale_out_service"],
            "downstream_throttling": ["throttle_ingress", "enable_queue_backpressure", "switch_to_fallback_dependency"],
            "unknown_transient_or_multi_factor": ["restart_unhealthy_pod", "collect_diagnostics_only"],
        }
        candidates = playbooks[root_cause]

        def policy_score(action: str) -> float:
            historical = self.memory_rewards.get(action, [])
            historical_avg = mean(historical) if historical else 0.0
            severity_bias = 0.2 if action in {"scale_out_service", "rollback_last_release"} and severity > 0.8 else 0.0
            safe_bias = 0.1 if action in {"collect_diagnostics_only", "throttle_ingress"} and severity < 0.7 else 0.0
            return historical_avg + severity_bias + safe_bias

        scored = [(a, policy_score(a)) for a in candidates]
        action, score = max(scored, key=lambda t: t[1])
        confidence = self._sigmoid(2.0 * score + 0.8 * severity)
        return action, confidence

    @staticmethod
    def _sigmoid(v: float) -> float:
        return 1 / (1 + exp(-v))
