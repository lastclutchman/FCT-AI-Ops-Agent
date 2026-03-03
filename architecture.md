# Architecture & Design Explanation

## 1) High-Level Flow

The prototype follows a classic AIOps closed-loop pattern:

1. **Observe**: ingest structured telemetry metrics.
2. **Detect**: hybrid anomaly scoring detects unusual behavior.
3. **Diagnose**: infer likely root cause from metric signatures.
4. **Decide**: choose best remediation action from a playbook.
5. **Act**: execute automated action if confidence threshold is met.
6. **Learn**: update action rewards from incident outcome feedback.

---

## 2) Components

### A) Telemetry Simulator (`src/simulator.py`)
- Produces healthy baseline windows and incident points.
- Encodes incident scenarios such as CPU/DB contention, bad release, and memory leak.
- Simulates action outcomes with rewards to create a feedback loop.

### B) AI Agent (`src/aiops_agent.py`)

#### Detection layer
- Normalizes incoming metrics by healthy baseline mean/std.
- Uses multivariate distance over normalized metrics as an unsupervised outlier signal.
- Combines distance signal + z-score severity into one anomaly severity value.

#### Diagnosis layer
- Applies interpretable root-cause signatures based on combinations of signals:
  - High CPU + high latency + high queue = saturation/DB contention.
  - High error + high latency = bad release/dependency failure.
  - High memory + latency = memory pressure/leak.

#### Decision layer
- Maps root cause to candidate playbook actions.
- Scores actions using:
  - historical reward memory,
  - severity-sensitive bias,
  - safety-sensitive bias.
- Produces action + confidence score.

#### Autonomous behavior
- Executes action only if confidence is above threshold.
- Otherwise escalates to human operator.

#### Learning loop
- Records reward for chosen action and reuses it in future decisions.

---

## 3) Why this is AI-Driven (even though simple)

- Uses **machine learning** for anomaly detection, not static thresholds.
- Uses **hybrid intelligence** (ML + symbolic heuristics) for practical interpretability.
- Uses **experience-based adaptation** through reward memory.
- Implements **autonomous policy gating** with confidence-based actioning.

---

## 4) Trade-offs

- Heuristic RCA is easy to explain but less expressive than causal/graph methods.
- Reward memory is simple and not full RL.
- Synthetic data speeds development but does not capture full production complexity.

---

## 5) Future Roadmap

- Replace heuristics with causal inference + topology-aware reasoning.
- Add incident timeline context (logs, traces, deployment events).
- Support multi-agent setup (detector agent, diagnostic agent, remediator agent).
- Integrate with real tools: Prometheus/Grafana/Kubernetes/Argo Rollouts.
- Add safety policies and rollback checks before autonomous execution.
