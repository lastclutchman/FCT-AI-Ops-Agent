# AI-Driven AIOps Agent (Summer Student Program Prototype)

A lightweight, **AI-focused AIOps agent** that demonstrates how anomaly detection, root-cause inference, and autonomous remediation can be combined into one operational workflow.

> Goal: show thoughtful AI design and decision logic in a small prototype, not production hardening.

## What this project demonstrates

- AI/ML on operations telemetry (CPU, memory, latency, error rate, queue depth)
- Intelligent anomaly detection with a hybrid approach:
  - Statistical drift scoring (z-score against rolling baseline)
  - Multivariate distance score on normalized telemetry
- Root-cause inference using a simple signature-based reasoning layer
- Autonomous remediation decisions using confidence and policy scoring
- Learning loop from outcomes (incident memory with action reward tracking)

## Repository Structure

- `src/aiops_agent.py` – core agent logic (detect → diagnose → decide → act → learn)
- `src/simulator.py` – synthetic telemetry generator + incident scenarios
- `demo.py` – runnable end-to-end simulation
- `architecture.md` – architecture and design rationale
- `tests/test_agent.py` – basic behavioral tests
- `requirements.txt` – dependencies

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo.py
```

## Example Output (abridged)

- Agent trains baseline model from healthy telemetry
- Injected incident appears (`cpu_spike_db_contention`)
- Agent detects anomaly and predicts likely root cause
- Agent selects remediation (`scale_out_service` or `restart_unhealthy_pod`)
- Simulator returns outcome; agent updates action reward memory

## AI Techniques Used

1. **Unsupervised anomaly detection**: multivariate distance on normalized telemetry.
2. **Time-local baseline modeling**: rolling mean/std deviation to capture metric drift.
3. **Hybrid scoring**: combines model anomaly signal + statistical severity.
4. **Heuristic root-cause classifier**: interpretable signal signatures.
5. **Policy scoring with memory**: select actions from playbook based on confidence and historic reward.

## Design Assumptions

- Input is structured metric stream sampled every minute.
- Scope is single service boundary (can be extended to multi-service graph).
- Action outcomes are available quickly (simulator feedback loop).
- Safety policy allows auto-remediation only above confidence threshold.

## Future Improvements

- Replace heuristic RCA with causal graph or LLM-assisted incident reasoning.
- Add change-point detection and seasonality decomposition.
- Integrate with real observability backends (Prometheus, OpenTelemetry).
- Introduce human-in-the-loop approvals and policy guardrails.
- Use reinforcement learning for sequential remediation planning.

## Creating a Separate GitHub Repository

1. Create an empty repository in GitHub (e.g., `summer-aiops-agent`).
2. Push this code:

```bash
git init
git add .
git commit -m "Initial AI-driven AIOps agent prototype"
git branch -M main
git remote add origin <YOUR_NEW_REPO_URL>
git push -u origin main
```

## Video Demonstration Checklist (5–10 min)

- Problem statement and goals
- Agent architecture walkthrough (`architecture.md`)
- Live run of `python demo.py`
- Explain AI logic behind detection + decision making
- Challenges, trade-offs, and future roadmap
