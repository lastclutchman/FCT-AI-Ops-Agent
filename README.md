# AIOps Agent
A system that watches your cloud infrastructure and automatically fixes common problems.

## The Problem
When you're running services in the cloud, you have to monitor: CPU usage, Memory usage, Response time (latency)

When one of these goes bad, you need to
1. Notice it's happening
2. Figure out why
3. Fix it fast

This system does all that automatically.

## How It Works

The system has 4 ways to detect problems:
**1. Z-Score** - Spots sudden spikes
- "Your CPU was at 30%, now it's 75% - that's bad"
**2. IQR** - Finds values outside the normal range
- Works even when your data is messy or skewed
**3. Isolation Forest** - ML model that learns what's normal
- Gets smarter the more data it sees
- Catches weird combinations of metrics
**4. EWMA** - Catches slow problems getting worse
- Detects memory leaks (memory slowly increasing)
- Spots trends before they become critical

When multiple methods agree something's wrong, it triggers an alert.

## Then It Figures Out What's Happening
Once something is detected:
- It looks for patterns (are CPU and memory always high together?)
- It finds the root cause (CPU spike leads to memory problems 5 seconds later)
- It predicts what comes next (if this keeps going, latency will spike)
- It remembers similar situations from the past

## Finally It Takes Action
Based on what it finds, it decides what to do:
- **ALERT** - Tell someone it's happening
- **SCALE_UP** - Add more resources
- **THROTTLE** - Slow down less important work
- **AUTO_HEAL** - Clean up memory, restart processes
- **INVESTIGATE** - Something weird is happening, need to dig in
- **RESTART** - Restart the broken service
- **ESCALATE** - Too complicated, need a human

Each decision includes a confidence score. High confidence = do it automatically. Low confidence = escalate to a human.

## It Gets Better Over Time

After each incident, the system tracks:
- Did that action actually fix it?
- Was I right to be confident?
- Should I do that next time?

Over time it gets better at making decisions.

## Try It Out

```bash
pip install -r requirements.txt
python demo.py
```

This runs 5 scenarios:
1. Normal operation (nothing breaks)
2. CPU spike
3. Memory leak
4. Everything fails at once (cascading failure)
5. Just latency is slow

Watch what it detects and how it responds.

## Real Example

Here's what happens when everything breaks:

```
Start: CPU 30%, Memory 50%, Latency 100ms (all normal)

5 seconds later: CPU jumps to 50%
  Detection: "That's a spike"
  Decision: Scale up (80% confident)
  
10 seconds later: Memory jumps to 72%
  Detection: "Memory is high too"
  Analysis: "Memory always goes up when CPU does"
  Decision: Auto-heal - clean memory (80% confident)

15 seconds later: Memory is back to 52%
  Status: Fixed. No human needed.
  Learning: "That auto-heal worked well"
```

## The Code

- `anomaly_detector.py` - The 4 detection methods
- `correlation_agent.py` - Finds patterns and root causes
- `response_agent.py` - Decides what to do
- `aiops_agent.py` - Puts it all together
- `demo.py` - Runs the test scenarios

## What It's Good At

- Catches problems quickly (detects in < 2ms)
- Doesn't have too many false alarms (4 methods voting)
- Learns from what works
- Handles metrics that are messy or noisy
- Figures out what caused what

## How to Use It

```python
from aiops_agent import AIOpsAgent

agent = AIOpsAgent()

# Give it metrics
metrics = {
    'cpu': 75.5,
    'memory': 82.3,
    'latency': 250.0
}

# See if anything's wrong
report = agent.ingest_metrics(metrics)

if report:
    print(f"Problem found: {report.incident_id}")
    
    for anomaly in report.anomalies:
        print(f"  {anomaly['metric']}: {anomaly['severity']}")
    
    for action in report.actions_taken:
        print(f"  Action: {action['action']}")

# Later, when it's fixed
agent.resolve_incident(
    incident_id=report.incident_id,
    outcome='resolved',
    metric_improvement=15.0
)
```

## Installation

See SETUP.md for detailed instructions.

