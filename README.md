# ardupilot-log-diagnosis

AI-assisted log diagnosis and root-cause detection for ArduPilot vehicles.

Pre-work for the **AI-Assisted Log Diagnosis & Root-Cause Detection** project in
[ArduPilot GSoC 2026](https://ardupilot.org/dev/docs/gsoc-ideas-list.html).

The goal is a service that reads an ArduPilot `.bin` or `.tlog` file and
outputs the most likely fault type, a confidence score, evidence segments
from the log, and a ranked list of suggested fixes with links to sources.

## Structure

```
ardupilot-log-diagnosis/
├── python/
│   ├── log_parser.py          # Parse DataFlash (.bin) and MAVLink (.tlog) logs
│   ├── feature_extractor.py   # Extract ML features (GPS, IMU, VIBE, ERR, EKF, ...)
│   ├── fault_classifier.py    # XGBoost classifier + heuristic fallback
│   ├── rag_pipeline.py        # RAG fix suggestions (builtin KB + ChromaDB)
│   └── diagnose.py            # CLI entry point
├── data/
│   └── README.md              # How to get logs + training data format
├── models/
│   └── README.md              # How to train + download model artifacts
├── requirements.txt
└── .gitignore
```

## Quick start

```bash
pip install -r requirements.txt

# Diagnose a flight log (heuristic mode — no trained model needed)
python python/diagnose.py path/to/flight.bin

# With trained model
python python/diagnose.py path/to/flight.bin --model models/fault_classifier.pkl

# Machine-readable JSON output
python python/diagnose.py path/to/flight.bin --json
```

Example output:

```
=== ArduPilot Log Diagnosis ===
File:     flight_2024_10_15.bin
Duration: 487 s

Fault Diagnosis
----------------------------------------
  1. vibration_fault              (confidence: 87.2%)
  2. ekf_failsafe                 (confidence:  9.1%)
  3. normal                       (confidence:  2.3%)

Suggested Fixes for: vibration_fault
----------------------------------------
  [1] High Vibration Causes EKF Divergence
      Source: https://ardupilot.org/copter/docs/vibration-damping.html
      Steps:
        - Balance all propellers with a prop balancer.
        - Check motor screws and bell for looseness.
        - Add foam vibration dampeners under the flight controller.

Evidence in log
----------------------------------------
  vibe_vibex_p95                      42.3 m/s^2    [threshold: >30.0 m/s^2]
  vibe_clip0_total                   214.0 clips    [threshold: >100.0 clips]
  err_ekf_check_count                  3.0 events   [threshold: >0.0 events]
```

## Fault taxonomy

| Label              | Description                                              |
|--------------------|----------------------------------------------------------|
| normal             | No notable anomaly found                                |
| gps_glitch         | GPS position/velocity innovation exceeded threshold     |
| compass_error      | Compass calibration failure or interference             |
| vibration_fault    | VIBE levels or clip counts exceeded safe limits         |
| ekf_failsafe       | EKF core switched or entered emergency mode             |
| battery_failsafe   | Battery voltage or capacity failsafe triggered          |
| rc_failsafe        | RC receiver signal lost                                 |
| baro_fault         | Barometer pressure spike or temperature anomaly         |
| motor_imbalance    | RCOU outputs show asymmetric motor loading              |

## ML pipeline

```
.bin / .tlog
     |
     v
log_parser.parse_log()          DataFlash -> dict of DataFrames per message type
     |
     v
feature_extractor.extract_all() Summary stats + domain counts -> flat feature vector
     |
     v
fault_classifier.predict()      XGBoost multi-class -> ranked fault list + confidence
     |
     v
rag_pipeline.suggest()          KB / vector retrieval -> ordered fix steps + links
```

## Why this matters for ArduPilot

Manual log diagnosis requires expert knowledge of DataFlash message types,
parameter tuning thresholds, and historical issue patterns. Automating this
lowers the barrier for new pilots and developers to identify and fix problems,
and could be integrated directly into Mission Planner or MAVProxy as a
one-click diagnostic tool.

## Technologies

**Primary:** Python, scikit-learn, XGBoost, pymavlink, pandas

**General topics:** Machine Learning, Robotics, ArduPilot, Log Analysis,
Retrieval-Augmented Generation, Fault Detection, Autonomous Systems

## Related work

- [ArduPilot DataFlash log documentation](https://ardupilot.org/dev/docs/common-downloading-and-analyzing-data-logs-in-mission-planner.html)
- [pymavlink DFReader](https://github.com/ArduPilot/pymavlink)
- [RAG with sentence-transformers](https://www.sbert.net/)
- Past GSoC: AI Chat WebTool for MP/QGC (GSoC 2025)
