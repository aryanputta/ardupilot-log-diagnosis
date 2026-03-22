# ArduPilot Log Diagnosis — AI-Assisted Root-Cause Detection

Pre-work prototype for the **AI-Assisted Log Diagnosis & Root-Cause Detection** project in [ArduPilot GSoC 2026](https://ardupilot.org/dev/docs/gsoc-ideas-list.html).

The ArduPilot brief asks for *"a model/service that flags likely root causes from logs and suggests fixes with confidence"* by *"learning from labeled log segments, known issue patterns, and parameter states."*  This prototype addresses all three:

| Brief requirement | How it is addressed |
|---|---|
| Labeled log segments | XGBoost multi-class classifier trained on SITL-generated logs with injected faults |
| Known issue patterns | RAG pipeline (ChromaDB + sentence-transformers) over ArduPilot wiki, discuss forum, GitHub issues |
| **Parameter states** | `param_checker.py` reads PARM messages and flags risky configurations linked to each fault type |
| Confidence score | `CalibratedClassifierCV(method='isotonic')` — calibrated probabilities, not raw softmax |
| Links to evidence in the log | `get_fault_windows()` returns exact timestamp ranges where anomaly signals were active |

---

## Quick start

```bash
pip install -r requirements.txt

# Heuristic mode — no trained model needed
python python/diagnose.py path/to/flight.bin

# With trained model
python python/diagnose.py path/to/flight.bin --model models/fault_classifier.pkl

# Machine-readable JSON (for API / Mission Planner integration)
python python/diagnose.py path/to/flight.bin --json
```

---

## Example output

```
=== ArduPilot Log Diagnosis ===
File:     flight_2024_10_15.bin
Duration: 487 s

Fault Diagnosis
----------------------------------------
  1. vibration_fault             (confidence: 87.2%)
  2. ekf_failsafe                (confidence:  9.1%)
  3. normal                      (confidence:  2.3%)

Fault Windows in Log  (jump to these regions in Mission Planner)
----------------------------------------
  T=312.4s - T=381.7s   VibeX > 30 m/s2
  T=334.1s - T=381.7s   VibeY > 30 m/s2
  T=378.2s - T=379.2s   ERR Subsys=EKFCHECK ECode=1

Evidence  (sensor features that crossed thresholds)
----------------------------------------
  vibe_vibex_p95                        42.3 m/s2    [threshold: >30.0 m/s2]
  vibe_clip0_total                     214.0 clips   [threshold: >100.0 clips]
  err_ekf_check_count                    3.0 events  [threshold: >0.0 events]

Parameter State Warnings
----------------------------------------
  [WARNING] INS_ACCEL_FILTER = 20
            INS_ACCEL_FILTER=20 Hz is high — reduces vibration filtering.
            Recommended <=10 Hz for most frames.
            Ref: https://ardupilot.org/copter/docs/common-imu-notch-filtering.html

  [CRITICAL] ARMING_CHECK = 0
             ARMING_CHECK=0 — ALL pre-arm checks are disabled. This masks
             configuration problems that would otherwise block arming.
             Ref: https://ardupilot.org/copter/docs/common-prearm-safety-checks.html

Suggested Fixes for: vibration_fault
----------------------------------------
  [1] High Vibration Causes EKF Divergence
       Source: https://ardupilot.org/copter/docs/vibration-damping.html
       Steps:
         - Balance all propellers with a prop balancer.
         - Check motor screws and bell housing for looseness.
         - Add foam vibration dampeners under the flight controller.
         - Set INS_ACCEL_FILTER to 10 Hz and re-test.
         - Enable In-Flight FFT notch filter if vibration persists.
```

---

## Architecture

```
.bin / .tlog
     |
     v
log_parser.py          parse_bin() / parse_tlog()
 +- per-type DataFrames (GPS, IMU, VIBE, BARO, EKF, ATT, BAT, ERR, RCIN, RCOU)
 +- parse_params()      -> Dict[param_name, value]    <- PARM messages
 +- get_fault_windows() -> [(start_s, end_s, reason)] <- timestamp-linked evidence
     |
     v
feature_extractor.py   extract_all()
 +- 60+ summary statistics per flight (mean / std / p95 / min / max per channel)
     |
     +---------------------------+
     v                           v
fault_classifier.py         param_checker.py
 XGBoost multi-class          check_params()
 + CalibratedClassifierCV     Flags known risky parameter
 -> [{fault, confidence}]     configs for the predicted fault
     |
     v
rag_pipeline.py        suggest()
 ChromaDB + sentence-transformers
 -> ordered fix steps + ArduPilot wiki links
     |
     v
diagnose.py            print_report() / --json
 Full report: faults | fault windows | evidence | param warnings | fixes
```

---

## param_checker.py — parameter state analysis

This module directly implements the **parameter states** requirement from the ArduPilot brief. It reads PARM messages logged at the start of every DataFlash flight and checks each value against rules derived from the ArduPilot wiki and known failure patterns.

The checks are fault-type specific:

- **vibration_fault** — `INS_ACCEL_FILTER`, `INS_GYRO_FILTER`, `MOT_THST_EXPO`
- **ekf_failsafe** — `EK2_CHECK_SCALE`, `EK3_CHECK_SCALE`, `FS_EKF_THRESH`
- **gps_glitch** — `GPS_HDOP_GOOD`, `GPS_MIN_ELEV`
- **compass_error** — `COMPASS_AUTODEC`, `COMPASS_MOT_X`
- **battery_failsafe** — `BATT_LOW_VOLT`, `BATT_LOW_MAH`
- **rc_failsafe** — `FS_THR_ENABLE`, `FS_THR_VALUE`
- **motor_imbalance** — `MOT_SPIN_MIN`, `MOT_BAT_VOLT_MAX`
- **baro_fault** — `GND_ALT_OFFSET`, `BARO_PROBE_EXT`
- **universal** (every run) — `ARMING_CHECK`, `LOG_BITMASK`

---

## Training data: SITL fault injection

Labels are generated automatically using ArduPilot SITL — not hand-annotated.

| Fault class | SITL parameter |
|---|---|
| `vibration_fault` | `SIM_VIB_FREQ` |
| `gps_glitch` | `SIM_GPS_GLITCH_X/Y` |
| `compass_error` | `SIM_MAG_ERROR` |
| `battery_failsafe` | `BATT_LOW_VOLT` |
| `rc_failsafe` | `SIM_RC_FAIL` |
| `baro_fault` | `SIM_BARO_DRIFT` |
| `ekf_failsafe` | `SIM_GPS_GLITCH` + `SIM_MAG_ERROR` combined |
| `motor_imbalance` | `SIM_MOT_FAIL_MSK` |

The injection and capture loop is scripted with pymavlink so it runs unattended. Target corpus for GSoC: 1,000+ labeled log segments.

---

## Fault taxonomy

| Label | Description |
|---|---|
| `normal` | No notable anomaly found |
| `gps_glitch` | GPS position/velocity innovation exceeded threshold |
| `compass_error` | Compass calibration failure or interference |
| `vibration_fault` | VIBE levels or clip counts exceeded safe limits |
| `ekf_failsafe` | EKF core switched or entered emergency mode |
| `battery_failsafe` | Battery voltage or capacity failsafe triggered |
| `rc_failsafe` | RC receiver signal lost |
| `baro_fault` | Barometer pressure spike or temperature anomaly |
| `motor_imbalance` | RCOU outputs show asymmetric motor loading |

---

## Repository structure

```
ardupilot-log-diagnosis/
+-- python/
|   +-- log_parser.py        # Parse .bin / .tlog + PARM messages + fault time windows
|   +-- feature_extractor.py # 60+ ML features: GPS, IMU, VIBE, EKF, ERR, BAT, RCIN, RCOU
|   +-- fault_classifier.py  # XGBoost multi-class + calibrated confidence + heuristic fallback
|   +-- param_checker.py     # Parameter state analysis — per-fault-type config rules
|   +-- rag_pipeline.py      # RAG fix suggestions (builtin KB + ChromaDB vector store)
|   +-- diagnose.py          # CLI entry point — full report or JSON output
+-- data/
|   +-- README.md            # Training data format, SITL injection instructions
+-- models/
|   +-- README.md            # Training workflow, model artifacts
+-- requirements.txt
```

---

## Technologies

**Python · XGBoost · scikit-learn · pymavlink · pandas · ChromaDB · sentence-transformers**

ArduPilot SITL used for automated training data generation via fault injection — no hand-labeling required.

## Related

- [ArduPilot DataFlash log documentation](https://ardupilot.org/copter/docs/common-downloading-and-analyzing-data-logs-in-mission-planner.html)
- [pymavlink DFReader](https://github.com/ArduPilot/pymavlink)
- [ArduPilot GSoC 2026 ideas list](https://ardupilot.org/dev/docs/gsoc-ideas-list.html)
