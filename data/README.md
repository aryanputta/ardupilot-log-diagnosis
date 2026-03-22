# data/

This directory holds two things:
1. Sample labeled log metadata used to train and evaluate the fault classifier.
2. Instructions for downloading real ArduPilot flight logs to build the corpus.

Actual `.bin` and `.tlog` files are not stored in this repository because they
are binary assets that are often hundreds of MB each.

## Getting sample logs

### From the ArduPilot community log server

ArduPilot maintains a public log archive. Community members regularly upload
flight logs, including logs from known-failure flights:

```bash
# Browse logs by vehicle type and date
https://autotest.ardupilot.org/
```

### From SITL (Software-In-The-Loop) simulation

The fastest way to generate labeled training data is to inject known faults
in SITL and record the resulting logs. Each SITL parameter set maps to a
known fault label.

```bash
# Install ArduPilot SITL
git clone https://github.com/ArduPilot/ardupilot
cd ardupilot
./Tools/environment_install/install-prereqs-ubuntu.sh -y
./waf configure --board sitl
./waf copter

# Simulate GPS glitch fault
sim_vehicle.py -v ArduCopter --console --map \
  --add-param-file Tools/autotest/default_params/copter.parm \
  -P SIM_GPS_GLITCH_X=0.05  # inject 50m GPS position error

# The log is written to logs/ in the current directory
```

**Fault injection parameters for SITL training data:**

| Fault label        | SITL parameter               | Value  |
|--------------------|------------------------------|--------|
| gps_glitch         | SIM_GPS_GLITCH_X/Y/Z         | 0.05   |
| compass_error      | SIM_MAG_ERROR                | 90     |
| vibration_fault    | SIM_VIB_FREQ                 | 30     |
| baro_fault         | SIM_BARO_DRIFT               | 2.0    |
| rc_failsafe        | SIM_RC_FAIL                  | 1      |
| battery_failsafe   | BATT_LOW_VOLT (set low)      | 14.0   |

### From the ArduPilot discuss forum

Users often post logs when reporting issues. The ArduPilot log diagnosis
project aims to build a community-labeled dataset from these posts.

Relevant threads (good sources of labeled failure logs):
- https://discuss.ardupilot.org/c/copter/6
- https://discuss.ardupilot.org/c/google-summer-of-code/131

## Expected layout after downloading

```
data/
  raw/
    labeled/
      gps_glitch/
        *.bin       # logs confirmed as GPS glitch failures
      vibration_fault/
        *.bin
      ekf_failsafe/
        *.bin
      normal/
        *.bin       # clean flights with no notable fault
      ...
  rag_corpus.jsonl      # knowledge base for fix suggestions (see rag_pipeline.py)
  sample_labels.json    # example structure for training labels
```

## sample_labels.json format

Each entry maps a log filename to its ground truth fault label and
the time range (seconds) where the fault occurred:

```json
[
  {
    "file": "raw/labeled/gps_glitch/2024-10-15_flight.bin",
    "label": "gps_glitch",
    "fault_start_s": 142.3,
    "fault_end_s": 178.1,
    "notes": "GPS position jumped ~40m east, EKF switched core at t=165"
  },
  {
    "file": "raw/labeled/vibration_fault/quad_vibe_test.bin",
    "label": "vibration_fault",
    "fault_start_s": 0.0,
    "fault_end_s": 610.0,
    "notes": "VibeX > 35 throughout flight, Clip0 count = 347"
  }
]
```

## rag_corpus.jsonl format

Each line is a JSON object that becomes one document in the RAG vector store:

```json
{"fault_type": "gps_glitch", "title": "GPS Glitch Protection", "url": "https://ardupilot.org/copter/docs/gps-failsafe-glitch-protection.html", "text": "...excerpt from wiki page..."}
{"fault_type": "vibration_fault", "title": "Vibration Damping", "url": "https://ardupilot.org/copter/docs/vibration-damping.html", "text": "..."}
```

Run `python python/build_corpus.py` (to be added in Week 4 of the GSoC
project) to automatically scrape and embed the ArduPilot wiki into this file.
