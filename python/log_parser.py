"""
log_parser.py

Reads ArduPilot DataFlash (.bin) and MAVLink telemetry (.tlog) files
and converts them into per-message-type pandas DataFrames ready for
feature extraction.

This is the first stage of the AI-Assisted Log Diagnosis pipeline that
this project proposes for ArduPilot GSoC 2026. Every downstream module
(feature_extractor, fault_classifier, rag_pipeline) operates on the
dictionary produced here.

ArduPilot log message types used in diagnosis (non-exhaustive):
  GPS   - satellite count, fix type, hdop, position error
  IMU   - raw accelerometer/gyro, vibration clips
  ATT   - roll/pitch/yaw, desired vs actual (control tracking)
  BARO  - altitude, pressure, temperature
  EKF2/3 - innovation variances, flags (compass, velocity, position)
  ERR   - ArduPilot error subsystem + error code (the primary fault signal)
  RCIN  - receiver channel values (RC failsafe detection)
  RCOU  - output PWM per motor (motor imbalance detection)
  POWR  - board voltage, servo voltage
  BAT   - battery voltage, current, consumed mAh
  VIBE  - vibration levels per axis + clipping counts
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ArduPilot ERR subsystem codes (ref: libraries/AP_Logger/LogStructure.h)
ERR_SUBSYSTEM = {
    1: "MAIN",
    2: "RADIO",
    3: "COMPASS",
    4: "OPTFLOW",
    5: "FAILSAFE_RADIO",
    6: "FAILSAFE_BATT",
    7: "FAILSAFE_GPS",
    8: "FAILSAFE_GCS",
    9: "FAILSAFE_FENCE",
    10: "FLIGHT_MODE",
    11: "GPS",
    12: "CRASH_CHECK",
    13: "FLIP",
    14: "AUTOTUNE",
    15: "PARACHUTE",
    16: "EKFCHECK",
    17: "FAILSAFE_EKF",
    18: "BAROMETER",
    19: "CPU",
    20: "ARMING",
    21: "WIND",
    22: "MOTOR",
}


def parse_bin(log_path: str) -> Dict[str, pd.DataFrame]:
    """
    Parse a DataFlash .bin log into a dict of DataFrames, one per message type.

    pymavlink's DFReader is the reference parser for DataFlash; this wrapper
    normalises timestamps and converts to pandas for downstream ML use.

    The 'TimeUS' column (microseconds since boot) is converted to seconds and
    added as 'time_s' so all message types share a common time axis.

    Raises FileNotFoundError if the path does not exist.
    Raises ImportError if pymavlink is not installed.
    """
    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise ImportError(
            "pymavlink is required for .bin parsing: pip install pymavlink"
        ) from exc

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log not found: {log_path}")

    log = mavutil.mavlink_connection(log_path, robust_parsing=True, dialect="ardupilotmega")

    raw: Dict[str, list] = {}
    while True:
        msg = log.recv_match()
        if msg is None:
            break
        msg_type = msg.get_type()
        if msg_type in ("BAD_DATA", "EMPTY"):
            continue
        d = msg.to_dict()
        d.pop("mavpackettype", None)
        raw.setdefault(msg_type, []).append(d)

    frames: Dict[str, pd.DataFrame] = {}
    for msg_type, records in raw.items():
        df = pd.DataFrame(records)
        if "TimeUS" in df.columns:
            df["time_s"] = df["TimeUS"] / 1e6
        frames[msg_type] = df

    return frames


def parse_tlog(log_path: str) -> Dict[str, pd.DataFrame]:
    """
    Parse a MAVLink telemetry (.tlog) log.

    .tlog files are ground-control-station recordings of MAVLink packets.
    They contain a subset of the vehicle state (not the full DataFlash set)
    but are the only log type available for some failure scenarios.

    The parsed output format matches parse_bin() so feature_extractor.py
    can treat both identically.
    """
    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise ImportError(
            "pymavlink is required for .tlog parsing: pip install pymavlink"
        ) from exc

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log not found: {log_path}")

    mlog = mavutil.mavlink_connection(log_path)
    raw: Dict[str, list] = {}
    while True:
        msg = mlog.recv_match(blocking=False)
        if msg is None:
            break
        msg_type = msg.get_type()
        if msg_type in ("BAD_DATA",):
            continue
        d = msg.to_dict()
        d.pop("mavpackettype", None)
        d["_timestamp"] = getattr(msg, "_timestamp", None)
        raw.setdefault(msg_type, []).append(d)

    frames: Dict[str, pd.DataFrame] = {}
    for msg_type, records in raw.items():
        df = pd.DataFrame(records)
        if "_timestamp" in df.columns:
            df["time_s"] = df["_timestamp"]
        frames[msg_type] = df

    return frames


def parse_log(log_path: str) -> Dict[str, pd.DataFrame]:
    """
    Dispatch to parse_bin or parse_tlog based on file extension.
    Accepted extensions: .bin, .BIN, .tlog, .log
    """
    ext = Path(log_path).suffix.lower()
    if ext in (".bin", ".log"):
        return parse_bin(log_path)
    elif ext == ".tlog":
        return parse_tlog(log_path)
    else:
        raise ValueError(f"Unrecognised log extension '{ext}'. Expected .bin or .tlog.")


def get_error_events(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Extract ERR messages and annotate them with subsystem names.

    ERR messages are the most direct fault signal in DataFlash logs.
    The fault_classifier uses the count and timing of ERR events as
    high-weight features (they appear at the moment of ArduPilot's own
    internal fault detection, so they are strong labels for training data).
    """
    if "ERR" not in frames:
        return pd.DataFrame(columns=["time_s", "Subsys", "ECode", "subsys_name"])

    err = frames["ERR"].copy()
    err["subsys_name"] = err["Subsys"].map(ERR_SUBSYSTEM).fillna("UNKNOWN")
    return err[["time_s", "Subsys", "ECode", "subsys_name"]] if "time_s" in err.columns else err


def log_duration_seconds(frames: Dict[str, pd.DataFrame]) -> Optional[float]:
    """
    Return the total recording duration in seconds, or None if unavailable.
    Used to normalise per-second event-rate features.
    """
    for msg_type in ("GPS", "IMU", "ATT"):
        if msg_type in frames and "time_s" in frames[msg_type].columns:
            col = frames[msg_type]["time_s"].dropna()
            if len(col) >= 2:
                return float(col.iloc[-1] - col.iloc[0])
    return None
