"""
log_parser.py
Reads ArduPilot DataFlash (.bin) and MAVLink telemetry (.tlog) files and
converts them into per-message-type pandas DataFrames ready for feature
extraction.

ArduPilot log message types used in diagnosis:
  GPS   - satellite count, fix type, hdop, position error
  IMU   - raw accelerometer/gyro, vibration clips
  ATT   - roll/pitch/yaw, desired vs actual (control tracking)
  BARO  - altitude, pressure, temperature
  EKF2/3- innovation variances, flags
  ERR   - ArduPilot error subsystem + error code  (primary fault signal)
  RCIN  - receiver channel values (RC failsafe detection)
  RCOU  - output PWM per motor (motor imbalance detection)
  BAT   - battery voltage, current, consumed mAh
  VIBE  - vibration levels per axis + clipping counts
  PARM  - all ArduPilot parameter values at the time of the flight
          (used by param_checker to detect mis-configurations)
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

ERR_SUBSYSTEM = {
    1: "MAIN", 2: "RADIO", 3: "COMPASS", 4: "OPTFLOW",
    5: "FAILSAFE_RADIO", 6: "FAILSAFE_BATT", 7: "FAILSAFE_GPS",
    8: "FAILSAFE_GCS", 9: "FAILSAFE_FENCE", 10: "FLIGHT_MODE",
    11: "GPS", 12: "CRASH_CHECK", 13: "FLIP", 14: "AUTOTUNE",
    15: "PARACHUTE", 16: "EKFCHECK", 17: "FAILSAFE_EKF",
    18: "BAROMETER", 19: "CPU", 20: "ARMING", 21: "WIND", 22: "MOTOR",
}


def parse_bin(log_path: str) -> Dict[str, pd.DataFrame]:
    """Parse a DataFlash .bin log into a dict of DataFrames, one per message type.

    pymavlink's DFReader is the reference parser for DataFlash; this wrapper
    normalises timestamps and converts to pandas for downstream ML use.
    TimeUS (microseconds since boot) is converted to time_s so all message
    types share a common time axis.
    """
    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise ImportError("pip install pymavlink") from exc

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
    """Parse a MAVLink telemetry (.tlog) log.

    .tlog files are ground-control-station recordings of MAVLink packets.
    Output format matches parse_bin() so feature_extractor can treat both
    identically.
    """
    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise ImportError("pip install pymavlink") from exc

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
    """Dispatch to parse_bin or parse_tlog based on file extension."""
    ext = Path(log_path).suffix.lower()
    if ext in (".bin", ".log"):
        return parse_bin(log_path)
    elif ext == ".tlog":
        return parse_tlog(log_path)
    else:
        raise ValueError(f"Unrecognised log extension '{ext}'. Expected .bin or .tlog.")


def parse_params(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Extract ArduPilot parameter values from PARM messages in the log.

    DataFlash logs include a PARM message for every parameter set at the time
    of the flight. param_checker.py uses this dict to flag known risky
    configurations (e.g. ARMING_CHECK=0, FS_THR_ENABLE=0, EK2_CHECK_SCALE<100)
    linked to specific fault types.

    This implements the 'parameter states' requirement from the ArduPilot GSoC
    brief: the classifier learns not just from sensor signals but from the
    vehicle's configuration at the time of the incident.

    Returns empty dict if no PARM messages are present (e.g. .tlog files).
    """
    if "PARM" not in frames:
        return {}
    parm_df = frames["PARM"]
    if "Name" not in parm_df.columns or "Value" not in parm_df.columns:
        return {}
    params: Dict[str, float] = {}
    for _, row in parm_df.iterrows():
        try:
            params[str(row["Name"])] = float(row["Value"])
        except (ValueError, TypeError):
            pass
    return params


def get_error_events(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extract ERR messages and annotate them with subsystem names.

    ERR messages are the most direct fault signal in DataFlash logs. The
    fault_classifier uses ERR counts and timing as high-weight features.
    """
    if "ERR" not in frames:
        return pd.DataFrame(columns=["time_s", "Subsys", "ECode", "subsys_name"])
    err = frames["ERR"].copy()
    err["subsys_name"] = err["Subsys"].map(ERR_SUBSYSTEM).fillna("UNKNOWN")
    return err[["time_s", "Subsys", "ECode", "subsys_name"]] if "time_s" in err.columns else err


def log_duration_seconds(frames: Dict[str, pd.DataFrame]) -> Optional[float]:
    """Return the total recording duration in seconds, or None if unavailable."""
    for msg_type in ("GPS", "IMU", "ATT"):
        if msg_type in frames and "time_s" in frames[msg_type].columns:
            col = frames[msg_type]["time_s"].dropna()
            if len(col) >= 2:
                return float(col.iloc[-1] - col.iloc[0])
    return None


def get_fault_windows(
    frames: Dict[str, pd.DataFrame],
    fault_type: str,
) -> List[Tuple[float, float, str]]:
    """Return (start_s, end_s, reason) tuples showing WHEN the fault was active.

    These are shown in the report so the user can jump directly to the relevant
    region in Mission Planner or UAV Log Viewer rather than searching manually.
    Provides the 'links to relevant evidence in the log' the ArduPilot brief
    requires — at timestamp resolution, not just feature-level statistics.
    """
    windows: List[Tuple[float, float, str]] = []

    if fault_type == "vibration_fault":
        if "VIBE" in frames:
            vibe = frames["VIBE"].dropna(subset=["time_s"])
            for axis in ("VibeX", "VibeY", "VibeZ"):
                if axis in vibe.columns:
                    high = vibe[vibe[axis] > 30.0]
                    if not high.empty:
                        windows.append((
                            round(float(high["time_s"].iloc[0]), 1),
                            round(float(high["time_s"].iloc[-1]), 1),
                            f"{axis} > 30 m/s2",
                        ))

    elif fault_type in ("ekf_failsafe", "gps_glitch", "compass_error",
                        "battery_failsafe", "rc_failsafe", "baro_fault",
                        "motor_imbalance"):
        subsys_map = {
            "ekf_failsafe":      [16, 17],
            "gps_glitch":        [7, 11],
            "compass_error":     [3],
            "battery_failsafe":  [6],
            "rc_failsafe":       [5],
            "baro_fault":        [18],
            "motor_imbalance":   [22],
        }
        targets = subsys_map.get(fault_type, [])
        err_df = get_error_events(frames)
        if not err_df.empty and targets:
            relevant = err_df[err_df["Subsys"].isin(targets)]
            for _, row in relevant.iterrows():
                windows.append((
                    round(float(row["time_s"]), 1),
                    round(float(row["time_s"]) + 1.0, 1),
                    f"ERR Subsys={row['subsys_name']} ECode={int(row['ECode'])}",
                ))

    return windows
