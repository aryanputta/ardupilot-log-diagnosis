"""
feature_extractor.py

Converts the DataFrames produced by log_parser.parse_log() into a flat
feature vector that fault_classifier.py can train on or score.

Why this design: ArduPilot logs are variable-length time series. A classifier
needs a fixed-size input. The approach here computes summary statistics
(mean, std, max, percentiles) plus domain-specific counts (error events,
vibration clip counts, EKF flag activations) for each relevant message type.
This keeps the feature set interpretable and avoids the need for a sequence
model, which would require far more labeled training data than is currently
available.

Feature groups:
  gps_*      - satellite count, hdop, fix quality, position innovation
  imu_*      - accelerometer/gyro mean & std per axis
  vibe_*     - vibration levels, clip counts (main cause of EKF divergence)
  baro_*     - altitude variance, pressure deviation
  ekf_*      - innovation variance flags, core switch count
  att_*      - tracking error (desired - actual roll/pitch), max error
  bat_*      - minimum voltage, current spikes
  rcin_*     - min channel value (RC failsafe triggers when ch3 drops to 0)
  err_*      - count per subsystem code (ERR messages are primary fault labels)
  meta_*     - flight duration, total ERR count, mode-change count
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from log_parser import get_error_events, log_duration_seconds


def _safe_stats(series: pd.Series, prefix: str) -> Dict[str, float]:
    """Compute mean, std, min, max, p95 for a numeric series."""
    s = series.dropna()
    if len(s) == 0:
        return {f"{prefix}_{k}": np.nan for k in ("mean", "std", "min", "max", "p95")}
    return {
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_std": float(s.std()),
        f"{prefix}_min": float(s.min()),
        f"{prefix}_max": float(s.max()),
        f"{prefix}_p95": float(np.percentile(s, 95)),
    }


def extract_gps_features(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for msg_type in ("GPS", "GPS2"):
        df = frames.get(msg_type)
        prefix = msg_type.lower()
        if df is None or df.empty:
            feats[f"{prefix}_present"] = 0.0
            continue
        feats[f"{prefix}_present"] = 1.0
        if "NSats" in df.columns:
            feats.update(_safe_stats(df["NSats"], f"{prefix}_nsats"))
            feats[f"{prefix}_low_sat_frac"] = float((df["NSats"] < 6).mean())
        if "HDop" in df.columns:
            feats.update(_safe_stats(df["HDop"], f"{prefix}_hdop"))
            feats[f"{prefix}_high_hdop_frac"] = float((df["HDop"] > 2.0).mean())
        if "Status" in df.columns:
            feats[f"{prefix}_fix3d_frac"] = float((df["Status"] >= 3).mean())
        if "VZ" in df.columns:
            feats.update(_safe_stats(df["VZ"].abs(), f"{prefix}_vz_abs"))
    return feats


def extract_imu_features(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for imu_id in ("IMU", "IMU2", "IMU3"):
        df = frames.get(imu_id)
        if df is None or df.empty:
            continue
        for axis in ("AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"):
            if axis in df.columns:
                feats.update(_safe_stats(df[axis], f"{imu_id.lower()}_{axis.lower()}"))
    return feats


def extract_vibration_features(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Vibration is the most common cause of EKF failures in multicopters.
    VIBE.VibeX/Y/Z > 30 m/s^2 indicates significant vibration.
    Clip counts > 100 per flight indicate accelerometer saturation.
    """
    feats: Dict[str, float] = {}
    df = frames.get("VIBE")
    if df is None or df.empty:
        return {"vibe_present": 0.0}
    feats["vibe_present"] = 1.0
    for axis in ("VibeX", "VibeY", "VibeZ"):
        if axis in df.columns:
            feats.update(_safe_stats(df[axis], f"vibe_{axis.lower()}"))
            feats[f"vibe_{axis.lower()}_high_frac"] = float((df[axis] > 30.0).mean())
    for clip_col in ("Clip0", "Clip1", "Clip2"):
        if clip_col in df.columns:
            feats[f"vibe_{clip_col.lower()}_total"] = float(df[clip_col].max())
    return feats


def extract_baro_features(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    df = frames.get("BARO")
    if df is None or df.empty:
        return {"baro_present": 0.0}
    feats["baro_present"] = 1.0
    if "Alt" in df.columns:
        feats.update(_safe_stats(df["Alt"], "baro_alt"))
    if "Press" in df.columns:
        feats.update(_safe_stats(df["Press"], "baro_press"))
        diff = df["Press"].diff().abs()
        feats["baro_press_spike_max"] = float(diff.max()) if not diff.empty else np.nan
    if "Temp" in df.columns:
        feats.update(_safe_stats(df["Temp"], "baro_temp"))
    return feats


def extract_ekf_features(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for ekf in ("EKF2", "EKF3", "NKF1", "NKF4", "NKF5", "XKF1", "XKF4"):
        df = frames.get(ekf)
        if df is None or df.empty:
            continue
        for col in df.columns:
            if col.startswith("IV") or "Inn" in col or "Var" in col:
                feats.update(_safe_stats(df[col], f"ekf_{col.lower()}"))
        if "Flags" in df.columns:
            feats[f"ekf_{ekf.lower()}_flag_set_frac"] = float((df["Flags"] != 0).mean())
    if "EKFS" in frames:
        feats["ekf_core_switch_count"] = float(len(frames["EKFS"]))
    return feats


def extract_attitude_features(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    df = frames.get("ATT")
    if df is None or df.empty:
        return {"att_present": 0.0}
    feats["att_present"] = 1.0
    for axis in ("Roll", "Pitch", "Yaw"):
        desired = f"Des{axis}"
        if axis in df.columns and desired in df.columns:
            tracking_err = (df[axis] - df[desired]).abs()
            feats.update(_safe_stats(tracking_err, f"att_{axis.lower()}_err"))
    return feats


def extract_battery_features(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    df = frames.get("BAT") or frames.get("CURR")
    if df is None or df.empty:
        return {"bat_present": 0.0}
    feats["bat_present"] = 1.0
    volt_col = "Volt" if "Volt" in df.columns else ("VoltR" if "VoltR" in df.columns else None)
    if volt_col:
        feats.update(_safe_stats(df[volt_col], "bat_volt"))
        feats["bat_volt_min"] = float(df[volt_col].min())
    if "Curr" in df.columns:
        feats.update(_safe_stats(df["Curr"], "bat_curr"))
    return feats


def extract_rc_features(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    df = frames.get("RCIN")
    if df is None or df.empty:
        return {"rcin_present": 0.0}
    feats["rcin_present"] = 1.0
    if "C3" in df.columns:
        feats["rcin_ch3_min"] = float(df["C3"].min())
        feats["rcin_ch3_near_failsafe_frac"] = float((df["C3"] < 1050).mean())
    for col in [c for c in df.columns if c.startswith("C") and c[1:].isdigit()]:
        zero_frac = float((df[col] == 0).mean())
        if zero_frac > 0:
            feats[f"rcin_{col.lower()}_zero_frac"] = zero_frac
    return feats


def extract_error_features(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    ERR messages are the highest-signal feature for fault classification.
    Each subsystem's error count is its own feature so the classifier can
    learn which subsystems matter most for each fault type.
    """
    feats: Dict[str, float] = {}
    err_df = get_error_events(frames)
    feats["err_total"] = float(len(err_df))
    if err_df.empty:
        return feats
    for subsys_code, subsys_name in {
        5: "radio_failsafe", 6: "batt_failsafe", 7: "gps_failsafe",
        11: "gps", 16: "ekf_check", 17: "ekf_failsafe",
        18: "baro", 3: "compass", 22: "motor",
    }.items():
        count = float((err_df["Subsys"] == subsys_code).sum())
        feats[f"err_{subsys_name}_count"] = count
    return feats


def extract_meta_features(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    duration = log_duration_seconds(frames)
    feats["meta_duration_s"] = duration if duration is not None else np.nan
    if "MODE" in frames:
        feats["meta_mode_change_count"] = float(len(frames["MODE"]))
    if "ARM" in frames:
        feats["meta_arm_event_count"] = float(len(frames["ARM"]))
    return feats


def extract_all(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Run all feature extractors and return one flat dict of floats.
    NaN values are preserved so the caller can impute them as needed.
    """
    feats: Dict[str, float] = {}
    for extractor in (
        extract_gps_features,
        extract_imu_features,
        extract_vibration_features,
        extract_baro_features,
        extract_ekf_features,
        extract_attitude_features,
        extract_battery_features,
        extract_rc_features,
        extract_error_features,
        extract_meta_features,
    ):
        feats.update(extractor(frames))
    return feats
