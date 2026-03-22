"""
param_checker.py
----------------
Reads PARM messages from an ArduPilot DataFlash log and checks each parameter
against known risky or misconfigured ranges for every fault type.

This directly addresses the "parameter states" requirement in the ArduPilot
GSoC brief: the tool learns not just from sensor signals but from the
vehicle's configuration at the time of the incident.
"""

from __future__ import annotations
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Known-risky parameter patterns, keyed by fault type.
# Each entry: param name, condition that flags a warning, message, wiki link.
# ---------------------------------------------------------------------------
PARAM_RULES: Dict[str, List[Dict[str, Any]]] = {
    "vibration_fault": [
        {
            "param": "INS_ACCEL_FILTER",
            "check": lambda v: v > 20,
            "message": "INS_ACCEL_FILTER={v:.0f} Hz is high — reduces vibration filtering. Recommended <=10 Hz for most frames.",
            "url": "https://ardupilot.org/copter/docs/common-imu-notch-filtering.html",
        },
        {
            "param": "INS_GYRO_FILTER",
            "check": lambda v: v > 30,
            "message": "INS_GYRO_FILTER={v:.0f} Hz is high — gyro noise may feed into EKF. Recommended <=20 Hz.",
            "url": "https://ardupilot.org/copter/docs/common-imu-notch-filtering.html",
        },
        {
            "param": "MOT_THST_EXPO",
            "check": lambda v: v < 0.35 or v > 0.75,
            "message": "MOT_THST_EXPO={v:.2f} is outside typical range 0.35-0.75. Incorrect expo curve causes uneven motor loading and vibration.",
            "url": "https://ardupilot.org/copter/docs/motor-thrust-scaling.html",
        },
    ],
    "ekf_failsafe": [
        {
            "param": "EK2_CHECK_SCALE",
            "check": lambda v: v < 100,
            "message": "EK2_CHECK_SCALE={v:.0f}% is below 100% — EKF innovation gate is tighter than default, raising false-failsafe risk.",
            "url": "https://ardupilot.org/copter/docs/ekf-inav-failures.html",
        },
        {
            "param": "EK3_CHECK_SCALE",
            "check": lambda v: v < 100,
            "message": "EK3_CHECK_SCALE={v:.0f}% is below 100% — same as EK2 issue.",
            "url": "https://ardupilot.org/copter/docs/ekf-inav-failures.html",
        },
        {
            "param": "FS_EKF_THRESH",
            "check": lambda v: v < 0.6,
            "message": "FS_EKF_THRESH={v:.2f} is very low — EKF failsafe triggers at minimal innovation error. Default 0.8 is safer.",
            "url": "https://ardupilot.org/copter/docs/ekf-inav-failures.html",
        },
    ],
    "gps_glitch": [
        {
            "param": "GPS_HDOP_GOOD",
            "check": lambda v: v > 2.3,
            "message": "GPS_HDOP_GOOD={v:.1f} — threshold for GPS is good is permissive; glitchy GPS positions will still be accepted.",
            "url": "https://ardupilot.org/copter/docs/gps-failsafe-glitch-protection.html",
        },
        {
            "param": "GPS_MIN_ELEV",
            "check": lambda v: v < 10,
            "message": "GPS_MIN_ELEV={v:.0f} deg — satellites below 10 deg elevation introduce multipath and position errors.",
            "url": "https://ardupilot.org/copter/docs/gps-failsafe-glitch-protection.html",
        },
    ],
    "compass_error": [
        {
            "param": "COMPASS_AUTODEC",
            "check": lambda v: v == 0,
            "message": "COMPASS_AUTODEC=0 — automatic magnetic declination is off. Manual declination must be set correctly for the flight location.",
            "url": "https://ardupilot.org/copter/docs/common-compass-calibration-in-mission-planner.html",
        },
        {
            "param": "COMPASS_MOT_X",
            "check": lambda v: abs(v) > 50,
            "message": "COMPASS_MOT_X={v:.1f} — large motor-interference compensation suggests significant compass-to-motor proximity.",
            "url": "https://ardupilot.org/copter/docs/common-compass-calibration-in-mission-planner.html",
        },
    ],
    "battery_failsafe": [
        {
            "param": "BATT_LOW_VOLT",
            "check": lambda v: 0 < v < 9.9,
            "message": "BATT_LOW_VOLT={v:.1f} V — low-voltage warning threshold may be set too low for the battery cell count.",
            "url": "https://ardupilot.org/copter/docs/failsafe-battery.html",
        },
        {
            "param": "BATT_LOW_MAH",
            "check": lambda v: v == 0,
            "message": "BATT_LOW_MAH=0 — capacity-based failsafe is disabled. Voltage-only failsafe can fail on high-C packs with a stiff voltage curve.",
            "url": "https://ardupilot.org/copter/docs/failsafe-battery.html",
        },
    ],
    "rc_failsafe": [
        {
            "param": "FS_THR_ENABLE",
            "check": lambda v: v == 0,
            "message": "FS_THR_ENABLE=0 — RC throttle failsafe is DISABLED. A lost RC link will not trigger any failsafe action.",
            "url": "https://ardupilot.org/copter/docs/radio-failsafe.html",
        },
        {
            "param": "FS_THR_VALUE",
            "check": lambda v: v > 975,
            "message": "FS_THR_VALUE={v:.0f} us — failsafe threshold is high; normal RC noise may trigger spurious failsafes.",
            "url": "https://ardupilot.org/copter/docs/radio-failsafe.html",
        },
    ],
    "motor_imbalance": [
        {
            "param": "MOT_SPIN_MIN",
            "check": lambda v: v < 0.05 or v > 0.2,
            "message": "MOT_SPIN_MIN={v:.3f} — outside typical range 0.05-0.15. Too low: motors stall. Too high: cannot hold steady hover.",
            "url": "https://ardupilot.org/copter/docs/motor-range.html",
        },
        {
            "param": "MOT_BAT_VOLT_MAX",
            "check": lambda v: v == 0,
            "message": "MOT_BAT_VOLT_MAX=0 — battery voltage compensation is off. As the pack discharges, effective motor output becomes asymmetric.",
            "url": "https://ardupilot.org/copter/docs/motor-range.html",
        },
    ],
    "baro_fault": [
        {
            "param": "GND_ALT_OFFSET",
            "check": lambda v: abs(v) > 5,
            "message": "GND_ALT_OFFSET={v:.1f} m — large altitude offset suggests the baro reading was manually corrected; investigate baro exposure.",
            "url": "https://ardupilot.org/copter/docs/common-barometer-calibration.html",
        },
        {
            "param": "BARO_PROBE_EXT",
            "check": lambda v: v == 0,
            "message": "BARO_PROBE_EXT=0 — only the internal board-mounted barometer is in use; it is susceptible to propwash pressure pulses.",
            "url": "https://ardupilot.org/copter/docs/common-barometer-calibration.html",
        },
    ],
}

# Universal checks that apply regardless of fault type
UNIVERSAL_RULES: List[Dict[str, Any]] = [
    {
        "param": "ARMING_CHECK",
        "check": lambda v: v == 0,
        "message": "ARMING_CHECK=0 — ALL pre-arm checks are disabled. This masks configuration problems that would otherwise block arming.",
        "url": "https://ardupilot.org/copter/docs/common-prearm-safety-checks.html",
        "severity": "critical",
    },
    {
        "param": "LOG_BITMASK",
        "check": lambda v: v < 830,
        "message": "LOG_BITMASK={v:.0f} — several important message types may not be logged; diagnosis accuracy will be lower.",
        "url": "https://ardupilot.org/copter/docs/common-downloading-and-analyzing-data-logs-in-mission-planner.html",
        "severity": "warning",
    },
]


def check_params(
    params: Dict[str, float],
    fault_type: str,
    include_universal: bool = True,
) -> List[Dict[str, str]]:
    """
    Check a parameter dict against known risky configurations for a fault type.

    Parameters
    ----------
    params : dict[str, float]
        Parameter name to value, as returned by log_parser.parse_params().
    fault_type : str
        One of the 9 fault taxonomy labels.
    include_universal : bool
        If True, also run universal safety checks regardless of fault type.

    Returns
    -------
    list of dicts with keys: param, value, message, url, severity
    """
    warnings: List[Dict[str, str]] = []

    for rule in PARAM_RULES.get(fault_type, []):
        name = rule["param"]
        if name not in params:
            continue
        val = params[name]
        try:
            if rule["check"](val):
                warnings.append({
                    "param": name,
                    "value": f"{val:.4g}",
                    "message": rule["message"].format(v=val),
                    "url": rule["url"],
                    "severity": rule.get("severity", "warning"),
                })
        except Exception:
            pass

    if include_universal:
        for rule in UNIVERSAL_RULES:
            name = rule["param"]
            if name not in params:
                continue
            val = params[name]
            try:
                if rule["check"](val):
                    warnings.append({
                        "param": name,
                        "value": f"{val:.4g}",
                        "message": rule["message"].format(v=val),
                        "url": rule["url"],
                        "severity": rule.get("severity", "warning"),
                    })
            except Exception:
                pass

    return warnings


def format_param_warnings(warnings: List[Dict[str, str]]) -> str:
    """Return a human-readable string of parameter warnings."""
    if not warnings:
        return "  No risky parameter configurations detected.\n"
    lines = []
    for w in warnings:
        sev = w.get("severity", "warning").upper()
        lines.append(f"  [{sev}] {w['param']} = {w['value']}")
        lines.append(f"         {w['message']}")
        lines.append(f"         Ref: {w['url']}")
    return "\n".join(lines) + "\n"
