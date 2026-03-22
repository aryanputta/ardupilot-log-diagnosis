"""
preflight_check.py  --  Pre-flight parameter health check

Inspired by NASA IVHM pre-launch checklists: catch configuration
problems BEFORE the vehicle leaves the ground, not after a crash.

Runs all param_checker rule sets simultaneously and outputs a
GO / NO-GO recommendation with CRITICAL / WARNING / INFO severity.
Preventing one crash with this module pays back the entire
development cost of this project.
"""

from __future__ import annotations
from typing import Dict, List
from enum import Enum

try:
    from param_checker import PARAM_RULES, UNIVERSAL_RULES
except ImportError:
    PARAM_RULES = {}
    UNIVERSAL_RULES = []


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING  = "WARNING"
    INFO     = "INFO"


FAULT_SEVERITY = {
    "rc_failsafe":      Severity.CRITICAL,
    "battery_failsafe": Severity.CRITICAL,
    "ekf_failsafe":     Severity.CRITICAL,
    "gps_glitch":       Severity.WARNING,
    "compass_error":    Severity.WARNING,
    "baro_fault":       Severity.WARNING,
    "vibration_fault":  Severity.INFO,
    "motor_imbalance":  Severity.INFO,
}

FAULT_DESCRIPTIONS = {
    "rc_failsafe":      "RC failsafe not configured -- loss of RC signal leaves vehicle with no safe response",
    "battery_failsafe": "Battery failsafe not configured -- low voltage may cause uncontrolled descent",
    "ekf_failsafe":     "EKF parameters too relaxed -- attitude estimation errors may not trigger protective action",
    "gps_glitch":       "GPS health checks insufficient -- position errors may go undetected",
    "compass_error":    "Compass configuration issue -- heading errors can cause flyaway",
    "baro_fault":       "Barometer configuration issue -- altitude hold may be unreliable",
    "vibration_fault":  "Vibration filtering not optimal -- sensor noise may accumulate during flight",
    "motor_imbalance":  "Motor configuration suboptimal -- asymmetric thrust may affect stability",
}


def run_preflight_check(params: Dict[str, float], include_universal: bool = True) -> Dict:
    """
    Run all parameter rule sets against the provided parameter dictionary.

    Returns dict with keys: go (bool), issues (list), summary (str),
    critical_count, warning_count, info_count.
    """
    if not params:
        return {
            "go": False, "issues": [],
            "summary": "No parameters found. Cannot assess pre-flight state.",
            "critical_count": 0, "warning_count": 0, "info_count": 0,
        }

    issues = []

    for fault_type, rules in PARAM_RULES.items():
        severity = FAULT_SEVERITY.get(fault_type, Severity.INFO)
        for rule in rules:
            param_name = rule.get("param")
            condition  = rule.get("condition", "")
            message    = rule.get("message", "")
            if param_name not in params:
                continue
            val = params[param_name]
            try:
                triggered = eval(f"{val} {condition}")
            except Exception:
                triggered = False
            if triggered:
                issues.append({
                    "fault_type":  fault_type,
                    "severity":    severity.value,
                    "param":       param_name,
                    "value":       val,
                    "message":     message,
                    "description": FAULT_DESCRIPTIONS.get(fault_type, ""),
                })

    if include_universal:
        for rule in UNIVERSAL_RULES:
            param_name = rule.get("param")
            condition  = rule.get("condition", "")
            message    = rule.get("message", "")
            if param_name not in params:
                continue
            val = params[param_name]
            try:
                triggered = eval(f"{val} {condition}")
            except Exception:
                triggered = False
            if triggered:
                issues.append({
                    "fault_type":  "universal",
                    "severity":    Severity.WARNING.value,
                    "param":       param_name,
                    "value":       val,
                    "message":     message,
                    "description": "Universal safety parameter is misconfigured",
                })

    critical_count = sum(1 for i in issues if i["severity"] == "CRITICAL")
    warning_count  = sum(1 for i in issues if i["severity"] == "WARNING")
    info_count     = sum(1 for i in issues if i["severity"] == "INFO")
    go = critical_count == 0

    if critical_count > 0:
        summary = f"NO-GO: {critical_count} critical issue(s). Do not fly until resolved."
    elif warning_count > 0:
        summary = f"GO WITH CAUTION: {warning_count} warning(s). Review before flight."
    elif info_count > 0:
        summary = f"GO: {info_count} minor suggestion(s). Safe to fly."
    else:
        summary = "GO: All safety parameters check out."

    return {
        "go": go, "issues": issues, "summary": summary,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "info_count": info_count,
    }


def print_preflight_report(result: Dict) -> None:
    go_str = "GO" if result["go"] else "NO-GO"
    print("\n" + "=" * 60)
    print(f"PRE-FLIGHT HEALTH CHECK  [{go_str}]")
    print("=" * 60)
    print(result["summary"])
    if result["issues"]:
        print()
        for issue in sorted(result["issues"], key=lambda x: x["severity"]):
            marker = {"CRITICAL": "[!!]", "WARNING": "[!] ", "INFO": "[ ] "}.get(issue["severity"], "    ")
            print(f"{marker} {issue['severity']:8s}  {issue['param']:25s} = {issue['value']}")
            print(f"           {issue['description']}")
            print(f"           Fix: {issue['message']}")
            print()
    print("=" * 60)
