"""
diagnose.py
Command-line entry point for the AI-Assisted Log Diagnosis service.

Usage:
    python diagnose.py path/to/flight.bin
    python diagnose.py path/to/flight.bin --model models/fault_classifier.pkl
    python diagnose.py path/to/flight.bin --top-k 5
    python diagnose.py path/to/flight.bin --json

Output per diagnosis
--------------------
  faults          Ranked fault types with calibrated confidence scores.
  fixes           Ordered fix steps with links to ArduPilot wiki pages.
  evidence        Sensor features that exceeded thresholds.
  fault_windows   Time ranges in the log where anomaly signals were active
                  (links to relevant evidence in the log, per ArduPilot brief).
  param_warnings  Parameter configurations known to contribute to the fault
                  (parameter states requirement from the ArduPilot brief).
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

from log_parser import parse_log, parse_params, log_duration_seconds, get_fault_windows
from feature_extractor import extract_all
from fault_classifier import FaultClassifier, heuristic_classify
from rag_pipeline import FixSuggestionPipeline
from param_checker import check_params, format_param_warnings

EVIDENCE_THRESHOLDS: Dict[str, tuple] = {
    "vibe_vibex_p95":              ("m/s2",    30.0,  "high"),
    "vibe_vibey_p95":              ("m/s2",    30.0,  "high"),
    "vibe_vibez_p95":              ("m/s2",    30.0,  "high"),
    "vibe_clip0_total":            ("clips",  100.0,  "high"),
    "vibe_clip1_total":            ("clips",  100.0,  "high"),
    "gps_hdop_p95":                ("hdop",    2.0,   "high"),
    "gps_low_sat_frac":            ("%",       0.1,   "high"),
    "bat_volt_min":                ("V",       3.5,   "low"),
    "rcin_ch3_near_failsafe_frac": ("%",       0.05,  "high"),
    "baro_press_spike_max":        ("Pa",    200.0,   "high"),
    "err_total":                   ("events",  0.0,   "high"),
    "ekf_core_switch_count":       ("switches",0.0,   "high"),
}


def build_evidence(features: Dict[str, float]) -> List[Dict]:
    evidence = []
    for feat_name, (unit, threshold, direction) in EVIDENCE_THRESHOLDS.items():
        val = features.get(feat_name)
        if val is None or math.isnan(val):
            continue
        exceeded = (val > threshold) if direction == "high" else (val < threshold)
        if exceeded:
            evidence.append({
                "feature":   feat_name,
                "value":     round(val, 3),
                "unit":      unit,
                "threshold": threshold,
                "direction": direction,
            })
    return evidence


def diagnose(
    log_path: str,
    model_path: Optional[str] = None,
    top_k: int = 3,
) -> Dict:
    """Run the full diagnosis pipeline on a single log file.

    Returns
    -------
    {
        "file":            str,
        "duration_s":      float | None,
        "faults":          [{fault, confidence}, ...],
        "fixes":           [{title, url, fix_steps, source}, ...],
        "evidence":        [{feature, value, unit, threshold, direction}, ...],
        "fault_windows":   [{start_s, end_s, reason}, ...],
        "param_warnings":  [{param, value, message, url, severity}, ...],
    }
    """
    frames   = parse_log(log_path)
    features = extract_all(frames)
    params   = parse_params(frames)

    try:
        classifier = FaultClassifier(model_path)
        classifier.load()
        faults = classifier.predict(features, top_k=top_k)
    except FileNotFoundError:
        faults = heuristic_classify(features)[:top_k]

    top_fault = faults[0]["fault"] if faults else "normal"

    rag   = FixSuggestionPipeline()
    fixes = rag.suggest(fault_type=top_fault, features=features, top_k=3)

    evidence      = build_evidence(features)
    fault_windows = get_fault_windows(frames, top_fault)
    param_warnings = check_params(params, top_fault, include_universal=True)

    return {
        "file":           str(Path(log_path).name),
        "duration_s":     log_duration_seconds(frames),
        "faults":         faults,
        "fixes":          fixes,
        "evidence":       evidence,
        "fault_windows":  [{"start_s": s, "end_s": e, "reason": r} for s, e, r in fault_windows],
        "param_warnings": param_warnings,
    }


def print_report(result: Dict) -> None:
    print("\n=== ArduPilot Log Diagnosis ===")
    print(f"File:     {result['file']}")
    dur = result.get("duration_s")
    print(f"Duration: {round(dur)} s" if dur else "Duration: unknown")

    print("\nFault Diagnosis")
    print("-" * 40)
    for i, f in enumerate(result["faults"], 1):
        pct = round(f["confidence"] * 100, 1)
        print(f"  {i}. {f['fault']:<25} (confidence: {pct}%)")

    if result.get("fault_windows"):
        print("\nFault Windows in Log  (jump to these regions in Mission Planner)")
        print("-" * 40)
        for w in result["fault_windows"]:
            print(f"  T={w['start_s']}s - T={w['end_s']}s   {w['reason']}")

    if result.get("evidence"):
        print("\nEvidence  (sensor features that crossed thresholds)")
        print("-" * 40)
        for ev in result["evidence"]:
            direction = ">" if ev["direction"] == "high" else "<"
            print(f"  {ev['feature']:<35} {ev['value']:>10} {ev['unit']:<8} "
                  f"[threshold: {direction}{ev['threshold']} {ev['unit']}]")

    if result.get("param_warnings"):
        print("\nParameter State Warnings")
        print("-" * 40)
        print(format_param_warnings(result["param_warnings"]), end="")

    if result.get("fixes"):
        top_fault = result["faults"][0]["fault"] if result["faults"] else "unknown"
        print(f"\nSuggested Fixes for: {top_fault}")
        print("-" * 40)
        for i, fix in enumerate(result["fixes"], 1):
            print(f"\n  [{i}] {fix['title']}")
            print(f"       Source: {fix['url']}")
            print("       Steps:")
            for step in fix.get("fix_steps", []):
                print(f"         - {step}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Assisted ArduPilot Log Diagnosis")
    parser.add_argument("log_path", help="Path to .bin or .tlog file")
    parser.add_argument("--model",  help="Path to trained model .pkl", default=None)
    parser.add_argument("--top-k",  type=int, default=3, help="Top N faults to show")
    parser.add_argument("--json",   action="store_true", help="Output as JSON")
    args = parser.parse_args()

    try:
        result = diagnose(args.log_path, model_path=args.model, top_k=args.top_k)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
