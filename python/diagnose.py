"""
diagnose.py

Command-line entry point for the AI-Assisted Log Diagnosis service.

Usage:
    python diagnose.py path/to/flight.bin
    python diagnose.py path/to/telemetry.tlog --top-k 5
    python diagnose.py path/to/flight.bin --model models/fault_classifier.pkl
    python diagnose.py path/to/flight.bin --json   # machine-readable output

This module wires together log_parser, feature_extractor, fault_classifier,
and rag_pipeline into a single callable diagnose() function so the same
logic can be reused in a web API, a Mission Planner plugin, or a companion-
computer daemon without copying code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from log_parser import parse_log, log_duration_seconds
from feature_extractor import extract_all
from fault_classifier import FaultClassifier, heuristic_classify
from rag_pipeline import FixSuggestionPipeline

# Feature thresholds used to build the evidence section of the report.
# Values taken from ArduPilot wiki recommended limits.
EVIDENCE_THRESHOLDS = {
    "vibe_vibex_p95":            ("m/s^2",    30.0, "high"),
    "vibe_vibey_p95":            ("m/s^2",    30.0, "high"),
    "vibe_vibez_p95":            ("m/s^2",    30.0, "high"),
    "vibe_clip0_total":          ("clips",   100.0, "high"),
    "vibe_clip1_total":          ("clips",   100.0, "high"),
    "gps_hdop_p95":              ("hdop",      2.0, "high"),
    "gps_low_sat_frac":          ("%",         0.1, "high"),
    "bat_volt_min":              ("V",         3.5, "low"),
    "rcin_ch3_near_failsafe_frac": ("%",       0.05, "high"),
    "baro_press_spike_max":      ("Pa",       200.0, "high"),
    "err_total":                 ("events",    0.0, "high"),
    "ekf_core_switch_count":     ("switches",  0.0, "high"),
}


def build_evidence(features: Dict[str, float]) -> List[Dict]:
    """
    Return a list of feature readings that exceeded their thresholds.
    These are shown to the user to explain *why* a fault was flagged.
    """
    evidence = []
    for feat_name, (unit, threshold, direction) in EVIDENCE_THRESHOLDS.items():
        val = features.get(feat_name)
        if val is None:
            continue
        import math
        if math.isnan(val):
            continue
        exceeded = (val > threshold) if direction == "high" else (val < threshold)
        if exceeded:
            evidence.append({
                "feature": feat_name,
                "value": round(val, 3),
                "unit": unit,
                "threshold": threshold,
                "direction": direction,
            })
    return evidence


def diagnose(
    log_path: str,
    model_path: Optional[str] = None,
    top_k: int = 3,
) -> Dict:
    """
    Run the full diagnosis pipeline on a single log file.

    Returns a structured dict:
    {
      "file":       str,
      "duration_s": float | None,
      "faults":     [{fault, confidence}, ...],
      "fixes":      [{title, url, fix_steps, source}, ...],
      "evidence":   [{feature, value, unit, threshold, direction}, ...],
    }
    """
    frames = parse_log(log_path)
    features = extract_all(frames)

    # Try trained model first; fall back to heuristic rules if none found.
    try:
        classifier = FaultClassifier(model_path)
        classifier.load()
        faults = classifier.predict(features, top_k=top_k)
    except FileNotFoundError:
        faults = heuristic_classify(features)[:top_k]

    top_fault = faults[0]["fault"] if faults else "normal"

    rag = FixSuggestionPipeline()
    fixes = rag.suggest(fault_type=top_fault, features=features, top_k=3)

    evidence = build_evidence(features)

    return {
        "file": str(Path(log_path).name),
        "duration_s": log_duration_seconds(frames),
        "faults": faults,
        "fixes": fixes,
        "evidence": evidence,
    }


def print_report(result: Dict) -> None:
    """Human-readable console report."""
    print("\n=== ArduPilot Log Diagnosis ===")
    print(f"File:     {result['file']}")
    dur = result.get("duration_s")
    print(f"Duration: {round(dur)} s" if dur else "Duration: unknown")

    print("\nFault Diagnosis")
    print("-" * 40)
    for i, f in enumerate(result["faults"], 1):
        pct = round(f["confidence"] * 100, 1)
        print(f"  {i}. {f['fault']:<25} (confidence: {pct}%)")

    if result["fixes"]:
        top_fault = result["faults"][0]["fault"] if result["faults"] else "unknown"
        print(f"\nSuggested Fixes for: {top_fault}")
        print("-" * 40)
        for i, fix in enumerate(result["fixes"], 1):
            print(f"\n  [{i}] {fix['title']}")
            print(f"      Source: {fix['url']}")
            print("      Steps:")
            for step in fix.get("fix_steps", []):
                print(f"        - {step}")

    if result["evidence"]:
        print("\nEvidence in log")
        print("-" * 40)
        for ev in result["evidence"]:
            direction = ">" if ev["direction"] == "high" else "<"
            print(
                f"  {ev['feature']:<35} {ev['value']:>10} {ev['unit']:<10} "
                f"[threshold: {direction}{ev['threshold']} {ev['unit']}]"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-Assisted ArduPilot Log Diagnosis"
    )
    parser.add_argument("log_path", help="Path to .bin or .tlog file")
    parser.add_argument("--model", help="Path to trained model .pkl", default=None)
    parser.add_argument("--top-k", type=int, default=3, help="Top N faults to show")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
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
