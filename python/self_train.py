"""
self_train.py  --  Self-training feedback loop for ardupilot-log-diagnosis

Every high-confidence diagnosis is saved to a local buffer.
When the buffer hits 50 samples, the model retrains automatically.
No manual labeling, no annotation costs, no cloud API needed.
The model gets smarter with every flight at zero added cost.
"""

from __future__ import annotations
import json, os, hashlib, pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import LabelEncoder
    _TRAIN_DEPS = True
except ImportError:
    _TRAIN_DEPS = False

DEFAULT_BUFFER_PATH = Path("data/training_buffer.jsonl")
DEFAULT_MODEL_DIR   = Path("models")
DEFAULT_BASE_DATA   = Path("data/labeled_segments.csv")

MIN_NEW_SAMPLES      = 50
CONFIDENCE_THRESHOLD = 0.82


class TrainingBuffer:
    def __init__(self, buffer_path: Path = DEFAULT_BUFFER_PATH):
        self.path = Path(buffer_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, features, label, confidence, log_hash, fault_windows=None):
        if confidence < CONFIDENCE_THRESHOLD:
            return False
        record = {
            "ts": datetime.utcnow().isoformat(),
            "label": label,
            "confidence": round(confidence, 4),
            "log_hash": log_hash,
            "features": {k: round(float(v), 6) for k, v in features.items()},
        }
        if fault_windows:
            record["fault_windows_count"] = len(fault_windows)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        return True

    def count(self):
        if not self.path.exists():
            return 0
        with open(self.path) as fh:
            return sum(1 for _ in fh)

    def load(self):
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def archive(self):
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        archive_path = self.path.parent / f"buffer_archive_{ts}.jsonl"
        self.path.rename(archive_path)
        return archive_path


def _log_hash(log_path):
    size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    return hashlib.md5(f"{log_path}:{size}".encode()).hexdigest()[:8]


def record_diagnosis(features, label, confidence, log_path,
                     fault_windows=None, buffer_path=DEFAULT_BUFFER_PATH):
    buf = TrainingBuffer(buffer_path)
    accepted = buf.add(features, label, confidence,
                       _log_hash(log_path), fault_windows)
    if not accepted:
        return False, (f"Skipped (confidence {confidence:.0%} below "
                       f"{CONFIDENCE_THRESHOLD:.0%} threshold)")
    n = buf.count()
    msg = f"Sample saved ({n} total in buffer)."
    if n >= MIN_NEW_SAMPLES:
        msg += " Triggering retrain."
        msg += " " + maybe_retrain(buf, DEFAULT_MODEL_DIR, DEFAULT_BASE_DATA)
    return True, msg


def maybe_retrain(buffer, model_dir=DEFAULT_MODEL_DIR,
                  base_data=DEFAULT_BASE_DATA):
    if not _TRAIN_DEPS:
        return "Skipped: xgboost/sklearn not installed."
    records = buffer.load()
    if not records:
        return "Skipped: buffer is empty."

    feature_keys = sorted(records[0]["features"].keys())
    X_new = np.array([[r["features"].get(k, 0.0) for k in feature_keys]
                      for r in records])
    y_new = np.array([r["label"] for r in records])

    X_base, y_base = np.empty((0, len(feature_keys))), np.array([])
    if Path(base_data).exists():
        try:
            import csv
            with open(base_data) as fh:
                rows = list(csv.DictReader(fh))
            X_base = np.array([[float(r.get(k, 0)) for k in feature_keys]
                               for r in rows])
            y_base = np.array([r["label"] for r in rows])
        except Exception:
            pass

    X = np.vstack([X_base, X_new]) if len(X_base) else X_new
    y = np.concatenate([y_base, y_new]) if len(y_base) else y_new

    le = LabelEncoder()
    clf = CalibratedClassifierCV(
        xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                          use_label_encoder=False, eval_metric="mlogloss",
                          random_state=42),
        method="isotonic", cv=3
    )
    clf.fit(X, le.fit_transform(y))

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_path = model_dir / f"xgb_model_{ts}.pkl"

    with open(model_path, "wb") as fh:
        pickle.dump({"clf": clf, "le": le, "feature_keys": feature_keys}, fh)

    with open(model_dir / f"xgb_model_{ts}_meta.json", "w") as fh:
        json.dump({"trained_at": ts, "n_samples": int(len(X)),
                   "n_buffer_samples": int(len(X_new)),
                   "classes": le.classes_.tolist()}, fh, indent=2)

    archive = buffer.archive()
    return (f"Retrained on {len(X)} samples ({len(X_new)} from live logs). "
            f"Model: {model_path.name}. Archive: {archive.name}.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--buffer",    default=str(DEFAULT_BUFFER_PATH))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--base-data", default=str(DEFAULT_BASE_DATA))
    args = p.parse_args()
    buf = TrainingBuffer(Path(args.buffer))
    print(f"Buffer: {buf.count()} samples")
    print(maybe_retrain(buf, Path(args.model_dir), Path(args.base_data)))
