"""
fault_classifier.py

Supervised classifier mapping a feature vector (from feature_extractor.py)
to a ranked list of probable fault types with confidence scores.

Why XGBoost: ArduPilot labeled log datasets are small (hundreds of examples).
Gradient-boosted trees outperform neural networks in this regime, handle NaN
features natively, and produce well-calibrated probabilities after isotonic
regression. The model runs on a Raspberry Pi / Jetson without a GPU.

Fault taxonomy:
  normal, gps_glitch, compass_error, vibration_fault, ekf_failsafe,
  battery_failsafe, rc_failsafe, baro_fault, motor_imbalance

The fit/predict interface is sklearn-compatible so the model can be swapped
for a neural classifier later without changing the downstream API.
"""

from __future__ import annotations
import json, os, pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

FAULT_LABELS = [
    "normal", "gps_glitch", "compass_error", "vibration_fault",
    "ekf_failsafe", "battery_failsafe", "rc_failsafe", "baro_fault",
    "motor_imbalance",
]
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "fault_classifier.pkl"


class FaultClassifier:
    """
    Thin wrapper around an XGBoost multi-class classifier with isotonic
    calibration so confidence values shown to users are meaningful.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._pipeline = None
        self._feature_names: List[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series) -> "FaultClassifier":
        """
        Fit on labeled feature data.
        X: DataFrame of features from extract_all().
        y: Series of fault label strings.
        Median imputation handles logs where some message types are absent.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost required: pip install xgboost") from exc
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.calibration import CalibratedClassifierCV

        self._label_encoder = LabelEncoder().fit(FAULT_LABELS)
        y_enc = self._label_encoder.transform(y)
        self._feature_names = list(X.columns)

        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            n_jobs=-1, random_state=42,
        )
        base = Pipeline([("imputer", SimpleImputer(strategy="median")), ("xgb", xgb)])
        self._pipeline = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self._pipeline.fit(X, y_enc)
        return self

    def predict(self, features: Dict[str, float], top_k: int = 3) -> List[Dict]:
        """
        Score one log and return the top_k probable faults.
        Returns: [{fault: str, confidence: float}, ...] sorted by confidence.
        """
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        row = pd.DataFrame([{k: features.get(k, np.nan) for k in self._feature_names}])
        probs = self._pipeline.predict_proba(row)[0]
        return [
            {"fault": self._label_encoder.classes_[i], "confidence": round(float(probs[i]), 4)}
            for i in np.argsort(probs)[::-1][:top_k]
        ]

    def save(self, path: Optional[str] = None) -> str:
        path = Path(path or self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self._pipeline,
                         "feature_names": self._feature_names,
                         "label_encoder": self._label_encoder}, f)
        return str(path)

    def load(self, path: Optional[str] = None) -> "FaultClassifier":
        path = Path(path or self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self._pipeline = obj["pipeline"]
        self._feature_names = obj["feature_names"]
        self._label_encoder = obj["label_encoder"]
        return self


def heuristic_classify(features: Dict[str, float]) -> List[Dict]:
    """
    Rule-based fallback classifier requiring no training data.

    Uses the same feature names as extract_all() with thresholds from
    ArduPilot wiki recommended limits. Serves as both a baseline to beat
    with the trained model and a fallback for novel fault types.
    """
    scores: Dict[str, float] = {label: 0.0 for label in FAULT_LABELS}
    scores["normal"] = 0.3

    if features.get("gps_hdop_p95", 0) > 3.0:         scores["gps_glitch"] += 0.4
    if features.get("err_gps_failsafe_count", 0) > 0:  scores["gps_glitch"] += 0.5
    if features.get("gps_low_sat_frac", 0) > 0.1:      scores["gps_glitch"] += 0.2
    if features.get("vibe_vibex_p95", 0) > 30:         scores["vibration_fault"] += 0.5
    if features.get("vibe_clip0_total", 0) > 100:      scores["vibration_fault"] += 0.3
    if features.get("err_ekf_failsafe_count", 0) > 0:  scores["ekf_failsafe"] += 0.7
    if features.get("ekf_core_switch_count", 0) > 0:   scores["ekf_failsafe"] += 0.4
    if features.get("err_batt_failsafe_count", 0) > 0: scores["battery_failsafe"] += 0.8
    if features.get("bat_volt_min", 99) < 3.5:         scores["battery_failsafe"] += 0.3
    if features.get("err_radio_failsafe_count", 0) > 0: scores["rc_failsafe"] += 0.8
    if features.get("rcin_ch3_near_failsafe_frac", 0) > 0.05: scores["rc_failsafe"] += 0.3
    if features.get("err_compass_count", 0) > 0:       scores["compass_error"] += 0.6
    if features.get("err_baro_count", 0) > 0:          scores["baro_fault"] += 0.6
    if features.get("baro_press_spike_max", 0) > 200:  scores["baro_fault"] += 0.3

    total = sum(scores.values())
    return sorted(
        [{"fault": k, "confidence": round(v / total, 4)} for k, v in scores.items() if v > 0],
        key=lambda x: x["confidence"], reverse=True
    )[:3]
