"""
rag_pipeline.py

Retrieval-Augmented Generation (RAG) pipeline for ArduPilot fault fix
suggestions.

Problem: A classifier can label a fault but it cannot explain *why* it
happened or what to check first. ArduPilot's wiki, issue tracker, and
discuss forum contain thousands of solved cases. RAG bridges the gap by:
  1. Embedding a corpus of ArduPilot wiki pages + discuss posts into a
     vector store (ChromaDB or FAISS).
  2. At query time, retrieving the top-k passages most similar to the
     fault description + relevant log features.
  3. Formatting the retrieved passages as a structured list of suggested
     fixes with confidence scores and links back to the source.

Why RAG instead of a fine-tuned LLM: ArduPilot diagnostics change with
firmware versions. A RAG index can be updated by re-ingesting the wiki
without retraining. This keeps the system accurate as ArduPilot evolves.

Knowledge base sources (initial corpus):
  - ArduPilot wiki pages: https://ardupilot.org/dev/docs/
  - ArduPilot discuss forums: https://discuss.ardupilot.org/
  - GitHub issue tracker: https://github.com/ArduPilot/ardupilot/issues
  - Pre-labeled log segments from community contributions

Usage:
    pipeline = FixSuggestionPipeline()
    pipeline.build_index(corpus_path="data/rag_corpus.jsonl")
    suggestions = pipeline.suggest(fault_type="vibration_fault",
                                   features=feature_dict,
                                   top_k=3)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

CORPUS_PATH = Path(__file__).parent.parent / "data" / "rag_corpus.jsonl"
INDEX_DIR = Path(__file__).parent.parent / "models" / "rag_index"

# Fallback knowledge base that works without any vector store.
# Each entry covers a fault type with: causes, checks, and fix steps.
# These are grounded in ArduPilot wiki content and community Q&A.
BUILTIN_KB: Dict[str, List[Dict]] = {
    "vibration_fault": [
        {
            "title": "High Vibration Causes EKF Divergence",
            "url": "https://ardupilot.org/copter/docs/vibration-damping.html",
            "text": (
                "VIBE.VibeX/Y/Z > 30 m/s^2 indicates excessive frame vibration. "
                "Common causes: loose motor bell, unbalanced propellers, worn motor "
                "bearings. Check VIBE.Clip0/1/2 - clip counts > 100 mean the "
                "accelerometer is saturating and EKF position estimates will drift."
            ),
            "fix_steps": [
                "Balance all propellers with a prop balancer.",
                "Check motor screws and bell for looseness.",
                "Inspect motor bearings for roughness.",
                "Add foam vibration dampeners under the flight controller.",
                "Verify VIBE.VibeZ < 15 m/s^2 in a hover test after changes.",
            ],
        },
    ],
    "gps_glitch": [
        {
            "title": "GPS Glitch Detected",
            "url": "https://ardupilot.org/copter/docs/gps-failsafe-glitch-protection.html",
            "text": (
                "GPS glitch triggers when the EKF GPS position innovation exceeds "
                "the GPS_GLITCH_RADIUS threshold. Common causes: multipath near "
                "buildings, electrical interference from ESCs or power wiring, "
                "GPS antenna placement too close to other electronics."
            ),
            "fix_steps": [
                "Move GPS antenna to highest point on the frame, away from power leads.",
                "Check GPS_HDOP_GOOD (default 1.4) - wait for lower HDOP before arming.",
                "Increase EK3_GPS_TYPE to use GPS + Baro fusion for better resilience.",
                "Use a GPS module with compass if compass interference is also suspected.",
                "Twist ESC signal wires together to reduce EMI.",
            ],
        },
    ],
    "ekf_failsafe": [
        {
            "title": "EKF Failsafe Triggered",
            "url": "https://ardupilot.org/copter/docs/ekf-inav-failsafe.html",
            "text": (
                "EKF failsafe fires when FS_EKF_THRESH innovation variances are "
                "exceeded or when the primary EKF core switches. The ERR subsystem "
                "17 (FAILSAFE_EKF) confirms this. Root causes vary: GPS glitch, "
                "compass interference, and high vibration are the most common."
            ),
            "fix_steps": [
                "Check ERR log for preceding subsystem failures (GPS, compass, VIBE).",
                "Verify COMPASS_ORIENT matches physical mounting orientation.",
                "Set EK3_CHECK_SCALE = 100 and re-fly to gather fresh innovation data.",
                "Consider enabling second IMU as primary (INS_USE2 = 1).",
            ],
        },
    ],
    "battery_failsafe": [
        {
            "title": "Battery Failsafe Triggered",
            "url": "https://ardupilot.org/copter/docs/failsafe-battery.html",
            "text": (
                "Battery failsafe fires when voltage drops below BATT_LOW_VOLT "
                "or capacity below BATT_LOW_MAH. Voltage sag under high current "
                "loads can trigger false positives on older batteries."
            ),
            "fix_steps": [
                "Check actual battery cell voltages with a cell checker after the flight.",
                "Calibrate voltage divider: set BATT_VOLT_MULT correctly.",
                "Replace batteries showing > 20% capacity loss compared to rated mAh.",
                "Set BATT_LOW_VOLT conservatively: 3.5V/cell is standard.",
                "Increase BATT_FS_LOW_ACT to RTL instead of LAND for better recovery.",
            ],
        },
    ],
    "rc_failsafe": [
        {
            "title": "RC Failsafe Triggered",
            "url": "https://ardupilot.org/copter/docs/radio-failsafe.html",
            "text": (
                "RC failsafe fires when the receiver stops sending valid pulses. "
                "This is ERR subsystem 5 (FAILSAFE_RADIO). Common causes: "
                "range exceeded, antenna orientation, or frequency interference."
            ),
            "fix_steps": [
                "Verify receiver failsafe is configured to output no signal (not PWM).",
                "Check RC_FS_THR: it should be set 10-50 PWM below hover throttle.",
                "Inspect receiver antenna orientation - both antennas should be 90 deg apart.",
                "Test range on the ground before flying near the boundary.",
                "Use diversity or long-range receiver (e.g., ExpressLRS) for critical work.",
            ],
        },
    ],
    "compass_error": [
        {
            "title": "Compass Calibration or Interference",
            "url": "https://ardupilot.org/copter/docs/common-compass-calibration-in-mission-planner.html",
            "text": (
                "Compass errors appear in ERR subsystem 3. The EKF uses compass "
                "heading at low speeds. Interference from high-current cables near "
                "the compass causes heading jumps and eventual EKF divergence."
            ),
            "fix_steps": [
                "Move compass/GPS module at least 10 cm away from high-current wiring.",
                "Re-run onboard compass calibration (COMPASS_CAL_FIT < 10 is acceptable).",
                "Enable MOT_PWM_TYPE = dshot and check if compass health improves.",
                "Disable internal compass if external compass health is good: COMPASS_USE = 0.",
            ],
        },
    ],
    "baro_fault": [
        {
            "title": "Barometer Anomaly",
            "url": "https://ardupilot.org/copter/docs/common-barometer-calibration.html",
            "text": (
                "Barometer faults show as ERR subsystem 18. Foam blocking the "
                "baro port or prop wash over the opening causes altitude spikes. "
                "Temperature gradients in direct sunlight also shift baro readings."
            ),
            "fix_steps": [
                "Cover the barometer with open-cell foam to block prop wash.",
                "Keep the flight controller out of direct sunlight.",
                "Check BARO.Press for large instantaneous spikes (> 50 Pa jump).",
                "Enable redundant baro via BARO_PRIMARY if available.",
            ],
        },
    ],
    "motor_imbalance": [
        {
            "title": "Motor / ESC Imbalance",
            "url": "https://ardupilot.org/copter/docs/motor-thrust-scaling.html",
            "text": (
                "Motor imbalance shows as one or more RCOU channels consistently "
                "higher than the others to maintain level flight. This can be "
                "caused by a damaged propeller, failing motor, or miscalibrated ESC."
            ),
            "fix_steps": [
                "Compare RCOU.C1-C4 (or C1-C8) - values should be within 5% in hover.",
                "Run ESC calibration if using analog ESCs.",
                "Swap propellers between motors to rule out prop damage.",
                "Check motor spin direction for each arm.",
                "Run MOT_SPIN_ARM test to verify all motors respond equally.",
            ],
        },
    ],
    "normal": [
        {
            "title": "No Fault Detected",
            "url": "https://ardupilot.org/copter/docs/common-logs.html",
            "text": "No significant anomaly was detected in this log.",
            "fix_steps": ["No action needed. Review log manually if unexpected behavior occurred."],
        },
    ],
}


class FixSuggestionPipeline:
    """
    Retrieval-augmented fix suggestion pipeline.

    When a vector index is available (built from rag_corpus.jsonl), it uses
    semantic similarity to retrieve the most relevant passages. When no index
    exists, it falls back to the built-in knowledge base dictionary above.
    """

    def __init__(self, use_vector_store: bool = False):
        self.use_vector_store = use_vector_store
        self._collection = None

    def build_index(self, corpus_path: Optional[str] = None) -> "FixSuggestionPipeline":
        """
        Embed the JSONL corpus and persist a ChromaDB vector collection.

        corpus_path: path to a .jsonl file where each line has:
          {"fault_type": ..., "title": ..., "url": ..., "text": ..., "fix_steps": [...]}

        Requires: chromadb, sentence-transformers (pip install chromadb sentence-transformers)
        """
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "chromadb and sentence-transformers required for vector search: "
                "pip install chromadb sentence-transformers"
            ) from exc

        corpus_path = Path(corpus_path or CORPUS_PATH)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}. See data/README.md.")

        model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self._collection = client.get_or_create_collection("ardupilot_kb")

        with open(corpus_path) as f:
            docs = [json.loads(line) for line in f if line.strip()]

        texts = [d["text"] for d in docs]
        embeddings = model.encode(texts, show_progress_bar=True).tolist()

        self._collection.upsert(
            ids=[str(i) for i in range(len(docs))],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{
                "fault_type": d.get("fault_type", ""),
                "title": d.get("title", ""),
                "url": d.get("url", ""),
            } for d in docs],
        )
        self.use_vector_store = True
        return self

    def suggest(
        self,
        fault_type: str,
        features: Optional[Dict[str, float]] = None,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Return top_k fix suggestions for the given fault type.

        Each suggestion has:
          title       - human-readable summary
          url         - link to source doc or forum thread
          fix_steps   - ordered list of recommended actions
          source      - "vector_store" or "builtin_kb"
        """
        if self.use_vector_store and self._collection is not None:
            return self._vector_suggest(fault_type, features, top_k)
        return self._builtin_suggest(fault_type, top_k)

    def _builtin_suggest(self, fault_type: str, top_k: int) -> List[Dict]:
        entries = BUILTIN_KB.get(fault_type, BUILTIN_KB.get("normal", []))
        results = []
        for entry in entries[:top_k]:
            results.append({
                "title": entry["title"],
                "url": entry["url"],
                "fix_steps": entry["fix_steps"],
                "source": "builtin_kb",
            })
        return results

    def _vector_suggest(
        self,
        fault_type: str,
        features: Optional[Dict[str, float]],
        top_k: int,
    ) -> List[Dict]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return self._builtin_suggest(fault_type, top_k)

        query = f"ArduPilot {fault_type.replace('_', ' ')} root cause fix"
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = model.encode([query]).tolist()
        results_raw = self._collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            where={"fault_type": fault_type} if fault_type != "normal" else None,
        )
        suggestions = []
        for doc, meta in zip(
            results_raw["documents"][0],
            results_raw["metadatas"][0],
        ):
            kb_entry = BUILTIN_KB.get(fault_type, [{}])[0]
            suggestions.append({
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "text_excerpt": doc[:200],
                "fix_steps": kb_entry.get("fix_steps", []),
                "source": "vector_store",
            })
        return suggestions
