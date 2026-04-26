#!/usr/bin/env python3
"""
engine/state_manager.py — The Enhanced Musical Blackboard
Centralized state, evidence reconciliation, and resource management for Grimlock 4.5.
"""

import time
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict


@dataclass
class Evidence:
    """Single piece of evidence with temporal and confidence context."""
    key: str
    value: Any
    confidence: float
    source: str
    start_time: float
    end_time: float
    timestamp: float = field(default_factory=time.time)
    ttl: float = 10.0

    def is_stale(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    def current_confidence(self) -> float:
        """Applies linear decay to confidence as evidence ages."""
        if self.is_stale():
            return 0.0
        age = time.time() - self.timestamp
        return self.confidence * (1.0 - (age / self.ttl))


@dataclass
class MusicalContext:
    """Calculated 'Global Truth' derived from evidence."""
    tempo: float = 120.0
    tempo_confidence: float = 0.0
    swing_ratio: float = 0.0
    time_sig: Tuple[int, int] = (4, 4)
    key_center: str = "C"
    mode: str = "major"
    feel: str = "straight"
    active_instruments: set = field(default_factory=set)


class StateManager:
    """The Brain of Grimlock. Orchestrates evidence, manages resources."""

    def __init__(self, task_id: str, guided_mode: bool = False):
        self.task_id = task_id
        self.guided_mode = guided_mode

        self._evidence: Dict[str, List[Evidence]] = defaultdict(list)
        self._locks: set = set()
        self._context = MusicalContext()
        self._observers: List[Callable] = []

        self._polyrhythms: List[Dict] = []
        self._phrase_boundaries: List[float] = []

        self._models: Dict[str, Any] = {}
        self._model_lock = asyncio.Lock()

    # ─────────────────────────────────────────────────────────
    # Evidence Management
    # ─────────────────────────────────────────────────────────

    def add_evidence(self, key: str, value: Any, confidence: float,
                     source: str, start_time: float = 0.0,
                     end_time: Optional[float] = None):
        """Injects new data into the blackboard."""
        if key in self._locks:
            return

        ev = Evidence(
            key=key, value=value, confidence=confidence, source=source,
            start_time=start_time, end_time=end_time or (start_time + 0.5)
        )

        self._evidence[key].append(ev)
        self._clean_stale_evidence()
        self._reconcile(key)
        self._notify_observers()

    def _clean_stale_evidence(self):
        """Remove expired evidence to prevent memory bloat."""
        for key in list(self._evidence.keys()):
            self._evidence[key] = [e for e in self._evidence[key] if not e.is_stale()]
            if not self._evidence[key]:
                del self._evidence[key]

    def _reconcile(self, key: str):
        """Dispatches to specific jazz-logic resolvers."""
        ev_list = [e for e in self._evidence[key] if not e.is_stale()]
        if not ev_list:
            return

        if key == "tempo":
            self._reconcile_tempo(ev_list)
        elif key == "key":
            self._reconcile_key(ev_list)
        elif key == "swing":
            self._reconcile_swing(ev_list)
        elif key == "instrument":
            self._context.active_instruments.update([e.value for e in ev_list])

    def _reconcile_tempo(self, ev_list: List[Evidence]):
        """Detects 3:2 polyrhythms (hemiola) common in modern jazz."""
        tempos = np.array([e.value for e in ev_list])
        confs = np.array([e.current_confidence() for e in ev_list])

        if len(tempos) > 1:
            ratio = np.max(tempos) / np.min(tempos)
            if 1.45 < ratio < 1.55:
                self._polyrhythms.append({"type": "3:2", "base": np.min(tempos)})

        self._context.tempo = float(np.average(tempos, weights=confs))
        self._context.tempo_confidence = float(np.mean(confs))
        self._infer_feel()

    def _reconcile_swing(self, ev_list: List[Evidence]):
        """Calculate swing ratio from evidence."""
        ratios = [e.value for e in ev_list if isinstance(e.value, (int, float))]
        if ratios:
            confs = [e.current_confidence() for e in ev_list]
            self._context.swing_ratio = float(np.average(ratios, weights=confs))
            self._infer_feel()

    def _reconcile_key(self, ev_list: List[Evidence]):
        """Uses Circle of Fifths proximity to stabilize key detection."""
        circle = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
        weights = defaultdict(float)

        for e in ev_list:
            raw_key = e.value

            # Parse root note
            if isinstance(raw_key, str):
                if len(raw_key) > 1 and raw_key[1] in ['b', '#']:
                    root = raw_key[:2]
                else:
                    root = raw_key[0]

                if 'm' in raw_key and 'major' not in raw_key:
                    self._context.mode = 'minor'
                else:
                    self._context.mode = 'major'

                if root in circle:
                    weights[root] += e.current_confidence()

        if weights:
            self._context.key_center = max(weights, key=weights.get)

    def _infer_feel(self):
        """Automatically detect musical feel from evidence."""
        has_ride = 'ride_cymbal' in self._context.active_instruments

        if self._context.swing_ratio > 0.6:
            self._context.feel = "heavy_swing"
        elif self._context.swing_ratio > 0.3:
            self._context.feel = "swing"
        elif self._context.swing_ratio < 0.1:
            self._context.feel = "straight"

        if 'brush' in self._context.active_instruments:
            self._context.feel = "ballad"

    # ─────────────────────────────────────────────────────────
    # Temporal Queries (for Deep Analysis)
    # ─────────────────────────────────────────────────────────

    def get_evidence_at_time(self, key: str, time_sec: float) -> List[Evidence]:
        """Get evidence active at a specific timestamp."""
        active = []
        for e in self._evidence.get(key, []):
            if e.start_time <= time_sec <= e.end_time:
                active.append(e)
        return active

    def get_problem_segments(self, threshold: float = 0.5) -> List[Tuple[float, float]]:
        """Identify low-confidence segments for deep analysis."""
        segments = []
        for ev_list in self._evidence.values():
            for e in ev_list:
                if e.current_confidence() < threshold:
                    segments.append((e.start_time, e.end_time))

        if not segments:
            return []

        segments.sort()
        merged = [list(segments[0])]
        for seg in segments[1:]:
            if seg[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], seg[1])
            else:
                merged.append(list(seg))

        return [(s, e) for s, e in merged]

    # ─────────────────────────────────────────────────────────
    # Resource Management
    # ─────────────────────────────────────────────────────────

    async def get_model(self, model_id: str, loader_func: Callable):
        """Ensures heavy AI models are only loaded once."""
        async with self._model_lock:
            if model_id not in self._models:
                print(f"📦 [State] Loading {model_id}...")
                self._models[model_id] = loader_func()
            return self._models[model_id]

    # ─────────────────────────────────────────────────────────
    # Guided Mode & UI Hooks
    # ─────────────────────────────────────────────────────────

    def lock_parameter(self, key: str, value: Any):
        """Allows Guided Mode to override all AI detection."""
        if not self.guided_mode:
            return
        self._locks.add(key)
        setattr(self._context, key, value)
        print(f"🔒 [State] Locked {key} to {value}")

    def get_snapshot(self) -> Dict:
        """Returns current 'Musical Truth' for UI/Fusion layer."""
        return {
            "tempo": round(self._context.tempo, 2),
            "key": f"{self._context.key_center} {self._context.mode}",
            "feel": self._context.feel,
            "swing_ratio": round(self._context.swing_ratio, 3),
            "instruments": list(self._context.active_instruments),
            "polyrhythm_detected": len(self._polyrhythms) > 0,
            "active_evidence_count": sum(len(v) for v in self._evidence.values())
        }

    def _notify_observers(self):
        for callback in self._observers:
            callback(self.get_snapshot())

    def subscribe(self, callback: Callable):
        self._observers.append(callback)