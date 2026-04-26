#!/usr/bin/env python3
"""
music_box.py — Forensic Logging System

Records every decision the engine makes for debugging and analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class MusicBoxEntry:
    """Single entry in the music box log."""
    timestamp: str
    event_type: str
    data: Dict[str, Any]


class MusicBox:
    """
    Forensic logger for transcription decisions.

    Records:
    - Every detection (time, type, confidence, source)
    - Every mirror result
    - Every consensus vote
    - Every grid snap decision
    """

    def __init__(self, task_id: str, output_dir: Path = Path("./music_box")):
        self.task_id = task_id
        self.output_dir = output_dir
        self.entries: List[MusicBoxEntry] = []
        self.start_time = datetime.now()

        output_dir.mkdir(exist_ok=True)

    def log_detection(self, time: float, drum_type: str, confidence: float,
                      source: str, features: Dict = None):
        """Log a drum detection event."""
        self.entries.append(MusicBoxEntry(
            timestamp=datetime.now().isoformat(),
            event_type="detection",
            data={
                "time": time,
                "type": drum_type,
                "confidence": confidence,
                "source": source,
                "features": features or {}
            }
        ))

    def log_mirror(self, time: float, mirror_name: str, passed: bool,
                   score: float, value: Any):
        """Log a Schoenberg mirror result."""
        self.entries.append(MusicBoxEntry(
            timestamp=datetime.now().isoformat(),
            event_type="mirror",
            data={
                "time": time,
                "mirror": mirror_name,
                "passed": passed,
                "score": score,
                "value": value
            }
        ))

    def log_consensus(self, time: float, final_type: str, confidence: float,
                      votes: Dict[str, float]):
        """Log a consensus vote result."""
        self.entries.append(MusicBoxEntry(
            timestamp=datetime.now().isoformat(),
            event_type="consensus",
            data={
                "time": time,
                "final_type": final_type,
                "confidence": confidence,
                "votes": votes
            }
        ))

    def log_grid_snap(self, time: float, snapped_time: float,
                      penalty: float, beat_position: int):
        """Log a grid snap decision."""
        self.entries.append(MusicBoxEntry(
            timestamp=datetime.now().isoformat(),
            event_type="grid_snap",
            data={
                "original_time": time,
                "snapped_time": snapped_time,
                "confidence_penalty": penalty,
                "beat_position": beat_position
            }
        ))

    def log_decision(self, decision: str, confidence: float, reason: str):
        """Log a confidence router decision."""
        self.entries.append(MusicBoxEntry(
            timestamp=datetime.now().isoformat(),
            event_type="decision",
            data={
                "decision": decision,
                "confidence": confidence,
                "reason": reason
            }
        ))

    def save(self) -> Path:
        """Save all entries to JSON file."""
        output_path = self.output_dir / f"musicbox_{self.task_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            "task_id": self.task_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_entries": len(self.entries),
            "entries": [
                {
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "data": e.data
                }
                for e in self.entries
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"📝 Music Box saved: {output_path}")
        return output_path


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

class MusicBoxIntegration:
    """Helper to integrate Music Box into the pipeline."""

    def __init__(self, pipeline, task_id: str):
        self.pipeline = pipeline
        self.music_box = MusicBox(task_id)

    def log_pipeline_event(self, stage: str, data: Dict):
        """Log a pipeline stage event."""
        self.music_box.entries.append(MusicBoxEntry(
            timestamp=datetime.now().isoformat(),
            event_type=f"pipeline_{stage}",
            data=data
        ))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.music_box.save()