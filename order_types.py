#!/usr/bin/env python3
"""
order_types.py — Shared dataclasses for Grimlock 4.5

All modules import from here. No circular dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class DrumType(str, Enum):
    """Standardized drum type names."""
    KICK = "KICK"
    SNARE = "SNARE"
    RIDE = "RIDE"
    HIHAT_CLOSED = "HIHAT_CLOSED"
    HIHAT_OPEN = "HIHAT_OPEN"
    HIHAT = "HIHAT"           # Generic/NMF hi-hat
    CRASH = "CRASH"
    TOM_HIGH = "TOM_HIGH"
    TOM_LOW = "TOM_LOW"
    TOM = "TOM"               # Generic/NMF tom
    CROSS_STICK = "CROSS_STICK"
    UNKNOWN = "UNKNOWN"


class InstrumentType(str, Enum):
    """Instrument types for pitch intelligence."""
    PIANO = "PIANO"
    BASS = "BASS"
    MELODY = "MELODY"
    VOCALS = "VOCALS"
    OTHER = "OTHER"


class SourceType(str, Enum):
    """Where a detection came from."""
    NMF = "NMF"
    ONSET = "ONSET"
    BAND = "BAND"
    CROSS_STICK = "CROSS_STICK"
    DEEP_ANALYSIS = "DEEP_ANALYSIS"
    USER = "USER"
    INFERRED = "INFERRED"


class MirrorType(str, Enum):
    """Schoenberg Mirror types."""
    ZCR = "ZCR"
    TEMPORAL = "TEMPORAL"
    SPECTRAL = "SPECTRAL"
    NMF = "NMF"


# ============================================================================
# CORE DATACLASSES
# ============================================================================

@dataclass
class Note:
    """A single transcribed note (pitched instrument)."""
    pitch: int  # MIDI note number (0-127)
    start: float  # onset time in seconds
    end: float  # offset time in seconds
    velocity: int  # 0-127
    confidence: float = 0.7
    instrument: InstrumentType = InstrumentType.OTHER
    source: SourceType = SourceType.ONSET
    voice_id: int = -1  # for voice assignment (-1 = unassigned)

    # Frequency intelligence
    frequency_hz: Optional[float] = None   # raw detected frequency (Hz)
    pitch_cents: float = 0.0               # deviation from nearest semitone (cents)

    def duration(self) -> float:
        """Note duration in seconds."""
        return max(0.0, self.end - self.start)

    def to_midi_event(self) -> tuple:
        """Convert to (pitch, velocity, start, end) tuple."""
        return (self.pitch, self.velocity, self.start, self.end)

    def is_blue_note(self, cents_threshold: float = 25.0) -> bool:
        """Returns True if note deviates significantly from equal temperament."""
        return abs(self.pitch_cents) >= cents_threshold


@dataclass
class DrumHit:
    """A single detected drum hit."""
    time: float
    drum_type: DrumType
    confidence: float
    velocity: int = 80
    source: SourceType = SourceType.ONSET
    beat_position: int = -1  # 0-based position in bar
    grid_deviation_ms: float = 0.0
    is_inferred: bool = False

    @property
    def midi_note(self) -> int:
        """GM MIDI note number for this drum type."""
        mapping = {
            DrumType.KICK: 36,
            DrumType.SNARE: 38,
            DrumType.RIDE: 51,
            DrumType.HIHAT_CLOSED: 42,
            DrumType.HIHAT_OPEN: 46,
            DrumType.HIHAT: 42,       # Default to closed for generic
            DrumType.CRASH: 49,
            DrumType.TOM_HIGH: 48,
            DrumType.TOM_LOW: 45,
            DrumType.TOM: 45,         # Default to low for generic
            DrumType.CROSS_STICK: 37,
            DrumType.UNKNOWN: 36,
        }
        return mapping.get(self.drum_type, 36)


@dataclass
class RhythmInfo:
    """Temporal information from rhythm engine."""
    tempo: float = 120.0
    beat_times: List[float] = field(default_factory=list)
    downbeats: List[float] = field(default_factory=list)
    time_signature: str = "4/4"
    beats_per_bar: int = 4
    confidence: float = 0.7
    grid: List[float] = field(default_factory=list)
    tracker_source: str = "unknown"

    # Swing and Key Metadata
    swing_ratio: float = 1.0          # 1.0=straight, ~1.5=jazz swing, ~2.0=hardswing
    swing_confidence: float = 0.5
    detected_key: str = "unknown"     # e.g. "Bb Minor"
    key_confidence: float = 0.3

    def beat_position(self, time: float, tolerance_ms: float = 65.0) -> int:
        """Get 0-based beat position for a given time."""
        if not self.grid:
            return -1
        tolerance = tolerance_ms / 1000.0
        for i, grid_time in enumerate(self.grid):
            if abs(grid_time - time) <= tolerance:
                return i % self.beats_per_bar
        return -1


@dataclass
class MirrorResult:
    """Results from a single Schoenberg Mirror."""
    passed: bool
    score: float
    value: float
    reason: Optional[str] = None


@dataclass
class SchoenbergResult:
    """Aggregated results from all four mirrors."""
    zcr: MirrorResult
    temporal: MirrorResult
    spectral: MirrorResult
    nmf: MirrorResult

    def passes_all(self) -> bool:
        """True if all mirrors passed."""
        return all([self.zcr.passed, self.temporal.passed,
                    self.spectral.passed, self.nmf.passed])

    def confidence_penalty(self) -> float:
        """Calculate penalty based on failure count."""
        failed = 4 - sum([self.zcr.passed, self.temporal.passed,
                          self.spectral.passed, self.nmf.passed])
        return max(0.0, 1.0 - (failed * 0.15))


@dataclass
class TranscriptionResult:
    """Complete transcription result for a file."""
    task_id: str
    duration_seconds: float
    tempo: float
    time_signature: str
    key: str
    notes: List[Note]
    drum_hits: List[DrumHit]
    rhythm_info: RhythmInfo
    confidence_score: float = 0.0
    deep_analysis_triggered: bool = False
    warnings: List[str] = field(default_factory=list)
    success: bool = False

    @property
    def total_notes(self) -> int:
        return len(self.notes) + len(self.drum_hits)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "task_id": self.task_id,
            "duration_seconds": self.duration_seconds,
            "tempo": self.tempo,
            "time_signature": self.time_signature,
            "key": self.key,
            "detected_key": self.rhythm_info.detected_key,
            "key_confidence": self.rhythm_info.key_confidence,
            "swing_ratio": self.rhythm_info.swing_ratio,
            "swing_confidence": self.rhythm_info.swing_confidence,
            "total_notes": self.total_notes,
            "piano_notes": len([n for n in self.notes if n.instrument == InstrumentType.PIANO]),
            "bass_notes": len([n for n in self.notes if n.instrument == InstrumentType.BASS]),
            "melody_notes": len([n for n in self.notes if n.instrument == InstrumentType.MELODY]),
            "drum_hits": len(self.drum_hits),
            "confidence_score": self.confidence_score,
            "deep_analysis_triggered": self.deep_analysis_triggered,
            "warnings": self.warnings,
            "success": self.success
        }