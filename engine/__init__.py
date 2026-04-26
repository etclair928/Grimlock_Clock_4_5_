#!/usr/bin/env python3
"""
Grimlock 4.5 - Engine Package
==============================

Neural Orchestration Engine for Music Transcription with:
- Hybrid stem separation (Demucs + BS-Roformer)
- Multi-path drum detection (Spectral + NMF)
- Pitch transcription with confidence scoring
- Evidence-based decision routing
- Drift-safe MIDI fusion with atomic writes

Version: 4.5.0
"""

# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = "4.5.0"
__author__ = "Grimlock Team"
__description__ = "Neural Orchestration Engine for Music Transcription"

# ============================================================================
# TYPE IMPORTS (lazy loading to avoid circular imports)
# ============================================================================

from typing import TYPE_CHECKING, Optional, Dict, List, Any, Tuple, Union
from pathlib import Path

if TYPE_CHECKING:
    from .order_types import (
        Note, DrumHit, RhythmInfo, TranscriptionResult, InstrumentType
    )
    from .confidence_router import ConfidenceRouter, Decision, ConfidenceConfig
    from .fusion_layer import GrimlockFusionLayer, FusionConfig, FusionResult
    from .pitch_intelligence import PitchIntelligence
    from .drum_intelligence import DrumIntelligence
    from .rhythm_engine import RhythmEngine
    from .hybrid_separator import HybridSeparator

# ============================================================================
# MODULE EXPORTS
# ============================================================================

# Order Types (data structures)
from .order_types import (
    Note,
    DrumHit,
    DrumType,
    RhythmInfo,
    TranscriptionResult,
    InstrumentType,
    ConfidenceLevel,
)

# Confidence Router (decision engine)
from .confidence_router import (
    ConfidenceRouter,
    Decision,
    ConfidenceConfig,
)

# Fusion Layer (MIDI assembly)
from .fusion_layer import (
    GrimlockFusionLayer,
    FusionConfig,
    FusionResult,
    NoteEvent,
    DrumEvent,
    RhythmData,
    create_fusion_layer,
)

# Core Intelligence Modules
from .pitch_intelligence import PitchIntelligence
from .drum_intelligence import DrumIntelligence
from .rhythm_engine import RhythmEngine

# Separation
from .hybrid_separator import HybridSeparator


# ============================================================================
# LAZY LOADING FOR HEAVY MODULES
# ============================================================================

def get_pitch_intelligence() -> 'PitchIntelligence':
    """Lazy load PitchIntelligence (avoids loading models until needed)"""
    from .pitch_intelligence import PitchIntelligence
    return PitchIntelligence()


def get_drum_intelligence() -> 'DrumIntelligence':
    """Lazy load DrumIntelligence"""
    from .drum_intelligence import DrumIntelligence
    return DrumIntelligence()


def get_rhythm_engine() -> 'RhythmEngine':
    """Lazy load RhythmEngine"""
    from .rhythm_engine import RhythmEngine
    return RhythmEngine()


def get_hybrid_separator() -> 'HybridSeparator':
    """Lazy load HybridSeparator"""
    from .hybrid_separator import HybridSeparator
    return HybridSeparator()


def get_confidence_router(config: Optional['ConfidenceConfig'] = None) -> 'ConfidenceRouter':
    """Lazy load ConfidenceRouter with optional config"""
    from .confidence_router import ConfidenceRouter
    return ConfidenceRouter(config)


def get_fusion_layer(config: Optional['FusionConfig'] = None) -> 'GrimlockFusionLayer':
    """Lazy load FusionLayer with optional config"""
    from .fusion_layer import create_fusion_layer
    return create_fusion_layer(
        output_dir=config.output_dir if config else "./output",
        quantization_division=config.quantization_division if config else 16,
        verbose=config.verbose if config else False
    )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary for all components"""
    return {
        "version": __version__,
        "separator": {
            "model": "htdemucs",
            "use_bs_roformer": True,
            "truncate_seconds": 60,
            "device": "cpu"
        },
        "pitch": {
            "basic_pitch_model": "default",
            "crepe_model": "tiny",
            "spice_enabled": True,
            "min_confidence": 0.35
        },
        "drums": {
            "nmf_enabled": True,
            "nmf_components": 5,
            "multiband_enabled": True,
            "min_confidence": 0.3
        },
        "rhythm": {
            "use_madmom": True,
            "min_bpm": 40,
            "max_bpm": 250,
            "swing_detection": True
        },
        "fusion": {
            "quantization_division": 16,
            "atomic_writes": True,
            "drift_tolerance_ms": 10,
            "min_note_duration_ms": 20
        },
        "routing": {
            "accept_threshold": 0.85,
            "deep_threshold": 0.60,
            "retry_threshold": 0.50
        }
    }


def validate_pipeline_ready() -> Dict[str, bool]:
    """
    Check if all required components are available.
    Returns dict with component availability status.
    """
    status = {
        "version": __version__,
        "hybrid_separator": False,
        "pitch_intelligence": False,
        "drum_intelligence": False,
        "rhythm_engine": False,
        "confidence_router": True,  # Always available
        "fusion_layer": True,  # Always available
    }

    # Check HybridSeparator
    try:
        from .hybrid_separator import HybridSeparator
        status["hybrid_separator"] = True
    except ImportError:
        pass

    # Check PitchIntelligence
    try:
        from .pitch_intelligence import PitchIntelligence
        status["pitch_intelligence"] = True
    except ImportError:
        pass

    # Check DrumIntelligence
    try:
        from .drum_intelligence import DrumIntelligence
        status["drum_intelligence"] = True
    except ImportError:
        pass

    # Check RhythmEngine
    try:
        from .rhythm_engine import RhythmEngine
        status["rhythm_engine"] = True
    except ImportError:
        pass

    return status


# ============================================================================
# GRIMLOCK PIPELINE INTEGRATION
# ============================================================================

class GrimlockEngine:
    """
    Unified interface for the Grimlock transcription engine.
    Provides a single entry point for all functionality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or create_default_config()
        self._separator = None
        self._pitch = None
        self._drums = None
        self._rhythm = None
        self._router = None
        self._fusion = None

    @property
    def separator(self):
        """Lazy-loaded separator"""
        if self._separator is None:
            self._separator = get_hybrid_separator()
        return self._separator

    @property
    def pitch_intelligence(self):
        """Lazy-loaded pitch intelligence"""
        if self._pitch is None:
            self._pitch = get_pitch_intelligence()
        return self._pitch

    @property
    def drum_intelligence(self):
        """Lazy-loaded drum intelligence"""
        if self._drums is None:
            self._drums = get_drum_intelligence()
        return self._drums

    @property
    def rhythm_engine(self):
        """Lazy-loaded rhythm engine"""
        if self._rhythm is None:
            self._rhythm = get_rhythm_engine()
        return self._rhythm

    @property
    def confidence_router(self):
        """Lazy-loaded confidence router"""
        if self._router is None:
            self._router = get_confidence_router()
        return self._router

    @property
    def fusion_layer(self):
        """Lazy-loaded fusion layer"""
        if self._fusion is None:
            fusion_config = FusionConfig(
                output_dir=self.config.get("output_dir", "./output"),
                quantization_division=self.config["fusion"]["quantization_division"],
                atomic_writes=self.config["fusion"]["atomic_writes"],
                verbose=self.config.get("verbose", False)
            )
            self._fusion = GrimlockFusionLayer(fusion_config)
        return self._fusion

    async def transcribe(self, audio_path: Path, task_id: str, duration: float) -> 'TranscriptionResult':
        """
        Complete transcription pipeline entry point.

        Args:
            audio_path: Path to audio file
            task_id: Unique task identifier
            duration: Audio duration in seconds

        Returns:
            TranscriptionResult with all data
        """
        # Step 1: Separate stems
        stems = await self.separator.separate(audio_path, truncate_seconds=min(duration, 60))

        # Step 2: Extract rhythm
        rhythm_info = await self.rhythm_engine.analyze(stems, duration)

        # Step 3: Detect drums
        drum_hits = await self.drum_intelligence.analyze(stems.get('drums'), rhythm_info)

        # Step 4: Detect pitched instruments
        pitch_notes = await self.pitch_intelligence.analyze(stems, duration)

        # Step 5: Routing decision
        decision = self.confidence_router.evaluate(
            drum_hits=drum_hits,
            notes=pitch_notes,
            rhythm_info=rhythm_info,
            duration=duration
        )

        # Step 6: Fusion
        result = await self.fusion_layer.fuse(
            notes=pitch_notes,
            drum_hits=drum_hits,
            rhythm_info=rhythm_info,
            task_id=task_id,
            duration_seconds=duration,
            decision=decision.value if decision else None
        )

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            "version": __version__,
            "config": self.config,
            "components": validate_pipeline_ready()
        }


# ============================================================================
# PACKAGE METADATA
# ============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__description__",

    # Types
    "Note",
    "DrumHit",
    "DrumType",
    "RhythmInfo",
    "TranscriptionResult",
    "InstrumentType",
    "ConfidenceLevel",

    # Core Components
    "PitchIntelligence",
    "DrumIntelligence",
    "RhythmEngine",
    "HybridSeparator",
    "ConfidenceRouter",
    "GrimlockFusionLayer",

    # Config
    "ConfidenceConfig",
    "FusionConfig",
    "FusionResult",
    "NoteEvent",
    "DrumEvent",
    "RhythmData",

    # Decision
    "Decision",

    # Factory Functions
    "create_fusion_layer",
    "get_pitch_intelligence",
    "get_drum_intelligence",
    "get_rhythm_engine",
    "get_hybrid_separator",
    "get_confidence_router",
    "get_fusion_layer",

    # Helpers
    "create_default_config",
    "validate_pipeline_ready",

    # Unified Interface
    "GrimlockEngine",
]


# ============================================================================
# INITIALIZATION LOGGING
# ============================================================================

def _print_banner():
    """Print Grimlock banner on import"""
    banner = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    GRIMLOCK 4.5 ENGINE                           ║
║              Neural Orchestration for Music Transcription        ║
║                                                                  ║
║  Version: {__version__:<46}║
║  Features:                                                       ║
║    • Hybrid Stem Separation (Demucs + BS-Roformer)              ║
║    • Multi-path Drum Detection (Spectral + NMF)                  ║
║    • Confidence-weighted Pitch Transcription                     ║
║    • Evidence-based Decision Routing                             ║
║    • Drift-safe MIDI Fusion                                      ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)

# Uncomment to show banner on import
# _print_banner()