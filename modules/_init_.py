#!/usr/bin/env python3
"""
modules/ — Analysis modules for Grimlock 4.5

Each module is independent and can be tested separately.
"""

from .schoenberg_mirror import (
    SchoenbergMirror,
    MirrorConfig,
    ZCRMirror,
    TemporalMirror,
    SpectralMirror,
    NMFMirror,
)

from .drum_intelligence import (
    DrumIntelligence,
    DrumConfig,
    NMFDrumDetector,
    OnsetDrumDetector,
    CrossStickDetector,
    DrumClassifier,
)

__all__ = [
    # Schoenberg Mirror
    "SchoenbergMirror",
    "MirrorConfig",
    "ZCRMirror",
    "TemporalMirror",
    "SpectralMirror",
    "NMFMirror",
    # Drum Intelligence
    "DrumIntelligence",
    "DrumConfig",
    "NMFDrumDetector",
    "OnsetDrumDetector",
    "CrossStickDetector",
    "DrumClassifier",
]