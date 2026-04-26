#!/usr/bin/env python3
"""
Grimlock Clock 4.5 — "Fascinating Rhythm"
===========================================

A modular, multi-agent transcription system where truth emerges from consensus,
not dominance.

Core philosophy: Trust-but-verify. Every sensor is a witness. The Schoenberg
Mirror validates before separation. The Confidence Router decides when to
trust and when to listen closer.

Phase 1: Foundation — stable utilities and shared types.
"""

__version__ = "4.5.0"
__author__ = "Grimlock Development"
__status__ = "Development"

# Core type exports
from .types import Note, DrumHit, RhythmInfo, TranscriptionResult

# Version check
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("Grimlock 4.5 requires Python 3.8 or higher")