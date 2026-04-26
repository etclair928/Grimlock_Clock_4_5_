#!/usr/bin/env python3
"""
engine/ — Decision and correction modules for Grimlock 4.5

The "Brain" of Grimlock — evaluates confidence and triggers deep analysis.
"""

from .confidence_router import ConfidenceRouter, ConfidenceConfig
from .deep_analysis import DeepAnalysisEngine, DeepAnalysisConfig

__all__ = [
    "ConfidenceRouter",
    "ConfidenceConfig",
    "DeepAnalysisEngine",
    "DeepAnalysisConfig",
]