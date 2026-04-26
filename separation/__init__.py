#!/usr/bin/env python3
"""
Grimlock 4.5 - Separation Package
==================================

Advanced stem separation modules for music source separation.

Available Modules:
------------------
- HybridSeparator      → Combined Demucs + BS-Roformer separation (recommended)
- DemucsEngine         → Pure Demucs (HTDemucs model) separation
- BSRoformerEngine     → Pure Band-Split Roformer separation

The HybridSeparator intelligently combines both for optimal results:
- Uses Demucs as baseline (fast, good general separation)
- Enhances with BS-Roformer (better quality for specific stems)
- Falls back gracefully if either model is unavailable

Version: 4.5.0
"""

# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = "4.5.0"
__all__ = [
    # Main Separators
    "HybridSeparator",
    "DemucsEngine",
    "BSRoformerEngine",

    # Factory Functions
    "create_separator",
    "get_hybrid_separator",
    "get_demucs_separator",
    "get_bs_roformer_separator",

    # Utilities
    "check_separation_availability",
    "get_available_models",
]

# ============================================================================
# LAZY LOADING (prevents GPU memory allocation until needed)
# ============================================================================

from typing import Optional, Dict, Any
import sys


class _LazySeparator:
    """Lazy loader for separator modules - prevents early model loading"""

    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._instance = None
        self._available = None

    def _get_module(self):
        """Import the actual module when needed"""
        if self._instance is None:
            try:
                module = __import__(
                    f"engine.separation.{self.module_name}",
                    fromlist=[self.class_name]
                )
                cls = getattr(module, self.class_name)
                self._instance = cls
                self._available = True
            except ImportError as e:
                self._available = False
                self._error = str(e)
        return self._instance

    def __call__(self, *args, **kwargs):
        cls = self._get_module()
        if cls is None:
            raise ImportError(
                f"Cannot import {self.class_name}: {getattr(self, '_error', 'Module not found')}"
            )
        return cls(*args, **kwargs)

    @property
    def available(self) -> bool:
        """Check if this separator is available"""
        self._get_module()
        return self._available


# Lazy-loaded separator classes
HybridSeparator = _LazySeparator("hybrid_separator", "HybridSeparator")
DemucsEngine = _LazySeparator("demucs", "DemucsEngine")
BSRoformerEngine = _LazySeparator("bs_roformer_engine", "BSRoformerEngine")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_separator(separator_type: str = "hybrid", **kwargs) -> Any:
    """
    Factory function to create a stem separator instance.

    Args:
        separator_type: One of "hybrid", "demucs", or "bs_roformer"
        **kwargs: Additional arguments to pass to the separator constructor

    Returns:
        Separator instance

    Example:
        # Create hybrid separator
        separator = create_separator("hybrid", device="cpu", truncate_seconds=60)

        # Create pure Demucs
        separator = create_separator("demucs", model_name="htdemucs")

        # Create BS-Roformer only
        separator = create_separator("bs_roformer")
    """
    if separator_type == "hybrid":
        return HybridSeparator(**kwargs)
    elif separator_type == "demucs":
        return DemucsEngine(**kwargs)
    elif separator_type == "bs_roformer":
        return BSRoformerEngine(**kwargs)
    else:
        raise ValueError(
            f"Unknown separator type: {separator_type}. "
            f"Available: hybrid, demucs, bs_roformer"
        )


def get_hybrid_separator(device: str = "cpu", truncate_seconds: int = 60, **kwargs) -> HybridSeparator:
    """
    Get configured HybridSeparator instance.

    Args:
        device: "cpu" or "cuda"
        truncate_seconds: Maximum audio length to process (seconds)
        **kwargs: Additional HybridSeparator arguments

    Returns:
        Configured HybridSeparator
    """
    return HybridSeparator(device=device, truncate_seconds=truncate_seconds, **kwargs)


def get_demucs_separator(model_name: str = "htdemucs", device: str = "cpu", **kwargs) -> DemucsEngine:
    """
    Get configured Demucs separator instance.

    Args:
        model_name: Demucs model name ("htdemucs", "htdemucs_ft", etc.)
        device: "cpu" or "cuda"
        **kwargs: Additional DemucsEngine arguments

    Returns:
        Configured DemucsEngine
    """
    return DemucsEngine(model_name=model_name, device=device, **kwargs)


def get_bs_roformer_separator(model_name: str = "bs_roformer", device: str = "cpu", **kwargs) -> BSRoformerEngine:
    """
    Get configured BS-Roformer separator instance.

    Args:
        model_name: BS-Roformer model name
        device: "cpu" or "cuda"
        **kwargs: Additional BSRoformerEngine arguments

    Returns:
        Configured BSRoformerEngine
    """
    return BSRoformerEngine(model_name=model_name, device=device, **kwargs)


# ============================================================================
# AVAILABILITY CHECKING
# ============================================================================

def check_separation_availability() -> Dict[str, Dict[str, Any]]:
    """
    Check which separation models are available and their dependencies.

    Returns:
        Dict with availability status for each separator

    Example:
        status = check_separation_availability()
        if status["hybrid"]["available"]:
            print("Hybrid separator ready")
    """
    results = {
        "hybrid": {
            "available": False,
            "dependencies": {"demucs": False, "bs_roformer": False},
            "error": None
        },
        "demucs": {
            "available": False,
            "dependencies": {"demucs": False, "torch": False},
            "error": None
        },
        "bs_roformer": {
            "available": False,
            "dependencies": {"bs_roformer": False, "torch": False},
            "error": None
        }
    }

    # Check Demucs
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        results["demucs"]["dependencies"]["demucs"] = True
        try:
            import torch
            results["demucs"]["dependencies"]["torch"] = True
            results["demucs"]["available"] = True
        except ImportError:
            results["demucs"]["error"] = "PyTorch not installed"
    except ImportError as e:
        results["demucs"]["error"] = str(e)

    # Check BS-Roformer
    try:
        # Try to import BS-Roformer - adjust import path as needed
        # This is a placeholder - update with actual BS-Roformer import
        from bs_roformer import BSRoformer
        results["bs_roformer"]["dependencies"]["bs_roformer"] = True
        try:
            import torch
            results["bs_roformer"]["dependencies"]["torch"] = True
            results["bs_roformer"]["available"] = True
        except ImportError:
            results["bs_roformer"]["error"] = "PyTorch not installed"
    except ImportError as e:
        results["bs_roformer"]["error"] = str(e)
        results["bs_roformer"]["available"] = False

    # Check Hybrid (requires at least Demucs)
    results["hybrid"]["available"] = results["demucs"]["available"]
    results["hybrid"]["dependencies"]["demucs"] = results["demucs"]["available"]
    results["hybrid"]["dependencies"]["bs_roformer"] = results["bs_roformer"]["available"]

    if not results["hybrid"]["available"]:
        results["hybrid"]["error"] = "Demucs is required for hybrid mode"

    return results


def get_available_models() -> List[str]:
    """
    Get list of separator models that are currently available.

    Returns:
        List of available model names
    """
    status = check_separation_availability()
    available = []

    if status["hybrid"]["available"]:
        available.append("hybrid")
    if status["demucs"]["available"]:
        available.append("demucs")
    if status["bs_roformer"]["available"]:
        available.append("bs_roformer")

    return available


# ============================================================================
# SEPARATION PIPELINE (for sequential processing)
# ============================================================================

class SeparationPipeline:
    """
    Pipeline for running multiple separators sequentially.
    Useful for comparing outputs or cascading separation.
    """

    def __init__(self, separators: Optional[List[str]] = None):
        """
        Initialize separation pipeline.

        Args:
            separators: List of separator types to use (default: ["hybrid"])
        """
        self.separators = separators or ["hybrid"]
        self._instances = {}

    def _get_separator(self, sep_type: str):
        """Lazy-load separator instance"""
        if sep_type not in self._instances:
            self._instances[sep_type] = create_separator(sep_type)
        return self._instances[sep_type]

    async def separate(self, audio_path, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Run all configured separators on the same audio.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments passed to each separator

        Returns:
            Dict mapping separator names to their stem outputs
        """
        results = {}

        for sep_type in self.separators:
            separator = self._get_separator(sep_type)
            if separator.available:
                stems = await separator.separate(audio_path, **kwargs)
                results[sep_type] = stems
            else:
                results[sep_type] = {"error": f"{sep_type} not available"}

        return results

    def get_best_stems(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Intelligently combine stems from multiple separators.
        Uses the best stem from each available separator.

        Args:
            results: Output from separate() method

        Returns:
            Combined stems picking best from each separator
        """
        combined = {}
        stem_types = ['drums', 'bass', 'vocals', 'other']

        for stem in stem_types:
            best_stem = None
            best_quality = -1

            for sep_type, stems in results.items():
                if stem in stems and stems[stem] is not None:
                    # Simple quality heuristic: higher RMS = louder = maybe better
                    quality = float(np.sqrt(np.mean(stems[stem] ** 2)))
                    if quality > best_quality:
                        best_quality = quality
                        best_stem = stems[stem]

            if best_stem is not None:
                combined[stem] = best_stem

        return combined


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_separate(audio_path: str, separator_type: str = "hybrid", **kwargs) -> Dict[str, Any]:
    """
    One-shot separation function.

    Args:
        audio_path: Path to audio file
        separator_type: Type of separator to use
        **kwargs: Additional separator arguments

    Returns:
        Dictionary of separated stems

    Example:
        stems = quick_separate("song.mp3", "hybrid", device="cpu")
    """
    separator = create_separator(separator_type, **kwargs)

    # Run separation (synchronous wrapper)
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # Already in async context
        return await separator.separate(audio_path)
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(separator.separate(audio_path))


# ============================================================================
# UTILITIES
# ============================================================================

def get_separator_info() -> Dict[str, Any]:
    """
    Get detailed information about all separators.
    """
    availability = check_separation_availability()

    return {
        "version": __version__,
        "separators": {
            "hybrid": {
                "class": "HybridSeparator",
                "file": "hybrid_separator.py",
                "description": "Combines Demucs + BS-Roformer for optimal separation",
                "available": availability["hybrid"]["available"],
                "dependencies": availability["hybrid"]["dependencies"],
                "error": availability["hybrid"].get("error")
            },
            "demucs": {
                "class": "DemucsEngine",
                "file": "demucs.py",
                "description": "Pure Demucs (HTDemucs model) separation",
                "available": availability["demucs"]["available"],
                "dependencies": availability["demucs"]["dependencies"],
                "error": availability["demucs"].get("error")
            },
            "bs_roformer": {
                "class": "BSRoformerEngine",
                "file": "bs_roformer_engine.py",
                "description": "Pure Band-Split Roformer separation",
                "available": availability["bs_roformer"]["available"],
                "dependencies": availability["bs_roformer"]["dependencies"],
                "error": availability["bs_roformer"].get("error")
            }
        }
    }


# ============================================================================
# INITIALIZATION LOGGING
# ============================================================================

def _log_separation_status():
    """Log separation availability on import (optional)"""
    available = get_available_models()

    if available:
        print(f"✅ Separation models available: {', '.join(available)}")
    else:
        print("⚠️  No separation models available. Install demucs:")
        print("   pip install demucs")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Main classes
    "HybridSeparator",
    "DemucsEngine",
    "BSRoformerEngine",

    # Pipeline
    "SeparationPipeline",

    # Factories
    "create_separator",
    "get_hybrid_separator",
    "get_demucs_separator",
    "get_bs_roformer_separator",

    # Utilities
    "check_separation_availability",
    "get_available_models",
    "quick_separate",
    "get_separator_info",
]

# Uncomment to log status
# _log_separation_status()