#!/usr/bin/env python3
"""
Grimlock 4.5 - Modules Package
===============================

Core intelligence modules for the Grimlock transcription engine.

Available Modules:
------------------
- DrumIntelligence      → Multi-path drum detection (spectral + NMF)
- PitchIntelligence     → Pitch transcription with confidence scoring
- RhythmEngine          → Tempo and beat tracking
- MadmomTracker         → Madmom-based beat/downbeat tracking
- LibrosaTracker        → Librosa fallback beat tracking
- SchoenbergMirror      → Music theory / key detection

Each module is designed to work independently and be testable in isolation.
"""

# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = "4.5.0"
__all__ = [
    # Core Intelligence
    "DrumIntelligence",
    "PitchIntelligence",
    "RhythmEngine",

    # Trackers
    "MadmomTracker",
    "LibrosaTracker",

    # Music Theory
    "SchoenbergMirror",

    # Factory Functions
    "get_drum_intelligence",
    "get_pitch_intelligence",
    "get_rhythm_engine",
    "get_beat_tracker",
    "get_key_detector",

    # Type Checking
    "check_module_availability",
]

# ============================================================================
# LAZY LOADING (prevents circular imports and reduces startup time)
# ============================================================================

from typing import Optional, Dict, Any
import sys


def __lazy_import(module_name: str, class_name: str):
    """
    Create a lazy loader for a module class.
    Only imports the module when the class is first accessed.
    """

    def loader(*args, **kwargs):
        module = __import__(f"engine.modules.{module_name}", fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return loader


# ============================================================================
# CORE MODULE EXPORTS (lazy loaded)
# ============================================================================

class _LazyLoader:
    """Helper class for lazy loading module attributes"""

    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._instance = None

    def __call__(self, *args, **kwargs):
        if self._instance is None:
            module = __import__(f"engine.modules.{self.module_name}", fromlist=[self.class_name])
            cls = getattr(module, self.class_name)
            self._instance = cls(*args, **kwargs)
        return self._instance


# Declare lazy-loaded classes
DrumIntelligence = _LazyLoader("drum_intelligence", "DrumIntelligence")
PitchIntelligence = _LazyLoader("pitch_intelligence", "PitchIntelligence")
RhythmEngine = _LazyLoader("rhythm_engine", "RhythmEngine")
MadmomTracker = _LazyLoader("madmom", "MadmomTracker")
LibrosaTracker = _LazyLoader("librosa_tracker", "LibrosaTracker")
SchoenbergMirror = _LazyLoader("schoenberg_mirror", "SchoenbergMirror")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def get_drum_intelligence(sr: int = 22050, use_nmf: bool = True) -> DrumIntelligence:
    """
    Get configured DrumIntelligence instance.

    Args:
        sr: Sample rate for audio processing
        use_nmf: Enable NMF-based drum detection (requires scikit-learn)

    Returns:
        Configured DrumIntelligence instance
    """
    return DrumIntelligence(sr=sr, use_nmf=use_nmf)


def get_pitch_intelligence(use_crepe: bool = True, use_spice: bool = True) -> PitchIntelligence:
    """
    Get configured PitchIntelligence instance.

    Args:
        use_crepe: Enable CREPE for melody transcription
        use_spice: Enable SPICE for polyphonic transcription

    Returns:
        Configured PitchIntelligence instance
    """
    return PitchIntelligence(use_crepe=use_crepe, use_spice=use_spice)


def get_rhythm_engine(use_madmom: bool = True) -> RhythmEngine:
    """
    Get configured RhythmEngine instance.

    Args:
        use_madmom: Use madmom for tracking (fallback to librosa if False/unavailable)

    Returns:
        Configured RhythmEngine instance
    """
    return RhythmEngine(use_madmom=use_madmom)


def get_beat_tracker(tracker_type: str = "madmom") -> Any:
    """
    Get specific beat tracker instance.

    Args:
        tracker_type: "madmom" or "librosa"

    Returns:
        Beat tracker instance
    """
    if tracker_type.lower() == "madmom":
        return MadmomTracker()
    else:
        return LibrosaTracker()


def get_key_detector(method: str = "schoenberg") -> SchoenbergMirror:
    """
    Get key detection module.

    Args:
        method: Detection method (currently only "schoenberg" supported)

    Returns:
        SchoenbergMirror instance
    """
    return SchoenbergMirror()


# ============================================================================
# MODULE AVAILABILITY CHECK
# ============================================================================

def check_module_availability(module_name: str) -> Dict[str, Any]:
    """
    Check if a specific module and its dependencies are available.

    Args:
        module_name: "drum", "pitch", "rhythm", "madmom", "librosa", "schoenberg"

    Returns:
        Dict with availability status and dependency information
    """
    result = {
        "module": module_name,
        "available": False,
        "dependencies": {},
        "error": None
    }

    if module_name == "drum":
        try:
            from engine.modules.drum_intelligence import DrumIntelligence
            result["available"] = True
            # Check NMF availability
            try:
                from sklearn.decomposition import NMF
                result["dependencies"]["nmf"] = True
            except ImportError:
                result["dependencies"]["nmf"] = False
        except ImportError as e:
            result["error"] = str(e)

    elif module_name == "pitch":
        try:
            from engine.modules.pitch_intelligence import PitchIntelligence
            result["available"] = True
            # Check CREPE
            try:
                import crepe
                result["dependencies"]["crepe"] = True
            except ImportError:
                result["dependencies"]["crepe"] = False
            # Check SPICE (TF Hub)
            try:
                import tensorflow_hub as hub
                result["dependencies"]["spice"] = True
            except ImportError:
                result["dependencies"]["spice"] = False
        except ImportError as e:
            result["error"] = str(e)

    elif module_name == "rhythm":
        try:
            from engine.modules.rhythm_engine import RhythmEngine
            result["available"] = True
            # Check madmom
            try:
                import madmom
                result["dependencies"]["madmom"] = True
            except ImportError:
                result["dependencies"]["madmom"] = False
        except ImportError as e:
            result["error"] = str(e)

    elif module_name == "madmom":
        try:
            from engine.modules.madmom import MadmomTracker
            result["available"] = True
            import madmom
            result["dependencies"]["madmom"] = True
        except ImportError as e:
            result["error"] = str(e)
            result["dependencies"]["madmom"] = False

    elif module_name == "librosa":
        try:
            from engine.modules.librosa_tracker import LibrosaTracker
            result["available"] = True
            import librosa
            result["dependencies"]["librosa"] = True
        except ImportError as e:
            result["error"] = str(e)
            result["dependencies"]["librosa"] = False

    elif module_name == "schoenberg":
        try:
            from engine.modules.schoenberg_mirror import SchoenbergMirror
            result["available"] = True
            result["dependencies"]["numpy"] = True
        except ImportError as e:
            result["error"] = str(e)

    return result


def check_all_modules() -> Dict[str, bool]:
    """
    Check availability of all core modules.

    Returns:
        Dict mapping module names to availability status
    """
    modules = ["drum", "pitch", "rhythm", "madmom", "librosa", "schoenberg"]
    return {mod: check_module_availability(mod)["available"] for mod in modules}


# ============================================================================
# PIPELINE INTEGRATION
# ============================================================================

class ModulePipeline:
    """
    Simplified pipeline using the modules directory components.
    This is a lightweight alternative to the full engine pipeline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize modules based on config
        self.drum_intelligence = get_drum_intelligence(
            sr=self.config.get("sample_rate", 22050),
            use_nmf=self.config.get("use_nmf", True)
        )

        self.pitch_intelligence = get_pitch_intelligence(
            use_crepe=self.config.get("use_crepe", True),
            use_spice=self.config.get("use_spice", True)
        )

        self.rhythm_engine = get_rhythm_engine(
            use_madmom=self.config.get("use_madmom", True)
        )

        self.key_detector = get_key_detector()

    async def process(self, stems: Dict[str, Any], duration: float, task_id: str) -> Dict[str, Any]:
        """
        Process stems through all modules.

        Args:
            stems: Dictionary of stem audio arrays ('drums', 'bass', 'other', 'vocals')
            duration: Audio duration in seconds
            task_id: Unique task identifier

        Returns:
            Dictionary with all transcription results
        """
        results = {
            "task_id": task_id,
            "duration": duration,
            "rhythm": None,
            "drums": None,
            "pitch": {
                "piano_notes": [],
                "bass_notes": [],
                "melody_notes": []
            },
            "key": "Cm",
            "confidence": 0.0
        }

        # 1. Rhythm analysis
        if self.rhythm_engine:
            rhythm_info = await self.rhythm_engine.analyze(stems, duration)
            results["rhythm"] = rhythm_info

        # 2. Drum detection
        if self.drum_intelligence and "drums" in stems:
            drum_hits, drum_rhythm = await self.drum_intelligence.classify(
                stems["drums"],
                sr=self.config.get("sample_rate", 22050),
                task_id=task_id
            )
            results["drums"] = drum_hits
            if drum_rhythm and not results["rhythm"]:
                results["rhythm"] = drum_rhythm

        # 3. Pitch transcription
        if self.pitch_intelligence:
            # Piano (stems['other'])
            if "other" in stems:
                pitch_notes = await self.pitch_intelligence.analyze_piano(stems["other"])
                results["pitch"]["piano_notes"] = pitch_notes

            # Bass (stems['bass'])
            if "bass" in stems:
                bass_notes = await self.pitch_intelligence.analyze_bass(stems["bass"])
                results["pitch"]["bass_notes"] = bass_notes

            # Melody (stems['vocals'])
            if "vocals" in stems:
                melody_notes = await self.pitch_intelligence.analyze_melody(stems["vocals"])
                results["pitch"]["melody_notes"] = melody_notes

        # 4. Key detection
        if self.key_detector:
            all_pitch_notes = (
                    results["pitch"]["piano_notes"] +
                    results["pitch"]["bass_notes"] +
                    results["pitch"]["melody_notes"]
            )
            if all_pitch_notes:
                results["key"] = self.key_detector.detect_key(all_pitch_notes)

        # 5. Overall confidence
        total_notes = sum(len(v) for v in results["pitch"].values())
        drum_count = len(results["drums"]) if results["drums"] else 0
        results["confidence"] = min(1.0, (total_notes + drum_count) / (duration * 10))

        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_module_info() -> Dict[str, Any]:
    """
    Get information about all available modules.
    """
    return {
        "version": __version__,
        "modules": {
            "drum_intelligence": {
                "file": "drum_intelligence.py",
                "class": "DrumIntelligence",
                "dependencies": ["numpy", "scipy", "librosa", "sklearn (optional)"]
            },
            "pitch_intelligence": {
                "file": "pitch_intelligence.py",
                "class": "PitchIntelligence",
                "dependencies": ["basic_pitch", "crepe", "tensorflow_hub"]
            },
            "rhythm_engine": {
                "file": "rhythm_engine.py",
                "class": "RhythmEngine",
                "dependencies": ["librosa", "madmom (optional)"]
            },
            "madmom": {
                "file": "madmom.py",
                "class": "MadmomTracker",
                "dependencies": ["madmom"]
            },
            "librosa_tracker": {
                "file": "librosa_tracker.py",
                "class": "LibrosaTracker",
                "dependencies": ["librosa"]
            },
            "schoenberg_mirror": {
                "file": "schoenberg_mirror.py",
                "class": "SchoenbergMirror",
                "dependencies": ["numpy"]
            }
        }
    }


# ============================================================================
# INITIALIZATION LOGGING
# ============================================================================

def _log_availability():
    """Log module availability on import (optional)"""
    status = check_all_modules()
    available = [k for k, v in status.items() if v]
    missing = [k for k, v in status.items() if not v]

    if available:
        print(f"✅ Modules available: {', '.join(available)}")
    if missing:
        print(f"⚠️  Modules missing: {', '.join(missing)}")

# Uncomment to log availability
# _log_availability()